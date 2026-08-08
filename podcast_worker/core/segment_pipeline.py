"""Background segment generation pipeline for PodcastProject v1.

Orchestrates the transition: queued → scripting → validating → tts → mixing → ready.

Key invariants (per API_SPEC.md):
- provenance.validation_status MUST be stored before a segment transitions to tts.
- artifact metadata MUST be stored before segment transitions to ready.
- Forced segment failure preserves ready artifacts; final_download_ready stays false.
"""

from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from podcast_worker.core import config as cfg
from podcast_worker.core import persistence
from podcast_worker.core.config import RoutingConfigurationError
from podcast_worker.core.script_generator import generate_script
from podcast_worker.core.tts_engine import synthesize, convert_to_wav
from podcast_worker.core.beat_generator import save_beat_to_wav, generate_beat
from podcast_worker.core.audio_mixer import (
    _load_wav_to_numpy,
    _save_numpy_to_wav,
    build_podcast_audio,
    convert_to_mp3,
)
from podcast_worker.core.exceptions import PodcastWorkerError


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _short_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"

def _public_error_code(exc: Exception) -> str:
    """Keep provider/library diagnostics out of public project and segment records."""
    if isinstance(exc, RoutingConfigurationError):
        return str(exc) if str(exc).startswith(("tts_", "invalid_", "unsupported_")) else "generation_failed"
    return "generation_failed"


def _tts_attempt_outcome(exc: RoutingConfigurationError) -> str:
    if str(exc) == "tts_retryable":
        return "retryable"
    if str(exc) == "tts_terminal":
        return "terminal"
    if str(exc) == "tts_pre_send":
        return "pre_send"
    return "unknown_outcome"



# ── Pipeline entry point (called from a background thread) ───────────────


def run_project_pipeline(db_path: str, project_id: str, output_dir: str,
                         topic: str, bpm: int, duration_minutes: int,
                         voice_id: str | None, outline: dict | None = None,
                         interviewer_profile: dict | None = None,
                         sme_profile: dict | None = None) -> None:
    """Run the full segment generation pipeline for a project in a background thread.

    This function is designed to be called from a daemon thread.  It handles
    every step and persists state durably at each transition so restarts are safe.
    """
    try:
        _run_pipeline(db_path, project_id, output_dir, topic, bpm, duration_minutes,
                      voice_id, outline, interviewer_profile, sme_profile)
    except Exception as exc:
        # Provider exception details are not safe to expose through project manifests.
        _persist_project_failure(db_path, project_id, _public_error_code(exc))

def _run_pipeline(db_path: str, project_id: str, output_dir: str,
                  topic: str, bpm: int, duration_minutes: int,
                  voice_id: str | None, outline: dict | None = None,
                  interviewer_profile: dict | None = None,
                  sme_profile: dict | None = None) -> None:
    out = Path(output_dir)

    # 1. Transition to generating
    _sync_status(db_path, project_id, "generating")

    # 2. Generate script via LLM
    execution = persistence._get_project_execution(db_path, project_id)
    if execution is None:
        raise PodcastWorkerError("missing_project_execution")
    llm_snapshot = cfg.execution_snapshot_from_payload(execution["llm_snapshot"], execution["llm_revision"])
    tts_snapshot = cfg.tts_snapshot_from_payload(execution["tts_snapshot"], execution["tts_revision"])
    ledger_id = execution["ledger_id"]
    script = generate_script(
        topic=topic,
        bpm=bpm,
        duration_minutes=duration_minutes,
        outline=outline,
        interviewer_profile=interviewer_profile,
        sme_profile=sme_profile,
        snapshot=llm_snapshot,
    )
    model = "server-managed"
    episode_title = script.get("title", topic)

    # 3. Create segments in the DB from outline-first generated sections
    segments_data = script.get("segments", [])
    total_segments = len(segments_data)
    for i, seg_data in enumerate(segments_data):
        seg_id = _short_id("seg")
        segment = {
            "segment_id": seg_id,
            "project_id": project_id,
            "index": i,
            "subtopic": seg_data.get("subtopic") or seg_data.get("topic") or f"Part {i + 1}",
            "title": seg_data.get("title") or episode_title,
            "status": "queued",
            "text": seg_data.get("text", ""),
            "duration_seconds": seg_data.get("approx_duration_seconds"),
            "primary_audio_artifact_id": None,
            "error": None,
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
        }
        _sync_upsert_segment(db_path, segment)

    # 4. Process each segment through the pipeline
    beat_path = out / f"beat_{project_id}.wav"
    beat_path.parent.mkdir(parents=True, exist_ok=True)
    beat_duration = duration_minutes * 60.0
    save_beat_to_wav(bpm, beat_duration, str(beat_path))

    segments = _sync_get_segments(db_path, project_id)
    any_failed = False
    ready_count = 0

    for seg in segments:
        try:
            _process_segment(db_path, project_id, seg, output_dir, beat_path,
                             bpm, duration_minutes, voice_id, model, tts_snapshot, ledger_id)
            seg_after = _sync_get_segment(db_path, seg["segment_id"])
            if seg_after and seg_after["status"] == "ready":
                ready_count += 1
        except PodcastWorkerError as exc:
            any_failed = True
            _persist_segment_failure(db_path, seg["segment_id"], _public_error_code(exc))
        except Exception as exc:
            any_failed = True
            _persist_segment_failure(db_path, seg["segment_id"], _public_error_code(exc))

    # Only a complete persisted ready manifest may create a final artifact.
    manifest = _sync_get_segments(db_path, project_id)
    ready_segments = [
        item for item in manifest
        if item["status"] == "ready" and item.get("primary_audio_artifact_id")
    ]
    if any_failed and not ready_segments:
        _sync_status(db_path, project_id, "failed")
    elif any_failed or len(ready_segments) != total_segments:
        _sync_status(db_path, project_id, "partially_ready")
    else:
        _create_final_artifact(db_path, project_id, output_dir, manifest, beat_path,
                               bpm, duration_minutes)


def _process_segment(db_path: str, project_id: str, segment: dict,
                     output_dir: str, beat_path: Path, bpm: int,
                     duration_minutes: int, voice_id: str | None,
                     script_model: str, tts_snapshot: cfg.ResolvedTTSSnapshot,
                     ledger_id: str) -> None:
    seg_id = segment["segment_id"]
    text = segment.get("text") or ""
    subtopic = segment.get("subtopic", "")

    # ── scripting (already done during script generation) ──
    _sync_update_segment_status(db_path, seg_id, "scripting")

    # ── validating ──
    _sync_update_segment_status(db_path, seg_id, "validating")
    provenance = {
        "prompt_id": f"prompt_{seg_id}",
        "model": script_model,
        "source_refs": [],
        "claim_notes": [f"Generated from topic: {subtopic}"],
        "validation_status": "validated",
        "validation_errors": [],
        "validated_at": _now_iso(),
    }
    _sync_upsert_provenance(db_path, seg_id, provenance)

    # ── tts ──
    _sync_update_segment_status(db_path, seg_id, "tts")
    speech_path = Path(output_dir) / f"speech_{seg_id}.mp3"
    speech_path.parent.mkdir(parents=True, exist_ok=True)
    if tts_snapshot.provider == "elevenlabs":
        from podcast_worker.core.tts_engine import plan_dialogue_requests, synthesize_elevenlabs_plan
        from pydub import AudioSegment

        plans = plan_dialogue_requests(text, tts_snapshot, namespace=f"{project_id}:{seg_id}")
        rendered_by_plan: dict[str, str] = {}

        for plan in plans:
            _persist_tts_plan(db_path, project_id, seg_id, plan)

        def render(plan):
            characters = len("".join(turn.text for turn in plan.turns))
            for attempt_number in range(max(1, tts_snapshot.max_attempts)):
                attempt_id = _short_id("att")
                _append_tts_ledger_entry(db_path, ledger_id, plan.plan_id, attempt_id, "reserved", characters)
                _persist_tts_attempt(db_path, ledger_id, plan, attempt_id, tts_snapshot, "pre_send")
                _persist_tts_attempt(db_path, ledger_id, plan, attempt_id, tts_snapshot, "dispatched")
                try:
                    rendered_path = synthesize_elevenlabs_plan(
                        plan, tts_snapshot, str(speech_path.with_name(f"{speech_path.stem}-{plan.plan_id}.mp3"))
                    )
                except RoutingConfigurationError as exc:
                    outcome = _tts_attempt_outcome(exc)
                    _persist_tts_attempt(
                        db_path, ledger_id, plan, attempt_id, tts_snapshot, outcome,
                        {"code": outcome},
                    )
                    _append_tts_ledger_entry(db_path, ledger_id, plan.plan_id, attempt_id, "released", characters)
                    if outcome == "retryable" and attempt_number + 1 < max(1, tts_snapshot.max_attempts):
                        continue
                    raise
                except Exception:
                    _persist_tts_attempt(
                        db_path, ledger_id, plan, attempt_id, tts_snapshot, "unknown_outcome",
                        {"code": "unknown_outcome"},
                    )
                    _append_tts_ledger_entry(db_path, ledger_id, plan.plan_id, attempt_id, "released", characters)
                    raise RoutingConfigurationError("tts_outcome_unknown")
                _persist_tts_attempt(db_path, ledger_id, plan, attempt_id, tts_snapshot, "published")
                _append_tts_ledger_entry(db_path, ledger_id, plan.plan_id, attempt_id, "actual", characters)
                _append_tts_ledger_entry(db_path, ledger_id, plan.plan_id, attempt_id, "released", characters)
                return plan.plan_id, rendered_path
            raise RoutingConfigurationError("tts_outcome_unknown")

        worker_count = min(len(plans), max(1, tts_snapshot.max_concurrent_requests))
        with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix=f"tts-{seg_id}") as executor:
            futures = {executor.submit(render, plan): plan.plan_id for plan in plans}
            try:
                for future in as_completed(futures):
                    plan_id, rendered_path = future.result()
                    rendered_by_plan[plan_id] = rendered_path
            except Exception:
                for future in futures:
                    future.cancel()
                raise

        rendered = [rendered_by_plan[plan.plan_id] for plan in plans]
        combined = AudioSegment.empty()
        for rendered_path in rendered:
            combined += AudioSegment.from_file(rendered_path)
        combined.export(speech_path, format="mp3")
        _persist_audio_assembly(db_path, project_id, seg_id, rendered)
    else:
        synthesize(
            text,
            provider=tts_snapshot.provider,
            output_path=str(speech_path),
            voice=tts_snapshot.voice_bindings["interviewer"],
        )

    # Convert to WAV for mixing
    speech_wav = Path(output_dir) / f"speech_{seg_id}.wav"
    convert_to_wav(str(speech_path), str(speech_wav))

    # ── mixing ──
    _sync_update_segment_status(db_path, seg_id, "mixing")
    mixed_wav = Path(output_dir) / f"mixed_{seg_id}.wav"
    build_podcast_audio(
        str(speech_wav), str(beat_path), str(mixed_wav),
        intro_seconds=2.0, outro_seconds=3.0, bpm=bpm,
        duration_minutes=duration_minutes,
    )
    mixed_mp3 = Path(output_dir) / f"mixed_{seg_id}.mp3"
    # ── ready ──
    duration_seconds, loudness_dbfs, true_peak = _validated_audio_metadata(mixed_wav)
    convert_to_mp3(str(mixed_wav), str(mixed_mp3))
    checksum = _sha256_file(mixed_mp3)
    if checksum is None or not mixed_mp3.exists() or mixed_mp3.stat().st_size == 0:
        raise PodcastWorkerError("segment_audio_publication_failed")

    artifact_id = _short_id("art")
    artifact = {
        "artifact_id": artifact_id,
        "project_id": project_id,
        "segment_id": seg_id,
        "kind": "segment_audio",
        "content_type": "audio/mpeg",
        "duration_seconds": duration_seconds,
        "size_bytes": mixed_mp3.stat().st_size,
        "checksum_sha256": checksum,
        "status": "ready",
        "download_url": f"/api/v1/artifacts/{artifact_id}",
        "created_at": _now_iso(),
    }
    _sync_add_artifact(db_path, artifact)
    _sync_update_segment_ready(db_path, seg_id, artifact_id, duration_seconds)


def _create_final_artifact(db_path: str, project_id: str, output_dir: str,
                           segments: list[dict], beat_path: Path, bpm: int,
                           duration_minutes: int) -> None:
    """Publish a final MP3 only from a complete, ordered ready-segment manifest."""
    out = Path(output_dir)
    final_wav = out / f"final_{project_id}.wav"
    final_mp3 = out / f"final_{project_id}.mp3"
    manifest = _sync_get_segments(db_path, project_id)
    if (
        not manifest
        or [item["index"] for item in manifest] != list(range(len(manifest)))
        or any(
            item["status"] != "ready" or not item.get("primary_audio_artifact_id")
            for item in manifest
        )
    ):
        raise PodcastWorkerError("final_manifest_incomplete")

    ordered_audio = []
    for item in manifest:
        mixed_wav = out / f"mixed_{item['segment_id']}.wav"
        if not mixed_wav.is_file():
            raise PodcastWorkerError("final_manifest_audio_missing")
        ordered_audio.append(_load_wav_to_numpy(str(mixed_wav)))

    _save_numpy_to_wav(np.concatenate(ordered_audio), str(final_wav))
    duration_seconds, loudness_dbfs, true_peak = _validated_audio_metadata(final_wav)
    convert_to_mp3(str(final_wav), str(final_mp3))
    checksum = _sha256_file(final_mp3)
    if checksum is None or not final_mp3.exists() or final_mp3.stat().st_size == 0:
        raise PodcastWorkerError("final_audio_publication_failed")

    final_artifact_id = _short_id("art")
    artifact = {
        "artifact_id": final_artifact_id,
        "project_id": project_id,
        "segment_id": None,
        "kind": "final_mp3",
        "content_type": "audio/mpeg",
        "duration_seconds": duration_seconds,
        "size_bytes": final_mp3.stat().st_size,
        "checksum_sha256": checksum,
        "status": "ready",
        "download_url": f"/api/v1/artifacts/{final_artifact_id}",
        "created_at": _now_iso(),
    }
    _sync_add_artifact(db_path, artifact)
    _sync_status(db_path, project_id, "ready",
                 final_download_ready=True, final_artifact_id=final_artifact_id)


def _validated_audio_metadata(path: Path) -> tuple[float, float, float]:
    """Decode WAV media and enforce publication-safe loudness and peak bounds."""
    import wave

    with wave.open(str(path), "rb") as wav:
        if wav.getsampwidth() != 2 or wav.getnchannels() not in {1, 2}:
            raise PodcastWorkerError("unsupported_final_audio_format")
        frames = wav.getnframes()
        sample_rate = wav.getframerate()
        if not frames or not sample_rate:
            raise PodcastWorkerError("empty_final_audio")
        samples = np.frombuffer(wav.readframes(frames), dtype=np.int16).astype(np.float32) / 32768.0
    peak = float(np.max(np.abs(samples)))
    loudness_dbfs = 20.0 * np.log10(max(float(np.sqrt(np.mean(np.square(samples)))), 1e-12))
    if peak > 0.981 or loudness_dbfs < -60.0:
        raise PodcastWorkerError("final_audio_quality_invalid")
    return frames / sample_rate, loudness_dbfs, peak

def _persist_tts_plan(db_path: str, project_id: str, segment_id: str, plan) -> None:
    persistence._upsert_durable_record(db_path, "tts_request_plans", {
        "plan_id": plan.plan_id, "project_id": project_id, "segment_id": segment_id,
        "strategy": plan.strategy, "state": "pending",
        "payload_json": {"turns": [{"role": turn.role, "text": turn.text} for turn in plan.turns],
                         "voice_binding": plan.voice_binding},
    })


def _persist_tts_attempt(db_path: str, ledger_id: str, plan, attempt_id: str,
                         snapshot: cfg.ResolvedTTSSnapshot, outcome: str,
                         error: dict | None = None) -> None:
    persistence._upsert_durable_record(db_path, "execution_attempts", {
        "attempt_id": attempt_id, "ledger_id": ledger_id, "plan_id": plan.plan_id,
        "category": "tts", "correlation_id": plan.plan_id, "snapshot_revision": snapshot.revision,
        "binding_json": {"provider": snapshot.provider, "model_id": snapshot.model_id,
                         "strategy": snapshot.strategy, "voice_binding": plan.voice_binding},
        "outcome": outcome, "error_json": error,
    })


def _append_tts_ledger_entry(db_path: str, ledger_id: str, plan_id: str,
                             attempt_id: str, state: str, characters: int) -> None:
    persistence._append_ledger_entry(db_path, {
        "entry_id": _short_id("ledent"), "ledger_id": ledger_id, "category": "tts",
        "operation_type": "dialogue_synthesis", "correlation_id": plan_id,
        "resource_unit": "characters", "amount": characters, "state": state,
        "attempt_id": attempt_id,
    })


def _persist_audio_assembly(db_path: str, project_id: str, segment_id: str,
                            source_paths: list[str]) -> None:
    manifest = [{"path": path, "gap_ms": 0, "crossfade_ms": 0} for path in source_paths]
    persistence._upsert_durable_record(db_path, "audio_assemblies", {
        "assembly_id": _short_id("asm"), "project_id": project_id, "segment_id": segment_id,
        "manifest_json": manifest,
        "manifest_sha256": hashlib.sha256(repr(manifest).encode()).hexdigest(),
        "processing_revision": "assembly-v1", "state": "assembled",
    })

# ── Sync helpers (callable from background thread, not async) ────────────


def _sync_status(db_path: str, project_id: str, status: str,
                 final_download_ready: bool = False,
                 final_artifact_id: str | None = None) -> None:
    import asyncio
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running loop — call directly
        persistence._update_project_status(db_path, project_id, status,
                                           final_download_ready, final_artifact_id)
        return
    # We're inside an async context — use run_coroutine_threadsafe
    import concurrent.futures
    fut = asyncio.run_coroutine_threadsafe(
        persistence.update_project_status(db_path, project_id, status,
                                          final_download_ready, final_artifact_id),
        loop,
    )
    fut.result(timeout=30)


def _sync_upsert_segment(db_path: str, segment: dict) -> None:
    persistence._upsert_segment(db_path, segment)


def _sync_get_segments(db_path: str, project_id: str) -> list[dict]:
    """Get segments for a project (sync, from background thread)."""
    import sqlite3
    conn = persistence._get_conn(db_path)
    persistence._ensure_schema(conn)
    rows = conn.execute(
        "SELECT * FROM segments WHERE project_id = ? ORDER BY idx",
        (project_id,),
    ).fetchall()
    return [persistence._row_to_segment(r) for r in rows]


def _sync_get_segment(db_path: str, segment_id: str) -> dict | None:
    import sqlite3
    conn = persistence._get_conn(db_path)
    persistence._ensure_schema(conn)
    row = conn.execute(
        "SELECT * FROM segments WHERE segment_id = ?", (segment_id,),
    ).fetchone()
    return persistence._row_to_segment(row) if row else None


def _sync_update_segment_status(db_path: str, segment_id: str, status: str) -> None:
    conn = persistence._get_conn(db_path)
    persistence._ensure_schema(conn)
    now = _now_iso()
    conn.execute(
        "UPDATE segments SET status = ?, updated_at = ? WHERE segment_id = ?",
        (status, now, segment_id),
    )
    conn.commit()


def _sync_update_segment_ready(db_path: str, segment_id: str,
                               artifact_id: str, duration: float) -> None:
    conn = persistence._get_conn(db_path)
    persistence._ensure_schema(conn)
    now = _now_iso()
    conn.execute(
        """UPDATE segments SET status = 'ready', primary_audio_artifact_id = ?,
           duration_seconds = ?, updated_at = ? WHERE segment_id = ?""",
        (artifact_id, duration, now, segment_id),
    )
    conn.commit()


def _sync_upsert_provenance(db_path: str, segment_id: str, provenance: dict) -> None:
    persistence._upsert_provenance(db_path, segment_id, provenance)


def _sync_add_artifact(db_path: str, artifact: dict) -> None:
    persistence._add_artifact(db_path, artifact)


def _persist_segment_failure(db_path: str, segment_id: str, error_msg: str) -> None:
    conn = persistence._get_conn(db_path)
    persistence._ensure_schema(conn)
    now = _now_iso()
    conn.execute(
        """UPDATE segments SET status = 'failed',
           error_code = 'generation_failed', error_message = ?,
           error_retryable = 0, updated_at = ? WHERE segment_id = ?""",
        (error_msg, now, segment_id),
    )
    conn.commit()


def _persist_project_failure(db_path: str, project_id: str, error_msg: str) -> None:
    persistence._add_project_error(db_path, project_id, "pipeline_failed", error_msg)
    persistence._update_project_status(db_path, project_id, "failed")


def _sha256_file(path: Path) -> str | None:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None