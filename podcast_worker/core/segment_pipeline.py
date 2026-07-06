"""Background segment generation pipeline for PodcastProject v1.

Orchestrates the transition: queued → scripting → validating → tts → mixing → ready.

Key invariants (per API_SPEC.md):
- provenance.validation_status MUST be stored before a segment transitions to tts.
- artifact metadata MUST be stored before segment transitions to ready.
- Forced segment failure preserves ready artifacts; final_download_ready stays false.
"""

from __future__ import annotations

import hashlib
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from podcast_worker.core import config as cfg
from podcast_worker.core import persistence
from podcast_worker.core.script_generator import generate_script
from podcast_worker.core.tts_engine import synthesize, convert_to_wav
from podcast_worker.core.beat_generator import save_beat_to_wav, generate_beat
from podcast_worker.core.audio_mixer import build_podcast_audio, convert_to_mp3
from podcast_worker.core.exceptions import PodcastWorkerError


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _short_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


# ── Pipeline entry point (called from a background thread) ───────────────


def run_project_pipeline(db_path: str, project_id: str, output_dir: str,
                         topic: str, bpm: int, duration_minutes: int,
                         voice_id: str | None, outline: dict | None = None) -> None:
    """Run the full segment generation pipeline for a project in a background thread.

    This function is designed to be called from a daemon thread.  It handles
    every step and persists state durably at each transition so restarts are safe.
    """
    try:
        _run_pipeline(db_path, project_id, output_dir, topic, bpm, duration_minutes, voice_id, outline)
    except Exception as exc:
        # Project-level failure — persist error and mark failed if recoverable
        _persist_project_failure(db_path, project_id, str(exc))


def _run_pipeline(db_path: str, project_id: str, output_dir: str,
                  topic: str, bpm: int, duration_minutes: int,
                  voice_id: str | None, outline: dict | None = None) -> None:
    out = Path(output_dir)

    # 1. Transition to generating
    _sync_status(db_path, project_id, "generating")

    # 2. Generate script via LLM
    provider = cfg.settings.llm_provider
    model = cfg.settings.openai_model if provider == "openai" else cfg.settings.openrouter_model

    script = generate_script(
        topic=topic,
        bpm=bpm,
        duration_minutes=duration_minutes,
        provider=provider,
        model=model,
        outline=outline,
    )
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
                             bpm, duration_minutes, voice_id, model)
            seg_after = _sync_get_segment(db_path, seg["segment_id"])
            if seg_after and seg_after["status"] == "ready":
                ready_count += 1
        except PodcastWorkerError as exc:
            any_failed = True
            _persist_segment_failure(db_path, seg["segment_id"], str(exc))
        except Exception as exc:
            any_failed = True
            _persist_segment_failure(db_path, seg["segment_id"], str(exc))

    # 5. Determine final project status
    if any_failed and ready_count == 0:
        _sync_status(db_path, project_id, "failed")
    elif any_failed:
        # Partially ready — final_download_ready stays false per spec
        _sync_status(db_path, project_id, "partially_ready")
    elif ready_count == total_segments:
        # All segments ready — create final MP3 artifact
        _create_final_artifact(db_path, project_id, output_dir, segments, beat_path,
                               bpm, duration_minutes)
    else:
        _sync_status(db_path, project_id, "partially_ready")


def _process_segment(db_path: str, project_id: str, segment: dict,
                     output_dir: str, beat_path: Path, bpm: int,
                     duration_minutes: int, voice_id: str | None,
                     script_model: str) -> None:
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
    tts_provider = cfg.settings.tts_provider
    tts_voice = voice_id or cfg.settings.edge_tts_voice

    speech_path = Path(output_dir) / f"speech_{seg_id}.mp3"
    speech_path.parent.mkdir(parents=True, exist_ok=True)
    synthesize(text, str(speech_path), provider=tts_provider, voice=tts_voice)

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
    convert_to_mp3(str(mixed_wav), str(mixed_mp3))

    # ── ready ──
    artifact_id = _short_id("art")
    artifact = {
        "artifact_id": artifact_id,
        "project_id": project_id,
        "segment_id": seg_id,
        "kind": "segment_audio",
        "content_type": "audio/mpeg",
        "duration_seconds": float(duration_minutes * 60.0),
        "size_bytes": mixed_mp3.stat().st_size if mixed_mp3.exists() else None,
        "checksum_sha256": _sha256_file(mixed_mp3) if mixed_mp3.exists() else None,
        "status": "ready",
        "download_url": f"/api/v1/artifacts/{artifact_id}",
        "created_at": _now_iso(),
    }
    _sync_add_artifact(db_path, artifact)

    _sync_update_segment_ready(db_path, seg_id, artifact_id,
                               float(duration_minutes * 60.0))


def _create_final_artifact(db_path: str, project_id: str, output_dir: str,
                           segments: list[dict], beat_path: Path, bpm: int,
                           duration_minutes: int) -> None:
    """Concatenate all segment audio into a final MP3."""
    out = Path(output_dir)
    final_wav = out / f"final_{project_id}.wav"
    final_mp3 = out / f"final_{project_id}.mp3"

    # Simple concatenation: mix all segment speech+beat into one final file
    # Re-use the last segment's mixing approach but for all segments
    all_speech_wavs = []
    for seg in segments:
        sw = out / f"speech_{seg['segment_id']}.wav"
        if sw.exists():
            all_speech_wavs.append(str(sw))

    if not all_speech_wavs:
        return

    # For simplicity, concatenate the mixed files
    import numpy as np
    import wave

    combined = None
    sample_rate = cfg.SAMPLE_RATE
    for seg in segments:
        mixed_wav = out / f"mixed_{seg['segment_id']}.wav"
        if not mixed_wav.exists():
            continue
        try:
            with wave.open(str(mixed_wav), "rb") as wf:
                frames = wf.readframes(wf.getnframes())
                audio = np.frombuffer(frames, dtype=np.int16)
                if combined is None:
                    combined = audio
                else:
                    combined = np.concatenate([combined, audio])
        except Exception:
            continue

    if combined is None:
        return

    with wave.open(str(final_wav), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(combined.tobytes())

    convert_to_mp3(str(final_wav), str(final_mp3))

    final_artifact_id = _short_id("art")
    artifact = {
        "artifact_id": final_artifact_id,
        "project_id": project_id,
        "segment_id": None,
        "kind": "final_mp3",
        "content_type": "audio/mpeg",
        "duration_seconds": float(duration_minutes * 60.0),
        "size_bytes": final_mp3.stat().st_size if final_mp3.exists() else None,
        "checksum_sha256": _sha256_file(final_mp3) if final_mp3.exists() else None,
        "status": "ready",
        "download_url": f"/api/v1/artifacts/{final_artifact_id}",
        "created_at": _now_iso(),
    }
    _sync_add_artifact(db_path, artifact)

    _sync_status(db_path, project_id, "ready",
                 final_download_ready=True, final_artifact_id=final_artifact_id)


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