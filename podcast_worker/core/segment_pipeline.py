"""Durable, fenced project generation worker.

The worker consumes the plan and immutable inputs committed by project creation. It
never creates or renumbers segments. Every externally visible write is guarded by
the project-pipeline owner/epoch fence.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from podcast_worker.core import config as cfg, persistence
from podcast_worker.core.config import RoutingConfigurationError
from podcast_worker.core.exceptions import PodcastWorkerError
from podcast_worker.core.audio_mixer import build_podcast_audio, convert_to_mp3
from podcast_worker.core.beat_generator import save_beat_to_wav
from podcast_worker.core.script_generator import (
    generate_research_brief,
    generate_section_draft,
    generate_subtopic_research,
    generate_verified_section,
)
from podcast_worker.core.tts_engine import (
    convert_to_wav,
    plan_dialogue_requests,
    synthesize,
    synthesize_elevenlabs_plan,
)


class FenceLost(RuntimeError):
    """The worker no longer owns its durable work lease."""


class PipelineInvariantError(RuntimeError):
    """Committed generation inputs violate the durable contract."""



class ProviderOutcomeUnknown(RuntimeError):
    """A dispatched provider operation cannot be safely replayed."""


def _provider_operation(
    db_path: str,
    work_id: str,
    owner: str,
    epoch: int,
    logical_operation_id: str,
    scope_id: str,
    stage_name: str,
    operation,
) -> dict:
    attempt = persistence._reserve_provider_attempt(
        db_path,
        work_id,
        owner,
        epoch,
        logical_operation_id,
        scope_id,
        stage_name,
    )
    if attempt is None:
        raise FenceLost("provider attempt reservation rejected")
    state = attempt["state"]
    if state == "completed":
        return json.loads(attempt["result_json"])
    if state == "dispatched":
        persistence._fail_dispatched_attempt_unknown(
            db_path, attempt["attempt_id"], work_id, owner, epoch
        )
        raise ProviderOutcomeUnknown("provider_outcome_unknown")
    if state == "failed_unknown":
        raise ProviderOutcomeUnknown("provider_outcome_unknown")
    if state != "reserved" or not persistence._mark_provider_attempt_dispatched(
        db_path, attempt["attempt_id"], work_id, owner, epoch
    ):
        raise FenceLost("provider dispatch fence rejected")
    try:
        result = operation()
    except Exception:
        persistence._fail_dispatched_attempt_unknown(
            db_path, attempt["attempt_id"], work_id, owner, epoch
        )
        raise
    if not isinstance(result, dict):
        result = {"result": result}
    if not persistence._complete_provider_attempt(
        db_path, attempt["attempt_id"], work_id, owner, epoch, result
    ):
        raise FenceLost("provider result fence rejected")
    return result
def _verification_payload(result) -> dict:
    if result.outcome == "blocked" or result.verified_text is None:
        raise RoutingConfigurationError("validation_failed")
    return {
        "outcome": result.outcome,
        "issues": list(result.issues),
        "verified_text": result.verified_text,
    }
def _synthesize_snapshot(
    text: str,
    snapshot: cfg.ResolvedTTSSnapshot,
    output_path: Path,
    namespace: str,
) -> str:
    if snapshot.provider != "elevenlabs":
        return synthesize(
            text,
            provider=snapshot.provider,
            output_path=str(output_path),
            voice=snapshot.voice_bindings["interviewer"],
        )

    from pydub import AudioSegment

    plans = plan_dialogue_requests(text, snapshot, namespace=namespace)
    if not plans:
        raise RoutingConfigurationError("tts_empty_plan")
    rendered_paths = [
        synthesize_elevenlabs_plan(
            plan,
            snapshot,
            str(output_path.with_name(f"{output_path.stem}-{index}.mp3")),
        )
        for index, plan in enumerate(plans)
    ]
    combined = AudioSegment.empty()
    for rendered_path in rendered_paths:
        combined += AudioSegment.from_file(rendered_path)
    combined.export(output_path, format="mp3")
    return str(output_path)





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



def _connect(db_path: str) -> sqlite3.Connection:
    conn = persistence._get_conn(db_path)
    persistence._ensure_schema(conn)
    return conn


def _hydrate(db_path: str, work_id: str, owner: str, epoch: int) -> dict[str, Any]:
    conn = _connect(db_path)
    work = conn.execute(
        """SELECT w.*, p.topic, p.bpm, p.duration_minutes, p.deleted_at,
                  g.accepted_outline_json, g.interviewer_profile_json,
                  g.sme_profile_json, g.llm_snapshot_id, g.tts_snapshot_id,
                  g.disposition
           FROM project_pipeline w
           JOIN projects p ON p.project_id = w.project_id
           JOIN project_generation g ON g.project_id = w.project_id
           WHERE w.work_id=? AND w.lease_owner=? AND w.lease_epoch=?
             AND w.state='leased' AND w.lease_expires_at>=?""",
        (work_id, owner, epoch, _now_iso()),
    ).fetchone()
    if work is None or work["deleted_at"] is not None:
        raise FenceLost("work lease is not current")
    segments = conn.execute(
        "SELECT * FROM segments WHERE project_id=? ORDER BY idx",
        (work["project_id"],),
    ).fetchall()
    outline = json.loads(work["accepted_outline_json"])
    sections = outline.get("sections", [])
    if not segments or len(segments) != len(sections):
        raise PipelineInvariantError("committed segment plan is inconsistent")
    for index, (segment, section) in enumerate(zip(segments, sections)):
        if segment["idx"] != index or section.get("index") != index:
            raise PipelineInvariantError("committed segment order is inconsistent")
        if segment["subtopic"] != section.get("topic"):
            raise PipelineInvariantError("committed segment topic is inconsistent")
    execution = persistence._get_project_execution(db_path, work["project_id"])
    if (
        execution is None
        or execution["llm_snapshot_id"] != work["llm_snapshot_id"]
        or execution["tts_snapshot_id"] != work["tts_snapshot_id"]
    ):
        raise PipelineInvariantError("committed execution snapshots are inconsistent")
    llm_snapshot = cfg.execution_snapshot_from_payload(
        execution["llm_snapshot"], execution["llm_revision"]
    )
    tts_snapshot = cfg.tts_snapshot_from_payload(
        execution["tts_snapshot"], execution["tts_revision"]
    )
    return {
        "work": dict(work),
        "outline": outline,
        "segments": [persistence._row_to_segment(row) for row in segments],
        "interviewer_profile": json.loads(work["interviewer_profile_json"]),
        "sme_profile": json.loads(work["sme_profile_json"]),
        "llm_snapshot": llm_snapshot,
        "tts_snapshot": tts_snapshot,
    }


def _transition(
    db_path: str,
    work_id: str,
    owner: str,
    epoch: int,
    scope_id: str,
    stage: str,
    state: str,
    result: dict | None = None,
    error_code: str | None = None,
) -> None:
    accepted = persistence._record_stage_transition(
        db_path, work_id, owner, epoch, scope_id, stage, state, result, error_code
    )
    if not accepted:
        raise FenceLost(f"fence rejected {stage}/{scope_id}/{state}")


def _stage_row(db_path: str, project_id: str, scope_id: str, stage: str) -> sqlite3.Row:
    row = _connect(db_path).execute(
        """SELECT * FROM generation_stage_results
           WHERE project_id=? AND scope_id=? AND stage_name=?""",
        (project_id, scope_id, stage),
    ).fetchone()
    if row is None:
        raise PipelineInvariantError(f"missing {stage}/{scope_id} stage row")
    return row


def _completed_result(db_path: str, project_id: str, scope_id: str, stage: str) -> dict | None:
    row = _stage_row(db_path, project_id, scope_id, stage)
    if row["state"] == "running":
        raise PipelineInvariantError(f"provider outcome unknown for {stage}/{scope_id}")
    if row["state"] != "completed":
        return None
    return json.loads(row["result_json"]) if row["result_json"] else {}


def _renew(db_path: str, work_id: str, owner: str, epoch: int) -> None:
    if not persistence._renew_work_lease(
        db_path, work_id, owner, epoch, cfg.settings.work_lease_seconds
    ):
        raise FenceLost("work lease renewal failed")


def _publish_verified_text(
    db_path: str,
    work_id: str,
    owner: str,
    epoch: int,
    segment_id: str,
    verification: dict,
) -> None:
    """Atomically publish verified text/provenance and complete fact checking."""
    conn = _connect(db_path)
    now = _now_iso()
    result_json = json.dumps(verification, sort_keys=True, separators=(",", ":"))
    try:
        conn.execute("BEGIN IMMEDIATE")
        fence = conn.execute(
            """SELECT w.project_id FROM project_pipeline w
               JOIN project_generation g ON g.project_id=w.project_id
               JOIN projects p ON p.project_id=w.project_id
               WHERE w.work_id=? AND w.lease_owner=? AND w.lease_epoch=?
                 AND w.state='leased' AND w.lease_expires_at>=?
                 AND g.disposition='active' AND p.deleted_at IS NULL""",
            (work_id, owner, epoch, now),
        ).fetchone()
        if fence is None:
            raise FenceLost("verified-text publication fence rejected")
        conn.execute(
            """INSERT INTO provenance
               (segment_id,prompt_id,model,source_refs,claim_notes,
                validation_status,validation_errors,validated_at)
               VALUES (?,?,?,?,?,'validated','[]',?)
               ON CONFLICT(segment_id) DO UPDATE SET
                 validation_status='validated', validation_errors='[]',
                 validated_at=excluded.validated_at""",
            (segment_id, f"prompt_{segment_id}", "server-profile", "[]", "[]", now),
        )
        conn.execute(
            "UPDATE segments SET text=?, status='tts', updated_at=? WHERE segment_id=?",
            (verification["verified_text"], now, segment_id),
        )
        conn.execute(
            """UPDATE generation_stage_results SET state='completed', result_json=?,
                 result_hash=?, completed_at=?, updated_at=?
               WHERE project_id=? AND scope_id=? AND stage_name='fact_checking'""",
            (
                result_json,
                hashlib.sha256(result_json.encode()).hexdigest(),
                now,
                now,
                fence["project_id"],
                segment_id,
            ),
        )
        conn.execute(
            """UPDATE project_generation SET progress_version=progress_version+1,
                 last_transition_at=? WHERE project_id=?""",
            (now, fence["project_id"]),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def _publish_audio(
    db_path: str,
    work_id: str,
    owner: str,
    epoch: int,
    segment_id: str,
    path: Path,
    duration: float,
) -> None:
    artifact_id = _short_id("art")
    now = _now_iso()
    checksum = _sha256_file(path)
    if not path.is_file() or not path.stat().st_size or checksum is None:
        raise PipelineInvariantError("mixed audio failed validation")
    conn = _connect(db_path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        fence = conn.execute(
            """SELECT w.project_id FROM project_pipeline w
               JOIN project_generation g ON g.project_id=w.project_id
               JOIN projects p ON p.project_id=w.project_id
               WHERE w.work_id=? AND w.lease_owner=? AND w.lease_epoch=?
                 AND w.state='leased' AND w.lease_expires_at>=?
                 AND g.disposition='active' AND p.deleted_at IS NULL""",
            (work_id, owner, epoch, now),
        ).fetchone()
        if fence is None:
            raise FenceLost("audio publication fence rejected")
        conn.execute(
            """INSERT INTO artifacts
               (artifact_id,project_id,segment_id,kind,content_type,duration_seconds,
                size_bytes,checksum_sha256,status,download_url,created_at)
               VALUES (?,?,?,'segment_audio','audio/mpeg',?,?,?,'ready',?,?)""",
            (artifact_id, fence["project_id"], segment_id, duration, path.stat().st_size,
             checksum, f"/api/v1/artifacts/{artifact_id}", now),
        )
        conn.execute(
            """UPDATE segments SET status='ready', primary_audio_artifact_id=?,
                 duration_seconds=?, updated_at=? WHERE segment_id=?""",
            (artifact_id, duration, now, segment_id),
        )
        result = json.dumps({"artifact_id": artifact_id, "checksum_sha256": checksum}, sort_keys=True)
        conn.execute(
            """UPDATE generation_stage_results SET state='completed', result_json=?,
                 result_hash=?, completed_at=?, updated_at=?
               WHERE project_id=? AND scope_id=? AND stage_name='mixing'""",
            (result, hashlib.sha256(result.encode()).hexdigest(), now, now,
             fence["project_id"], segment_id),
        )
        conn.execute(
            """UPDATE project_generation SET progress_version=progress_version+1,
                 last_transition_at=? WHERE project_id=?""",
            (now, fence["project_id"]),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def run_project_pipeline(
    db_path: str,
    work_id: str,
    lease_owner: str,
    lease_epoch: int,
    output_dir: str,
) -> None:
    """Resume one claimed work item from committed durable stage results."""
    try:
        _run_pipeline(db_path, work_id, lease_owner, lease_epoch, output_dir)
    except FenceLost:
        _settle_cancellation(db_path, work_id, lease_owner, lease_epoch)
    except Exception as exc:
        _settle_failure(db_path, work_id, lease_owner, lease_epoch, exc)


def _run_pipeline(db_path: str, work_id: str, owner: str, epoch: int, output_dir: str) -> None:
    hydrated = _hydrate(db_path, work_id, owner, epoch)
    work = hydrated["work"]
    project_id = work["project_id"]
    outline = hydrated["outline"]
    llm_snapshot = hydrated["llm_snapshot"]
    tts_snapshot = hydrated["tts_snapshot"]
    provider_args = {"snapshot": llm_snapshot}

    research = _completed_result(db_path, project_id, "project", "research")
    if research is None:
        _transition(db_path, work_id, owner, epoch, "project", "research", "running")
        research = _provider_operation(
            db_path, work_id, owner, epoch, "lead_research", "project", "research",
            lambda: generate_research_brief(work["topic"], outline, **provider_args),
        )
        _transition(db_path, work_id, owner, epoch, "project", "research", "completed", research)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    beat_path = out / f"beat_{project_id}.wav"
    if not beat_path.exists():
        save_beat_to_wav(work["bpm"], work["duration_minutes"] * 60.0, str(beat_path))

    previous_text = ""
    for segment, section in zip(hydrated["segments"], outline["sections"]):
        _renew(db_path, work_id, owner, epoch)
        segment_id = segment["segment_id"]
        section_research = _completed_result(db_path, project_id, segment_id, "research")
        if section_research is None:
            _transition(db_path, work_id, owner, epoch, segment_id, "research", "running")
            section_research = _provider_operation(
                db_path, work_id, owner, epoch,
                f"{segment_id}:research", segment_id, "research",
                lambda: generate_subtopic_research(
                    work["topic"], outline, section, research, **provider_args
                ),
            )
            _transition(db_path, work_id, owner, epoch, segment_id, "research", "completed", section_research)

        draft = _completed_result(db_path, project_id, segment_id, "text_generation")
        if draft is None:
            _transition(db_path, work_id, owner, epoch, segment_id, "text_generation", "running")
            draft = _provider_operation(
                db_path, work_id, owner, epoch,
                f"{segment_id}:draft", segment_id, "text_generation",
                lambda: generate_section_draft(
                    work["topic"], work["bpm"], work["duration_minutes"], outline,
                    section, research, section_research, previous_text,
                    hydrated["interviewer_profile"], hydrated["sme_profile"],
                    **provider_args,
                ),
            )
            _transition(db_path, work_id, owner, epoch, segment_id, "text_generation", "completed", draft)

        verification = _completed_result(db_path, project_id, segment_id, "fact_checking")
        if verification is None:
            _transition(db_path, work_id, owner, epoch, segment_id, "fact_checking", "running")
            verification = _provider_operation(
                db_path, work_id, owner, epoch,
                f"{segment_id}:verify", segment_id, "fact_checking",
                lambda: _verification_payload(
                    generate_verified_section(
                        work["topic"],
                        outline,
                        section,
                        research,
                        section_research,
                        draft["text"],
                        **provider_args,
                    )
                ),
            )
            _publish_verified_text(db_path, work_id, owner, epoch, segment_id, verification)
        previous_text = verification["verified_text"]

        if _completed_result(db_path, project_id, segment_id, "mixing") is not None:
            continue
        _transition(db_path, work_id, owner, epoch, segment_id, "tts", "running")
        speech_mp3 = out / f"speech_{segment_id}.mp3"
        speech_wav = out / f"speech_{segment_id}.wav"
        _provider_operation(
            db_path, work_id, owner, epoch,
            f"{segment_id}:tts", segment_id, "tts",
            lambda: {
                "output": _synthesize_snapshot(
                    previous_text,
                    tts_snapshot,
                    speech_mp3,
                    namespace=f"{project_id}:{segment_id}",
                )
            },
        )
        convert_to_wav(str(speech_mp3), str(speech_wav))
        _transition(db_path, work_id, owner, epoch, segment_id, "tts", "completed",
                    {"assembled": True})
        _transition(db_path, work_id, owner, epoch, segment_id, "mixing", "running")
        mixed_wav = out / f"mixed_{segment_id}.wav"
        mixed_mp3 = out / f"mixed_{segment_id}.mp3"
        build_podcast_audio(str(speech_wav), str(beat_path), str(mixed_wav),
                            intro_seconds=2.0, outro_seconds=3.0, bpm=work["bpm"],
                            duration_minutes=work["duration_minutes"])
        convert_to_mp3(str(mixed_wav), str(mixed_mp3))
        _publish_audio(db_path, work_id, owner, epoch, segment_id, mixed_mp3,
                       float(segment.get("planned_duration_seconds") or 0))

    _finalize(db_path, work_id, owner, epoch, project_id, hydrated["segments"], out)


def _finalize(db_path: str, work_id: str, owner: str, epoch: int, project_id: str,
              segments: list[dict], out: Path) -> None:
    if _completed_result(db_path, project_id, "project", "finalizing") is not None:
        _settle_ready(db_path, work_id, owner, epoch, None)
        return
    _transition(db_path, work_id, owner, epoch, "project", "finalizing", "running")
    arrays: list[np.ndarray] = []
    sample_rate = cfg.SAMPLE_RATE
    for segment in segments:
        mixed = out / f"mixed_{segment['segment_id']}.wav"
        with wave.open(str(mixed), "rb") as stream:
            arrays.append(np.frombuffer(stream.readframes(stream.getnframes()), dtype=np.int16))
    final_wav = out / f"final_{project_id}.wav"
    final_mp3 = out / f"final_{project_id}.mp3"
    with wave.open(str(final_wav), "wb") as stream:
        stream.setnchannels(1); stream.setsampwidth(2); stream.setframerate(sample_rate)
        stream.writeframes(np.concatenate(arrays).tobytes())
    convert_to_mp3(str(final_wav), str(final_mp3))
    if not final_mp3.is_file() or not final_mp3.stat().st_size:
        raise PipelineInvariantError("final audio failed validation")
    _publish_final_artifact(
        db_path,
        work_id,
        owner,
        epoch,
        project_id,
        final_mp3,
    )


def _publish_final_artifact(
    db_path: str,
    work_id: str,
    owner: str,
    epoch: int,
    project_id: str,
    final_mp3: Path,
) -> None:
    """Atomically publish final audio, complete finalizing, and settle ready."""
    artifact_id = _short_id("art")
    checksum = _sha256_file(final_mp3)
    if checksum is None:
        raise PipelineInvariantError("final audio checksum failed")
    now = _now_iso()
    result_json = json.dumps({"artifact_id": artifact_id}, sort_keys=True)
    conn = _connect(db_path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        fence = conn.execute(
            """SELECT w.project_id FROM project_pipeline w
               JOIN project_generation g ON g.project_id=w.project_id
               JOIN projects p ON p.project_id=w.project_id
               WHERE w.work_id=? AND w.project_id=? AND w.lease_owner=?
                 AND w.lease_epoch=? AND w.state='leased'
                 AND w.lease_expires_at>=? AND g.disposition='active'
                 AND p.deleted_at IS NULL""",
            (work_id, project_id, owner, epoch, now),
        ).fetchone()
        if fence is None:
            raise FenceLost("final publication fence rejected")
        conn.execute(
            """INSERT INTO artifacts
               (artifact_id,project_id,segment_id,kind,content_type,duration_seconds,
                size_bytes,checksum_sha256,status,download_url,created_at)
               VALUES (?,?,NULL,'final_mp3','audio/mpeg',NULL,?,?,'ready',?,?)""",
            (
                artifact_id,
                project_id,
                final_mp3.stat().st_size,
                checksum,
                f"/api/v1/artifacts/{artifact_id}",
                now,
            ),
        )
        updated = conn.execute(
            """UPDATE generation_stage_results
               SET state='completed',result_json=?,result_hash=?,
                   completed_at=?,updated_at=?
               WHERE project_id=? AND scope_id='project'
                 AND stage_name='finalizing' AND state='running'""",
            (
                result_json,
                hashlib.sha256(result_json.encode()).hexdigest(),
                now,
                now,
                project_id,
            ),
        )
        if updated.rowcount != 1:
            raise PipelineInvariantError("finalizing stage was not running")
        conn.execute(
            """UPDATE project_pipeline
               SET state='succeeded',settled_at=?,updated_at=?
               WHERE work_id=?""",
            (now, now, work_id),
        )
        conn.execute(
            """UPDATE project_generation
               SET disposition='terminal',terminal_outcome='ready',terminal_at=?,
                   last_transition_at=?,progress_version=progress_version+1
               WHERE project_id=?""",
            (now, now, project_id),
        )
        conn.execute(
            """UPDATE projects
               SET status='ready',final_download_ready=1,final_artifact_id=?,
                   updated_at=? WHERE project_id=?""",
            (artifact_id, now, project_id),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def _settle_ready(
    db_path: str,
    work_id: str,
    owner: str,
    epoch: int,
    artifact_id: str | None,
) -> None:
    """Settle an already-published final artifact during idempotent recovery."""
    conn = _connect(db_path)
    now = _now_iso()
    conn.execute("BEGIN IMMEDIATE")
    row = conn.execute(
        """SELECT w.project_id FROM project_pipeline w
           JOIN project_generation g ON g.project_id=w.project_id
           JOIN projects p ON p.project_id=w.project_id
           WHERE w.work_id=? AND w.lease_owner=? AND w.lease_epoch=?
             AND w.state='leased' AND w.lease_expires_at>=?
             AND g.disposition='active' AND p.deleted_at IS NULL""",
        (work_id, owner, epoch, now),
    ).fetchone()
    if row is None:
        conn.rollback()
        raise FenceLost("ready settlement fence rejected")
    conn.execute(
        "UPDATE project_pipeline SET state='succeeded',settled_at=?,updated_at=? WHERE work_id=?",
        (now, now, work_id),
    )
    conn.execute(
        """UPDATE project_generation
           SET disposition='terminal',terminal_outcome='ready',terminal_at=?,
               last_transition_at=?,progress_version=progress_version+1
           WHERE project_id=?""",
        (now, now, row["project_id"]),
    )
    conn.execute(
        """UPDATE projects SET status='ready',final_download_ready=1,
           final_artifact_id=COALESCE(?,final_artifact_id),updated_at=?
           WHERE project_id=?""",
        (artifact_id, now, row["project_id"]),
    )
    conn.commit()


def _settle_cancellation(
    db_path: str,
    work_id: str,
    owner: str,
    epoch: int,
) -> bool:
    """Observe and atomically terminalize a requested cancellation."""
    conn = _connect(db_path)
    now = _now_iso()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            """SELECT w.project_id FROM project_pipeline w
               JOIN project_generation g ON g.project_id=w.project_id
               JOIN projects p ON p.project_id=w.project_id
               WHERE w.work_id=? AND w.lease_owner=? AND w.lease_epoch=?
                 AND w.state='leased' AND w.lease_expires_at>=?
                 AND g.disposition='cancellation_requested'
                 AND p.deleted_at IS NULL""",
            (work_id, owner, epoch, now),
        ).fetchone()
        if row is None:
            conn.rollback()
            return False
        project_id = row["project_id"]
        conn.execute(
            """UPDATE generation_stage_results
               SET state='cancelled',cancelled_at=?,updated_at=?
               WHERE project_id=? AND state NOT IN ('completed','failed','cancelled')""",
            (now, now, project_id),
        )
        conn.execute(
            """UPDATE segments
               SET status='failed',error_code='cancelled',
                   error_message='Generation was cancelled.',
                   error_retryable=0,updated_at=?
               WHERE project_id=? AND status!='ready'""",
            (now, project_id),
        )
        conn.execute(
            """UPDATE project_pipeline
               SET state='cancelled',settled_at=?,updated_at=? WHERE work_id=?""",
            (now, now, work_id),
        )
        conn.execute(
            """UPDATE project_generation
               SET disposition='terminal',terminal_outcome='cancelled',
                   cancellation_observed_at=?,terminal_at=?,last_transition_at=?,
                   progress_version=progress_version+1
               WHERE project_id=?""",
            (now, now, now, project_id),
        )
        conn.execute(
            """UPDATE projects
               SET status=CASE WHEN EXISTS(
                   SELECT 1 FROM segments WHERE project_id=? AND status='ready'
               ) THEN 'partially_ready' ELSE 'failed' END,updated_at=?
               WHERE project_id=?""",
            (project_id, now, project_id),
        )
        conn.commit()
        return True
    except Exception:
        conn.rollback()
        raise


def _settle_failure(db_path: str, work_id: str, owner: str, epoch: int, exc: Exception) -> None:
    conn = _connect(db_path); now = _now_iso()
    row = conn.execute("SELECT project_id FROM project_pipeline WHERE work_id=? AND lease_owner=? AND lease_epoch=? AND state='leased'",
                       (work_id, owner, epoch)).fetchone()
    if row is None:
        return
    safe_code = "provider_outcome_unknown" if "outcome unknown" in str(exc) else "pipeline_failed"
    conn.execute("UPDATE project_pipeline SET state='failed',settled_at=?,updated_at=? WHERE work_id=? AND lease_owner=? AND lease_epoch=?",
                 (now, now, work_id, owner, epoch))
    conn.execute("UPDATE project_generation SET disposition='terminal',terminal_outcome='failed',terminal_at=?,last_transition_at=?,progress_version=progress_version+1 WHERE project_id=? AND disposition='active'",
                 (now, now, row["project_id"]))
    conn.execute("UPDATE projects SET status=CASE WHEN EXISTS(SELECT 1 FROM segments WHERE project_id=? AND status='ready') THEN 'partially_ready' ELSE 'failed' END,updated_at=? WHERE project_id=?",
                 (row["project_id"], now, row["project_id"]))
    conn.commit()
    persistence._add_project_error(db_path, row["project_id"], safe_code, "Generation could not safely continue")


def _sha256_file(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(65536), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


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
