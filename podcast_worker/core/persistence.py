"""SQLite-backed durable persistence for PodcastProject v1.

Replaces the legacy in-memory dict store with sqlite3 via ThreadPoolExecutor
so async handlers never block the event loop on sync SQLite calls.

Schema invariants:
- provenance.validation_status MUST be stored before a segment transitions to tts
- artifact metadata MUST be stored before segment transitions to ready
- all required segments ready before final_download_ready=true
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Single-threaded executor — SQLite in WAL mode handles concurrent reads,
# but writes are serialised through one thread to avoid SQLITE_BUSY.
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sqlite")

# Thread-local connections keyed by db path so every OS thread gets its own
# connection, keeping the simple executor model safe.
_conns: dict[str, sqlite3.Connection] = {}
_conn_lock = threading.Lock()


def _get_conn(db_path: str) -> sqlite3.Connection:
    tid = threading.get_ident()
    key = f"{db_path}:{tid}"
    with _conn_lock:
        if key not in _conns:
            conn = sqlite3.connect(db_path, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.row_factory = sqlite3.Row
            _conns[key] = conn
        return _conns[key]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS projects (
            project_id      TEXT PRIMARY KEY,
            owner_id        TEXT NOT NULL DEFAULT 'single-user',
            topic           TEXT NOT NULL,
            bpm             INTEGER NOT NULL,
            duration_minutes INTEGER NOT NULL DEFAULT 5,
            status          TEXT NOT NULL DEFAULT 'queued',
            revision_token  TEXT NOT NULL DEFAULT 'rev_0001',
            final_download_ready INTEGER NOT NULL DEFAULT 0,
            final_artifact_id    TEXT,
            created_at      TEXT NOT NULL,
            updated_at      TEXT NOT NULL,
            deleted_at      TEXT
        );

        CREATE TABLE IF NOT EXISTS segments (
            segment_id              TEXT PRIMARY KEY,
            project_id              TEXT NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
            idx                     INTEGER NOT NULL,
            subtopic                TEXT NOT NULL,
            title                   TEXT,
            status                  TEXT NOT NULL DEFAULT 'queued',
            duration_seconds        REAL,
            text                    TEXT,
            primary_audio_artifact_id TEXT,
            error_code              TEXT,
            error_message           TEXT,
            error_retryable         INTEGER DEFAULT 0,
            created_at              TEXT NOT NULL,
            updated_at              TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS provenance (
            segment_id          TEXT PRIMARY KEY REFERENCES segments(segment_id) ON DELETE CASCADE,
            prompt_id           TEXT NOT NULL,
            model               TEXT NOT NULL,
            source_refs         TEXT NOT NULL DEFAULT '[]',
            claim_notes         TEXT NOT NULL DEFAULT '[]',
            validation_status   TEXT NOT NULL DEFAULT 'pending',
            validation_errors   TEXT NOT NULL DEFAULT '[]',
            validated_at        TEXT
        );

        CREATE TABLE IF NOT EXISTS artifacts (
            artifact_id             TEXT PRIMARY KEY,
            project_id              TEXT NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
            segment_id              TEXT,
            kind                    TEXT NOT NULL,
            content_type            TEXT NOT NULL,
            duration_seconds        REAL,
            size_bytes              INTEGER,
            checksum_sha256         TEXT,
            status                  TEXT NOT NULL DEFAULT 'pending',
            download_url            TEXT NOT NULL,
            signed_transfer_url     TEXT,
            signed_transfer_expires_at TEXT,
            created_at              TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS project_errors (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id  TEXT NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
            code        TEXT NOT NULL,
            message     TEXT NOT NULL,
            retryable   INTEGER NOT NULL DEFAULT 0,
            created_at  TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_segments_project ON segments(project_id);
        CREATE INDEX IF NOT EXISTS idx_artifacts_project ON artifacts(project_id);
        CREATE INDEX IF NOT EXISTS idx_artifacts_segment ON artifacts(segment_id);
        CREATE INDEX IF NOT EXISTS idx_errors_project ON project_errors(project_id);
    """)


# ── project helpers ─────────────────────────────────────────────────────

async def _run_in_executor(fn, *args):
    """Run a sync function in the SQLite executor and return its result."""
    # Using asyncio's run_in_executor requires the event loop, so we expose
    # this for routers that import persistence directly.
    import asyncio
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor, fn, *args)


# ── Project CRUD ────────────────────────────────────────────────────────


def _create_project(db_path: str, project_id: str, owner_id: str, topic: str,
                    bpm: int, duration_minutes: int) -> dict:
    conn = _get_conn(db_path)
    _ensure_schema(conn)
    now = _now_iso()
    conn.execute(
        """INSERT INTO projects (project_id, owner_id, topic, bpm,
           duration_minutes, status, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, 'queued', ?, ?)""",
        (project_id, owner_id, topic, bpm, duration_minutes, now, now),
    )
    conn.commit()
    return _get_project_by_id(db_path, project_id)


def _list_projects(db_path: str, owner_id: str) -> list[dict]:
    conn = _get_conn(db_path)
    _ensure_schema(conn)
    rows = conn.execute(
        """SELECT p.*,
           (SELECT COUNT(*) FROM segments s WHERE s.project_id = p.project_id) AS segment_count,
           (SELECT COUNT(*) FROM segments s WHERE s.project_id = p.project_id AND s.status = 'ready') AS ready_segment_count
           FROM projects p
           WHERE p.owner_id = ? AND p.deleted_at IS NULL
           ORDER BY p.created_at DESC""",
        (owner_id,),
    ).fetchall()
    return [_row_to_project_summary(r) for r in rows]


def _get_project_by_id(db_path: str, project_id: str) -> dict | None:
    conn = _get_conn(db_path)
    _ensure_schema(conn)
    row = conn.execute(
        "SELECT * FROM projects WHERE project_id = ? AND deleted_at IS NULL",
        (project_id,),
    ).fetchone()
    if row is None:
        return None
    project = _row_to_project_full(row)

    # Attach segments
    seg_rows = conn.execute(
        "SELECT * FROM segments WHERE project_id = ? ORDER BY idx",
        (project_id,),
    ).fetchall()
    project["segments"] = []
    for sr in seg_rows:
        seg = _row_to_segment(sr)
        # Attach provenance
        prov_row = conn.execute(
            "SELECT * FROM provenance WHERE segment_id = ?", (seg["segment_id"],),
        ).fetchone()
        seg["provenance"] = _row_to_provenance(prov_row) if prov_row else None
        project["segments"].append(seg)

    # Attach artifacts
    art_rows = conn.execute(
        "SELECT * FROM artifacts WHERE project_id = ? ORDER BY created_at",
        (project_id,),
    ).fetchall()
    project["artifacts"] = [_row_to_artifact(ar) for ar in art_rows]

    # Attach errors
    err_rows = conn.execute(
        "SELECT * FROM project_errors WHERE project_id = ? ORDER BY created_at",
        (project_id,),
    ).fetchall()
    project["errors"] = [_row_to_error(er) for er in err_rows]

    return project


def _delete_project(db_path: str, project_id: str) -> dict | None:
    conn = _get_conn(db_path)
    _ensure_schema(conn)
    now = _now_iso()
    cur = conn.execute(
        "UPDATE projects SET status = 'deleted', deleted_at = ? WHERE project_id = ? AND deleted_at IS NULL",
        (now, project_id),
    )
    conn.commit()
    if cur.rowcount == 0:
        return None
    return {"project_id": project_id, "status": "deleted", "deleted_at": now}


# ── Segment helpers ─────────────────────────────────────────────────────


def _upsert_segment(db_path: str, segment: dict) -> None:
    conn = _get_conn(db_path)
    _ensure_schema(conn)
    now = _now_iso()
    conn.execute(
        """INSERT INTO segments (segment_id, project_id, idx, subtopic, title,
           status, duration_seconds, text, primary_audio_artifact_id,
           error_code, error_message, error_retryable, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(segment_id) DO UPDATE SET
               status = excluded.status,
               duration_seconds = excluded.duration_seconds,
               text = excluded.text,
               primary_audio_artifact_id = excluded.primary_audio_artifact_id,
               error_code = excluded.error_code,
               error_message = excluded.error_message,
               error_retryable = excluded.error_retryable,
               updated_at = excluded.updated_at""",
        (
            segment["segment_id"], segment["project_id"], segment["index"],
            segment["subtopic"], segment.get("title"), segment["status"],
            segment.get("duration_seconds"), segment.get("text"),
            segment.get("primary_audio_artifact_id"),
            segment.get("error", {}).get("code") if segment.get("error") else None,
            segment.get("error", {}).get("message") if segment.get("error") else None,
            segment.get("error", {}).get("retryable", False) if segment.get("error") else 0,
            segment.get("created_at", now), now,
        ),
    )
    conn.commit()


def _upsert_provenance(db_path: str, segment_id: str, provenance: dict) -> None:
    conn = _get_conn(db_path)
    _ensure_schema(conn)
    conn.execute(
        """INSERT INTO provenance (segment_id, prompt_id, model, source_refs,
           claim_notes, validation_status, validation_errors, validated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(segment_id) DO UPDATE SET
               prompt_id = excluded.prompt_id,
               model = excluded.model,
               source_refs = excluded.source_refs,
               claim_notes = excluded.claim_notes,
               validation_status = excluded.validation_status,
               validation_errors = excluded.validation_errors,
               validated_at = excluded.validated_at""",
        (
            segment_id,
            provenance["prompt_id"],
            provenance["model"],
            json.dumps(provenance.get("source_refs", [])),
            json.dumps(provenance.get("claim_notes", [])),
            provenance["validation_status"],
            json.dumps(provenance.get("validation_errors", [])),
            provenance.get("validated_at"),
        ),
    )
    conn.commit()


def _add_artifact(db_path: str, artifact: dict) -> None:
    conn = _get_conn(db_path)
    _ensure_schema(conn)
    conn.execute(
        """INSERT OR REPLACE INTO artifacts (artifact_id, project_id, segment_id,
           kind, content_type, duration_seconds, size_bytes, checksum_sha256,
           status, download_url, signed_transfer_url, signed_transfer_expires_at,
           created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            artifact["artifact_id"], artifact["project_id"],
            artifact.get("segment_id"), artifact["kind"],
            artifact["content_type"], artifact.get("duration_seconds"),
            artifact.get("size_bytes"), artifact.get("checksum_sha256"),
            artifact["status"], artifact["download_url"],
            artifact.get("signed_transfer_url"),
            artifact.get("signed_transfer_expires_at"),
            artifact.get("created_at", _now_iso()),
        ),
    )
    conn.commit()


def _get_artifact_by_id(db_path: str, artifact_id: str) -> dict | None:
    conn = _get_conn(db_path)
    _ensure_schema(conn)
    row = conn.execute(
        "SELECT * FROM artifacts WHERE artifact_id = ?", (artifact_id,),
    ).fetchone()
    return _row_to_artifact(row) if row else None


def _update_project_status(db_path: str, project_id: str, status: str,
                           final_download_ready: bool = False,
                           final_artifact_id: str | None = None) -> None:
    conn = _get_conn(db_path)
    _ensure_schema(conn)
    now = _now_iso()
    conn.execute(
        """UPDATE projects SET status = ?, updated_at = ?,
           final_download_ready = ?, final_artifact_id = ?
           WHERE project_id = ?""",
        (status, now, 1 if final_download_ready else 0, final_artifact_id, project_id),
    )
    conn.commit()


def _add_project_error(db_path: str, project_id: str, code: str, message: str,
                       retryable: bool = False) -> None:
    conn = _get_conn(db_path)
    _ensure_schema(conn)
    conn.execute(
        """INSERT INTO project_errors (project_id, code, message, retryable, created_at)
           VALUES (?, ?, ?, ?, ?)""",
        (project_id, code, message, 1 if retryable else 0, _now_iso()),
    )
    conn.commit()


# ── Row converters ──────────────────────────────────────────────────────


def _row_to_project_summary(row: sqlite3.Row) -> dict:
    return {
        "project_id": row["project_id"],
        "topic": row["topic"],
        "bpm": row["bpm"],
        "duration_minutes": row["duration_minutes"],
        "status": row["status"],
        "revision_token": row["revision_token"],
        "segment_count": row["segment_count"],
        "ready_segment_count": row["ready_segment_count"],
        "final_download_ready": bool(row["final_download_ready"]),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def _row_to_project_full(row: sqlite3.Row) -> dict:
    return {
        "project_id": row["project_id"],
        "owner_id": row["owner_id"],
        "topic": row["topic"],
        "bpm": row["bpm"],
        "duration_minutes": row["duration_minutes"],
        "status": row["status"],
        "revision_token": row["revision_token"],
        "final_download_ready": bool(row["final_download_ready"]),
        "final_artifact_id": row["final_artifact_id"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "deleted_at": row["deleted_at"],
    }


def _row_to_segment(row: sqlite3.Row) -> dict:
    error = None
    if row["error_code"]:
        error = {
            "code": row["error_code"],
            "message": row["error_message"],
            "retryable": bool(row["error_retryable"]),
        }
    return {
        "segment_id": row["segment_id"],
        "index": row["idx"],
        "subtopic": row["subtopic"],
        "title": row["title"],
        "status": row["status"],
        "duration_seconds": row["duration_seconds"],
        "text": row["text"],
        "provenance": None,  # filled in by caller
        "artifact_ids": [],  # filled in by caller or artifact query
        "primary_audio_artifact_id": row["primary_audio_artifact_id"],
        "error": error,
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def _row_to_provenance(row: sqlite3.Row) -> dict | None:
    if row is None:
        return None
    return {
        "prompt_id": row["prompt_id"],
        "model": row["model"],
        "source_refs": json.loads(row["source_refs"]),
        "claim_notes": json.loads(row["claim_notes"]),
        "validation_status": row["validation_status"],
        "validation_errors": json.loads(row["validation_errors"]),
        "validated_at": row["validated_at"],
    }


def _row_to_artifact(row: sqlite3.Row) -> dict:
    return {
        "artifact_id": row["artifact_id"],
        "kind": row["kind"],
        "segment_id": row["segment_id"],
        "content_type": row["content_type"],
        "duration_seconds": row["duration_seconds"],
        "size_bytes": row["size_bytes"],
        "checksum_sha256": row["checksum_sha256"],
        "status": row["status"],
        "download_url": row["download_url"],
        "signed_transfer_url": row["signed_transfer_url"],
        "signed_transfer_expires_at": row["signed_transfer_expires_at"],
        "created_at": row["created_at"],
    }


def _row_to_error(row: sqlite3.Row) -> dict:
    return {
        "code": row["code"],
        "message": row["message"],
        "retryable": bool(row["retryable"]),
    }


# ── Async wrappers (for use in routers) ─────────────────────────────────

async def create_project(db_path: str, project_id: str, owner_id: str, topic: str,
                         bpm: int, duration_minutes: int) -> dict:
    return await _run_in_executor(_create_project, db_path, project_id, owner_id,
                                  topic, bpm, duration_minutes)


async def list_projects(db_path: str, owner_id: str) -> list[dict]:
    return await _run_in_executor(_list_projects, db_path, owner_id)


async def get_project(db_path: str, project_id: str) -> dict | None:
    return await _run_in_executor(_get_project_by_id, db_path, project_id)


async def delete_project(db_path: str, project_id: str) -> dict | None:
    return await _run_in_executor(_delete_project, db_path, project_id)


async def upsert_segment(db_path: str, segment: dict) -> None:
    return await _run_in_executor(_upsert_segment, db_path, segment)


async def upsert_provenance(db_path: str, segment_id: str, provenance: dict) -> None:
    return await _run_in_executor(_upsert_provenance, db_path, segment_id, provenance)


async def add_artifact(db_path: str, artifact: dict) -> None:
    return await _run_in_executor(_add_artifact, db_path, artifact)


async def get_artifact(db_path: str, artifact_id: str) -> dict | None:
    return await _run_in_executor(_get_artifact_by_id, db_path, artifact_id)


async def update_project_status(db_path: str, project_id: str, status: str,
                                final_download_ready: bool = False,
                                final_artifact_id: str | None = None) -> None:
    return await _run_in_executor(_update_project_status, db_path, project_id, status,
                                  final_download_ready, final_artifact_id)


async def add_project_error(db_path: str, project_id: str, code: str, message: str,
                            retryable: bool = False) -> None:
    return await _run_in_executor(_add_project_error, db_path, project_id, code, message,
                                  retryable)