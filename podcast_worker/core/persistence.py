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
    resolved_path = Path(db_path)
    if not resolved_path.is_absolute():
        project_root = Path(__file__).resolve().parent.parent.parent
        resolved_path = project_root / resolved_path
    resolved_path.parent.mkdir(parents=True, exist_ok=True)

    db_key = str(resolved_path)
    tid = threading.get_ident()
    key = f"{db_key}:{tid}"
    with _conn_lock:
        if key not in _conns:
            conn = sqlite3.connect(db_key, check_same_thread=False, timeout=30)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=30000")
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
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version INTEGER PRIMARY KEY,
            applied_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS execution_snapshots (
            snapshot_id TEXT PRIMARY KEY,
            snapshot_type TEXT NOT NULL CHECK(snapshot_type IN ('llm', 'tts')),
            profile_id TEXT NOT NULL,
            revision TEXT NOT NULL,
            canonical_json TEXT NOT NULL,
            sha256 TEXT NOT NULL,
            created_at TEXT NOT NULL,
            UNIQUE(snapshot_type, profile_id, revision, sha256)
        );
        CREATE TABLE IF NOT EXISTS outline_previews (
            outline_preview_id TEXT PRIMARY KEY,
            owner_id TEXT NOT NULL,
            topic TEXT NOT NULL,
            bpm INTEGER NOT NULL,
            duration_minutes INTEGER NOT NULL,
            outline_json TEXT NOT NULL,
            llm_profile_id TEXT NOT NULL,
            tts_profile_id TEXT NOT NULL,
            llm_snapshot_id TEXT NOT NULL REFERENCES execution_snapshots(snapshot_id),
            tts_snapshot_id TEXT NOT NULL REFERENCES execution_snapshots(snapshot_id),
            routing_revision TEXT NOT NULL,
            tts_routing_revision TEXT NOT NULL,
            ledger_id TEXT NOT NULL,
            expires_at TEXT NOT NULL,
            consumed_at TEXT,
            project_id TEXT REFERENCES projects(project_id),
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS episode_ledgers (
            ledger_id TEXT PRIMARY KEY,
            owner_id TEXT NOT NULL,
            project_id TEXT REFERENCES projects(project_id),
            llm_snapshot_id TEXT NOT NULL REFERENCES execution_snapshots(snapshot_id),
            tts_snapshot_id TEXT NOT NULL REFERENCES execution_snapshots(snapshot_id),
            policy_json TEXT NOT NULL,
            currency TEXT,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS episode_ledger_entries (
            entry_id TEXT PRIMARY KEY,
            ledger_id TEXT NOT NULL REFERENCES episode_ledgers(ledger_id) ON DELETE CASCADE,
            category TEXT NOT NULL CHECK(category IN ('llm', 'tts')),
            operation_type TEXT NOT NULL,
            correlation_id TEXT NOT NULL,
            resource_unit TEXT NOT NULL,
            amount INTEGER NOT NULL,
            state TEXT NOT NULL CHECK(state IN ('reserved', 'actual', 'released', 'rejected', 'observed')),
            pricing_json TEXT,
            attempt_id TEXT,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS work_items (
            work_id TEXT PRIMARY KEY,
            project_id TEXT REFERENCES projects(project_id) ON DELETE CASCADE,
            ledger_id TEXT REFERENCES episode_ledgers(ledger_id),
            kind TEXT NOT NULL,
            state TEXT NOT NULL DEFAULT 'pending',
            payload_json TEXT NOT NULL,
            lease_owner TEXT,
            lease_expires_at TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS dialogue_turns (
            turn_id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
            segment_id TEXT REFERENCES segments(segment_id) ON DELETE CASCADE,
            idx INTEGER NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('interviewer', 'guest')),
            text TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS dialogue_fragments (
            fragment_id TEXT PRIMARY KEY,
            turn_id TEXT NOT NULL REFERENCES dialogue_turns(turn_id) ON DELETE CASCADE,
            idx INTEGER NOT NULL,
            text TEXT NOT NULL,
            plan_id TEXT,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS tts_request_plans (
            plan_id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
            segment_id TEXT REFERENCES segments(segment_id) ON DELETE CASCADE,
            strategy TEXT NOT NULL,
            state TEXT NOT NULL DEFAULT 'pending',
            payload_json TEXT NOT NULL,
            output_artifact_id TEXT REFERENCES artifacts(artifact_id),
            lease_owner TEXT,
            lease_expires_at TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS execution_attempts (
            attempt_id TEXT PRIMARY KEY,
            ledger_id TEXT NOT NULL REFERENCES episode_ledgers(ledger_id),
            plan_id TEXT,
            category TEXT NOT NULL CHECK(category IN ('llm', 'tts')),
            correlation_id TEXT NOT NULL,
            snapshot_revision TEXT NOT NULL,
            binding_json TEXT NOT NULL,
            outcome TEXT NOT NULL,
            usage_json TEXT,
            cost_micros INTEGER,
            error_json TEXT,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS audio_assemblies (
            assembly_id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
            segment_id TEXT REFERENCES segments(segment_id) ON DELETE CASCADE,
            manifest_json TEXT NOT NULL,
            manifest_sha256 TEXT NOT NULL,
            processing_revision TEXT NOT NULL,
            artifact_id TEXT REFERENCES artifacts(artifact_id),
            checksum_sha256 TEXT,
            duration_seconds REAL,
            state TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_preview_owner ON outline_previews(owner_id, expires_at);
        CREATE INDEX IF NOT EXISTS idx_ledger_entries_ledger ON episode_ledger_entries(ledger_id);
        CREATE INDEX IF NOT EXISTS idx_work_reconcile ON work_items(state, lease_expires_at);
        CREATE INDEX IF NOT EXISTS idx_tts_plans_reconcile ON tts_request_plans(state, lease_expires_at);
        INSERT OR IGNORE INTO schema_migrations(version, applied_at) VALUES (3, CURRENT_TIMESTAMP);

        CREATE INDEX IF NOT EXISTS idx_segments_project ON segments(project_id);
        CREATE INDEX IF NOT EXISTS idx_artifacts_project ON artifacts(project_id);
        CREATE INDEX IF NOT EXISTS idx_artifacts_segment ON artifacts(segment_id);
        CREATE INDEX IF NOT EXISTS idx_errors_project ON project_errors(project_id);
    """)
    # Existing installations may predate durable routing columns.  Keep this
    # additive migration serialized with all other writers.
    conn.execute("BEGIN IMMEDIATE")
    try:
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(outline_previews)")}
        for name, definition in (
            ("llm_snapshot_id", "TEXT REFERENCES execution_snapshots(snapshot_id)"),
            ("tts_snapshot_id", "TEXT REFERENCES execution_snapshots(snapshot_id)"),
            ("tts_routing_revision", "TEXT"),
        ):
            if name not in columns:
                conn.execute(f"ALTER TABLE outline_previews ADD COLUMN {name} {definition}")
        conn.execute("INSERT OR IGNORE INTO schema_migrations(version, applied_at) VALUES (3, ?)", (_now_iso(),))
    except Exception:
        conn.rollback()
        raise
    conn.commit()


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
def _persist_execution_snapshot(conn: sqlite3.Connection, snapshot: dict, kind: str, now: str) -> str:
    """Insert an immutable snapshot or return the canonical row's existing ID."""
    canonical = json.dumps(snapshot["payload"], sort_keys=True, separators=(",", ":"))
    conn.execute(
        """INSERT OR IGNORE INTO execution_snapshots
           (snapshot_id, snapshot_type, profile_id, revision, canonical_json, sha256, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            snapshot["snapshot_id"],
            kind,
            snapshot["profile_id"],
            snapshot["revision"],
            canonical,
            snapshot["sha256"],
            now,
        ),
    )
    row = conn.execute(
        """SELECT snapshot_id FROM execution_snapshots
           WHERE snapshot_type = ? AND profile_id = ? AND revision = ? AND sha256 = ?""",
        (kind, snapshot["profile_id"], snapshot["revision"], snapshot["sha256"]),
    ).fetchone()
    if row is None:
        raise sqlite3.IntegrityError("execution snapshot identity conflict")
    return str(row["snapshot_id"])


def _create_preview_binding(db_path: str, preview: dict, llm_snapshot: dict,
                            tts_snapshot: dict, ledger: dict) -> None:
    """Atomically persist immutable paired snapshots, preview, and episode ledger."""
    conn = _get_conn(db_path)
    _ensure_schema(conn)
    now = _now_iso()
    conn.execute("BEGIN IMMEDIATE")
    try:
        snapshot_ids = [
            _persist_execution_snapshot(conn, snapshot, kind, now)
            for snapshot, kind in ((llm_snapshot, "llm"), (tts_snapshot, "tts"))
        ]
        conn.execute("""INSERT INTO episode_ledgers
            (ledger_id, owner_id, llm_snapshot_id, tts_snapshot_id, policy_json, currency, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (ledger["ledger_id"], preview["owner_id"], *snapshot_ids,
             json.dumps(ledger["policy"], sort_keys=True, separators=(",", ":")),
             ledger.get("currency"), now))
        conn.execute("""INSERT INTO outline_previews
            (outline_preview_id, owner_id, topic, bpm, duration_minutes, outline_json,
             llm_profile_id, tts_profile_id, llm_snapshot_id, tts_snapshot_id, routing_revision,
             tts_routing_revision, ledger_id, expires_at, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (preview["outline_preview_id"], preview["owner_id"], preview["topic"], preview["bpm"],
             preview["duration_minutes"], json.dumps(preview["outline"], sort_keys=True),
             preview["llm_profile_id"], preview["tts_profile_id"], *snapshot_ids,
             preview["routing_revision"], preview["tts_routing_revision"], ledger["ledger_id"],
             preview["expires_at"], now))
    except Exception:
        conn.rollback()
        raise
    conn.commit()


def _get_preview_binding(db_path: str, preview_id: str, owner_id: str) -> dict | None:
    conn = _get_conn(db_path)
    _ensure_schema(conn)
    row = conn.execute("SELECT * FROM outline_previews WHERE outline_preview_id = ? AND owner_id = ?",
                       (preview_id, owner_id)).fetchone()
    if row is None:
        return None
    result = dict(row)
    result["outline"] = json.loads(result.pop("outline_json"))
    return result


def _consume_preview_binding(db_path: str, preview_id: str, owner_id: str, project_id: str,
                             topic: str, bpm: int, duration_minutes: int, llm_profile_id: str,
                             tts_profile_id: str, routing_revision: str, tts_routing_revision: str) -> str:
    conn = _get_conn(db_path)
    _ensure_schema(conn)
    conn.execute("BEGIN IMMEDIATE")
    try:
        row = conn.execute("SELECT * FROM outline_previews WHERE outline_preview_id = ? AND owner_id = ?",
                           (preview_id, owner_id)).fetchone()
        if row is None:
            result = "preview_not_found"
        elif row["consumed_at"] is not None or row["expires_at"] <= _now_iso():
            result = "preview_expired"
        elif not row["tts_snapshot_id"]:
            result = "preview_binding_required"
        elif (row["topic"], row["bpm"], row["duration_minutes"], row["llm_profile_id"], row["tts_profile_id"],
              row["routing_revision"], row["tts_routing_revision"]) != (topic, bpm, duration_minutes,
              llm_profile_id, tts_profile_id, routing_revision, tts_routing_revision):
            result = "preview_profile_mismatch"
        else:
            conn.execute("UPDATE outline_previews SET consumed_at = ?, project_id = ? WHERE outline_preview_id = ?",
                         (_now_iso(), project_id, preview_id))
            conn.execute("UPDATE episode_ledgers SET project_id = ? WHERE ledger_id = ?",
                         (project_id, row["ledger_id"]))
            result = "ok"
    except Exception:
        conn.rollback()
        raise
    conn.commit()
    return result


def _append_ledger_entry(db_path: str, entry: dict) -> None:
    conn = _get_conn(db_path)
    _ensure_schema(conn)
    conn.execute("""INSERT INTO episode_ledger_entries
        (entry_id, ledger_id, category, operation_type, correlation_id, resource_unit, amount,
         state, pricing_json, attempt_id, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (entry["entry_id"], entry["ledger_id"], entry["category"], entry["operation_type"],
         entry["correlation_id"], entry["resource_unit"], entry["amount"], entry["state"],
         json.dumps(entry["pricing"], sort_keys=True) if entry.get("pricing") else None,
         entry.get("attempt_id"), entry.get("created_at", _now_iso())))
    conn.commit()
def _reserve_ledger_operation(db_path: str, entry: dict, attempt: dict) -> bool:
    """Atomically record an accepted attempt and its reservation before dispatch."""
    conn = _get_conn(db_path)
    _ensure_schema(conn)
    conn.execute("BEGIN IMMEDIATE")
    try:
        ledger = conn.execute("SELECT policy_json FROM episode_ledgers WHERE ledger_id = ?",
                              (entry["ledger_id"],)).fetchone()
        if ledger is None:
            raise ValueError("unknown ledger")
        policy = json.loads(ledger["policy_json"])
        cap = policy.get("caps", {}).get(entry["category"])
        used = conn.execute("""SELECT COALESCE(SUM(amount), 0) AS total FROM episode_ledger_entries
            WHERE ledger_id = ? AND category = ? AND state IN ('reserved', 'actual')""",
                            (entry["ledger_id"], entry["category"])).fetchone()["total"]
        accepted = policy.get("mode", policy.get("enforcement", "off")) != "enforced" or cap is None or used + entry["amount"] <= cap
        if accepted:
            conn.execute("""INSERT INTO execution_attempts
                (attempt_id, ledger_id, plan_id, category, correlation_id, snapshot_revision, binding_json, outcome, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, 'accepted', ?)""",
                (attempt["attempt_id"], entry["ledger_id"], attempt.get("plan_id"), entry["category"],
                 entry["correlation_id"], attempt["snapshot_revision"],
                 json.dumps(attempt["binding"], sort_keys=True), _now_iso()))
            conn.execute("""INSERT INTO episode_ledger_entries
                (entry_id, ledger_id, category, operation_type, correlation_id, resource_unit, amount, state, pricing_json, attempt_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, 'reserved', ?, ?, ?)""",
                (entry["entry_id"], entry["ledger_id"], entry["category"], entry["operation_type"],
                 entry["correlation_id"], entry["resource_unit"], entry["amount"],
                 json.dumps(entry.get("pricing", {}), sort_keys=True), attempt["attempt_id"], _now_iso()))
    except Exception:
        conn.rollback()
        raise
    conn.commit()
    return accepted


def _settle_ledger_operation(db_path: str, ledger_id: str, attempt_id: str, entry: dict,
                             outcome: str, usage: dict | None = None, error: dict | None = None) -> None:
    conn = _get_conn(db_path)
    _ensure_schema(conn)
    conn.execute("BEGIN IMMEDIATE")
    try:
        conn.execute("""UPDATE execution_attempts SET outcome = ?, usage_json = ?, cost_micros = ?, error_json = ?
            WHERE attempt_id = ? AND ledger_id = ?""",
            (outcome, json.dumps(usage, sort_keys=True) if usage else None, entry.get("cost_micros"),
             json.dumps(error, sort_keys=True) if error else None, attempt_id, ledger_id))
        state = "actual" if outcome == "succeeded" else "released"
        conn.execute("""UPDATE episode_ledger_entries SET state = ? WHERE ledger_id = ? AND attempt_id = ?
            AND state = 'reserved'""", (state, ledger_id, attempt_id))
        if conn.execute("SELECT changes()").fetchone()[0] != 1:
            raise ValueError("missing ledger reservation")
    except Exception:
        conn.rollback()
        raise
    conn.commit()


def _update_preview_outline(db_path: str, preview_id: str, outline: dict) -> None:
    conn = _get_conn(db_path)
    _ensure_schema(conn)
    conn.execute("UPDATE outline_previews SET outline_json = ? WHERE outline_preview_id = ?",
                 (json.dumps(outline, sort_keys=True), preview_id))
    conn.commit()


def _upsert_durable_record(db_path: str, table: str, record: dict) -> None:
    allowed = {"work_items", "tts_request_plans", "execution_attempts", "audio_assemblies",
               "dialogue_turns", "dialogue_fragments"}
    if table not in allowed:
        raise ValueError("unsupported durable table")
    conn = _get_conn(db_path)
    _ensure_schema(conn)
    now = _now_iso()
    values = dict(record)
    for key, value in values.items():
        if isinstance(value, (dict, list)):
            values[key] = json.dumps(value, sort_keys=True)
    if table in {"work_items", "tts_request_plans", "audio_assemblies"}:
        values.setdefault("created_at", now)
        values.setdefault("updated_at", now)
    else:
        values.setdefault("created_at", now)
    columns = list(values)
    conn.execute(f"INSERT OR REPLACE INTO {table} ({', '.join(columns)}) VALUES ({', '.join('?' for _ in columns)})",
                 [values[column] for column in columns])
    conn.commit()
def _claim_reconcilable_work(db_path: str, table: str, worker_id: str, lease_expires_at: str) -> list[dict]:
    if table not in {"work_items", "tts_request_plans"}:
        raise ValueError("unsupported lease table")
    conn = _get_conn(db_path)
    _ensure_schema(conn)
    conn.execute("BEGIN IMMEDIATE")
    try:
        rows = conn.execute(
            f"""SELECT * FROM {table} WHERE state = 'pending'
                OR (state = 'leased' AND lease_expires_at IS NOT NULL AND lease_expires_at <= ?)""",
            (_now_iso(),),
        ).fetchall()
        for row in rows:
            key = "work_id" if table == "work_items" else "plan_id"
            conn.execute(f"UPDATE {table} SET state = 'leased', lease_owner = ?, lease_expires_at = ?, updated_at = ? WHERE {key} = ?",
                         (worker_id, lease_expires_at, _now_iso(), row[key]))
    except Exception:
        conn.rollback()
        raise
    conn.commit()
    return [dict(row) for row in rows]
def _create_project_execution(db_path: str, project_id: str, owner_id: str, llm_snapshot: dict,
                              tts_snapshot: dict, ledger: dict) -> None:
    """Persist server-only paired snapshots and the project's single episode ledger."""
    conn = _get_conn(db_path)
    _ensure_schema(conn)
    now = _now_iso()
    conn.execute("BEGIN IMMEDIATE")
    try:
        snapshot_ids = [
            _persist_execution_snapshot(conn, snapshot, kind, now)
            for snapshot, kind in ((llm_snapshot, "llm"), (tts_snapshot, "tts"))
        ]
        conn.execute("""INSERT INTO episode_ledgers
            (ledger_id, owner_id, project_id, llm_snapshot_id, tts_snapshot_id, policy_json, currency, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (ledger["ledger_id"], owner_id, project_id, *snapshot_ids,
             json.dumps(ledger["policy"], sort_keys=True), ledger.get("currency"), now))
    except Exception:
        conn.rollback()
        raise
    conn.commit()
def _get_project_execution(db_path: str, project_id: str) -> dict | None:
    """Hydrate server-only paired snapshots and ledger for pipeline/reconciliation workers."""
    conn = _get_conn(db_path)
    _ensure_schema(conn)
    row = conn.execute("""SELECT l.ledger_id, l.policy_json, l.currency,
        ls.snapshot_id AS llm_snapshot_id, ls.profile_id AS llm_profile_id,
        ls.revision AS llm_revision, ls.canonical_json AS llm_snapshot_json,
        ts.snapshot_id AS tts_snapshot_id, ts.profile_id AS tts_profile_id,
        ts.revision AS tts_revision, ts.canonical_json AS tts_snapshot_json
        FROM episode_ledgers l
        JOIN execution_snapshots ls ON ls.snapshot_id = l.llm_snapshot_id
        JOIN execution_snapshots ts ON ts.snapshot_id = l.tts_snapshot_id
        WHERE l.project_id = ?""", (project_id,)).fetchone()
    if row is None:
        return None
    result = dict(row)
    result["policy"] = json.loads(result.pop("policy_json"))
    result["llm_snapshot"] = json.loads(result.pop("llm_snapshot_json"))
    result["tts_snapshot"] = json.loads(result.pop("tts_snapshot_json"))
    return result


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
        "model": "server-managed",
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
async def create_preview_binding(db_path: str, preview: dict, llm_snapshot: dict,
                                 tts_snapshot: dict, ledger: dict) -> None:
    return await _run_in_executor(_create_preview_binding, db_path, preview, llm_snapshot, tts_snapshot, ledger)
async def create_project_execution(db_path: str, project_id: str, owner_id: str, llm_snapshot: dict,
                                   tts_snapshot: dict, ledger: dict) -> None:
    return await _run_in_executor(_create_project_execution, db_path, project_id, owner_id,
                                  llm_snapshot, tts_snapshot, ledger)


async def get_preview_binding(db_path: str, preview_id: str, owner_id: str) -> dict | None:
    return await _run_in_executor(_get_preview_binding, db_path, preview_id, owner_id)


async def consume_preview_binding(db_path: str, preview_id: str, owner_id: str, project_id: str,
                                  topic: str, bpm: int, duration_minutes: int, llm_profile_id: str,
                                  tts_profile_id: str, routing_revision: str, tts_routing_revision: str) -> str:
    return await _run_in_executor(_consume_preview_binding, db_path, preview_id, owner_id, project_id,
                                  topic, bpm, duration_minutes, llm_profile_id, tts_profile_id,
                                  routing_revision, tts_routing_revision)


async def append_ledger_entry(db_path: str, entry: dict) -> None:
    return await _run_in_executor(_append_ledger_entry, db_path, entry)
async def reserve_ledger_operation(db_path: str, entry: dict, attempt: dict) -> bool:
    return await _run_in_executor(_reserve_ledger_operation, db_path, entry, attempt)


async def settle_ledger_operation(db_path: str, ledger_id: str, attempt_id: str, entry: dict,
                                  outcome: str, usage: dict | None = None, error: dict | None = None) -> None:
    return await _run_in_executor(_settle_ledger_operation, db_path, ledger_id, attempt_id, entry, outcome, usage, error)


async def update_preview_outline(db_path: str, preview_id: str, outline: dict) -> None:
    return await _run_in_executor(_update_preview_outline, db_path, preview_id, outline)


async def upsert_durable_record(db_path: str, table: str, record: dict) -> None:
    return await _run_in_executor(_upsert_durable_record, db_path, table, record)
async def claim_reconcilable_work(db_path: str, table: str, worker_id: str,
                                  lease_expires_at: str) -> list[dict]:
    return await _run_in_executor(_claim_reconcilable_work, db_path, table, worker_id, lease_expires_at)
async def get_project_execution(db_path: str, project_id: str) -> dict | None:
    return await _run_in_executor(_get_project_execution, db_path, project_id)