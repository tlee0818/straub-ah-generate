"""SQLite-backed durable persistence for PodcastProject v1.

Replaces the legacy in-memory dict store with sqlite3 via ThreadPoolExecutor
so async handlers never block the event loop on sync SQLite calls.

Schema invariants:
- provenance.validation_status MUST be stored before a segment transitions to tts
- artifact metadata MUST be stored before segment transitions to ready
- all required segments ready before final_download_ready=true
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import NAMESPACE_URL, uuid4, uuid5
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

        CREATE TABLE IF NOT EXISTS project_generation (
            project_id TEXT PRIMARY KEY REFERENCES projects(project_id) ON DELETE CASCADE,
            accepted_outline_json TEXT NOT NULL,
            interviewer_profile_json TEXT NOT NULL,
            sme_profile_json TEXT NOT NULL,
            input_schema_hash TEXT NOT NULL,
            llm_snapshot_id TEXT NOT NULL,
            tts_snapshot_id TEXT NOT NULL,
            generation_contract_version INTEGER NOT NULL DEFAULT 1,
            planned_segment_count INTEGER NOT NULL CHECK (planned_segment_count > 0),
            progress_version INTEGER NOT NULL DEFAULT 0 CHECK (progress_version >= 0),
            disposition TEXT NOT NULL DEFAULT 'active',
            terminal_outcome TEXT,
            cancellation_requested_at TEXT,
            cancellation_observed_at TEXT,
            terminal_at TEXT,
            last_transition_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS generation_stage_results (
            project_id TEXT NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
            scope_id TEXT NOT NULL,
            stage_name TEXT NOT NULL,
            state TEXT NOT NULL DEFAULT 'pending',
            result_json TEXT,
            result_hash TEXT,
            safe_error_code TEXT,
            attempt_id TEXT,
            started_at TEXT,
            completed_at TEXT,
            cancelled_at TEXT,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (project_id, scope_id, stage_name)
        );

        CREATE TABLE IF NOT EXISTS project_pipeline (
            work_id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL UNIQUE REFERENCES projects(project_id) ON DELETE CASCADE,
            state TEXT NOT NULL DEFAULT 'pending',
            lease_owner TEXT,
            lease_epoch INTEGER NOT NULL DEFAULT 0,
            lease_expires_at TEXT,
            heartbeat_at TEXT,
            settled_at TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_stage_results_project
            ON generation_stage_results(project_id, stage_name);
        CREATE INDEX IF NOT EXISTS idx_pipeline_claim
            ON project_pipeline(state, lease_expires_at, created_at);

        CREATE TABLE IF NOT EXISTS provider_attempts (
            attempt_id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL REFERENCES projects(project_id) ON DELETE CASCADE,
            work_id TEXT NOT NULL REFERENCES project_pipeline(work_id) ON DELETE CASCADE,
            logical_operation_id TEXT NOT NULL,
            scope_id TEXT NOT NULL,
            stage_name TEXT NOT NULL,
            state TEXT NOT NULL CHECK (
                state IN ('reserved','dispatched','completed','failed_unknown')
            ),
            lease_owner TEXT NOT NULL,
            lease_epoch INTEGER NOT NULL,
            result_json TEXT,
            result_hash TEXT,
            reserved_at TEXT NOT NULL,
            dispatched_at TEXT,
            completed_at TEXT,
            failed_at TEXT,
            UNIQUE(project_id, logical_operation_id)
        );

        CREATE INDEX IF NOT EXISTS idx_provider_attempts_recovery
            ON provider_attempts(work_id, state);

        CREATE TABLE IF NOT EXISTS outline_preview_bindings (
            outline_preview_id TEXT PRIMARY KEY,
            owner_id TEXT NOT NULL,
            topic TEXT NOT NULL,
            bpm INTEGER NOT NULL,
            duration_minutes INTEGER NOT NULL,
            planned_segment_count INTEGER NOT NULL,
            outline_json TEXT NOT NULL,
            outline_hash TEXT NOT NULL,
            request_hash TEXT NOT NULL,
            llm_profile_id TEXT NOT NULL,
            tts_profile_id TEXT NOT NULL,
            llm_routing_revision TEXT NOT NULL,
            tts_routing_revision TEXT NOT NULL,
            expires_at TEXT NOT NULL,
            created_at TEXT NOT NULL,
            consumed_at TEXT,
            consumed_project_id TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_preview_binding_expiry
            ON outline_preview_bindings(owner_id, expires_at, consumed_at);
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

    # Additive migration for databases created by earlier service versions.
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(segments)")}
    if "segment_type" not in columns:
        conn.execute("ALTER TABLE segments ADD COLUMN segment_type TEXT NOT NULL DEFAULT 'content'")
    if "planned_duration_seconds" not in columns:
        conn.execute("ALTER TABLE segments ADD COLUMN planned_duration_seconds REAL")
    conn.execute("PRAGMA user_version = 4")
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


def _create_outline_preview_binding(
    db_path: str, outline_preview_id: str, owner_id: str, topic: str,
    bpm: int, duration_minutes: int, outline: dict,
    llm_profile_id: str, tts_profile_id: str,
    llm_routing_revision: str, tts_routing_revision: str, expires_at: str,
) -> dict:
    conn = _get_conn(db_path)
    _ensure_schema(conn)
    sections = outline.get("sections") if isinstance(outline, dict) else None
    expected_count = min(max((duration_minutes + 1) // 2, 1), 12)
    if not isinstance(sections, list) or len(sections) != expected_count:
        raise ValueError("preview_outline_count_mismatch")
    outline_json = json.dumps(outline, sort_keys=True, separators=(",", ":"))
    outline_hash = hashlib.sha256(outline_json.encode()).hexdigest()
    request_value = {
        "topic": " ".join(topic.split()), "bpm": bpm,
        "duration_minutes": duration_minutes,
        "planned_segment_count": expected_count,
        "llm_profile_id": llm_profile_id, "tts_profile_id": tts_profile_id,
        "llm_routing_revision": llm_routing_revision,
        "tts_routing_revision": tts_routing_revision,
    }
    request_hash = hashlib.sha256(
        json.dumps(request_value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    now = _now_iso()
    conn.execute(
        """INSERT INTO outline_preview_bindings
           (outline_preview_id,owner_id,topic,bpm,duration_minutes,
            planned_segment_count,outline_json,outline_hash,request_hash,
            llm_profile_id,tts_profile_id,llm_routing_revision,
            tts_routing_revision,expires_at,created_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (outline_preview_id, owner_id, request_value["topic"], bpm,
         duration_minutes, expected_count, outline_json, outline_hash,
         request_hash, llm_profile_id, tts_profile_id, llm_routing_revision,
         tts_routing_revision, expires_at, now),
    )
    conn.commit()
    return {
        "outline_preview_id": outline_preview_id,
        "llm_profile_id": llm_profile_id,
        "tts_profile_id": tts_profile_id,
        "llm_routing_revision": llm_routing_revision,
        "tts_routing_revision": tts_routing_revision,
        "expires_at": expires_at,
        "request_hash": request_hash,
        "outline_hash": outline_hash,
    }


def _create_progress_project(
    db_path: str,
    project_id: str,
    owner_id: str,
    topic: str,
    bpm: int,
    duration_minutes: int,
    accepted_outline: dict,
    interviewer_profile: dict | None,
    sme_profile: dict | None,
    llm_snapshot_id: str,
    tts_snapshot_id: str,
    outline_preview_id: str | None = None,
) -> dict:
    """Atomically materialize a progress-capable project and its pending work."""
    conn = _get_conn(db_path)
    _ensure_schema(conn)
    sections = accepted_outline.get("sections") if isinstance(accepted_outline, dict) else None
    expected_count = min(max((duration_minutes + 1) // 2, 1), 12)
    if not isinstance(sections, list) or len(sections) != expected_count:
        raise ValueError(f"accepted outline must contain exactly {expected_count} sections")
    if [section.get("index") for section in sections] != list(range(expected_count)):
        raise ValueError("accepted outline section indices must be contiguous from zero")
    for section in sections:
        if section.get("segment_type", "content") not in {"intro", "content", "outro"}:
            raise ValueError("accepted outline contains an invalid segment_type")
        if not str(section.get("topic", "")).strip():
            raise ValueError("accepted outline contains a blank topic")
        if not str(section.get("title", "")).strip():
            raise ValueError("accepted outline contains a blank title")
        duration = section.get("approx_duration_seconds")
        if isinstance(duration, bool) or not isinstance(duration, (int, float)) or duration <= 0:
            raise ValueError("accepted outline contains an invalid planned duration")

    now = _now_iso()
    outline_json = json.dumps(accepted_outline, sort_keys=True, separators=(",", ":"))
    immutable_input = {
        "accepted_outline": accepted_outline,
        "interviewer_profile": interviewer_profile or {},
        "sme_profile": sme_profile or {},
        "llm_snapshot_id": llm_snapshot_id,
        "tts_snapshot_id": tts_snapshot_id,
        "generation_contract_version": 1,
    }
    input_hash = hashlib.sha256(
        json.dumps(immutable_input, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    segment_rows = [
        (
            f"seg_{uuid5(NAMESPACE_URL, '{}:{}'.format(project_id, section['index'])).hex}",
            section,
        )
        for section in sections
    ]

    try:
        conn.execute("BEGIN IMMEDIATE")
        if outline_preview_id is not None:
            llm_snapshot = conn.execute(
                "SELECT profile_id FROM execution_snapshots WHERE snapshot_id=? AND snapshot_type='llm'",
                (llm_snapshot_id,),
            ).fetchone()
            tts_snapshot = conn.execute(
                "SELECT profile_id FROM execution_snapshots WHERE snapshot_id=? AND snapshot_type='tts'",
                (tts_snapshot_id,),
            ).fetchone()
            if llm_snapshot is None or tts_snapshot is None:
                raise ValueError("preview_binding_not_found")
            binding = conn.execute(
                "SELECT * FROM outline_preview_bindings WHERE outline_preview_id=? AND owner_id=?",
                (outline_preview_id, owner_id),
            ).fetchone()
            if binding is None:
                raise ValueError("preview_binding_not_found")
            if binding["consumed_at"] is not None:
                raise ValueError("preview_binding_consumed")
            if binding["expires_at"] <= now:
                raise ValueError("preview_binding_expired")
            if (
                binding["topic"] != " ".join(topic.split())
                or binding["bpm"] != bpm
                or binding["duration_minutes"] != duration_minutes
                or binding["planned_segment_count"] != expected_count
                or binding["outline_hash"] != hashlib.sha256(outline_json.encode()).hexdigest()
                or binding["llm_profile_id"] != llm_snapshot["profile_id"]
                or binding["tts_profile_id"] != tts_snapshot["profile_id"]
            ):
                raise ValueError("preview_binding_mismatch")
            durable_binding = conn.execute(
                "SELECT * FROM outline_previews WHERE outline_preview_id=? AND owner_id=?",
                (outline_preview_id, owner_id),
            ).fetchone()
            if (
                durable_binding is None
                or durable_binding["consumed_at"] is not None
                or durable_binding["expires_at"] <= now
                or durable_binding["llm_snapshot_id"] != llm_snapshot_id
                or durable_binding["tts_snapshot_id"] != tts_snapshot_id
            ):
                raise ValueError("preview_binding_mismatch")
            consumed = conn.execute(
                """UPDATE outline_preview_bindings
                   SET consumed_at=?,consumed_project_id=?
                   WHERE outline_preview_id=? AND consumed_at IS NULL""",
                (now, project_id, outline_preview_id),
            )
            if consumed.rowcount != 1:
                raise ValueError("preview_binding_consumed")
        conn.execute(
            """INSERT INTO projects (
                project_id, owner_id, topic, bpm, duration_minutes, status,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, 'queued', ?, ?)""",
            (project_id, owner_id, topic, bpm, duration_minutes, now, now),
        )
        if outline_preview_id is not None:
            conn.execute(
                "UPDATE outline_previews SET consumed_at=?, project_id=? WHERE outline_preview_id=?",
                (now, project_id, outline_preview_id),
            )
            conn.execute(
                "UPDATE episode_ledgers SET project_id=? WHERE ledger_id=?",
                (project_id, durable_binding["ledger_id"]),
            )
        for segment_id, section in segment_rows:
            conn.execute(
                """INSERT INTO segments (
                    segment_id, project_id, idx, subtopic, title, segment_type,
                    planned_duration_seconds, status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'queued', ?, ?)""",
                (
                    segment_id,
                    project_id,
                    section["index"],
                    section["topic"].strip(),
                    section["title"].strip(),
                    section.get("segment_type", "content"),
                    float(section["approx_duration_seconds"]),
                    now,
                    now,
                ),
            )
        conn.execute(
            """INSERT INTO project_generation (
                project_id, accepted_outline_json, interviewer_profile_json,
                sme_profile_json, input_schema_hash, llm_snapshot_id,
                tts_snapshot_id, generation_contract_version,
                planned_segment_count, progress_version, disposition,
                last_transition_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, 0, 'active', ?)""",
            (
                project_id,
                outline_json,
                json.dumps(interviewer_profile or {}, sort_keys=True),
                json.dumps(sme_profile or {}, sort_keys=True),
                input_hash,
                llm_snapshot_id,
                tts_snapshot_id,
                expected_count,
                now,
            ),
        )
        stage_scopes = [("research", "project")]
        stage_scopes.extend(("research", segment_id) for segment_id, _ in segment_rows)
        for stage in ("text_generation", "fact_checking", "tts", "mixing"):
            stage_scopes.extend((stage, segment_id) for segment_id, _ in segment_rows)
        stage_scopes.append(("finalizing", "project"))
        conn.executemany(
            """INSERT INTO generation_stage_results (
                project_id, scope_id, stage_name, state, updated_at
            ) VALUES (?, ?, ?, 'pending', ?)""",
            [(project_id, scope_id, stage, now) for stage, scope_id in stage_scopes],
        )
        work_id = f"work_{uuid5(NAMESPACE_URL, f'{project_id}:pipeline').hex}"
        conn.execute(
            """INSERT INTO project_pipeline (
                work_id, project_id, state, created_at, updated_at
            ) VALUES (?, ?, 'pending', ?, ?)""",
            (work_id, project_id, now, now),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
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
    projects = []
    for row in rows:
        project = _row_to_project_summary(row)
        project["generation_progress"] = _get_generation_progress(conn, row["project_id"])
        projects.append(project)
    return projects


_STAGE_ORDER = (
    "research",
    "text_generation",
    "fact_checking",
    "tts",
    "mixing",
    "finalizing",
)


def _get_generation_progress(conn: sqlite3.Connection, project_id: str) -> dict | None:
    generation = conn.execute(
        "SELECT * FROM project_generation WHERE project_id = ?",
        (project_id,),
    ).fetchone()
    if generation is None:
        return None
    result_rows = conn.execute(
        """SELECT * FROM generation_stage_results
           WHERE project_id = ? ORDER BY stage_name, scope_id""",
        (project_id,),
    ).fetchall()
    grouped = {
        stage: [row for row in result_rows if row["stage_name"] == stage]
        for stage in _STAGE_ORDER
    }
    stages = []
    for stage_name in _STAGE_ORDER:
        rows = grouped[stage_name]
        completed = sum(row["state"] == "completed" for row in rows)
        states = {row["state"] for row in rows}
        if generation["terminal_outcome"] == "cancelled" and completed < len(rows):
            state = "cancelled"
        elif "failed" in states:
            state = "failed"
        elif rows and completed == len(rows):
            state = "completed"
        elif "running" in states or completed > 0:
            state = "running"
        else:
            state = "pending"
        current = next(
            (
                row["scope_id"]
                for row in rows
                if row["state"] == "running" and row["scope_id"] != "project"
            ),
            None,
        )
        updated_at = max(
            (row["updated_at"] for row in rows),
            default=generation["last_transition_at"],
        )
        stages.append(
            {
                "name": stage_name,
                "state": state,
                "completed_units": completed,
                "total_units": len(rows),
                "current_segment_id": current,
                "updated_at": updated_at,
            }
        )

    disposition = generation["disposition"]
    if disposition == "terminal":
        current_activity = None
    elif disposition == "cancellation_requested":
        current_activity = {"kind": "cancellation", "stage": None, "segment_id": None}
    else:
        active = next(
            (stage for stage in stages if stage["state"] in {"running", "pending"}),
            stages[-1],
        )
        current_activity = {
            "kind": "stage",
            "stage": active["name"],
            "segment_id": active["current_segment_id"],
        }
    return {
        "schema_version": 1,
        "progress_version": generation["progress_version"],
        "disposition": disposition,
        "terminal_outcome": generation["terminal_outcome"],
        "is_terminal": disposition == "terminal",
        "planned_segment_count": generation["planned_segment_count"],
        "cancellation_requested_at": generation["cancellation_requested_at"],
        "terminal_at": generation["terminal_at"],
        "last_transition_at": generation["last_transition_at"],
        "current_activity": current_activity,
        "stages": stages,
    }


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

    project["generation_progress"] = _get_generation_progress(conn, project_id)
    return project


def _claim_next_work(
    db_path: str,
    lease_owner: str,
    lease_seconds: int = 300,
    work_id: str | None = None,
) -> dict | None:
    conn = _get_conn(db_path)
    _ensure_schema(conn)
    now = datetime.now(timezone.utc)
    now_iso = now.isoformat()
    expires_at = (now + timedelta(seconds=lease_seconds)).isoformat()
    try:
        conn.execute("BEGIN IMMEDIATE")
        params: list[object] = [now_iso]
        work_filter = ""
        if work_id is not None:
            work_filter = " AND work_id = ?"
            params.append(work_id)
        row = conn.execute(
            f"""SELECT * FROM project_pipeline
                WHERE (state = 'pending'
                    OR (state = 'leased' AND lease_expires_at < ?))
                {work_filter}
                ORDER BY created_at
                LIMIT 1""",
            params,
        ).fetchone()
        if row is None:
            conn.rollback()
            return None
        epoch = row["lease_epoch"] + 1
        updated = conn.execute(
            """UPDATE project_pipeline
               SET state = 'leased', lease_owner = ?, lease_epoch = ?,
                   lease_expires_at = ?, heartbeat_at = ?, updated_at = ?
               WHERE work_id = ? AND lease_epoch = ? AND
                   (state = 'pending' OR (state = 'leased' AND lease_expires_at < ?))""",
            (
                lease_owner,
                epoch,
                expires_at,
                now_iso,
                now_iso,
                row["work_id"],
                row["lease_epoch"],
                now_iso,
            ),
        )
        if updated.rowcount != 1:
            conn.rollback()
            return None
        conn.commit()
        return {
            "work_id": row["work_id"],
            "project_id": row["project_id"],
            "lease_owner": lease_owner,
            "lease_epoch": epoch,
            "lease_expires_at": expires_at,
        }
    except Exception:
        conn.rollback()
        raise


def _renew_work_lease(
    db_path: str,
    work_id: str,
    lease_owner: str,
    lease_epoch: int,
    lease_seconds: int = 300,
) -> bool:
    conn = _get_conn(db_path)
    _ensure_schema(conn)
    now = datetime.now(timezone.utc)
    now_iso = now.isoformat()
    expires_at = (now + timedelta(seconds=lease_seconds)).isoformat()
    updated = conn.execute(
        """UPDATE project_pipeline
           SET lease_expires_at = ?, heartbeat_at = ?, updated_at = ?
           WHERE work_id = ? AND lease_owner = ? AND lease_epoch = ?
             AND state = 'leased' AND lease_expires_at >= ?""",
        (expires_at, now_iso, now_iso, work_id, lease_owner, lease_epoch, now_iso),
    )
    conn.commit()
    return updated.rowcount == 1


def _record_stage_transition(
    db_path: str,
    work_id: str,
    lease_owner: str,
    lease_epoch: int,
    scope_id: str,
    stage_name: str,
    state: str,
    result: dict | None = None,
    safe_error_code: str | None = None,
) -> bool:
    """Publish a stage transition only under the current nonterminal fence."""
    if state not in {"pending", "running", "completed", "failed", "cancelled"}:
        raise ValueError("invalid stage state")
    conn = _get_conn(db_path)
    _ensure_schema(conn)
    now = _now_iso()
    result_json = (
        json.dumps(result, sort_keys=True, separators=(",", ":"))
        if result is not None
        else None
    )
    result_hash = (
        hashlib.sha256(result_json.encode()).hexdigest() if result_json is not None else None
    )
    try:
        conn.execute("BEGIN IMMEDIATE")
        work = conn.execute(
            """SELECT w.*, g.disposition
               FROM project_pipeline w
               JOIN project_generation g ON g.project_id = w.project_id
               JOIN projects p ON p.project_id = w.project_id
               WHERE w.work_id = ? AND w.lease_owner = ? AND w.lease_epoch = ?
                 AND w.state = 'leased' AND w.lease_expires_at >= ?
                 AND g.disposition = 'active' AND p.deleted_at IS NULL""",
            (work_id, lease_owner, lease_epoch, now),
        ).fetchone()
        if work is None:
            conn.rollback()
            return False
        completed_at = now if state == "completed" else None
        cancelled_at = now if state == "cancelled" else None
        started_at = now if state == "running" else None
        updated = conn.execute(
            """UPDATE generation_stage_results
               SET state = ?, result_json = COALESCE(?, result_json),
                   result_hash = COALESCE(?, result_hash),
                   safe_error_code = ?, started_at = COALESCE(started_at, ?),
                   completed_at = COALESCE(?, completed_at),
                   cancelled_at = COALESCE(?, cancelled_at), updated_at = ?
               WHERE project_id = ? AND scope_id = ? AND stage_name = ?""",
            (
                state,
                result_json,
                result_hash,
                safe_error_code,
                started_at,
                completed_at,
                cancelled_at,
                now,
                work["project_id"],
                scope_id,
                stage_name,
            ),
        )
        if updated.rowcount != 1:
            conn.rollback()
            return False
        conn.execute(
            """UPDATE project_generation
               SET progress_version = progress_version + 1, last_transition_at = ?
               WHERE project_id = ?""",
            (now, work["project_id"]),
        )
        conn.execute(
            """UPDATE project_pipeline SET heartbeat_at = ?, updated_at = ?
               WHERE work_id = ?""",
            (now, now, work_id),
        )
        conn.commit()
        return True
    except Exception:
        conn.rollback()
        raise


def _reserve_provider_attempt(
    db_path: str, work_id: str, lease_owner: str, lease_epoch: int,
    logical_operation_id: str, scope_id: str, stage_name: str,
) -> dict | None:
    """Reserve a stable logical operation under the current publication fence."""
    conn = _get_conn(db_path)
    _ensure_schema(conn)
    now = _now_iso()
    try:
        conn.execute("BEGIN IMMEDIATE")
        work = conn.execute(
            """SELECT w.project_id FROM project_pipeline w
               JOIN project_generation g ON g.project_id=w.project_id
               JOIN projects p ON p.project_id=w.project_id
               WHERE w.work_id=? AND w.lease_owner=? AND w.lease_epoch=?
                 AND w.state='leased' AND w.lease_expires_at>=?
                 AND g.disposition='active' AND p.deleted_at IS NULL""",
            (work_id, lease_owner, lease_epoch, now),
        ).fetchone()
        if work is None:
            conn.rollback()
            return None
        existing = conn.execute(
            "SELECT * FROM provider_attempts WHERE project_id=? AND logical_operation_id=?",
            (work["project_id"], logical_operation_id),
        ).fetchone()
        if existing is not None:
            conn.commit()
            return dict(existing)
        attempt_id = f"attempt_{uuid4().hex}"
        conn.execute(
            """INSERT INTO provider_attempts
               (attempt_id,project_id,work_id,logical_operation_id,scope_id,
                stage_name,state,lease_owner,lease_epoch,reserved_at)
               VALUES (?,?,?,?,?,?,'reserved',?,?,?)""",
            (attempt_id, work["project_id"], work_id, logical_operation_id,
             scope_id, stage_name, lease_owner, lease_epoch, now),
        )
        conn.commit()
        return dict(conn.execute(
            "SELECT * FROM provider_attempts WHERE attempt_id=?", (attempt_id,)
        ).fetchone())
    except Exception:
        conn.rollback()
        raise


def _mark_provider_attempt_dispatched(
    db_path: str, attempt_id: str, work_id: str,
    lease_owner: str, lease_epoch: int,
) -> bool:
    """Commit dispatch before the external call; this transition is irreversible."""
    conn = _get_conn(db_path)
    _ensure_schema(conn)
    now = _now_iso()
    updated = conn.execute(
        """UPDATE provider_attempts SET state='dispatched',dispatched_at=?
           WHERE attempt_id=? AND work_id=? AND state='reserved'
             AND EXISTS (
               SELECT 1 FROM project_pipeline w
               JOIN project_generation g ON g.project_id=w.project_id
               JOIN projects p ON p.project_id=w.project_id
               WHERE w.work_id=provider_attempts.work_id AND w.lease_owner=?
                 AND w.lease_epoch=? AND w.state='leased' AND w.lease_expires_at>=?
                 AND g.disposition='active' AND p.deleted_at IS NULL
             )""",
        (now, attempt_id, work_id, lease_owner, lease_epoch, now),
    )
    conn.commit()
    return updated.rowcount == 1


def _complete_provider_attempt(
    db_path: str, attempt_id: str, work_id: str,
    lease_owner: str, lease_epoch: int, result: dict,
) -> bool:
    """Store a provider result only while the dispatch fence is still current."""
    conn = _get_conn(db_path)
    _ensure_schema(conn)
    now = _now_iso()
    result_json = json.dumps(result, sort_keys=True, separators=(",", ":"))
    updated = conn.execute(
        """UPDATE provider_attempts
           SET state='completed',result_json=?,result_hash=?,completed_at=?
           WHERE attempt_id=? AND work_id=? AND state='dispatched'
             AND EXISTS (
               SELECT 1 FROM project_pipeline w
               JOIN project_generation g ON g.project_id=w.project_id
               JOIN projects p ON p.project_id=w.project_id
               WHERE w.work_id=provider_attempts.work_id AND w.lease_owner=?
                 AND w.lease_epoch=? AND w.state='leased' AND w.lease_expires_at>=?
                 AND g.disposition='active' AND p.deleted_at IS NULL
             )""",
        (result_json, hashlib.sha256(result_json.encode()).hexdigest(), now,
         attempt_id, work_id, lease_owner, lease_epoch, now),
    )
    conn.commit()
    return updated.rowcount == 1


def _fail_dispatched_attempt_unknown(
    db_path: str, attempt_id: str, work_id: str,
    lease_owner: str, lease_epoch: int,
) -> bool:
    """Terminalize an ambiguous dispatched operation without replaying it."""
    conn = _get_conn(db_path)
    _ensure_schema(conn)
    now = _now_iso()
    try:
        conn.execute("BEGIN IMMEDIATE")
        attempt = conn.execute(
            """SELECT a.*,w.project_id FROM provider_attempts a
               JOIN project_pipeline w ON w.work_id=a.work_id
               JOIN projects p ON p.project_id=w.project_id
               WHERE a.attempt_id=? AND a.work_id=? AND a.state='dispatched'
                 AND w.lease_owner=? AND w.lease_epoch=? AND w.state='leased'
                 AND w.lease_expires_at>=? AND p.deleted_at IS NULL""",
            (attempt_id, work_id, lease_owner, lease_epoch, now),
        ).fetchone()
        if attempt is None:
            conn.rollback()
            return False
        conn.execute(
            "UPDATE provider_attempts SET state='failed_unknown',failed_at=? WHERE attempt_id=?",
            (now, attempt_id),
        )
        conn.execute(
            """UPDATE generation_stage_results SET state='failed',
               safe_error_code='provider_outcome_unknown',updated_at=?
               WHERE project_id=? AND scope_id=? AND stage_name=? AND state!='completed'""",
            (now, attempt["project_id"], attempt["scope_id"], attempt["stage_name"]),
        )
        conn.execute(
            "UPDATE project_pipeline SET state='failed',settled_at=?,updated_at=? WHERE work_id=?",
            (now, now, work_id),
        )
        conn.execute(
            """UPDATE project_generation SET disposition='terminal',
               terminal_outcome='failed',terminal_at=?,last_transition_at=?,
               progress_version=progress_version+1
               WHERE project_id=? AND disposition='active'""",
            (now, now, attempt["project_id"]),
        )
        conn.execute(
            """UPDATE projects SET status=CASE WHEN EXISTS(
               SELECT 1 FROM segments WHERE project_id=? AND status='ready')
               THEN 'partially_ready' ELSE 'failed' END,updated_at=? WHERE project_id=?""",
            (attempt["project_id"], now, attempt["project_id"]),
        )
        conn.commit()
        return True
    except Exception:
        conn.rollback()
        raise


def _request_project_cancellation(
    db_path: str, project_id: str, owner_id: str
) -> dict | None:
    conn = _get_conn(db_path)
    _ensure_schema(conn)
    now = _now_iso()
    try:
        conn.execute("BEGIN IMMEDIATE")
        project = conn.execute(
            """SELECT project_id FROM projects
               WHERE project_id=? AND owner_id=? AND deleted_at IS NULL""",
            (project_id, owner_id),
        ).fetchone()
        if project is None:
            conn.rollback()
            return None
        generation = conn.execute(
            "SELECT * FROM project_generation WHERE project_id=?",
            (project_id,),
        ).fetchone()
        if generation is None:
            conn.rollback()
            return None
        if generation["disposition"] == "terminal":
            conn.rollback()
            if generation["terminal_outcome"] != "cancelled":
                raise ValueError("project_not_cancellable")
            state = "cancelled"
            requested_at = generation["cancellation_requested_at"]
            observed_at = generation["cancellation_observed_at"]
            status_code = 200
        elif generation["disposition"] == "cancellation_requested":
            conn.rollback()
            state = "requested"
            requested_at = generation["cancellation_requested_at"]
            observed_at = generation["cancellation_observed_at"]
            status_code = 202
        else:
            conn.execute(
                """UPDATE project_generation
                   SET disposition='cancellation_requested',
                       cancellation_requested_at=?, last_transition_at=?,
                       progress_version=progress_version+1
                   WHERE project_id=? AND disposition='active'""",
                (now, now, project_id),
            )
            conn.execute(
                "UPDATE projects SET updated_at=? WHERE project_id=?",
                (now, project_id),
            )
            conn.commit()
            state = "requested"
            requested_at = now
            observed_at = None
            status_code = 202
        canonical = _get_project_by_id(db_path, project_id)
        return {
            "project": canonical,
            "state": state,
            "requested_at": requested_at,
            "observed_at": observed_at,
            "http_status": status_code,
        }
    except Exception:
        if conn.in_transaction:
            conn.rollback()
        raise


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
           segment_type, planned_duration_seconds, status, duration_seconds,
           text, primary_audio_artifact_id, error_code, error_message,
           error_retryable, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            segment["subtopic"], segment.get("title"),
            segment.get("segment_type", "content"),
            segment.get("planned_duration_seconds"),
            segment["status"], segment.get("duration_seconds"), segment.get("text"),
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
        "segment_type": row["segment_type"],
        "planned_duration_seconds": row["planned_duration_seconds"],
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
        "project_id": row["project_id"],
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

async def create_outline_preview_binding(
    db_path: str, outline_preview_id: str, owner_id: str, topic: str,
    bpm: int, duration_minutes: int, outline: dict,
    llm_profile_id: str, tts_profile_id: str,
    llm_routing_revision: str, tts_routing_revision: str, expires_at: str,
) -> dict:
    return await _run_in_executor(
        _create_outline_preview_binding, db_path, outline_preview_id, owner_id,
        topic, bpm, duration_minutes, outline, llm_profile_id, tts_profile_id,
        llm_routing_revision, tts_routing_revision, expires_at,
    )



async def create_progress_project(
    db_path: str,
    project_id: str,
    owner_id: str,
    topic: str,
    bpm: int,
    duration_minutes: int,
    accepted_outline: dict,
    interviewer_profile: dict | None = None,
    sme_profile: dict | None = None,
    llm_snapshot_id: str = "default",
    tts_snapshot_id: str = "default",
    outline_preview_id: str | None = None,
) -> dict:
    return await _run_in_executor(
        _create_progress_project,
        db_path,
        project_id,
        owner_id,
        topic,
        bpm,
        duration_minutes,
        accepted_outline,
        interviewer_profile,
        sme_profile,
        llm_snapshot_id,
        tts_snapshot_id,
        outline_preview_id,
    )


async def list_projects(db_path: str, owner_id: str) -> list[dict]:
    return await _run_in_executor(_list_projects, db_path, owner_id)


async def get_project(db_path: str, project_id: str) -> dict | None:
    return await _run_in_executor(_get_project_by_id, db_path, project_id)


async def delete_project(db_path: str, project_id: str) -> dict | None:
    return await _run_in_executor(_delete_project, db_path, project_id)


async def request_project_cancellation(
    db_path: str, project_id: str, owner_id: str
) -> dict | None:
    return await _run_in_executor(
        _request_project_cancellation, db_path, project_id, owner_id
    )


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


async def claim_next_work(
    db_path: str,
    lease_owner: str,
    lease_seconds: int = 300,
    work_id: str | None = None,
) -> dict | None:
    return await _run_in_executor(
        _claim_next_work, db_path, lease_owner, lease_seconds, work_id
    )


async def renew_work_lease(
    db_path: str,
    work_id: str,
    lease_owner: str,
    lease_epoch: int,
    lease_seconds: int = 300,
) -> bool:
    return await _run_in_executor(
        _renew_work_lease,
        db_path,
        work_id,
        lease_owner,
        lease_epoch,
        lease_seconds,
    )


async def record_stage_transition(
    db_path: str,
    work_id: str,
    lease_owner: str,
    lease_epoch: int,
    scope_id: str,
    stage_name: str,
    state: str,
    result: dict | None = None,
    safe_error_code: str | None = None,
) -> bool:
    return await _run_in_executor(
        _record_stage_transition,
        db_path,
        work_id,
        lease_owner,
        lease_epoch,
        scope_id,
        stage_name,
        state,
        result,
        safe_error_code,
    )


async def reserve_provider_attempt(
    db_path: str, work_id: str, lease_owner: str, lease_epoch: int,
    logical_operation_id: str, scope_id: str, stage_name: str,
) -> dict | None:
    return await _run_in_executor(
        _reserve_provider_attempt, db_path, work_id, lease_owner, lease_epoch,
        logical_operation_id, scope_id, stage_name,
    )


async def mark_provider_attempt_dispatched(
    db_path: str, attempt_id: str, work_id: str,
    lease_owner: str, lease_epoch: int,
) -> bool:
    return await _run_in_executor(
        _mark_provider_attempt_dispatched, db_path, attempt_id, work_id,
        lease_owner, lease_epoch,
    )


async def complete_provider_attempt(
    db_path: str, attempt_id: str, work_id: str,
    lease_owner: str, lease_epoch: int, result: dict,
) -> bool:
    return await _run_in_executor(
        _complete_provider_attempt, db_path, attempt_id, work_id,
        lease_owner, lease_epoch, result,
    )


async def fail_dispatched_attempt_unknown(
    db_path: str, attempt_id: str, work_id: str,
    lease_owner: str, lease_epoch: int,
) -> bool:
    return await _run_in_executor(
        _fail_dispatched_attempt_unknown, db_path, attempt_id, work_id,
        lease_owner, lease_epoch,
    )
