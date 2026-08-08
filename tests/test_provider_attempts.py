import sqlite3

import pytest

from podcast_worker.core import persistence
from podcast_worker.core.segment_pipeline import FenceLost, _provider_operation


def _leased_project(tmp_path):
    db_path = str(tmp_path / "provider-attempts.db")
    persistence._create_progress_project(
        db_path,
        "prj_provider",
        "single-user",
        "durable provider calls",
        120,
        1,
        {
            "project_id": "preview",
            "topic": "durable provider calls",
            "title": "Durable provider calls",
            "sections": [
                {
                    "index": 0,
                    "segment_type": "content",
                    "topic": "Attempts",
                    "title": "Attempts",
                    "approx_duration_seconds": 60,
                }
            ],
        },
        None,
        None,
        "default",
        "default",
    )
    claim = persistence._claim_next_work(db_path, "worker-a")
    assert claim is not None
    return db_path, claim


def test_completed_provider_attempt_is_reused_without_replay(tmp_path):
    db_path, claim = _leased_project(tmp_path)
    calls = 0

    def provider():
        nonlocal calls
        calls += 1
        return {"brief": "persisted result"}

    args = (
        db_path,
        claim["work_id"],
        claim["lease_owner"],
        claim["lease_epoch"],
        "lead_research",
        "project",
        "research",
        provider,
    )
    assert _provider_operation(*args) == {"brief": "persisted result"}
    assert _provider_operation(*args) == {"brief": "persisted result"}
    assert calls == 1


def test_provider_crash_fails_unknown_and_cannot_replay_after_restart(tmp_path):
    db_path, claim = _leased_project(tmp_path)
    calls = 0

    def crashing_provider():
        nonlocal calls
        calls += 1
        raise RuntimeError("provider connection dropped after dispatch")

    args = (
        db_path,
        claim["work_id"],
        claim["lease_owner"],
        claim["lease_epoch"],
        "lead_research",
        "project",
        "research",
        crashing_provider,
    )
    with pytest.raises(RuntimeError, match="connection dropped"):
        _provider_operation(*args)

    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    attempt = connection.execute(
        "SELECT state FROM provider_attempts WHERE logical_operation_id='lead_research'"
    ).fetchone()
    stage = connection.execute(
        """SELECT state, safe_error_code FROM generation_stage_results
           WHERE scope_id='project' AND stage_name='research'"""
    ).fetchone()
    pipeline = connection.execute(
        "SELECT state FROM project_pipeline WHERE work_id=?", (claim["work_id"],)
    ).fetchone()
    generation = connection.execute(
        "SELECT disposition, terminal_outcome FROM project_generation WHERE project_id='prj_provider'"
    ).fetchone()
    project_error = connection.execute(
        "SELECT code, message, retryable FROM project_errors WHERE project_id='prj_provider'"
    ).fetchone()
    connection.close()

    assert dict(attempt) == {"state": "failed_unknown"}
    assert dict(stage) == {"state": "failed", "safe_error_code": "provider_outcome_unknown"}
    assert pipeline["state"] == "failed"
    assert dict(generation) == {"disposition": "terminal", "terminal_outcome": "failed"}
    assert dict(project_error) == {
        "code": "generation_failed",
        "message": "Generation could not safely continue",
        "retryable": 0,
    }
    assert persistence._claim_next_work(db_path, "worker-restarted") is None
    with pytest.raises(FenceLost):
        _provider_operation(*args)
    assert calls == 1
