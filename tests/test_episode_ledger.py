"""Durability tests for paired snapshots, previews, and the shared episode ledger."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone

from podcast_worker.core import persistence
from podcast_worker.core.config import execution_snapshot_from_payload, tts_snapshot_from_payload


def _snapshots():
    llm_payload = {
        "profile_id": "llm-a",
        "routes": {
            purpose: {"purpose": purpose, "provider": "openai", "model": "server-model", "dialect": "openai_json_object"}
            for purpose in ("outline", "research_brief", "subtopic_research", "dialogue_draft", "fact_verification")
        },
    }
    tts_payload = {
        "profile_id": "tts-a", "provider": "elevenlabs", "strategy": "text_to_dialogue_v3",
        "model_id": "eleven_v3", "output_format": "mp3", "max_scene_characters": 100,
        "max_scene_turns": 2, "max_fragment_characters": 0,
        "voice_bindings": {"interviewer": "raw-host", "guest": "raw-guest"}, "max_attempts": 1,
    }
    return (
        {"snapshot_id": "lsn_a", "profile_id": "llm-a", "revision": "llm-r1", "payload": llm_payload, "sha256": hashlib.sha256(json.dumps(llm_payload, sort_keys=True).encode()).hexdigest()},
        {"snapshot_id": "tsn_a", "profile_id": "tts-a", "revision": "tts-r1", "payload": tts_payload, "sha256": hashlib.sha256(json.dumps(tts_payload, sort_keys=True).encode()).hexdigest()},
    )


def _preview(expires_at: str):
    return {
        "outline_preview_id": "opv_a", "owner_id": "owner-a", "topic": "topic", "bpm": 120,
        "duration_minutes": 5, "outline": {"title": "Outline", "sections": []},
        "llm_profile_id": "llm-a", "tts_profile_id": "tts-a", "routing_revision": "llm-r1",
        "tts_routing_revision": "tts-r1", "expires_at": expires_at,
    }


def test_preview_binding_is_paired_immutable_owner_scoped_and_one_time(tmp_path):
    db_path = str(tmp_path / "worker.db")
    llm, tts = _snapshots()
    persistence._create_preview_binding(
        db_path, _preview((datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()), llm, tts,
        {"ledger_id": "led_a", "policy": {"enforcement": "off"}, "currency": None},
    )

    stored = persistence._get_preview_binding(db_path, "opv_a", "owner-a")
    assert stored["llm_snapshot_id"] == "lsn_a"
    assert stored["tts_snapshot_id"] == "tsn_a"
    assert persistence._get_preview_binding(db_path, "opv_a", "other-owner") is None
    persistence._create_project(db_path, "prj_a", "owner-a", "topic", 120, 5)
    assert persistence._consume_preview_binding(db_path, "opv_a", "owner-a", "prj_a", "llm-a", "tts-b", "llm-r1", "tts-r1") == "preview_profile_mismatch"
    assert persistence._consume_preview_binding(db_path, "opv_a", "owner-a", "prj_a", "llm-a", "tts-a", "llm-r1", "tts-r1") == "ok"
    assert persistence._consume_preview_binding(db_path, "opv_a", "owner-a", "prj_b", "llm-a", "tts-a", "llm-r1", "tts-r1") == "preview_expired"


def test_expired_preview_cannot_be_consumed(tmp_path):
    db_path = str(tmp_path / "worker.db")
    llm, tts = _snapshots()
    persistence._create_preview_binding(
        db_path, _preview((datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()), llm, tts,
        {"ledger_id": "led_a", "policy": {"enforcement": "off"}, "currency": None},
    )
    assert persistence._consume_preview_binding(db_path, "opv_a", "owner-a", "prj_a", "llm-a", "tts-a", "llm-r1", "tts-r1") == "preview_expired"


def test_project_execution_hydrates_persisted_snapshots_and_records_shared_categories(tmp_path):
    db_path = str(tmp_path / "worker.db")
    persistence._create_project(db_path, "prj_a", "owner-a", "topic", 120, 5)
    llm, tts = _snapshots()
    persistence._create_project_execution(
        db_path, "prj_a", "owner-a", llm, tts,
        {"ledger_id": "led_a", "policy": {"enforcement": "off"}, "currency": None},
    )
    execution = persistence._get_project_execution(db_path, "prj_a")
    hydrated_llm = execution_snapshot_from_payload(execution["llm_snapshot"], execution["llm_revision"])
    hydrated_tts = tts_snapshot_from_payload(execution["tts_snapshot"], execution["tts_revision"])
    assert hydrated_llm.route_for("outline").model == "server-model"
    assert hydrated_tts.voice_bindings == {"interviewer": "raw-host", "guest": "raw-guest"}

    for entry_id, category in (("entry-llm", "llm"), ("entry-tts", "tts")):
        persistence._append_ledger_entry(db_path, {
            "entry_id": entry_id, "ledger_id": "led_a", "category": category,
            "operation_type": "generation", "correlation_id": entry_id,
            "resource_unit": "tokens" if category == "llm" else "characters", "amount": 1, "state": "actual",
        })
    rows = persistence._get_conn(db_path).execute(
        "SELECT category FROM episode_ledger_entries WHERE ledger_id = ? ORDER BY entry_id", ("led_a",)
    ).fetchall()
    assert [row["category"] for row in rows] == ["llm", "tts"]
