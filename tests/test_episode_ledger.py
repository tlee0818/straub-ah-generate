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
    assert persistence._consume_preview_binding(db_path, "opv_a", "owner-a", "prj_a", "topic", 120, 5, "llm-a", "tts-b", "llm-r1", "tts-r1") == "preview_profile_mismatch"
    assert persistence._consume_preview_binding(db_path, "opv_a", "owner-a", "prj_a", "topic", 120, 5, "llm-a", "tts-a", "llm-r1", "tts-r1") == "ok"
    assert persistence._consume_preview_binding(db_path, "opv_a", "owner-a", "prj_b", "topic", 120, 5, "llm-a", "tts-a", "llm-r1", "tts-r1") == "preview_expired"


def test_repeated_preview_reuses_canonical_snapshots(tmp_path):
    db_path = str(tmp_path / "worker.db")
    llm, tts = _snapshots()
    expires_at = (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()
    persistence._create_preview_binding(
        db_path, _preview(expires_at), llm, tts,
        {"ledger_id": "led_a", "policy": {"mode": "off"}, "currency": None},
    )

    second_preview = {**_preview(expires_at), "outline_preview_id": "opv_b"}
    second_llm = {**llm, "snapshot_id": "lsn_b"}
    second_tts = {**tts, "snapshot_id": "tsn_b"}
    persistence._create_preview_binding(
        db_path, second_preview, second_llm, second_tts,
        {"ledger_id": "led_b", "policy": {"mode": "off"}, "currency": None},
    )

    stored = persistence._get_preview_binding(db_path, "opv_b", "owner-a")
    assert stored["llm_snapshot_id"] == "lsn_a"
    assert stored["tts_snapshot_id"] == "tsn_a"


def test_expired_preview_cannot_be_consumed(tmp_path):
    db_path = str(tmp_path / "worker.db")
    llm, tts = _snapshots()
    persistence._create_preview_binding(
        db_path, _preview((datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()), llm, tts,
        {"ledger_id": "led_a", "policy": {"enforcement": "off"}, "currency": None},
    )
    assert persistence._consume_preview_binding(db_path, "opv_a", "owner-a", "prj_a", "topic", 120, 5, "llm-a", "tts-a", "llm-r1", "tts-r1") == "preview_expired"


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
def test_preview_consumption_requires_canonical_topic_bpm_and_duration(tmp_path):
    db_path = str(tmp_path / "worker.db")
    llm, tts = _snapshots()
    persistence._create_preview_binding(
        db_path, _preview((datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()), llm, tts,
        {"ledger_id": "led_a", "policy": {"mode": "off"}, "currency": None},
    )
    persistence._create_project(db_path, "prj_a", "owner-a", "other topic", 121, 6)
    assert persistence._consume_preview_binding(
        db_path, "opv_a", "owner-a", "prj_a", "other topic", 121, 6, "llm-a", "tts-a", "llm-r1", "tts-r1"
    ) == "preview_profile_mismatch"


def test_atomic_ledger_reservation_enforces_shared_cap(tmp_path):
    db_path = str(tmp_path / "worker.db")
    persistence._create_project(db_path, "prj_a", "owner-a", "topic", 120, 5)
    llm, tts = _snapshots()
    persistence._create_project_execution(
        db_path, "prj_a", "owner-a", llm, tts,
        {"ledger_id": "led_a", "policy": {"mode": "enforced", "caps": {"llm": 1}}, "currency": "USD"},
    )
    entry = {"ledger_id": "led_a", "category": "llm", "operation_type": "outline",
             "correlation_id": "outline", "resource_unit": "request", "amount": 1}
    assert persistence._reserve_ledger_operation(
        db_path, {**entry, "entry_id": "entry_a"}, {"attempt_id": "attempt_a", "snapshot_revision": "llm-r1", "binding": {}}
    )
    assert not persistence._reserve_ledger_operation(
        db_path, {**entry, "entry_id": "entry_b"}, {"attempt_id": "attempt_b", "snapshot_revision": "llm-r1", "binding": {}}
    )
def test_restart_reconciliation_claims_pending_and_expired_leases(tmp_path):
    db_path = str(tmp_path / "worker.db")
    persistence._upsert_durable_record(db_path, "work_items", {
        "work_id": "pending", "kind": "project_pipeline", "state": "pending", "payload_json": {},
    })
    persistence._upsert_durable_record(db_path, "work_items", {
        "work_id": "stale", "kind": "project_pipeline", "state": "leased", "payload_json": {},
        "lease_owner": "dead", "lease_expires_at": (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat(),
    })
    claimed = persistence._claim_reconcilable_work(
        db_path, "work_items", "restart", (datetime.now(timezone.utc) + timedelta(minutes=1)).isoformat()
    )
    assert {row["work_id"] for row in claimed} == {"pending", "stale"}
