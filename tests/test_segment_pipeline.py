"""Tests for purpose routing and the strict script verification gate."""

from __future__ import annotations

import sqlite3
import sys
import threading
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest

from podcast_worker.core import persistence, script_generator, segment_pipeline
from podcast_worker.core.config import LLMRoute, ResolvedExecutionSnapshot, ResolvedTTSSnapshot, RoutingConfigurationError


def _snapshot() -> ResolvedExecutionSnapshot:
    routes = {
        purpose: LLMRoute(purpose, "openai" if purpose != "research_brief" else "ollama", f"{purpose}-model", "openai_json_object" if purpose != "research_brief" else "ollama_format_json")
        for purpose in ("outline", "research_brief", "subtopic_research", "dialogue_draft", "fact_verification")
    }
    return ResolvedExecutionSnapshot("routing-profile", "rte_test", MappingProxyType(routes))
  

def _tts_snapshot(max_concurrent_requests: int) -> ResolvedTTSSnapshot:
    return ResolvedTTSSnapshot(
        "parallel-profile", "tts-test-revision", "elevenlabs", "text_to_dialogue_v3",
        "eleven_v3", "mp3_44100_128", 100, 1, 0,
        MappingProxyType({"interviewer": "host", "guest": "guest"}), 1, 0, 0, None,
        max_concurrent_requests,
    )
def test_verification_result_is_persistable_and_blocks_rejected_text():
    accepted = SimpleNamespace(
        outcome="accepted",
        issues=(),
        verified_text="Interviewer: Verified",
    )

    assert segment_pipeline._verification_payload(accepted) == {
        "outcome": "accepted",
        "issues": [],
        "verified_text": "Interviewer: Verified",
    }

    blocked = SimpleNamespace(
        outcome="blocked",
        issues=("unsupported",),
        verified_text=None,
    )
    with pytest.raises(RoutingConfigurationError, match="validation_failed"):
        segment_pipeline._verification_payload(blocked)

def test_pipeline_uses_persisted_llm_snapshot(monkeypatch):
    snapshot = _snapshot()
    captured = {}

    monkeypatch.setattr(
        segment_pipeline,
        "_hydrate",
        lambda *_: {
            "work": {"project_id": "project", "topic": "topic", "bpm": 120, "duration_minutes": 1},
            "outline": {"sections": [{"index": 0, "topic": "section"}]},
            "segments": [{"segment_id": "segment"}],
            "interviewer_profile": None,
            "sme_profile": None,
            "llm_snapshot": snapshot,
            "tts_snapshot": _tts_snapshot(1),
        },
    )
    monkeypatch.setattr(segment_pipeline, "_completed_result", lambda *_: None)
    monkeypatch.setattr(segment_pipeline, "_transition", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        segment_pipeline,
        "_provider_operation",
        lambda *_args: _args[-1](),
    )

    def research(*_args, **kwargs):
        captured.update(kwargs)
        raise RuntimeError("stop after routing assertion")

    monkeypatch.setattr(segment_pipeline, "generate_research_brief", research)

    with pytest.raises(RuntimeError, match="stop after routing assertion"):
        segment_pipeline._run_pipeline("db", "work", "owner", 1, "output")

    assert captured["snapshot"] is snapshot



def _parallel_pipeline_db(tmp_path, text: str) -> tuple[str, dict]:
    db_path = str(tmp_path / "parallel.sqlite")
    persistence._create_project(db_path, "project", "owner", "topic", 120, 1)
    persistence._upsert_segment(db_path, {
        "segment_id": "segment", "project_id": "project", "index": 0, "subtopic": "topic",
        "status": "queued", "text": text,
    })
    conn = persistence._get_conn(db_path)
    now = persistence._now_iso()
    for snapshot_id, snapshot_type in (("llm-snapshot", "llm"), ("tts-snapshot", "tts")):
        conn.execute(
            """INSERT INTO execution_snapshots
               (snapshot_id, snapshot_type, profile_id, revision, canonical_json, sha256, created_at)
               VALUES (?, ?, ?, ?, '{}', ?, ?)""",
            (snapshot_id, snapshot_type, snapshot_type, "revision", f"{snapshot_type}-sha", now),
        )
    conn.execute(
        """INSERT INTO episode_ledgers
           (ledger_id, owner_id, project_id, llm_snapshot_id, tts_snapshot_id, policy_json, currency, created_at)
           VALUES ('ledger', 'owner', 'project', 'llm-snapshot', 'tts-snapshot', '{}', 'USD', ?)""",
        (now,),
    )
    conn.commit()
    return db_path, {
        "segment_id": "segment", "subtopic": "topic", "text": text,
    }


def _install_audio_fakes(monkeypatch, assembled_paths: list[str]) -> None:
    class Audio:
        def __init__(self, paths=()):
            self.paths = list(paths)

        def __iadd__(self, other):
            self.paths.extend(other.paths)
            return self

        def export(self, output_path, format):
            assembled_paths.extend(self.paths)
            with open(output_path, "wb") as output:
                output.write(b"combined")

    class AudioSegment:
        @staticmethod
        def empty():
            return Audio()

        @staticmethod
        def from_file(path):
            return Audio([path])

    monkeypatch.setitem(sys.modules, "pydub", SimpleNamespace(AudioSegment=AudioSegment))
    monkeypatch.setattr(segment_pipeline, "convert_to_wav", lambda *_: None)
    monkeypatch.setattr(segment_pipeline, "build_podcast_audio", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        segment_pipeline, "convert_to_mp3",
        lambda _wav, mp3: Path(mp3).write_bytes(b"combined"),
    )
    monkeypatch.setattr(
        segment_pipeline, "_validated_audio_metadata",
        lambda _path: (0.5, -18.0, 0.5),
    )
    monkeypatch.setattr(segment_pipeline, "_sync_add_artifact", lambda *_: None)
    monkeypatch.setattr(segment_pipeline, "_sync_update_segment_status", lambda *_: None)
    monkeypatch.setattr(segment_pipeline, "_sync_upsert_provenance", lambda *_: None)
  

class TestParallelElevenLabsSegments:
    def test_bounded_parallel_rendering_persists_each_plan_and_assembles_plan_order(self, monkeypatch, tmp_path):
        from podcast_worker.core import tts_engine

        text = "Interviewer: Zero.\nInterviewer: One.\nInterviewer: Two."
        db_path, segment = _parallel_pipeline_db(tmp_path, text)
        assembled, starts, active = [], [], {"current": 0, "maximum": 0}
        first_two_started, release, second_finished, release_zero = (
            threading.Event(), threading.Event(), threading.Event(), threading.Event()
        )
        lock = threading.Lock()
        _install_audio_fakes(monkeypatch, assembled)

        def synthesize(plan, _snapshot, output_path):
            with lock:
                starts.append(plan.plan_id)
                active["current"] += 1
                active["maximum"] = max(active["maximum"], active["current"])
                if len(starts) == 2:
                    first_two_started.set()
            try:
                assert release.wait(2), "test did not release blocked providers"
                if plan.plan_id == "scene-0":
                    assert release_zero.wait(2), "test did not release scene zero"
                else:
                    second_finished.set()
                with open(output_path, "wb") as output:
                    output.write(plan.plan_id.encode())
                return output_path
            finally:
                with lock:
                    active["current"] -= 1

        monkeypatch.setattr(tts_engine, "synthesize_elevenlabs_plan", synthesize)
        worker_error = []

        def process():
            try:
                segment_pipeline._process_segment(
                    db_path, "project", segment, str(tmp_path), tmp_path / "beat.mp3",
                    120, 1, None, "model", _tts_snapshot(2), "ledger",
                )
            except Exception as exc:
                worker_error.append(exc)

        worker = None
        try:
            worker = threading.Thread(target=process)
            worker.start()
            assert first_two_started.wait(2)
            assert active["maximum"] == 2
            release.set()
            assert second_finished.wait(2)
            release_zero.set()
            worker.join(2)
            assert not worker.is_alive()
        finally:
            release.set()
            release_zero.set()
            if worker is not None:
                worker.join(2)

        assert worker_error == []
        assert {plan_id.rsplit(":", 1)[-1] for plan_id in starts} == {"scene-0", "scene-1", "scene-2"}
        assert active["maximum"] == 2
        assert len(set(assembled)) == 3
        assert [Path(path).name.rsplit(":", 1)[-1] for path in assembled] == ["scene-0.mp3", "scene-1.mp3", "scene-2.mp3"]
        conn = sqlite3.connect(db_path)
        attempts = conn.execute(
            "SELECT attempt_id, plan_id, outcome FROM execution_attempts ORDER BY plan_id, created_at"
        ).fetchall()
        ledger_entries = conn.execute(
            "SELECT plan_id FROM (SELECT correlation_id AS plan_id FROM episode_ledger_entries) ORDER BY plan_id"
        ).fetchall()
        conn.close()
        assert len({attempt[0] for attempt in attempts}) == 3
        assert {(attempt[1], attempt[2]) for attempt in attempts} == {
            ("project:segment:scene-0", "published"), ("project:segment:scene-1", "published"), ("project:segment:scene-2", "published"),
        }
        assert len(ledger_entries) == 9
        assert {entry[0].rsplit(":", 1)[-1] for entry in ledger_entries} == {"scene-0", "scene-1", "scene-2"}

    def test_maximum_one_is_sequential(self, monkeypatch, tmp_path):
        from podcast_worker.core import tts_engine

        text = "Interviewer: Zero.\nInterviewer: One.\nInterviewer: Two."
        db_path, segment = _parallel_pipeline_db(tmp_path, text)
        assembled, starts, active = [], [], {"current": 0, "maximum": 0}
        lock = threading.Lock()
        _install_audio_fakes(monkeypatch, assembled)

        def synthesize(plan, _snapshot, output_path):
            with lock:
                starts.append(plan.plan_id)
                active["current"] += 1
                active["maximum"] = max(active["maximum"], active["current"])
            try:
                with open(output_path, "wb") as output:
                    output.write(plan.plan_id.encode())
                return output_path
            finally:
                with lock:
                    active["current"] -= 1

        monkeypatch.setattr(tts_engine, "synthesize_elevenlabs_plan", synthesize)
        segment_pipeline._process_segment(
            db_path, "project", segment, str(tmp_path), tmp_path / "beat.mp3",
            120, 1, None, "model", _tts_snapshot(1), "ledger",
        )
        assert starts == ["project:segment:scene-0", "project:segment:scene-1", "project:segment:scene-2"]
        assert active["maximum"] == 1

    def test_dispatched_failure_marks_only_that_attempt_unknown_and_blocks_assembly(self, monkeypatch, tmp_path):
        from podcast_worker.core import tts_engine

        text = "Interviewer: Zero.\nInterviewer: One.\nInterviewer: Two."
        db_path, segment = _parallel_pipeline_db(tmp_path, text)
        assembled, dispatched, statuses, publications = [], [], [], []
        _install_audio_fakes(monkeypatch, assembled)
        monkeypatch.setattr(segment_pipeline, "_sync_update_segment_status", lambda _db, _segment, status: statuses.append(status))
        monkeypatch.setattr(segment_pipeline, "_sync_add_artifact", lambda *_: publications.append(True))

        def synthesize(plan, _snapshot, _output_path):
            dispatched.append(plan.plan_id)
            raise RoutingConfigurationError("tts_outcome_unknown")

        monkeypatch.setattr(tts_engine, "synthesize_elevenlabs_plan", synthesize)
        with pytest.raises(RoutingConfigurationError, match="tts_outcome_unknown"):
            segment_pipeline._process_segment(
                db_path, "project", segment, str(tmp_path), tmp_path / "beat.mp3",
                120, 1, None, "model", _tts_snapshot(1), "ledger",
            )

        conn = sqlite3.connect(db_path)
        attempts = conn.execute("SELECT plan_id, outcome FROM execution_attempts").fetchall()
        assemblies = conn.execute("SELECT COUNT(*) FROM audio_assemblies").fetchone()[0]
        conn.close()
        assert dispatched[0] == "project:segment:scene-0"
        assert set(dispatched).issubset({"project:segment:scene-0", "project:segment:scene-1"})
        assert set(attempts) == {(plan_id, "unknown_outcome") for plan_id in dispatched}
        assert assemblies == 0
        assert "ready" not in statuses
        assert publications == []
        assert assembled == []
    def test_provider_diagnostic_is_redacted_from_segment_manifest(self, tmp_path):
        db_path, segment = _parallel_pipeline_db(tmp_path, "Interviewer: Zero.")
        segment_pipeline._persist_segment_failure(
            db_path, segment["segment_id"], segment_pipeline._public_error_code(RuntimeError("secret token"))
        )

        conn = sqlite3.connect(db_path)
        error = conn.execute("SELECT error_message FROM segments WHERE segment_id = ?", (segment["segment_id"],)).fetchone()[0]
        conn.close()
        assert error == "generation_failed"



class TestPurposeRouting:
    def test_each_operation_uses_its_snapshot_purpose_route(self, monkeypatch):
        calls = []

        def openai(*args, dialect=None, **kwargs):
            calls.append(("openai", args[3], dialect))
            return {"ok": True}

        def ollama(*args, model=None, **kwargs):
            calls.append(("ollama", model, None))
            return {"ok": True}

        monkeypatch.setattr(script_generator, "_call_openai", openai)
        monkeypatch.setattr(script_generator, "_call_ollama", ollama)
        snapshot = _snapshot()
        for purpose in snapshot.routes:
            assert script_generator._call_provider("system", "user", snapshot=snapshot, purpose=purpose) == {"ok": True}

        assert calls == [
            ("openai", "outline-model", "openai_json_object"),
            ("ollama", "research_brief-model", None),
            ("openai", "subtopic_research-model", "openai_json_object"),
            ("openai", "dialogue_draft-model", "openai_json_object"),
            ("openai", "fact_verification-model", "openai_json_object"),
        ]
    def test_section_draft_selects_dialogue_route(self, monkeypatch):
        captured = {}

        def provider(*_args, **kwargs):
            captured.update(kwargs)
            return {"segment": {"text": "Interviewer: Hello"}}

        monkeypatch.setattr(script_generator, "_call_provider", provider)
        outline = {"sections": [{"index": 0, "topic": "Section", "title": "Section"}]}

        script_generator.generate_section_draft(
            "Topic",
            120,
            1,
            outline,
            outline["sections"][0],
            {},
            {},
            snapshot=_snapshot(),
        )

        assert captured["purpose"] == "dialogue_draft"



class TestVerificationGate:
    @pytest.mark.parametrize(
        ("payload", "outcome", "text"),
        [
            ({"outcome": "accepted", "issues": [], "verified_text": "Interviewer: Draft"}, "accepted", "Interviewer: Draft"),
            ({"outcome": "accepted", "issues": [], "verified_text": "Interviewer: Factful rewrite"}, "accepted", "Interviewer: Factful rewrite"),
            ({"outcome": "corrected", "issues": ["Unsupported claim"], "verified_text": "Interviewer: Corrected"}, "corrected", "Interviewer: Corrected"),
            ({"outcome": "blocked", "issues": ["Cannot verify"], "verified_text": None}, "blocked", None),
        ],
    )
    def test_verifier_accepts_only_explicit_valid_outcomes(self, payload, outcome, text):
        result = script_generator.parse_verification_result(payload, "Interviewer: Draft")
        assert (result.outcome, result.verified_text) == (outcome, text)

    @pytest.mark.parametrize(
        "payload",
        [
            {"outcome": "corrected", "issues": [], "verified_text": "corrected"},
            {"outcome": "blocked", "issues": ["reason"], "verified_text": "draft"},
            {"outcome": "unknown", "issues": [], "verified_text": "draft"},
            {"outcome": "accepted", "issues": "none", "verified_text": "draft"},
        ],
    )
    def test_verifier_rejects_malformed_or_inconsistent_output(self, payload):
        with pytest.raises(RoutingConfigurationError, match="structured_output_failure"):
            script_generator.parse_verification_result(payload, "draft")

    def test_blocked_verification_stops_before_any_tts_boundary(self, monkeypatch):
        calls = []

        def provider(system_prompt, *args, **kwargs):
            calls.append(system_prompt)
            if "lead research" in system_prompt:
                return {"research_brief": "brief"}
            if "subtopic research" in system_prompt:
                return {"key_points": ["point"]}
            if "factfulness" in system_prompt:
                return {"outcome": "blocked", "issues": ["unverified"], "verified_text": None}
            return {"segment_type": "content", "text": "Interviewer: Draft", "approx_duration_seconds": 10}

        monkeypatch.setattr(script_generator, "_call_provider", provider)
        with pytest.raises(RoutingConfigurationError, match="validation_failed"):
            script_generator.generate_script(
                "topic", 100, outline={"title": "T", "sections": [{"topic": "one", "approx_duration_seconds": 10}]}
            )
        assert any("factfulness" in call for call in calls)
        assert not any("tts" in call.lower() for call in calls)
