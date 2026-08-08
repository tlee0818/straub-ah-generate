"""Focused tests for immutable ElevenLabs request routing."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from podcast_worker.core.config import ResolvedTTSSnapshot, RoutingConfigurationError, tts_snapshot_from_payload
from podcast_worker.core.tts_engine import (
    classify_tts_failure,
    plan_dialogue_requests,
    synthesize_elevenlabs_plan,
)


def _snapshot(strategy: str, **overrides) -> ResolvedTTSSnapshot:
    values = {
        "profile_id": "eleven-dialogue",
        "revision": "tts_revision_1",
        "provider": "elevenlabs",
        "strategy": strategy,
        "model_id": "eleven_v3" if strategy == "text_to_dialogue_v3" else "eleven_multilingual_v2",
        "output_format": "mp3_44100_128",
        "max_scene_characters": 20 if strategy == "text_to_dialogue_v3" else 0,
        "max_scene_turns": 2 if strategy == "text_to_dialogue_v3" else 0,
        "max_fragment_characters": 12 if strategy == "stitched_text_to_speech" else 0,
        "voice_bindings": {"interviewer": "eleven-host", "guest": "eleven-guest"},
        "max_attempts": 1,
        "context_max_request_ids": 2,
        "context_max_age_seconds": 60,
        "continuity_after_expiry": "reset",
    }
    values.update(overrides)
    return ResolvedTTSSnapshot(**values)
  

def test_persisted_snapshot_keeps_its_concurrency_limit_immutable():
    snapshot = tts_snapshot_from_payload({
        "profile_id": "eleven-dialogue",
        "provider": "elevenlabs",
        "strategy": "text_to_dialogue_v3",
        "model_id": "eleven_v3",
        "output_format": "mp3_44100_128",
        "max_scene_characters": 20,
        "max_scene_turns": 2,
        "max_fragment_characters": 0,
        "voice_bindings": {"interviewer": "eleven-host", "guest": "eleven-guest"},
        "max_attempts": 1,
        "max_concurrent_requests": 2,
    }, "persisted-revision")

    assert snapshot.max_concurrent_requests == 2
    with pytest.raises(AttributeError):
        snapshot.max_concurrent_requests = 1


class TestElevenLabsPlanning:
    def test_v3_groups_multiple_dialogue_turns_without_losing_roles(self):
        plans = plan_dialogue_requests(
            "Interviewer: Hello.\nSME: A concise answer.\nInterviewer: Next question.",
            _snapshot("text_to_dialogue_v3", max_scene_characters=30, max_scene_turns=2),
        )

        assert [plan.plan_id for plan in plans] == ["scene-0", "scene-1"]
        assert [(turn.role, turn.text) for turn in plans[0].turns] == [
            ("interviewer", "Hello."),
            ("guest", "A concise answer."),
        ]
        assert [(turn.role, turn.text) for turn in plans[1].turns] == [("interviewer", "Next question.")]
        assert all(plan.voice_binding is None for plan in plans)

    def test_stitched_strategy_fragments_each_speaker_and_keeps_voice_binding(self):
        plans = plan_dialogue_requests(
            "Interviewer: One two three four.\nSME: Five six seven eight.",
            _snapshot("stitched_text_to_speech", max_fragment_characters=12),
        )

        assert [(plan.turns[0].role, plan.voice_binding) for plan in plans] == [
            ("interviewer", "eleven-host"),
            ("interviewer", "eleven-host"),
            ("guest", "eleven-guest"),
            ("guest", "eleven-guest"),
        ]
        assert all(len(plan.turns[0].text) <= 12 for plan in plans)
        assert [plan.plan_id for plan in plans] == [f"fragment-{index}" for index in range(4)]
    def test_persisted_plan_namespace_prevents_cross_segment_collisions(self):
        snapshot = _snapshot("text_to_dialogue_v3")
        first = plan_dialogue_requests("Interviewer: Hello.", snapshot, namespace="project-a:segment-a")
        second = plan_dialogue_requests("Interviewer: Hello.", snapshot, namespace="project-b:segment-b")

        assert [plan.plan_id for plan in first] == ["project-a:segment-a:scene-0"]
        assert [plan.plan_id for plan in second] == ["project-b:segment-b:scene-0"]
        assert first[0].plan_id != second[0].plan_id


    def test_oversized_v3_turn_is_rejected_before_provider_dispatch(self):
        with pytest.raises(RoutingConfigurationError, match="tts_turn_too_long"):
            plan_dialogue_requests("Interviewer: This turn is deliberately too long.", _snapshot("text_to_dialogue_v3"))


class TestElevenLabsExecution:
    def test_v3_request_uses_one_multi_turn_dialogue_call(self, monkeypatch, tmp_path):
        from podcast_worker.core import tts_engine

        calls = []

        class Response:
            ok = True
            status_code = 200

            @staticmethod
            def iter_content(_size):
                return [b"audio"]

        def post(*args, **kwargs):
            calls.append((args, kwargs))
            return Response()

        monkeypatch.setattr(tts_engine.config.settings, "elevenlabs_api_key", "server-secret")
        monkeypatch.setitem(sys.modules, "requests", SimpleNamespace(post=post, RequestException=Exception))
        plan = plan_dialogue_requests("Interviewer: Hello.\nSME: Welcome.", _snapshot("text_to_dialogue_v3"))[0]

        output = tmp_path / "scene.mp3"
        assert synthesize_elevenlabs_plan(plan, _snapshot("text_to_dialogue_v3"), str(output)) == str(output)
        assert output.read_bytes() == b"audio"
        assert calls[0][0][0] == "https://api.elevenlabs.io/v1/text-to-dialogue"
        assert calls[0][1]["json"]["inputs"] == [
            {"text": "Hello.", "voice_id": "eleven-host"},
            {"text": "Welcome.", "voice_id": "eleven-guest"},
        ]
        assert "output_format" not in calls[0][1]["json"]
        assert calls[0][1]["params"] == {"output_format": "mp3_44100_128"}
        assert calls[0][1]["headers"]["X-Request-Id"] == plan.plan_id

    def test_ambiguous_provider_outcome_is_not_blindly_retried(self, monkeypatch, tmp_path):
        from podcast_worker.core import tts_engine

        calls = []

        def post(*args, **kwargs):
            calls.append((args, kwargs))
            raise RuntimeError("socket closed after dispatch")

        monkeypatch.setattr(tts_engine.config.settings, "elevenlabs_api_key", "server-secret")
        monkeypatch.setitem(sys.modules, "requests", SimpleNamespace(post=post, RequestException=RuntimeError))
        plan = plan_dialogue_requests("Interviewer: Hello.", _snapshot("text_to_dialogue_v3"))[0]

        with pytest.raises(RoutingConfigurationError, match="tts_outcome_unknown"):
            synthesize_elevenlabs_plan(plan, _snapshot("text_to_dialogue_v3"), str(tmp_path / "scene.mp3"))
        assert len(calls) == 1
        assert classify_tts_failure(None, dispatched=True) == "unknown_outcome"
    @pytest.mark.parametrize(
        ("status_code", "outcome"),
        [(429, "retryable"), (500, "retryable"), (503, "retryable"), (400, "terminal")],
    )
    def test_only_proven_provider_rejections_are_classified_for_safe_retry(self, status_code, outcome):
        assert classify_tts_failure(status_code, dispatched=True) == outcome

    def test_pre_send_failure_is_not_misclassified_as_a_provider_rejection(self):
        assert classify_tts_failure(None, dispatched=False) == "pre_send"
