from __future__ import annotations

import asyncio
import hashlib
import json

import httpx
import pytest

from endpoint_verifier import BoundedEndpointVerifier, VerificationFailure, TOPIC


def progress(version: int, terminal: bool = False) -> dict:
    totals = {"research": 2, "text_generation": 1, "fact_checking": 1, "tts": 1, "mixing": 1, "finalizing": 1}
    return {"progress_version": version, "planned_segment_count": 1, "is_terminal": terminal, "disposition": "terminal" if terminal else "active", "terminal_outcome": "ready" if terminal else None, "stages": [{"name": name, "state": "completed" if terminal else "running", "completed_units": total if terminal else 0, "total_units": total} for name, total in totals.items()]}


def project(version: int, terminal: bool = False, verified: bool = False) -> dict:
    audio = b"fake-mp3"
    checksum = hashlib.sha256(audio).hexdigest()
    segment = {"segment_id": "seg_1", "index": 0, "status": "ready" if terminal else "scripting", "text": "verified generated text" if terminal or verified else None, "provenance": {"validation_status": "validated"} if terminal or verified else None, "primary_audio_artifact_id": "art_seg" if terminal else None}
    artifacts = [{"artifact_id": "art_seg", "kind": "segment_audio", "segment_id": "seg_1", "status": "ready", "checksum_sha256": checksum, "size_bytes": len(audio), "duration_seconds": 60}] if terminal else []
    if terminal:
        artifacts.append({"artifact_id": "art_final", "kind": "final_mp3", "segment_id": None, "status": "ready", "checksum_sha256": checksum, "size_bytes": len(audio), "duration_seconds": 60})
    return {"project_id": "prj_1", "status": "ready" if terminal else "generating", "segments": [segment], "artifacts": artifacts, "final_download_ready": terminal, "final_artifact_id": "art_final" if terminal else None, "generation_progress": progress(version, terminal)}


@pytest.mark.asyncio
async def test_full_one_minute_flow_is_sanitized_and_cleans_exact_terminal_id() -> None:
    calls: list[tuple[str, str]] = []
    polls = 0
    deleted = False

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal polls, deleted
        calls.append((request.method, request.url.path))
        path = request.url.path
        if path == "/api/v1/health": return httpx.Response(200, json={"status": "ok", "version": "v1", "uptime_seconds": 1})
        if path == "/api/v1/config": return httpx.Response(200, json={"llm_profiles": [{"id": "llm-default", "label": "Default"}], "tts_profiles": [{"id": "tts-default", "label": "Default"}], "voices": [], "default_llm_profile_id": "llm-default", "default_tts_profile_id": "tts-default", "bpm_range": {"min": 60, "max": 180}, "duration_minutes_range": {"min": 1, "max": 30}})
        if path.endswith("outline-preview"): return httpx.Response(200, json={"topic": TOPIC, "title": "Safe", "sections": [{"index": 0, "segment_type": "intro", "topic": "section", "title": "One", "approx_duration_seconds": 60}], "binding": {"outline_preview_id": "prev_1", "llm_profile_id": "llm-default", "tts_profile_id": "tts-default", "llm_routing_revision": "r1", "tts_routing_revision": "r2", "expires_at": "2099-01-01T00:00:00Z"}})
        if request.method == "POST" and path == "/api/v1/projects": return httpx.Response(202, json={"project": project(0)})
        if path == "/api/v1/projects":
            polls += 1
            value = project(1 if polls < 3 else 2, polls >= 3)
            return httpx.Response(200, json={"projects": [{"project_id": "prj_1", "segment_count": 1, "ready_segment_count": int(polls >= 3), "generation_progress": value["generation_progress"]}]})
        if path == "/api/v1/projects/prj_1" and request.method == "GET":
            return httpx.Response(404, json={"error": {"code": "not_found"}}) if deleted else httpx.Response(200, json=project(1 if polls < 3 else 2, polls >= 3, polls == 2))
        if path.startswith("/api/v1/artifacts/"):
            return httpx.Response(404, json={"error": {"code": "not_found"}}) if deleted else httpx.Response(200, content=b"fake-mp3", headers={"content-type": "audio/mpeg"})
        if request.method == "DELETE":
            deleted = True
            return httpx.Response(200, json={"project_id": "prj_1", "status": "deleted", "deleted_at": "2099-01-01T00:00:00Z"})
        raise AssertionError(path)

    async def sleep(_: float) -> None: pass
    verifier = BoundedEndpointVerifier("https://example.test", "secret-token", transport=httpx.MockTransport(handler), sleep=sleep)
    record = await verifier.run_scenario()
    assert ("DELETE", "/api/v1/projects/prj_1") in calls
    assert "secret-token" not in json.dumps(record)
    assert TOPIC not in json.dumps(record)
    assert any(item.get("checksum_matches") for item in record["observations"])
    delete_index = calls.index(("DELETE", "/api/v1/projects/prj_1"))
    assert calls[delete_index + 1:] == [
        ("GET", "/api/v1/projects/prj_1"),
        ("GET", "/api/v1/artifacts/art_final"),
        ("GET", "/api/v1/artifacts/art_seg"),
    ]
    assert any(item.get("cleanup_verified") and item.get("retired_artifact_count") == 2 for item in record["observations"])


@pytest.mark.asyncio
async def test_invariant_failure_does_not_cleanup() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/health": return httpx.Response(200, json={"status": "ok", "version": "v1"})
        if request.url.path == "/api/v1/config": return httpx.Response(200, json={"llm_profiles": [{"id": "llm"}], "tts_profiles": [{"id": "tts"}], "default_llm_profile_id": "llm", "default_tts_profile_id": "tts"})
        if request.url.path.endswith("outline-preview"): return httpx.Response(200, json={"sections": [], "binding": {}})
        raise AssertionError("network continued after invariant")
    verifier = BoundedEndpointVerifier("https://example.test", "token", transport=httpx.MockTransport(handler))
    with pytest.raises(VerificationFailure, match="invalid_outline_preview"):
        await verifier.run_scenario()


@pytest.mark.asyncio
async def test_polling_request_budget_is_bounded() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429)

    async def sleep(_: float) -> None: pass
    verifier = BoundedEndpointVerifier("https://example.test", "token", transport=httpx.MockTransport(handler), sleep=sleep, request_budget=1)
    with pytest.raises(VerificationFailure, match="unrecognized_error_envelope"):
        await verifier.request("health", "GET", "/api/v1/health", authenticated=False, attempts=2)


def test_cleanup_requires_captured_terminal_id_and_redacts_topic() -> None:
    verifier = BoundedEndpointVerifier("https://example.test", "token-value")
    verifier.claim_active_project("prj_one")
    with pytest.raises(VerificationFailure, match="cleanup_guard"):
        verifier.exact_cleanup_route("prj_one", terminal=False)
    assert verifier.exact_cleanup_route("prj_one", terminal=True) == "/api/v1/projects/prj_one"
    record = json.dumps(verifier.sanitized_record())
    assert "token-value" not in record and TOPIC not in record
@pytest.mark.asyncio
async def test_retryable_http_error_fails_without_retry() -> None:
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(429, json={"error": {"code": "provider_quota_exceeded"}})

    verifier = BoundedEndpointVerifier("https://example.test", "token", transport=httpx.MockTransport(handler))
    with pytest.raises(VerificationFailure, match="provider_quota_or_budget_exceeded"):
        await verifier.request("health", "GET", "/api/v1/health", authenticated=False)
    assert calls == 1


def test_progress_and_config_invariants() -> None:
    verifier = BoundedEndpointVerifier("https://example.test", "token")
    verifier._validate_progress(progress(0), initial=True)
    regressed = progress(1)
    regressed["stages"][0]["completed_units"] = -1
    with pytest.raises(VerificationFailure, match="invalid_progress_counters"):
        verifier._validate_progress(regressed, initial=False)
    with pytest.raises(VerificationFailure, match="unsafe_config_payload"):
        verifier._config_profiles({"llm_profiles": [{"id": "llm"}], "tts_profiles": [{"id": "tts", "nested": {"provider_route": "x"}}], "default_llm_profile_id": "llm", "default_tts_profile_id": "tts"})
    with pytest.raises(VerificationFailure, match="invalid_default_profiles"):
        verifier._config_profiles({"llm_profiles": [{"id": "other"}], "tts_profiles": [{"id": "tts"}], "default_llm_profile_id": "llm", "default_tts_profile_id": "tts"})
@pytest.mark.asyncio
async def test_recognized_retryable_error_honors_retry_after_then_succeeds() -> None:
    calls = 0
    delays: list[float] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, headers={"Retry-After": "7"}, json={"error": {"code": "generation_capacity_full"}})
        return httpx.Response(200, json={"status": "ok"})

    async def sleep(delay: float) -> None:
        delays.append(delay)

    verifier = BoundedEndpointVerifier("https://example.test", "token", transport=httpx.MockTransport(handler), sleep=sleep)
    assert (await verifier.request("health", "GET", "/api/v1/health", authenticated=False)).status_code == 200
    assert calls == 2
    assert delays == [7.0]


def test_stage_state_rank_cannot_regress_when_counter_advances() -> None:
    verifier = BoundedEndpointVerifier("https://example.test", "token")
    first = progress(0)
    first["stages"][0].update({"completed_units": 1, "state": "completed"})
    verifier._validate_progress(first, initial=False)
    regressed = progress(1)
    regressed["stages"][0].update({"completed_units": 2, "state": "running"})
    with pytest.raises(VerificationFailure, match="stage_progress_regression"):
        verifier._validate_progress(regressed, initial=False)


@pytest.mark.asyncio
async def test_terminal_stage_states_must_all_be_completed() -> None:
    verifier = BoundedEndpointVerifier("https://example.test", "token")
    terminal = project(1, terminal=True)
    terminal["generation_progress"]["stages"][0]["state"] = "failed"
    verifier._validate_progress(terminal["generation_progress"], initial=False)
    with pytest.raises(VerificationFailure, match="terminal_stages_incomplete"):
        await verifier._validate_terminal(terminal, "prj_1")
