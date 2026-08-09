from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Callable
from urllib.parse import urlsplit

import httpx

CONNECT_TIMEOUT = 5.0
TOTAL_TIMEOUTS = {"health": 15.0, "config": 15.0, "list": 15.0, "detail": 15.0, "artifact_metadata": 15.0, "outline_preview": 90.0, "create": 30.0, "cancel": 30.0, "delete": 30.0, "artifact_transfer": 60.0}
BACKOFF = (2.0, 4.0, 8.0, 16.0, 30.0)
RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}
SAFE_ERROR_CODES = {"generation_capacity_full", "invalid_request", "invalid_outline", "preview_expired", "preview_mismatch", "provider_quota_exceeded", "provider_budget_exceeded", "budget_exceeded", "project_not_cancellable", "unauthorized", "forbidden", "not_found"}
TOPIC = "how to get over a long term relationship in your 20s"
STAGES = ("research", "text_generation", "fact_checking", "tts", "mixing", "finalizing")


@dataclass(frozen=True)
class Evidence:
    method: str
    route: str
    attempt: int
    started_monotonic: float
    ended_monotonic: float
    elapsed_ms: int
    status: int | None
    retry_after: float | None
    error_code: str | None
    project_id: str | None
    progress_version: int | None
    outcome: str
    observed_at: str


class VerificationFailure(RuntimeError):
    pass


class BoundedEndpointVerifier:
    """No-hang, single-project live pipeline verifier with allowlisted evidence."""

    def __init__(self, base_url: str, token: str, *, transport: httpx.AsyncBaseTransport | None = None, sleep: Callable[[float], Any] = asyncio.sleep, deadline_seconds: float = 20 * 60, request_budget: int = 400) -> None:
        parts = urlsplit(base_url)
        if parts.scheme != "https" or not parts.hostname or parts.username or parts.password:
            raise VerificationFailure("BASE_URL must be HTTPS without user-info")
        if not token.strip():
            raise VerificationFailure("TOKEN must be set and nonblank")
        self.base_url = f"https://{parts.netloc}".rstrip("/")
        self._token, self._transport, self._sleep = token, transport, sleep
        self._deadline_seconds, self._request_budget = deadline_seconds, request_budget
        self._started, self._requests = time.monotonic(), 0
        self._active_project_id: str | None = None
        self._last_version = -1
        self._last_versions = {"list": -1, "detail": -1}
        self._segment_ids: list[str] = []
        self._verified_text_version: int | None = None
        self._ready_version: int | None = None
        self._stage_progress_by_surface: dict[str, dict[str, tuple[int, str]]] = {
            "list": {},
            "detail": {},
        }
        self.evidence: list[Evidence] = []
        self.observations: list[dict[str, Any]] = []

    @property
    def target_host(self) -> str:
        return urlsplit(self.base_url).hostname or ""

    def _ensure_budget(self) -> None:
        if self._requests >= self._request_budget:
            raise VerificationFailure("request_budget_exhausted")
        if time.monotonic() - self._started >= self._deadline_seconds:
            raise VerificationFailure("scenario_deadline_exceeded")

    async def request(self, operation: str, method: str, route: str, *, body: dict[str, Any] | None = None, authenticated: bool = True, attempts: int = 6, allowed_statuses: frozenset[int] = frozenset()) -> httpx.Response:
        if operation not in TOTAL_TIMEOUTS:
            raise VerificationFailure("unknown_operation")
        if not route.startswith("/") or "?" in route:
            raise VerificationFailure("route_must_be_safe_path")
        headers = {"Accept": "application/json"}
        if authenticated:
            headers["Authorization"] = f"Bearer {self._token}"
        timeout = httpx.Timeout(TOTAL_TIMEOUTS[operation], connect=CONNECT_TIMEOUT)
        async with httpx.AsyncClient(base_url=self.base_url, headers=headers, timeout=timeout, transport=self._transport, follow_redirects=False) as client:
            for attempt in range(1, attempts + 1):
                self._ensure_budget()
                self._requests += 1
                started, response, retry_after, error_code = time.monotonic(), None, None, None
                outcome = "bounded_response"
                try:
                    response = await client.request(method, route, json=body)
                    retry_after = self._parse_retry_after(response.headers.get("Retry-After"))
                    payload = self._safe_json(response)
                    error_code = self._safe_error_code(payload)
                    if response.status_code in allowed_statuses:
                        self._record(method, route, attempt, started, response, retry_after, error_code, "expected_http")
                        return response
                    if response.status_code >= 400:
                        if error_code is None:
                            self._record(method, route, attempt, started, response, retry_after, None, "unrecognized_error")
                            raise VerificationFailure("unrecognized_error_envelope")
                        if error_code in {"provider_quota_exceeded", "provider_budget_exceeded", "budget_exceeded"}:
                            self._record(method, route, attempt, started, response, retry_after, error_code, "http_error")
                            raise VerificationFailure("provider_quota_or_budget_exceeded")
                        if response.status_code in RETRYABLE_STATUS:
                            self._record(method, route, attempt, started, response, retry_after, error_code, "retryable_http")
                            if attempt >= attempts:
                                raise VerificationFailure("retryable_http")
                            await self._sleep(max(BACKOFF[min(attempt - 1, len(BACKOFF) - 1)], retry_after or 0))
                            continue
                        self._record(method, route, attempt, started, response, retry_after, error_code, "http_error")
                        raise VerificationFailure("http_error")
                    self._record(method, route, attempt, started, response, retry_after, error_code, outcome)
                    return response
                except httpx.ConnectTimeout:
                    outcome = "transport_timeout"
                except (httpx.ReadTimeout, httpx.WriteTimeout, httpx.PoolTimeout):
                    outcome = "server_no_response"
                except httpx.TransportError:
                    outcome = "transport_failure"
                self._record(method, route, attempt, started, response, retry_after, error_code, outcome)
                if attempt >= attempts:
                    raise VerificationFailure(outcome)
                await self._sleep(max(BACKOFF[min(attempt - 1, len(BACKOFF) - 1)], retry_after or 0))
        raise VerificationFailure("unreachable")

    async def run_scenario(self) -> dict[str, Any]:
        await self._health_and_config()
        preview, llm_id, tts_id = await self._preview()
        project = await self._create(preview, llm_id, tts_id)
        project_id = self._project_id(project)
        self.claim_active_project(project_id)
        self._validate_initial(project, project_id)
        await self._immediate_manifest_check(project_id)
        terminal = await self._poll(project_id)
        await self._validate_terminal(terminal, project_id)
        await self._cleanup(terminal, project_id)
        return self.sanitized_record()

    async def _health_and_config(self) -> None:
        health = await self.request("health", "GET", "/api/v1/health", authenticated=False)
        payload = self._require_status(health, 200, "health_preflight_failed")
        if payload.get("status") != "ok" or not isinstance(payload.get("version"), str):
            raise VerificationFailure("invalid_health_payload")
        config = self._require_status(await self.request("config", "GET", "/api/v1/config"), 200, "config_preflight_failed")
        self._config_profiles(config)

    async def _preview(self) -> tuple[dict[str, Any], str, str]:
        config = self._last_payload
        llm_id, tts_id = self._config_profiles(config)
        body = {"topic": TOPIC, "bpm": 120, "duration_minutes": 1, "llm_profile_id": llm_id, "tts_profile_id": tts_id}
        preview = self._require_status(await self.request("outline_preview", "POST", "/api/v1/projects/outline-preview", body=body), 200, "outline_preview_failed")
        binding = preview.get("binding")
        sections = preview.get("sections")
        if not isinstance(binding, dict) or not isinstance(sections, list) or len(sections) != 1:
            raise VerificationFailure("invalid_outline_preview")
        section = sections[0]
        required = ("outline_preview_id", "llm_profile_id", "tts_profile_id", "llm_routing_revision", "tts_routing_revision", "expires_at")
        if any(not isinstance(binding.get(key), str) or not binding[key] for key in required):
            raise VerificationFailure("incomplete_outline_binding")
        if binding["llm_profile_id"] != llm_id or binding["tts_profile_id"] != tts_id or not self._valid_section(section, 0):
            raise VerificationFailure("invalid_outline_preview")
        return preview, llm_id, tts_id

    async def _create(self, preview: dict[str, Any], llm_id: str, tts_id: str) -> dict[str, Any]:
        body = {"topic": TOPIC, "bpm": 120, "duration_minutes": 1, "llm_profile_id": llm_id, "tts_profile_id": tts_id, "outline_preview_id": preview["binding"]["outline_preview_id"], "approved_outline": preview}
        payload = self._require_status(await self.request("create", "POST", "/api/v1/projects", body=body), 202, "create_failed")
        project = payload.get("project")
        if not isinstance(project, dict):
            raise VerificationFailure("invalid_create_response")
        return project

    async def _immediate_manifest_check(self, project_id: str) -> None:
        listing = self._require_status(await self.request("list", "GET", "/api/v1/projects"), 200, "list_failed")
        projects = listing.get("projects")
        summary = next((item for item in projects if isinstance(item, dict) and item.get("project_id") == project_id), None) if isinstance(projects, list) else None
        if not isinstance(summary, dict):
            raise VerificationFailure("project_missing_from_list")
        self._validate_summary(summary, project_id, initial=True)
        detail = self._require_status(await self.request("detail", "GET", f"/api/v1/projects/{project_id}"), 200, "detail_failed")
        self._validate_manifest(detail, project_id, initial=True)

    async def _poll(self, project_id: str) -> dict[str, Any]:
        while True:
            listing = self._require_status(await self.request("list", "GET", "/api/v1/projects"), 200, "list_failed")
            projects = listing.get("projects", [])
            summary = next((x for x in projects if isinstance(x, dict) and x.get("project_id") == project_id), None)
            if not isinstance(summary, dict):
                raise VerificationFailure("project_missing_from_list")
            self._validate_summary(summary, project_id)
            detail = self._require_status(await self.request("detail", "GET", f"/api/v1/projects/{project_id}"), 200, "detail_failed")
            self._validate_manifest(detail, project_id)
            progress = detail["generation_progress"]
            if progress.get("is_terminal"):
                return detail
            await self._sleep(3.0)

    def _validate_initial(self, project: dict[str, Any], project_id: str) -> None:
        self._validate_manifest(project, project_id, initial=True)

    def _validate_summary(self, summary: dict[str, Any], project_id: str, *, initial: bool = False) -> None:
        if summary.get("project_id") != project_id or summary.get("segment_count") != 1 or summary.get("ready_segment_count") not in (0, 1):
            raise VerificationFailure("invalid_list_counts")
        self._validate_progress(summary.get("generation_progress"), initial=initial, surface="list")

    def _validate_manifest(self, project: dict[str, Any], project_id: str, initial: bool = False) -> None:
        if project.get("project_id") != project_id:
            raise VerificationFailure("project_id_mismatch")
        segments = project.get("segments")
        if not isinstance(segments, list) or len(segments) != 1 or not self._valid_segment(segments[0], 0):
            raise VerificationFailure("invalid_segment_order")
        segment_id = segments[0]["segment_id"]
        if self._segment_ids and self._segment_ids != [segment_id]:
            raise VerificationFailure("segment_id_mutation")
        self._segment_ids = [segment_id]
        self._validate_progress(project.get("generation_progress"), initial=initial, surface="detail")
        self._observe(project)
        self._validate_text_and_artifacts(project, segments[0])

    def _validate_progress(self, progress: Any, *, initial: bool, surface: str = "detail") -> None:
        if not isinstance(progress, dict) or progress.get("planned_segment_count") != 1:
            raise VerificationFailure("invalid_progress")
        version = progress.get("progress_version")
        if not isinstance(version, int) or version < self._last_versions[surface]:
            raise VerificationFailure("progress_version_regression")
        self._last_versions[surface] = version
        if surface == "detail":
            self._last_version = version
        stages = {item.get("name"): item for item in progress.get("stages", []) if isinstance(item, dict)}
        if set(stages) != set(STAGES):
            raise VerificationFailure("invalid_progress_stages")
        expected = {"research": 2, "text_generation": 1, "fact_checking": 1, "tts": 1, "mixing": 1, "finalizing": 1}
        for name, total in expected.items():
            stage = stages[name]
            completed, state = stage.get("completed_units"), stage.get("state")
            if stage.get("total_units") != total or not isinstance(completed, int) or not 0 <= completed <= total or not isinstance(state, str):
                raise VerificationFailure("invalid_progress_counters")
            previous = self._stage_progress_by_surface[surface].get(name)
            if previous is not None and (completed < previous[0] or self._stage_state_rank(state) < self._stage_state_rank(previous[1])):
                raise VerificationFailure(
                    f"stage_progress_regression:{surface}:{name}:{previous[0]}:{previous[1]}:{completed}:{state}:v{version}"
                )
            self._stage_progress_by_surface[surface][name] = (completed, state)
            if initial and completed != 0:
                raise VerificationFailure("initial_progress_not_zero")

    def _validate_text_and_artifacts(self, project: dict[str, Any], segment: dict[str, Any]) -> None:
        text, provenance, ready = segment.get("text"), segment.get("provenance"), segment.get("status") == "ready"
        if text is not None:
            if not isinstance(text, str) or not text or not isinstance(provenance, dict) or provenance.get("validation_status") != "validated":
                raise VerificationFailure("invalid_verified_text")
            self._verified_text_version = self._last_version if self._verified_text_version is None else self._verified_text_version
            self.observations.append({"segment_id": segment["segment_id"], "text_length": len(text), "text_sha256": hashlib.sha256(text.encode()).hexdigest(), "validation_status": "validated", "progress_version": self._last_version})
        if ready:
            if text is None or self._verified_text_version is None or self._verified_text_version >= self._last_version:
                raise VerificationFailure("ready_without_prior_verified_text")
            self._ready_version = self._last_version
            artifact_id = segment.get("primary_audio_artifact_id")
            artifact = self._artifact(project, artifact_id, "segment_audio", segment["segment_id"])
            if artifact is None:
                raise VerificationFailure("ready_without_segment_artifact")

    async def _validate_terminal(self, project: dict[str, Any], project_id: str) -> None:
        progress = project.get("generation_progress", {})
        if project.get("status") != "ready" or not progress.get("is_terminal") or progress.get("disposition") != "terminal" or progress.get("terminal_outcome") != "ready" or not project.get("final_download_ready"):
            raise VerificationFailure("terminal_project_not_ready")
        expected_totals = {"research": 2, "text_generation": 1, "fact_checking": 1, "tts": 1, "mixing": 1, "finalizing": 1}
        if any(self._stage_progress_by_surface["detail"].get(name) != (expected_totals[name], "completed") for name in STAGES):
            raise VerificationFailure("terminal_stages_incomplete")
        segment = project["segments"][0]
        segment_artifact = self._artifact(project, segment.get("primary_audio_artifact_id"), "segment_audio", segment["segment_id"])
        final_artifact = self._artifact(project, project.get("final_artifact_id"), "final_mp3", None)
        if segment_artifact is None or final_artifact is None or self._verified_text_version is None or self._ready_version is None:
            raise VerificationFailure("missing_terminal_artifact")
        await self._download_artifact(segment_artifact)
        await self._download_artifact(final_artifact)

    async def _download_artifact(self, artifact: dict[str, Any]) -> None:
        response = await self.request("artifact_transfer", "GET", f"/api/v1/artifacts/{artifact['artifact_id']}")
        if response.status_code != 200 or not response.content or not response.headers.get("content-type", "").startswith("audio/"):
            raise VerificationFailure("artifact_download_failed")
        if len(response.content) != artifact["size_bytes"] or hashlib.sha256(response.content).hexdigest() != artifact["checksum_sha256"]:
            raise VerificationFailure("artifact_checksum_mismatch")
        self.observations.append({"artifact_id": artifact["artifact_id"], "kind": artifact["kind"], "byte_count": len(response.content), "checksum_matches": True})

    def _artifact(self, project: dict[str, Any], artifact_id: Any, kind: str, segment_id: str | None) -> dict[str, Any] | None:
        artifacts = project.get("artifacts", [])
        artifact = next((x for x in artifacts if isinstance(x, dict) and x.get("artifact_id") == artifact_id), None)
        if not isinstance(artifact, dict) or artifact.get("kind") != kind or artifact.get("status") != "ready" or artifact.get("segment_id") != segment_id:
            return None
        if not isinstance(artifact.get("checksum_sha256"), str) or not isinstance(artifact.get("size_bytes"), int) or artifact["size_bytes"] <= 0 or not isinstance(artifact.get("duration_seconds"), (int, float)) or artifact["duration_seconds"] <= 0:
            return None
        return artifact

    async def _cleanup(self, terminal: dict[str, Any], project_id: str) -> None:
        artifact_ids = {
            artifact.get("artifact_id")
            for artifact in terminal.get("artifacts", [])
            if isinstance(artifact, dict) and isinstance(artifact.get("artifact_id"), str)
        }
        payload = self._require_status(await self.request("delete", "DELETE", self.exact_cleanup_route(project_id, terminal=True)), 200, "cleanup_failed")
        if payload.get("project_id") != project_id or not payload.get("deleted_at"):
            raise VerificationFailure("invalid_cleanup_response")

        project_response = await self.request(
            "detail",
            "GET",
            f"/api/v1/projects/{project_id}",
            attempts=1,
            allowed_statuses=frozenset({404}),
        )
        if project_response.status_code != 404:
            raise VerificationFailure("deleted_project_still_available")
        for artifact_id in sorted(artifact_ids):
            artifact_response = await self.request(
                "artifact_transfer",
                "GET",
                f"/api/v1/artifacts/{artifact_id}",
                attempts=1,
                allowed_statuses=frozenset({404}),
            )
            if artifact_response.status_code != 404:
                raise VerificationFailure("deleted_artifact_still_available")
        self.observations.append({
            "cleanup_verified": True,
            "project_retired": True,
            "retired_artifact_count": len(artifact_ids),
        })

    def claim_active_project(self, project_id: str) -> None:
        if self._active_project_id is not None and self._active_project_id != project_id:
            raise VerificationFailure("second_active_project_rejected")
        if not project_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" for c in project_id):
            raise VerificationFailure("unsafe_project_id")
        self._active_project_id = project_id

    def exact_cleanup_route(self, project_id: str, *, terminal: bool) -> str:
        if not terminal or self._active_project_id != project_id:
            raise VerificationFailure("cleanup_guard_rejected")
        return f"/api/v1/projects/{project_id}"

    def sanitized_record(self) -> dict[str, Any]:
        return {"target_host": self.target_host, "topic_sha256": hashlib.sha256(TOPIC.encode()).hexdigest(), "requests": [asdict(item) for item in self.evidence], "observations": self.observations}

    def _observe(self, project: dict[str, Any]) -> None:
        progress = project["generation_progress"]
        self.observations.append({"project_id": project["project_id"], "progress_version": progress["progress_version"], "stages": [{key: stage.get(key) for key in ("name", "state", "completed_units", "total_units")} for stage in progress["stages"]]})

    def _config_profiles(self, config: dict[str, Any]) -> tuple[str, str]:
        self._last_payload = config
        if self._contains_unsafe_config_field(config):
            raise VerificationFailure("unsafe_config_payload")
        llm, tts = config.get("default_llm_profile_id"), config.get("default_tts_profile_id")
        llm_profiles, tts_profiles = config.get("llm_profiles"), config.get("tts_profiles")
        if not isinstance(llm_profiles, list) or not isinstance(tts_profiles, list):
            raise VerificationFailure("missing_profile_lists")
        llm_ids = {item.get("id") for item in llm_profiles if isinstance(item, dict)}
        tts_ids = {item.get("id") for item in tts_profiles if isinstance(item, dict)}
        if not isinstance(llm, str) or not isinstance(tts, str) or not llm or not tts or llm not in llm_ids or tts not in tts_ids:
            raise VerificationFailure("invalid_default_profiles")
        return llm, tts

    @staticmethod
    def _contains_unsafe_config_field(value: Any) -> bool:
        forbidden = ("key", "secret", "provider", "route", "token")
        if isinstance(value, dict):
            return any(any(word in str(key).lower() for word in forbidden) or BoundedEndpointVerifier._contains_unsafe_config_field(item) for key, item in value.items())
        if isinstance(value, list):
            return any(BoundedEndpointVerifier._contains_unsafe_config_field(item) for item in value)
        return False
    @staticmethod
    def _stage_state_rank(state: str) -> int:
        ranks = {"pending": 0, "queued": 0, "running": 1, "completed": 2, "failed": 2, "cancelled": 2}
        if state not in ranks:
            raise VerificationFailure("invalid_stage_state")
        return ranks[state]

    @staticmethod
    def _valid_section(section: Any, index: int) -> bool:
        return isinstance(section, dict) and section.get("index") == index and section.get("segment_type") in {"intro", "content", "outro"} and all(isinstance(section.get(key), str) and section[key].strip() for key in ("topic", "title")) and isinstance(section.get("approx_duration_seconds"), (int, float)) and not isinstance(section.get("approx_duration_seconds"), bool) and section["approx_duration_seconds"] > 0

    @staticmethod
    def _valid_segment(segment: Any, index: int) -> bool:
        return isinstance(segment, dict) and segment.get("index") == index and isinstance(segment.get("segment_id"), str) and bool(segment["segment_id"])

    @staticmethod
    def _project_id(project: dict[str, Any]) -> str:
        value = project.get("project_id")
        if not isinstance(value, str):
            raise VerificationFailure("invalid_project_id")
        return value

    @staticmethod
    def _require_status(response: httpx.Response, status: int, failure: str) -> dict[str, Any]:
        if response.status_code != status:
            raise VerificationFailure(failure)
        payload = BoundedEndpointVerifier._safe_json(response)
        if not payload:
            raise VerificationFailure(f"{failure}_payload")
        return payload

    def _record(self, method: str, route: str, attempt: int, started: float, response: httpx.Response | None, retry_after: float | None, error_code: str | None, outcome: str) -> None:
        ended, payload = time.monotonic(), self._safe_json(response)
        project = payload.get("project") if isinstance(payload.get("project"), dict) else payload
        progress = project.get("generation_progress") if isinstance(project, dict) else None
        self.evidence.append(Evidence(method.upper(), route, attempt, started, ended, max(0, round((ended - started) * 1000)), response.status_code if response else None, retry_after, error_code, project.get("project_id") if isinstance(project, dict) else None, progress.get("progress_version") if isinstance(progress, dict) else None, outcome, datetime.now(timezone.utc).isoformat()))

    @staticmethod
    def _parse_retry_after(value: str | None) -> float | None:
        try:
            parsed = float(value) if value is not None else None
        except ValueError:
            return None
        return parsed if parsed is not None and parsed >= 0 else None

    @staticmethod
    def _safe_json(response: httpx.Response | None) -> dict[str, Any]:
        try:
            value = response.json() if response is not None else {}
        except (ValueError, json.JSONDecodeError):
            return {}
        return value if isinstance(value, dict) else {}

    @staticmethod
    def _safe_error_code(payload: dict[str, Any]) -> str | None:
        error = payload.get("error")
        code = error.get("code") if isinstance(error, dict) else None
        return code if code in SAFE_ERROR_CODES else None


async def main() -> None:
    verifier = BoundedEndpointVerifier(os.environ.get("BASE_URL", ""), os.environ.get("TOKEN", ""))
    print(json.dumps(await verifier.run_scenario(), sort_keys=True))


if __name__ == "__main__":
    asyncio.run(main())
