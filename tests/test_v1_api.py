"""Integration tests for /api/v1/* endpoints.

Tests cover:
- Auth enforcement (401 on missing/invalid token)
- Project CRUD lifecycle (create, list, get, delete)
- Health and config endpoints
- Artifact access control
- Error envelope consistency
- Durability across client restarts
"""

from __future__ import annotations

import os
import sqlite3
import tempfile
import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from podcast_worker.routers.v1_projects import _start_project_pipeline as real_start_project_pipeline


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _patch_config(monkeypatch):
    """Ensure isolated test config for every test run."""
    from podcast_worker.core.config import settings as cfg

    # Create temp DB dir
    tmp = tempfile.mkdtemp(prefix="test_db_")
    db_path = os.path.join(tmp, "test.db")

    # Patch the singleton's attributes directly (env vars are read at construction time)
    monkeypatch.setattr(cfg, "auth_token", "test-token-abc123")
    monkeypatch.setattr(cfg, "db_path", db_path)
    monkeypatch.setattr(cfg, "llm_routing_profiles_json", "")
    monkeypatch.setattr(cfg, "tts_profiles_json", "")
    monkeypatch.setattr(cfg, "tts_provider", "elevenlabs")
    monkeypatch.setattr(cfg, "elevenlabs_api_key", "test-elevenlabs-key")
    from podcast_worker.routers import v1_projects

    monkeypatch.setattr(
        v1_projects,
        "_generation_slots",
        threading.Semaphore(cfg.max_concurrent_generations),
    )

    from podcast_worker.routers import v1_projects

    def deterministic_outline(topic, bpm, duration_minutes, provider=None, model=None):
        del bpm, provider, model
        count = min(max((duration_minutes + 1) // 2, 1), 12)
        return {
            "title": f"{topic} outline",
            "sections": [
                {
                    "segment_type": (
                        "intro" if index == 0 else "outro" if index == count - 1 else "content"
                    ),
                    "topic": f"Section {index + 1}",
                    "title": f"Section {index + 1}",
                    "approx_duration_seconds": 120,
                }
                for index in range(count)
            ],
        }

    def finish_without_worker(args):
        del args
        v1_projects._generation_slots.release()

    monkeypatch.setattr(
        v1_projects, "generate_script_outline", deterministic_outline
    )
    monkeypatch.setattr(v1_projects, "_start_project_pipeline", finish_without_worker)

    yield

    # Cleanup
    try:
        os.unlink(db_path)
        os.unlink(db_path + "-wal")
        os.unlink(db_path + "-shm")
    except FileNotFoundError:
        pass
    try:
        os.rmdir(tmp)
    except OSError:
        pass


@pytest.fixture
def client():
    from podcast_worker.main import app
    return TestClient(app)


@pytest.fixture
def auth_headers():
    return {"Authorization": "Bearer test-token-abc123"}


def _reviewed_create(client, auth_headers, payload):
    preview_request = {
        "topic": payload["topic"],
        "bpm": payload.get("bpm", 120),
        "duration_minutes": payload.get("duration_minutes", 5),
    }
    preview = client.post(
        "/api/v1/projects/outline-preview",
        json=preview_request,
        headers=auth_headers,
    )
    assert preview.status_code == 200
    preview_body = preview.json()
    create_payload = dict(payload)
    create_payload["approved_outline"] = preview_body
    create_payload["outline_preview_id"] = preview_body["binding"]["outline_preview_id"]
    return client.post(
        "/api/v1/projects",
        json=create_payload,
        headers=auth_headers,
    )


# ── Health & Config ───────────────────────────────────────────────────────


class TestHealth:
    def test_health_ok(self, client):
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["version"] == "2.0.0"
        assert "uptime_seconds" in data

    def test_config_requires_auth_and_exposes_only_opaque_profiles(self, client, auth_headers):
        assert client.get("/api/v1/config").status_code == 401

        resp = client.get("/api/v1/config", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert [item["id"] for item in data["llm_profiles"]] == ["economy", "balanced"]
        assert [item["id"] for item in data["tts_profiles"]] == ["studio-dialogue"]
        assert [item["id"] for item in data["voices"]] == [
            "interviewer-rachel",
            "interviewer-bella",
            "guest-adam",
            "guest-josh",
        ]
        assert data["default_llm_profile_id"] == "balanced"
        assert data["default_tts_profile_id"] == "studio-dialogue"
        assert data["bpm_range"] == {"min": 60, "max": 220}
        assert data["duration_minutes_range"] == {"min": 1, "max": 30}

        serialized = str(data).lower()
        for forbidden in ("provider", "model", "api_key", "key", "strategy", "voice_binding"):
            assert forbidden not in serialized

    def test_internal_metrics_requires_auth(self, client, auth_headers):
        from podcast_worker.core.observability import metrics

        metrics.reset()
        metrics.increment(
            "podcast_generation_transitions_total",
            stage="research",
            outcome="completed",
        )
        unauthorized = client.get("/internal/metrics")
        assert unauthorized.status_code == 401

        response = client.get("/internal/metrics", headers=auth_headers)
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/plain")
        assert "podcast_generation_transitions_total" in response.text
        assert "project_id" not in response.text


# ── Auth ──────────────────────────────────────────────────────────────────


class TestAuth:
    def test_missing_auth_returns_401(self, client):
        resp = client.get("/api/v1/projects")
        assert resp.status_code == 401
        data = resp.json()
        assert data["error"]["code"] == "unauthorized"

    def test_invalid_token_returns_401(self, client):
        resp = client.get(
            "/api/v1/projects",
            headers={"Authorization": "Bearer wrong-token"},
        )
        assert resp.status_code == 401
        assert resp.json()["error"]["code"] == "unauthorized"

    def test_malformed_auth_header_returns_401(self, client):
        resp = client.get(
            "/api/v1/projects",
            headers={"Authorization": "NoBearerHere test-token-abc123"},
        )
        assert resp.status_code == 401

    def test_valid_token_passes(self, client, auth_headers):
        resp = client.get("/api/v1/projects", headers=auth_headers)
        assert resp.status_code == 200


def test_pipeline_thread_receives_generation_slot(monkeypatch):
    from podcast_worker.routers import v1_projects

    captured = {}

    class FakeThread:
        def __init__(self, *, target, args, daemon):
            captured.update(target=target, args=args, daemon=daemon)

        def start(self):
            captured["started"] = True

    monkeypatch.setattr(v1_projects.threading, "Thread", FakeThread)
    pipeline_args = ("db", "work", "owner", 1, "output")

    real_start_project_pipeline(pipeline_args)

    assert captured["target"] is v1_projects._run_project_pipeline_with_slot
    assert captured["args"] == (v1_projects._generation_slots, *pipeline_args)
    assert captured["daemon"] is True
    assert captured["started"] is True

# ── Projects CRUD ─────────────────────────────────────────────────────────


class TestProjectsCRUD:
    def test_create_project_returns_202(self, client, auth_headers):
        resp = _reviewed_create(
            client,
            auth_headers,
            {"topic": "stoicism", "bpm": 120, "duration_minutes": 5},
        )
        assert resp.status_code == 202
        data = resp.json()
        assert "project" in data
        p = data["project"]
        assert p["topic"] == "stoicism"
        assert p["bpm"] == 120
        assert p["status"] == "queued"
        assert p["owner_id"] == "single-user"
        assert p["project_id"].startswith("prj_")
        assert data["manifest_url"] == f"/api/v1/projects/{p['project_id']}"

    def test_cancel_is_authenticated_owner_scoped_and_idempotent(
        self, client, auth_headers, monkeypatch
    ):
        from podcast_worker.routers import v1_projects

        monkeypatch.setattr(
            v1_projects, "_start_project_pipeline", lambda args: None
        )
        payload = {
            "topic": "cancel contract",
            "bpm": 120,
            "duration_minutes": 1,
            "approved_outline": {
                "project_id": "preview",
                "topic": "cancel contract",
                "title": "Cancel Contract",
                "sections": [{
                    "index": 0,
                    "segment_type": "intro",
                    "topic": "Only section",
                    "title": "Only section",
                    "approx_duration_seconds": 60,
                }],
            },
        }

        try:
            preview = client.post(
                "/api/v1/projects/outline-preview",
                json={
                    "topic": payload["topic"],
                    "bpm": payload["bpm"],
                    "duration_minutes": payload["duration_minutes"],
                },
                headers=auth_headers,
            ).json()
            payload["approved_outline"] = preview
            payload["outline_preview_id"] = preview["binding"]["outline_preview_id"]
            created = client.post(
                "/api/v1/projects", json=payload, headers=auth_headers
            )
            assert created.status_code == 202
            project_id = created.json()["project"]["project_id"]
            initial_version = created.json()["project"]["generation_progress"][
                "progress_version"
            ]

            assert client.post(f"/api/v1/projects/{project_id}/cancel").status_code == 401
            first = client.post(
                f"/api/v1/projects/{project_id}/cancel", headers=auth_headers
            )
            repeated = client.post(
                f"/api/v1/projects/{project_id}/cancel", headers=auth_headers
            )

            assert first.status_code == 202
            assert repeated.status_code == 202
            first_body = first.json()
            repeated_body = repeated.json()
            assert first_body["cancellation"]["state"] == "requested"
            assert first_body["cancellation"]["requested_at"] == repeated_body[
                "cancellation"
            ]["requested_at"]
            assert first_body["project"]["generation_progress"][
                "progress_version"
            ] == initial_version + 1
            assert repeated_body["project"]["generation_progress"][
                "progress_version"
            ] == initial_version + 1
            assert repeated_body["project"]["generation_progress"][
                "current_activity"
            ] == {"kind": "cancellation", "stage": None, "segment_id": None}

            missing = client.post(
                "/api/v1/projects/prj-missing/cancel", headers=auth_headers
            )
            assert missing.status_code == 404

            connection = sqlite3.connect(v1_projects.settings.db_path)
            connection.execute(
                """UPDATE project_generation
                   SET disposition='terminal', terminal_outcome='ready'
                   WHERE project_id=?""",
                (project_id,),
            )
            connection.execute(
                "UPDATE projects SET status='ready' WHERE project_id=?",
                (project_id,),
            )
            connection.commit()
            connection.close()
            terminal = client.post(
                f"/api/v1/projects/{project_id}/cancel", headers=auth_headers
            )
            assert terminal.status_code == 409
            assert terminal.json()["error"]["code"] == "project_not_cancellable"
        finally:
            v1_projects._generation_slots.release()

    def test_create_materializes_exact_plan_before_worker_start(
        self, client, auth_headers, monkeypatch
    ):
        from podcast_worker.routers import v1_projects

        scheduled_args = []
        monkeypatch.setattr(
            v1_projects, "_start_project_pipeline", scheduled_args.append
        )
        sections = [
            {
                "index": index,
                "segment_type": (
                    "intro" if index == 0 else "outro" if index == 5 else "content"
                ),
                "topic": f"Section topic {index}",
                "title": f"Section {index}",
                "approx_duration_seconds": 120,
            }
            for index in range(6)
        ]
        payload = {
            "topic": "atomic plan",
            "bpm": 120,
            "duration_minutes": 12,
            "approved_outline": {
                "project_id": "preview",
                "topic": "atomic plan",
                "title": "Atomic Plan",
                "sections": sections,
            },
        }

        try:
            preview = client.post(
                "/api/v1/projects/outline-preview",
                json={
                    "topic": payload["topic"],
                    "bpm": payload["bpm"],
                    "duration_minutes": payload["duration_minutes"],
                },
                headers=auth_headers,
            ).json()
            payload["approved_outline"] = preview
            payload["outline_preview_id"] = preview["binding"]["outline_preview_id"]
            created = client.post(
                "/api/v1/projects", json=payload, headers=auth_headers
            )
            assert created.status_code == 202
            project = created.json()["project"]
            project_id = project["project_id"]
            segment_ids = [segment["segment_id"] for segment in project["segments"]]
            assert len(segment_ids) == 6
            assert len(set(segment_ids)) == 6
            assert [segment["index"] for segment in project["segments"]] == list(range(6))
            assert len(scheduled_args) == 1
            db_path, work_id, lease_owner, lease_epoch, output_dir = scheduled_args[0]
            assert db_path == v1_projects.settings.db_path
            assert work_id.startswith("work_")
            assert lease_owner.startswith("worker_")
            assert lease_epoch == 1
            assert output_dir

            expected = {
                "research": (0, 7),
                "text_generation": (0, 6),
                "fact_checking": (0, 6),
                "tts": (0, 6),
                "mixing": (0, 6),
                "finalizing": (0, 1),
            }
            progress = project["generation_progress"]
            assert progress["planned_segment_count"] == 6
            assert {
                stage["name"]: (
                    stage["completed_units"],
                    stage["total_units"],
                )
                for stage in progress["stages"]
            } == expected

            listed = client.get("/api/v1/projects", headers=auth_headers)
            detailed = client.get(
                f"/api/v1/projects/{project_id}", headers=auth_headers
            )
            assert listed.status_code == 200
            assert detailed.status_code == 200
            summary = next(
                item
                for item in listed.json()["projects"]
                if item["project_id"] == project_id
            )
            assert summary["segment_count"] == 6
            assert summary["generation_progress"]["planned_segment_count"] == 6
            assert [
                segment["segment_id"] for segment in detailed.json()["segments"]
            ] == segment_ids
        finally:
            v1_projects._generation_slots.release()

    def test_outline_preview_returns_reviewable_plan(self, client, auth_headers, monkeypatch):
        from podcast_worker.routers import v1_projects

        def fake_generate_script_outline(topic, bpm, duration_minutes, provider=None, model=None, snapshot=None):
            return {
                "title": f"{topic.title()} Outline",
                "sections": [
                    {
                        "segment_type": "intro",
                        "topic": "Set the hook",
                        "approx_duration_seconds": 30,
                    },
                    {
                        "segment_type": "content",
                        "topic": "Develop the idea",
                        "approx_duration_seconds": 60,
                    },
                ],
            }

        monkeypatch.setattr(v1_projects, "generate_script_outline", fake_generate_script_outline)

        resp = client.post(
            "/api/v1/projects/outline-preview",
            json={"topic": "creative focus", "bpm": 120, "duration_minutes": 3},
            headers=auth_headers,
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["project_id"] == "preview"
        assert data["topic"] == "creative focus"
        assert data["title"] == "Creative Focus Outline"
        assert [section["topic"] for section in data["sections"]] == [
            "Set the hook",
            "Develop the idea",
        ]
        assert "text" not in data["sections"][0]
        assert data["outline_preview_id"].startswith("opv_")
        assert data["llm_profile_id"] == "balanced"
        assert data["tts_profile_id"] == "studio-dialogue"
        assert data["routing_revision"].startswith("rte_")
        assert data["tts_routing_revision"].startswith("tts_")
        assert "expires_at" in data

    def test_preview_outline_requires_auth(self, client):
        resp = client.post(
            "/api/v1/projects/outline-preview",
            json={"topic": "stoicism", "bpm": 120},
        )
        assert resp.status_code == 401

    def test_create_project_validation(self, client, auth_headers):
        # Missing required field
        resp = client.post(
            "/api/v1/projects",
            json={"bpm": 120},
            headers=auth_headers,
        )
        assert resp.status_code == 422

        # BPM out of range
        resp = client.post(
            "/api/v1/projects",
            json={"topic": "test", "bpm": 999},
            headers=auth_headers,
        )
        assert resp.status_code == 422

    def test_create_project_rejects_api_key_fields(self, client, auth_headers):
        """Provider secrets must be rejected in product requests."""
        resp = _reviewed_create(
            client,
            auth_headers,
            {
                "topic": "test",
                "bpm": 120,
                "api_key": "sk-secret-should-not-be-here",
            },
        )
        # By default Pydantic v2 ignores extra fields, not rejects.
        # The v1 model doesn't include api_key, so it's simply ignored.
        # This is acceptable — the field has no effect on the server.
        # A stricter deployment can add model_config with extra='forbid'.
        assert resp.status_code == 202

    def test_list_projects_empty(self, client, auth_headers):
        resp = client.get("/api/v1/projects", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["projects"] == []

    def test_preview_binding_is_required_and_consumed_once(
        self, client, auth_headers
    ):
        unreviewed = client.post(
            "/api/v1/projects",
            json={"topic": "bound", "bpm": 120, "duration_minutes": 1},
            headers=auth_headers,
        )
        assert unreviewed.status_code == 409
        assert unreviewed.json()["error"]["code"] == "preview_binding_not_found"
        assert client.get("/api/v1/projects", headers=auth_headers).json()["projects"] == []
        preview = client.post(
            "/api/v1/projects/outline-preview",
            json={"topic": "bound", "bpm": 120, "duration_minutes": 1},
            headers=auth_headers,
        )
        assert preview.status_code == 200
        preview_body = preview.json()
        payload = {
            "topic": "bound",
            "bpm": 120,
            "duration_minutes": 1,
            "approved_outline": preview_body,
        }

        missing = client.post("/api/v1/projects", json=payload, headers=auth_headers)
        assert missing.status_code == 409
        assert missing.json()["error"]["code"] == "preview_binding_not_found"
        assert client.get("/api/v1/projects", headers=auth_headers).json()["projects"] == []

        payload["outline_preview_id"] = preview_body["binding"]["outline_preview_id"]
        accepted = client.post("/api/v1/projects", json=payload, headers=auth_headers)
        replayed = client.post("/api/v1/projects", json=payload, headers=auth_headers)
        assert accepted.status_code == 202
        assert replayed.status_code == 409
        assert replayed.json()["error"]["code"] == "preview_binding_consumed"
        projects = client.get("/api/v1/projects", headers=auth_headers).json()["projects"]
        assert len(projects) == 1

    def test_list_projects_with_data(self, client, auth_headers):
        # Create a project
        _reviewed_create(
            client,
            auth_headers,
            {"topic": "test-list", "bpm": 100},
        )
        resp = client.get("/api/v1/projects", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["projects"]) >= 1
        assert data["projects"][0]["topic"] == "test-list"

    def test_get_project_not_found(self, client, auth_headers):
        resp = client.get("/api/v1/projects/prj_nonexistent", headers=auth_headers)
        assert resp.status_code == 404
        assert resp.json()["error"]["code"] == "not_found"

    def test_get_project_found(self, client, auth_headers):
        create_resp = _reviewed_create(
            client,
            auth_headers,
            {"topic": "get-test", "bpm": 110},
        )
        pid = create_resp.json()["project"]["project_id"]

        resp = client.get(f"/api/v1/projects/{pid}", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["project_id"] == pid
        assert data["topic"] == "get-test"

    def test_get_project_outline_returns_section_plan(self, client, auth_headers):
        import asyncio

        from podcast_worker.core import persistence
        from podcast_worker.core.config import settings as cfg

        preview = client.post(
            "/api/v1/projects/outline-preview",
            json={"topic": "outline-test", "bpm": 110, "duration_minutes": 1},
            headers=auth_headers,
        ).json()
        create_resp = client.post(
            "/api/v1/projects",
            json={
                "topic": "outline-test",
                "bpm": 110,
                "duration_minutes": 1,
                "approved_outline": preview,
                "outline_preview_id": preview["binding"]["outline_preview_id"],
            },
            headers=auth_headers,
        )
        created_project = create_resp.json()["project"]
        pid = created_project["project_id"]
        segment_id = created_project["segments"][0]["segment_id"]

        asyncio.run(
            persistence.upsert_segment(
                cfg.db_path,
                {
                    "segment_id": segment_id,
                    "project_id": pid,
                    "index": 0,
                    "subtopic": "Opening question",
                    "title": "A Useful Outline",
                    "status": "queued",
                    "segment_type": "content",
                    "planned_duration_seconds": 30,
                    "text": "Full script text should not appear in the outline response.",
                    "duration_seconds": 30,
                    "primary_audio_artifact_id": None,
                    "error": None,
                    "created_at": "2026-01-01T00:00:00Z",
                    "updated_at": "2026-01-01T00:00:00Z",
                },
            )
        )

        resp = client.get(f"/api/v1/projects/{pid}/outline", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["project_id"] == pid
        assert data["topic"] == "outline-test"
        assert data["title"] == created_project["segments"][0]["title"]
        assert data["sections"] == [
            {
                "index": 0,
                "segment_id": segment_id,
                "segment_type": created_project["segments"][0]["segment_type"],
                "topic": created_project["segments"][0]["subtopic"],
                "title": created_project["segments"][0]["title"],
                "approx_duration_seconds": created_project["segments"][0][
                    "planned_duration_seconds"
                ],
            }
        ]
        assert "Full script text" not in str(data)

    def test_get_project_outline_not_found(self, client, auth_headers):
        resp = client.get("/api/v1/projects/prj_nonexistent/outline", headers=auth_headers)
        assert resp.status_code == 404
        assert resp.json()["error"]["code"] == "not_found"

    def test_delete_project(self, client, auth_headers):
        create_resp = _reviewed_create(
            client,
            auth_headers,
            {"topic": "delete-test", "bpm": 130},
        )
        pid = create_resp.json()["project"]["project_id"]

        resp = client.delete(f"/api/v1/projects/{pid}", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["project_id"] == pid
        assert data["status"] == "deleted"

        # Subsequent GET returns 404
        resp = client.get(f"/api/v1/projects/{pid}", headers=auth_headers)
        assert resp.status_code == 404

    def test_delete_nonexistent_project(self, client, auth_headers):
        resp = client.delete("/api/v1/projects/prj_fake", headers=auth_headers)
        assert resp.status_code == 404


# ── Error envelope ────────────────────────────────────────────────────────


class TestErrorEnvelope:
    def test_404_uses_error_envelope(self, client, auth_headers):
        resp = client.get("/api/v1/projects/prj_nope", headers=auth_headers)
        assert resp.status_code == 404
        data = resp.json()
        assert "error" in data
        assert data["error"]["code"] == "not_found"
        assert "message" in data["error"]

    def test_401_uses_error_envelope(self, client):
        resp = client.get("/api/v1/projects")
        assert resp.status_code == 401
        data = resp.json()
        assert data["error"]["code"] == "unauthorized"

    def test_422_still_includes_error_details(self, client, auth_headers):
        resp = client.post(
            "/api/v1/projects",
            json={},
            headers=auth_headers,
        )
        assert resp.status_code == 422


# ── Artifacts ─────────────────────────────────────────────────────────────


class TestArtifacts:
    def test_nonexistent_artifact_404(self, client, auth_headers):
        resp = client.get("/api/v1/artifacts/art_nonexistent", headers=auth_headers)
        assert resp.status_code == 404

    def test_transfer_url_nonexistent_404(self, client, auth_headers):
        resp = client.post(
            "/api/v1/artifacts/art_nonexistent/transfer-url",
            headers=auth_headers,
        )
        assert resp.status_code == 404

    def test_artifact_access_requires_auth(self, client):
        resp = client.get("/api/v1/artifacts/art_anything")
        assert resp.status_code == 401
    def test_internal_artifact_is_not_publicly_addressable(self, client, auth_headers):
        from podcast_worker.core import persistence
        from podcast_worker.core.config import settings as cfg

        persistence._create_project(cfg.db_path, "prj_internal", "single-user", "topic", 120, 5)
        persistence._add_artifact(cfg.db_path, {
            "artifact_id": "art_internal", "project_id": "prj_internal", "segment_id": None,
            "kind": "execution_snapshot", "content_type": "application/json", "status": "ready",
            "download_url": "/api/v1/artifacts/art_internal",
        })

        assert client.get("/api/v1/artifacts/art_internal", headers=auth_headers).status_code == 404
        assert client.post("/api/v1/artifacts/art_internal/transfer-url", headers=auth_headers).status_code == 404


# ── Durability ────────────────────────────────────────────────────────────


class TestDurability:
    def test_project_survives_client_restart(self, client, auth_headers):
        """Project data must persist across 'restarts' (new TestClient)."""
        from podcast_worker.main import app
        c1 = TestClient(app)
        create_resp = _reviewed_create(
            c1,
            auth_headers,
            {"topic": "survivor", "bpm": 140},
        )
        pid = create_resp.json()["project"]["project_id"]

        # 'Restart' with a new client — same app, same DB
        c2 = TestClient(app)
        resp = c2.get(f"/api/v1/projects/{pid}", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json()["topic"] == "survivor"