"""
Tests for script API endpoints — script storage, retrieval, follow-up, summary.
"""

import uuid

import pytest
from fastapi.testclient import TestClient

# Import and monkey-patch the scripts store before importing the app
from podcast_worker import main as app_module

# Clear and populate the scripts store with a known script for testing
app_module.state.scripts.clear()
_known_id = str(uuid.uuid4())
app_module.state.scripts[_known_id] = {
    "script_id": _known_id,
    "topic": "AI",
    "bpm": 120,
    "duration_minutes": 5,
    "script": {
        "title": "Test Episode",
        "segments": [
            {"segment_type": "intro", "text": "Welcome.", "approx_duration_seconds": 10},
            {"segment_type": "content", "text": "AI is cool.", "approx_duration_seconds": 20},
            {"segment_type": "outro", "text": "Bye.", "approx_duration_seconds": 10},
        ],
    },
    "created_at": "2026-01-01T00:00:00Z",
    "follow_up_questions": None,
    "summary": None,
}

from podcast_worker.main import app

client = TestClient(app)


class TestListScripts:
    def test_list_scripts_returns_metadata(self):
        resp = client.get("/api/services/scripts")
        assert resp.status_code == 200
        data = resp.json()
        assert "scripts" in data
        assert data["count"] >= 1
        # Should not include full script content in listing
        for s in data["scripts"]:
            assert "title" in s
            assert "topic" in s
            assert "bpm" in s
            assert "script_id" in s
            assert "has_follow_up" in s
            assert "has_summary" in s
            assert "script" not in s  # full content should not be in list


class TestGetScript:
    def test_get_known_script(self):
        resp = client.get(f"/api/services/scripts/{_known_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["script_id"] == _known_id
        assert data["topic"] == "AI"
        assert "script" in data
        assert data["script"]["title"] == "Test Episode"

    def test_get_unknown_script(self):
        resp = client.get("/api/services/scripts/nonexistent-id")
        assert resp.status_code == 404


class TestCreateScript:
    def test_create_script_validates_required_fields(self):
        resp = client.post("/api/services/scripts", json={"topic": "", "bpm": 120})
        assert resp.status_code == 422  # Validation error

    def test_create_script_rejects_no_topic(self):
        resp = client.post("/api/services/scripts", json={"bpm": 120, "duration_minutes": 5})
        assert resp.status_code == 422

    def test_create_script_rejects_bad_bpm(self):
        resp = client.post("/api/services/scripts", json={"topic": "test", "bpm": 10})
        assert resp.status_code == 422  # Below min BPM


class TestFollowUp:
    def test_follow_up_unknown_script(self):
        resp = client.post(
            "/api/services/scripts/nonexistent/follow-up",
            json={"topic": "AI"},
        )
        assert resp.status_code == 404

    def test_follow_up_missing_topic(self):
        resp = client.post(
            f"/api/services/scripts/{_known_id}/follow-up",
            json={},
        )
        assert resp.status_code == 422  # topic is required


class TestSummary:
    def test_summary_unknown_script(self):
        resp = client.post(
            "/api/services/scripts/nonexistent/summary",
            json={},
        )
        assert resp.status_code == 404

    def test_summary_known_script(self):
        """With no provider set it uses the default (openai), which fails without key."""
        resp = client.post(
            f"/api/services/scripts/{_known_id}/summary",
            json={},
        )
        # Should fail with 500 because no LLM key is configured
        assert resp.status_code == 500


class TestScriptStoreState:
    def test_follow_up_and_summary_are_none_initially(self):
        resp = client.get(f"/api/services/scripts/{_known_id}")
        data = resp.json()
        assert data["follow_up_questions"] is None
        assert data["summary"] is None