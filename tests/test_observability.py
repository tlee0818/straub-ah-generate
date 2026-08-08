from __future__ import annotations

import json
import logging

import pytest

from podcast_worker.core.observability import MetricsRegistry, log_event


def test_metrics_reject_high_cardinality_labels():
    registry = MetricsRegistry()
    with pytest.raises(ValueError):
        registry.increment(
            "podcast_generation_transitions_total",
            project_id="prj-sensitive",
        )


def test_metrics_render_only_named_low_cardinality_labels():
    registry = MetricsRegistry()
    registry.increment(
        "podcast_generation_transitions_total",
        stage="research",
        outcome="completed",
    )
    registry.set("podcast_active_projects", 2)

    rendered = registry.render()
    assert 'podcast_generation_transitions_total{outcome="completed",stage="research"} 1' in rendered
    assert "podcast_active_projects 2" in rendered
    assert "project_id" not in rendered


def test_structured_log_redacts_sensitive_fields(caplog):
    with caplog.at_level(logging.INFO, logger="podcast_worker.observability"):
        log_event(
            "stage_transition",
            project_id="prj-safe-id",
            stage="research",
            outcome="completed",
            authorization="Bearer secret",
            provider_exception="raw provider failure",
            prompt_text="private prompt",
        )

    payload = json.loads(caplog.records[-1].message)
    assert payload == {
        "event": "stage_transition",
        "outcome": "completed",
        "project_id": "prj-safe-id",
        "stage": "research",
    }
