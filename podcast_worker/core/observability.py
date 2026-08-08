"""Low-cardinality metrics and redacted structured generation logging."""
from __future__ import annotations

import json
import logging
import threading
from collections import defaultdict
from typing import Mapping

_METRIC_NAMES = {
    "podcast_generation_transitions_total",
    "podcast_work_lease_events_total",
    "podcast_provider_attempts_total",
    "podcast_fence_rejections_total",
    "podcast_cancel_requests_total",
    "podcast_active_projects",
    "podcast_progress_staleness_seconds",
}
_ALLOWED_LABELS = {"stage", "outcome", "operation", "route"}
_SENSITIVE_PARTS = ("token", "authorization", "secret", "api_key", "prompt", "text", "exception", "url", "path")


def _safe_fields(fields: Mapping[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in fields.items()
        if not any(part in key.lower() for part in _SENSITIVE_PARTS)
    }


def log_event(event: str, **fields: object) -> None:
    """Emit one allowlisted JSON event without payloads, credentials, or provider errors."""
    payload = {"event": event, **_safe_fields(fields)}
    logging.getLogger("podcast_worker.observability").info(
        json.dumps(payload, sort_keys=True, separators=(",", ":"))
    )


class MetricsRegistry:
    """Small in-process Prometheus text collector with bounded label names."""

    def __init__(self) -> None:
        self._values: dict[tuple[str, tuple[tuple[str, str], ...]], float] = defaultdict(float)
        self._lock = threading.Lock()

    def _key(self, name: str, labels: Mapping[str, str]) -> tuple[str, tuple[tuple[str, str], ...]]:
        if name not in _METRIC_NAMES:
            raise ValueError("unknown metric")
        if not set(labels).issubset(_ALLOWED_LABELS):
            raise ValueError("high-cardinality or unknown metric label")
        return name, tuple(sorted((key, str(value)) for key, value in labels.items()))

    def increment(self, name: str, amount: float = 1, **labels: str) -> None:
        key = self._key(name, labels)
        with self._lock:
            self._values[key] += amount

    def set(self, name: str, value: float, **labels: str) -> None:
        key = self._key(name, labels)
        with self._lock:
            self._values[key] = value

    def render(self) -> str:
        with self._lock:
            values = sorted(self._values.items())
        lines = []
        for (name, labels), value in values:
            suffix = ""
            if labels:
                encoded = ",".join(
                    f'{key}="{label.replace(chr(92), chr(92) * 2).replace(chr(34), chr(92) + chr(34))}"'
                    for key, label in labels
                )
                suffix = "{" + encoded + "}"
            lines.append(f"{name}{suffix} {value:g}")
        return "\n".join(lines) + ("\n" if lines else "")

    def reset(self) -> None:
        with self._lock:
            self._values.clear()


metrics = MetricsRegistry()
