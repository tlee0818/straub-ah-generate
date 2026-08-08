"""Health and config endpoints for the v1 API.

GET /api/v1/health — service health (auth optional)
GET /api/v1/config — safe client profiles (auth optional, no secrets)
"""

from __future__ import annotations

import json
import time

from fastapi import APIRouter, Depends, HTTPException, Response

from podcast_worker.core.auth import optional_auth, require_auth
from podcast_worker.core.config import (
    RoutingConfigurationError,
    llm_profile_document,
    resolve_llm_profile,
    resolve_tts_profile,
    settings,
    tts_profile_document,
)
from podcast_worker.core.observability import metrics
from podcast_worker.core.models_v1 import (
    ConfigResponse,
    HealthResponse,
    LLMProfileSummary,
    TTSProfileSummary,
    VoiceProfile,
)

router = APIRouter(prefix="/api/v1", tags=["v1-health"])
internal_router = APIRouter(prefix="/internal", tags=["internal"])


@internal_router.get("/metrics", response_class=Response)
async def internal_metrics(owner_id: str = Depends(require_auth)) -> Response:
    """Return authenticated low-cardinality Prometheus metrics."""
    del owner_id
    return Response(content=metrics.render(), media_type="text/plain; version=0.0.4")

_start_time = time.time()


@router.get("/health", response_model=HealthResponse)
async def health(owner_id: str | None = Depends(optional_auth)):
    """Service health check. Auth is optional for deployment probes."""
    return HealthResponse(
        status="ok",
        version="2.0.0",
        uptime_seconds=time.time() - _start_time,
    )


@router.get("/config", response_model=ConfigResponse)
async def get_config(owner_id: str = Depends(require_auth)):
    """Return authenticated safe summaries derived from server-owned profile configuration."""
    try:
        default_llm = resolve_llm_profile()
        default_tts = resolve_tts_profile()
    except RoutingConfigurationError as exc:
        raise HTTPException(status_code=503, detail={"error": {"code": str(exc)}}) from exc
    llm_source = llm_profile_document()
    tts_source = tts_profile_document()
    llm_profiles = [
        {"id": item["id"], "label": item.get("label", item["id"]), "description": item.get("description")}
        for item in llm_source.get("profiles", [])
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    ]
    tts_profiles = [
        {"id": item["id"], "label": item.get("label", item["id"]), "description": item.get("description")}
        for item in tts_source.get("profiles", [])
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    ]
    voices = [
        {"id": item["id"], "label": item.get("label", item["id"]), "roles": item.get("roles", [])}
        for item in tts_source.get("voices", [])
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    ]
    return ConfigResponse(
        llm_profiles=[LLMProfileSummary(**item) for item in llm_profiles],
        tts_profiles=[TTSProfileSummary(**item) for item in tts_profiles],
        voices=[VoiceProfile(**item) for item in voices],
        default_llm_profile_id=default_llm.profile_id,
        default_tts_profile_id=default_tts.profile_id,
        bpm_range={"min": settings.min_bpm, "max": settings.max_bpm},
        duration_minutes_range={"min": 1, "max": 30},
    )
