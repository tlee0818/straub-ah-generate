"""Health and config endpoints for the v1 API.

GET /api/v1/health — service health (auth optional)
GET /api/v1/config — safe client profiles (auth optional, no secrets)
"""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends

from podcast_worker.core.auth import optional_auth
from podcast_worker.core.config import settings
from podcast_worker.core.models_v1 import (
    ConfigResponse,
    HealthResponse,
    VoiceProfile,
)

router = APIRouter(prefix="/api/v1", tags=["v1-health"])

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
async def get_config(owner_id: str | None = Depends(optional_auth)):
    """Return server-enabled generation options safe for client display."""
    # Build profiles from server config — never expose secrets
    llm_profiles = [
        VoiceProfile(id="default", label="Default script generator"),
    ]

    tts_profiles = [
        VoiceProfile(id="default", label="Default voice"),
    ]

    voices = [
        VoiceProfile(id=settings.edge_tts_voice, label=f"{settings.edge_tts_voice} — Edge TTS"),
        VoiceProfile(id=settings.openai_tts_voice, label=f"{settings.openai_tts_voice} — OpenAI TTS"),
        VoiceProfile(id=settings.openrouter_tts_voice, label=f"{settings.openrouter_tts_voice} — OpenRouter TTS"),
    ]

    return ConfigResponse(
        llm_profiles=llm_profiles,
        tts_profiles=tts_profiles,
        voices=voices,
        bpm_range={"min": settings.min_bpm, "max": settings.max_bpm},
        duration_minutes_range={"min": 1, "max": 30},
    )