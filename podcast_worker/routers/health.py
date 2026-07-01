"""Health and config endpoints for the Podcast Worker Service."""

import time

from fastapi import APIRouter

from podcast_worker.core import config as cfg
from podcast_worker.core.models import HealthResponse
from podcast_worker.main import state

router = APIRouter(tags=["health"])


@router.get("/api/services/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        version="2.0.0",
        uptime=time.time() - state.start_time,
    )


@router.get("/api/services/config")
async def get_config():
    """Return available providers, voices, models, and constraints."""
    return {
        "llm_providers": ["openai", "ollama", "openrouter"],
        "tts_providers": ["edge", "openai", "openrouter"],
        "voices": {
            "edge": ["en-US-GuyNeural", "en-US-JennyNeural", "en-GB-RyanNeural"],
            "openai": ["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
            "openrouter": ["en_paul_neutral", "en_paul_happy", "en_amy_happy"],
        },
        "models": {
            "llm": {
                "openai": ["gpt-4o", "gpt-4o-mini"],
                "openrouter": ["openai/gpt-4o", "anthropic/claude-sonnet-4-20250514"],
            },
            "tts": {
                "openai": ["tts-1", "tts-1-hd"],
                "openrouter": ["mistralai/voxtral-mini-tts-2603"],
            },
        },
        "bpm_range": {"min": cfg.MIN_BPM, "max": cfg.MAX_BPM},
        "max_duration_minutes": 30,
    }