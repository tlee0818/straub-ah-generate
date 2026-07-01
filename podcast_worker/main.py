"""FastAPI application for the Podcast Worker Service.

Endpoints:
  POST /api/services/generate          — Full pipeline (async, returns job_id)
  GET  /api/services/jobs/{job_id}     — Job status polling
  GET  /api/services/jobs/{job_id}/result  — Download result
  POST /api/services/generate-script   — Script generation only (sync)
  POST /api/services/generate-audio    — Audio generation only (sync)
  POST /api/services/generate-beat     — Beat generation only (sync)
  GET  /api/services/health            — Health check
  GET  /api/services/config            — Available providers, voices, models
  POST /api/services/scripts           — Generate and store a script
  GET  /api/services/scripts           — List stored scripts
  GET  /api/services/scripts/{id}      — Retrieve a stored script
  POST /api/services/scripts/{id}/follow-up — Generate follow-up questions
  POST /api/services/scripts/{id}/summary   — Generate script summary

The iOS app uses PodcastServiceClient to talk to this service.
"""

import json
import os
import threading
import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

# Import core modules
from podcast_worker.core import config as cfg
from podcast_worker.core.exceptions import (
    PodcastWorkerError,
    ConfigurationError,
    JobNotFoundError,
    JobNotCompleteError,
    ScriptNotFoundError,
)
from podcast_worker.core.models import (
    AudioRequest,
    BeatRequest,
    FollowUpRequest,
    GenerateRequest,
    HealthResponse,
    JobStatusResponse,
    OverlayRequest,
    ScriptRequest as ScriptRequestModel,
    ScriptStoreRequest,
    SpeechRequest,
    SummaryRequest,
)
from podcast_worker.core.script_generator import (
    generate_script,
    flatten_script,
    generate_follow_up_questions,
    generate_script_summary,
)
from podcast_worker.core.tts_engine import synthesize, convert_to_wav
from podcast_worker.core.beat_generator import save_beat_to_wav, generate_beat
from podcast_worker.core.audio_mixer import build_podcast_audio, convert_to_mp3

# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------

class AppState:
    """Shared mutable state for the FastAPI application."""

    def __init__(self) -> None:
        self.start_time: float = 0.0
        # In-memory job store; for production use Redis or a DB
        self.jobs: dict[str, dict] = {}
        # In-memory script store
        self.scripts: dict[str, dict] = {}
        # Default output directory
        project_root = Path(__file__).resolve().parent.parent
        self.output_dir = project_root / "output"


state = AppState()


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan — startup/shutdown logic."""
    state.start_time = time.time()
    state.output_dir.mkdir(parents=True, exist_ok=True)
    yield
    # Cleanup on shutdown (if needed)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Straub AH — Podcast Worker Service",
    version="2.0.0",
    description="Heavy-lift service for script generation, TTS, beat generation, and audio mixing.",
    lifespan=lifespan,
)

# CORS — configurable via PODCAST_CORS_ORIGINS env var
_cors_origins = cfg.CORS_ORIGINS.split(",") if cfg.CORS_ORIGINS != "*" else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Register routers
# ---------------------------------------------------------------------------

from podcast_worker.routers import health, jobs, audio, scripts

app.include_router(health.router)
app.include_router(jobs.router)
app.include_router(audio.router)
app.include_router(scripts.router)

# ---------------------------------------------------------------------------
# Job TTL eviction (lightweight — runs on each request)
# ---------------------------------------------------------------------------


def _evict_stale_jobs():
    """Remove jobs older than TTL threshold."""
    ttl = timedelta(hours=cfg.JOB_TTL_HOURS)
    now = datetime.now(timezone.utc)
    stale_ids = [
        jid for jid, job in state.jobs.items()
        if job.get("completed_at") and (
            now - datetime.fromisoformat(job["completed_at"])
        ) > ttl
    ]
    for jid in stale_ids:
        state.jobs.pop(jid, None)


@app.middleware("http")
async def _evict_middleware(request, call_next):
    _evict_stale_jobs()
    response = await call_next(request)
    return response


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    print(f"📡 Podcast Worker Service starting on http://0.0.0.0:8100")
    print(f"   Output directory: {state.output_dir}")
    uvicorn.run("podcast_worker.main:app", host="0.0.0.0", port=8100, log_level="info")
