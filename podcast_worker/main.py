"""FastAPI application for the Podcast Worker Service.

Production (v1) endpoints:
  GET    /api/v1/health                    — Health check
  GET    /api/v1/config                    — Safe client profiles (no secrets)
  POST   /api/v1/projects                  — Create project, start generation
  GET    /api/v1/projects                  — List projects
  GET    /api/v1/projects/{id}             — Full PodcastProject manifest
  DELETE /api/v1/projects/{id}             — Delete project
  GET    /api/v1/artifacts/{id}            — Download artifact
  POST   /api/v1/artifacts/{id}/transfer-url — Refresh signed transfer URL

Legacy / dev-only endpoints under /api/services/* are quarantined in a
legacy router and are NOT part of the v1 product contract.
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

from fastapi import Depends, FastAPI, HTTPException, Request, UploadFile, File, Form
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
from podcast_worker.core.models_v1 import ErrorEnvelope, ErrorResponse

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
_cors_origins = (
    ["*"] if cfg.CORS_ORIGINS.strip() == "*"
    else [origin.strip() for origin in cfg.CORS_ORIGINS.split(",") if origin.strip()]
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=_cors_origins != ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Register v1 product routers
# ---------------------------------------------------------------------------

from podcast_worker.routers import v1_health, v1_projects, v1_artifacts

app.include_router(v1_health.router)
app.include_router(v1_projects.router)
app.include_router(v1_artifacts.router)

# ---------------------------------------------------------------------------
# Legacy / dev-only routers (NOT part of v1 product contract)
# ---------------------------------------------------------------------------

from podcast_worker.routers import health, jobs, audio, scripts
from podcast_worker.core.auth import require_auth

app.include_router(health.router)
app.include_router(jobs.router, dependencies=[Depends(require_auth)])
app.include_router(audio.router, dependencies=[Depends(require_auth)])
app.include_router(scripts.router, dependencies=[Depends(require_auth)])

# ---------------------------------------------------------------------------
# v1 error handlers
# ---------------------------------------------------------------------------


@app.exception_handler(PodcastWorkerError)
async def podcast_worker_error_handler(request: Request, exc: PodcastWorkerError):
    return JSONResponse(
        status_code=502,
        content=ErrorEnvelope(
            error=ErrorResponse(code="provider_error", message=str(exc)),
        ).model_dump(),
    )


@app.exception_handler(HTTPException)
async def v1_http_exception_handler(request: Request, exc: HTTPException):
    """Preserve existing HTTPException behavior for non-v1 paths,
    wrap v1 paths in the standard error envelope."""
    path = request.url.path
    if path.startswith("/api/v1/"):
        # Map status code to error code
        code_map = {
            400: "bad_request",
            401: "unauthorized",
            403: "forbidden",
            404: "not_found",
            409: "conflict",
            422: "validation_error",
            500: "internal_error",
            502: "provider_error",
        }
        code = code_map.get(exc.status_code, "unknown_error")
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorEnvelope(
                error=ErrorResponse(code=code, message=str(exc.detail)),
            ).model_dump(),
        )
    # Fall through to default handler for legacy paths
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )


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
    print(f"   V1 API: /api/v1/*")
    print(f"   Legacy (dev-only): /api/services/*")
    print(f"   Output directory: {state.output_dir}")
    print(f"   Auth: {'enabled' if cfg.settings.auth_token else 'disabled (dev mode)'}")
    uvicorn.run("podcast_worker.main:app", host="0.0.0.0", port=8100, log_level="info")