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
# Helpers
# ---------------------------------------------------------------------------


def _resolve_llm_key(req, provider_field: str = "provider") -> Optional[str]:
    """Resolve the LLM API key, handling OpenRouter special case."""
    provider = getattr(req, provider_field, None)
    api_key = getattr(req, "api_key", None)
    openrouter_key = getattr(req, "openrouter_key", None)

    if provider == "openrouter":
        return openrouter_key or os.environ.get("OPENROUTER_API_KEY", "")
    return api_key


# ---------------------------------------------------------------------------
# Background workers
# ---------------------------------------------------------------------------


def _run_full_generation(job_id: str, req: GenerateRequest):
    """Run the full generation pipeline in a background thread."""
    state.jobs[job_id]["status"] = "running"
    try:
        state.jobs[job_id]["progress"] = "Generating script..."

        # 1. Script
        llm_key = _resolve_llm_key(req, "llm_provider")
        script = generate_script(
            topic=req.topic,
            bpm=req.bpm,
            duration_minutes=req.duration_minutes,
            provider=req.llm_provider,
            api_key=llm_key,
            model=req.llm_model,
        )
        state.jobs[job_id]["progress"] = "Script generated. Synthesizing speech..."

        # 2. TTS
        flat_text = flatten_script(script)
        tts_key = _resolve_llm_key(req, "tts_provider")

        speech_path = str(state.output_dir / f"{job_id}_speech.mp3")
        speech_mp3 = synthesize(
            flat_text,
            provider=req.tts_provider,
            output_path=speech_path,
            api_key=tts_key,
            voice=req.voice,
            model=req.tts_model,
        )
        speech_wav = convert_to_wav(speech_mp3)
        state.jobs[job_id]["progress"] = "Speech synthesized. Generating beat..."

        # 3. Beat
        estimated_speech_seconds = req.duration_minutes * 60 * 1.2
        beat_path = str(state.output_dir / f"{job_id}_beat.wav")
        save_beat_to_wav(req.bpm, estimated_speech_seconds, beat_path)
        state.jobs[job_id]["progress"] = "Beat generated. Mixing audio..."

        # 4. Mix
        podcast_wav = str(state.output_dir / f"{job_id}_podcast.wav")
        build_podcast_audio(
            speech_wav_path=speech_wav,
            beat_wav_path=beat_path,
            output_path=podcast_wav,
            bpm=req.bpm,
        )

        # 5. MP3 conversion
        podcast_mp3 = str(state.output_dir / f"{job_id}_podcast.mp3")
        try:
            final_path = convert_to_mp3(podcast_wav, podcast_mp3)
        except Exception:
            final_path = podcast_wav

        # Get actual duration
        actual_duration_seconds = req.duration_minutes * 60
        try:
            import subprocess
            result = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", final_path],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                actual_duration_seconds = int(float(result.stdout.strip()))
        except Exception:
            pass

        state.jobs[job_id].update({
            "status": "completed",
            "progress": "Complete.",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "result": {
                "title": script.get("title", f"Learn {req.topic}"),
                "topic": req.topic,
                "bpm": req.bpm,
                "duration_minutes": req.duration_minutes,
                "duration_seconds": actual_duration_seconds,
                "script": script,
                "audio_path": podcast_wav,
                "mp3_path": final_path if final_path != podcast_wav else None,
            },
        })
    except Exception as e:
        state.jobs[job_id].update({
            "status": "failed",
            "progress": "Failed.",
            "error": str(e),
            "completed_at": datetime.now(timezone.utc).isoformat(),
        })


# ---------------------------------------------------------------------------
# Health & Config
# ---------------------------------------------------------------------------


@app.get("/api/services/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        version="2.0.0",
        uptime=time.time() - state.start_time,
    )


@app.get("/api/services/config")
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


# ---------------------------------------------------------------------------
# Async generation endpoints
# ---------------------------------------------------------------------------


@app.post("/api/services/generate", status_code=202)
async def generate_podcast(req: GenerateRequest):
    """Start full podcast generation asynchronously. Returns job_id for polling."""
    job_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()

    state.jobs[job_id] = {
        "status": "pending",
        "progress": "Queued...",
        "error": None,
        "result": None,
        "created_at": created_at,
        "completed_at": None,
    }

    thread = threading.Thread(
        target=_run_full_generation,
        args=(job_id, req),
        daemon=True,
    )
    thread.start()

    return {
        "job_id": job_id,
        "status": "pending",
        "status_url": f"/api/services/jobs/{job_id}",
    }


@app.get("/api/services/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Poll job status. Returns status, progress, error, and result if completed."""
    job = state.jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")

    resp = {
        "job_id": job_id,
        "status": job["status"],
        "progress": job.get("progress"),
        "error": job.get("error"),
        "created_at": job.get("created_at"),
        "completed_at": job.get("completed_at"),
    }

    if job["status"] == "completed" and job.get("result"):
        resp["result"] = job["result"]

    return resp


@app.get("/api/services/jobs/{job_id}/result")
async def download_result(job_id: str):
    """Download the final podcast audio file."""
    job = state.jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not yet completed.")
    if not job.get("result"):
        raise HTTPException(status_code=500, detail="No result data.")

    result = job["result"]
    audio_path = result.get("mp3_path") or result.get("audio_path")
    if not audio_path or not os.path.isfile(audio_path):
        raise HTTPException(status_code=404, detail="Audio file not found.")

    return FileResponse(
        audio_path,
        media_type="audio/mpeg" if audio_path.endswith(".mp3") else "audio/wav",
        filename=os.path.basename(audio_path),
    )


@app.get("/api/services/jobs/{job_id}/script")
async def download_script(job_id: str):
    """Download the script JSON for a completed job."""
    job = state.jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not yet completed.")
    if not job.get("result") or not job["result"].get("script"):
        raise HTTPException(status_code=404, detail="No script data.")

    return JSONResponse(content=job["result"]["script"])


# ---------------------------------------------------------------------------
# Synchronous endpoints (for testing / direct calls)
# ---------------------------------------------------------------------------


@app.post("/api/services/generate-script")
async def generate_script_endpoint(req: ScriptRequestModel):
    """Generate a podcast script only. Returns the script JSON synchronously."""
    try:
        llm_key = _resolve_llm_key(req, "provider")
        script = generate_script(
            topic=req.topic,
            bpm=req.bpm,
            duration_minutes=req.duration_minutes,
            provider=req.provider,
            api_key=llm_key,
            model=req.model,
        )
        return {"status": "ok", "script": script}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/services/generate-audio")
async def generate_audio_endpoint(req: AudioRequest):
    """Generate audio from text: TTS + beat + mix. Returns paths synchronously."""
    try:
        job_id = str(uuid.uuid4())
        tts_key = _resolve_llm_key(req, "tts_provider")

        # TTS
        speech_path = str(state.output_dir / f"{job_id}_speech.mp3")
        speech_mp3 = synthesize(
            req.speech_text,
            provider=req.tts_provider,
            output_path=speech_path,
            api_key=tts_key,
            voice=req.voice,
            model=req.tts_model,
        )
        speech_wav = convert_to_wav(speech_mp3)

        # Beat
        estimated_speech_seconds = req.duration_minutes * 60 * 1.2
        beat_path = str(state.output_dir / f"{job_id}_beat.wav")
        save_beat_to_wav(req.bpm, estimated_speech_seconds, beat_path)

        # Mix
        podcast_wav = str(state.output_dir / f"{job_id}_podcast.wav")
        build_podcast_audio(
            speech_wav_path=speech_wav,
            beat_wav_path=beat_path,
            output_path=podcast_wav,
            bpm=req.bpm,
        )

        # MP3
        podcast_mp3 = str(state.output_dir / f"{job_id}_podcast.mp3")
        try:
            final_path = convert_to_mp3(podcast_wav, podcast_mp3)
        except Exception:
            final_path = podcast_wav

        return {
            "status": "ok",
            "job_id": job_id,
            "audio_path": final_path,
            "speech_wav": speech_wav,
            "beat_wav": beat_path,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/services/generate-beat")
async def generate_beat_endpoint(req: BeatRequest):
    """Generate a beat only. Returns the WAV file synchronously."""
    try:
        job_id = str(uuid.uuid4())
        beat_path = str(state.output_dir / f"{job_id}_beat.wav")
        save_beat_to_wav(req.bpm, req.duration_seconds, beat_path)
        return FileResponse(
            beat_path,
            media_type="audio/wav",
            filename=f"beat_{req.bpm}bpm.wav",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/services/generate-speech")
async def generate_speech_endpoint(req: SpeechRequest):
    """Generate speech audio from text (TTS only — no beat, no mix).

    Returns a WAV file suitable for later overlay with /api/services/overlay-audio.
    """
    try:
        job_id = str(uuid.uuid4())
        tts_key = _resolve_llm_key(req, "tts_provider")

        speech_path = str(state.output_dir / f"{job_id}_speech.mp3")
        speech_mp3 = synthesize(
            req.speech_text,
            provider=req.tts_provider,
            output_path=speech_path,
            api_key=tts_key,
            voice=req.voice,
            model=req.tts_model,
        )
        speech_wav = convert_to_wav(speech_mp3)

        return FileResponse(
            speech_wav,
            media_type="audio/wav",
            filename=f"speech_{job_id}.wav",
            headers={"X-Job-Id": job_id},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/services/overlay-audio")
async def overlay_audio_endpoint(
    speech_wav: UploadFile = File(...),
    beat_wav: UploadFile = File(...),
    bpm: int = Form(...),
    duration_minutes: int = Form(default=5),
    intro_seconds: float = Form(default=4.0),
    outro_seconds: float = Form(default=6.0),
):
    """Overlay pre-generated speech and beat WAVs into a finished podcast.

    Accepts two WAV files as multipart upload:
      - speech_wav: the spoken-word track (from /api/services/generate-speech or other)
      - beat_wav: the BPM beat track (from /api/services/generate-beat or other)

    Returns the mixed podcast with intro/outro, ducking, and MP3 conversion.
    """
    try:
        job_id = str(uuid.uuid4())

        speech_path = str(state.output_dir / f"{job_id}_upload_speech.wav")
        beat_path = str(state.output_dir / f"{job_id}_upload_beat.wav")

        speech_content = await speech_wav.read()
        beat_content = await beat_wav.read()

        with open(speech_path, "wb") as f:
            f.write(speech_content)
        with open(beat_path, "wb") as f:
            f.write(beat_content)

        podcast_wav = str(state.output_dir / f"{job_id}_podcast.wav")
        build_podcast_audio(
            speech_wav_path=speech_path,
            beat_wav_path=beat_path,
            output_path=podcast_wav,
            bpm=bpm,
            intro_seconds=intro_seconds,
            outro_seconds=outro_seconds,
        )

        podcast_mp3 = str(state.output_dir / f"{job_id}_podcast.mp3")
        try:
            final_path = convert_to_mp3(podcast_wav, podcast_mp3)
        except Exception:
            final_path = podcast_wav

        # Cleanup uploaded temp files
        try:
            os.remove(speech_path)
            os.remove(beat_path)
        except OSError:
            pass

        return FileResponse(
            final_path,
            media_type="audio/mpeg" if final_path.endswith(".mp3") else "audio/wav",
            filename=f"podcast_{bpm}bpm_{job_id[:8]}.mp3",
            headers={
                "X-Job-Id": job_id,
                "X-Bpm": str(bpm),
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Script endpoints
# ---------------------------------------------------------------------------


@app.post("/api/services/scripts", status_code=201)
async def create_script(req: ScriptStoreRequest):
    """Generate a podcast script, store it, and return a script_id."""
    try:
        llm_key = _resolve_llm_key(req, "provider")
        script = generate_script(
            topic=req.topic,
            bpm=req.bpm,
            duration_minutes=req.duration_minutes,
            provider=req.provider,
            api_key=llm_key,
            model=req.model,
        )

        script_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        state.scripts[script_id] = {
            "script_id": script_id,
            "topic": req.topic,
            "bpm": req.bpm,
            "duration_minutes": req.duration_minutes,
            "script": script,
            "created_at": now,
            "follow_up_questions": None,
            "summary": None,
        }

        return {
            "status": "ok",
            "script_id": script_id,
            "script": script,
            "created_at": now,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/services/scripts")
async def list_scripts():
    """List all stored scripts with metadata (no full script content)."""
    return {
        "scripts": [
            {
                "script_id": s["script_id"],
                "topic": s["topic"],
                "bpm": s["bpm"],
                "duration_minutes": s["duration_minutes"],
                "title": s["script"].get("title", "Untitled"),
                "created_at": s["created_at"],
                "has_follow_up": s["follow_up_questions"] is not None,
                "has_summary": s["summary"] is not None,
            }
            for s in state.scripts.values()
        ],
        "count": len(state.scripts),
    }


@app.get("/api/services/scripts/{script_id}")
async def get_script(script_id: str):
    """Retrieve a stored script by its script_id. Returns full script content."""
    entry = state.scripts.get(script_id)
    if not entry:
        raise HTTPException(status_code=404, detail=f"Script {script_id} not found.")
    return entry


@app.post("/api/services/scripts/{script_id}/follow-up")
async def generate_follow_up(script_id: str, req: FollowUpRequest):
    """Generate follow-up questions for a stored script."""
    entry = state.scripts.get(script_id)
    if not entry:
        raise HTTPException(status_code=404, detail=f"Script {script_id} not found.")

    try:
        llm_key = _resolve_llm_key(req, "provider")
        questions = generate_follow_up_questions(
            topic=req.topic,
            script=entry["script"],
            provider=req.provider,
            api_key=llm_key,
            model=req.model,
        )
        entry["follow_up_questions"] = questions

        return {
            "status": "ok",
            "script_id": script_id,
            "follow_up_questions": questions,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/services/scripts/{script_id}/summary")
async def generate_summary(script_id: str, req: SummaryRequest):
    """Generate a summary for a stored script."""
    entry = state.scripts.get(script_id)
    if not entry:
        raise HTTPException(status_code=404, detail=f"Script {script_id} not found.")

    try:
        llm_key = _resolve_llm_key(req, "provider")
        summary = generate_script_summary(
            script=entry["script"],
            provider=req.provider,
            api_key=llm_key,
            model=req.model,
        )
        entry["summary"] = summary

        return {
            "status": "ok",
            "script_id": script_id,
            "summary": summary,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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


# Schedule eviction before each request via middleware
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
