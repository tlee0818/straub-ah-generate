"""
FastAPI application for the Podcast Worker Service.

Endpoints:
  POST /api/services/generate          — Full pipeline (async, returns job_id)
  GET  /api/services/jobs/{job_id}     — Job status polling
  GET  /api/services/jobs/{job_id}/result  — Download result
  POST /api/services/generate-script   — Script generation only (sync)
  POST /api/services/generate-audio    — Audio generation only (sync)
  POST /api/services/generate-beat     — Beat generation only (sync)
  GET  /api/services/health            — Health check
  GET  /api/services/config            — Available providers, voices, models

The iOS app uses PodcastServiceClient to talk to this service.
"""

import json
import os
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

# Import core modules from inside the worker package
from .core import config as cfg
from .core.script_generator import generate_script, flatten_script
from .core.tts_engine import synthesize, convert_to_wav
from .core.beat_generator import save_beat_to_wav, generate_beat
from .core.audio_mixer import build_podcast_audio, convert_to_mp3

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Straub AH — Podcast Worker Service",
    version="2.0.0",
    description="Heavy-lift service for script generation, TTS, beat generation, and audio mixing.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Job store (in-memory; for production use Redis or a DB)
# ---------------------------------------------------------------------------
jobs: dict[str, dict] = {}

# Default output directory for generated files
_project_root = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = _project_root / "output"
DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class ScriptRequest(BaseModel):
    topic: str = Field(..., min_length=1)
    bpm: int = Field(..., ge=cfg.MIN_BPM, le=cfg.MAX_BPM)
    duration_minutes: int = Field(default=5, ge=1, le=30)
    provider: Optional[str] = None
    api_key: Optional[str] = None
    model: Optional[str] = None
    openrouter_key: Optional[str] = None


class AudioRequest(BaseModel):
    speech_text: str = Field(..., min_length=1)
    bpm: int = Field(..., ge=cfg.MIN_BPM, le=cfg.MAX_BPM)
    duration_minutes: int = Field(default=5, ge=1, le=30)
    tts_provider: Optional[str] = None
    voice: Optional[str] = None
    api_key: Optional[str] = None
    openrouter_key: Optional[str] = None
    tts_model: Optional[str] = None


class GenerateRequest(BaseModel):
    topic: str = Field(..., min_length=1)
    bpm: int = Field(..., ge=cfg.MIN_BPM, le=cfg.MAX_BPM)
    duration_minutes: int = Field(default=5, ge=1, le=30)
    llm_provider: Optional[str] = None
    tts_provider: Optional[str] = None
    voice: Optional[str] = None
    llm_model: Optional[str] = None
    tts_model: Optional[str] = None
    api_key: Optional[str] = None
    openrouter_key: Optional[str] = None


class BeatRequest(BaseModel):
    bpm: int = Field(..., ge=cfg.MIN_BPM, le=cfg.MAX_BPM)
    duration_seconds: float = Field(..., gt=0)


class SpeechRequest(BaseModel):
    """Request for TTS-only generation (no beat, no mix)."""
    speech_text: str = Field(..., min_length=1)
    tts_provider: Optional[str] = None
    voice: Optional[str] = None
    api_key: Optional[str] = None
    openrouter_key: Optional[str] = None
    tts_model: Optional[str] = None


class OverlayRequest(BaseModel):
    """Request to overlay pre-generated speech and beat WAVs."""
    bpm: int = Field(..., ge=cfg.MIN_BPM, le=cfg.MAX_BPM)
    duration_minutes: int = Field(default=5, ge=1, le=30)
    intro_seconds: float = Field(default=4.0, ge=0)
    outro_seconds: float = Field(default=6.0, ge=0)


class JobStatusResponse(BaseModel):
    job_id: str
    status: str  # "pending" | "running" | "completed" | "failed"
    progress: Optional[str] = None
    error: Optional[str] = None
    created_at: str
    completed_at: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    version: str
    uptime: float


# ---------------------------------------------------------------------------
# Background workers
# ---------------------------------------------------------------------------


def _run_full_generation(job_id: str, req: GenerateRequest):
    """Run the full generation pipeline in a background thread."""
    jobs[job_id]["status"] = "running"
    try:
        jobs[job_id]["progress"] = "Generating script..."

        # 1. Script
        llm_key = req.api_key
        if req.llm_provider == "openrouter":
            llm_key = req.openrouter_key or os.environ.get("OPENROUTER_API_KEY", "")
        script = generate_script(
            topic=req.topic,
            bpm=req.bpm,
            duration_minutes=req.duration_minutes,
            provider=req.llm_provider,
            api_key=llm_key,
            model=req.llm_model,
        )
        jobs[job_id]["progress"] = "Script generated. Synthesizing speech..."

        # 2. TTS
        flat_text = flatten_script(script)
        tts_key = req.api_key
        if req.tts_provider == "openrouter":
            tts_key = req.openrouter_key or os.environ.get("OPENROUTER_API_KEY", "")

        speech_path = str(DEFAULT_OUTPUT_DIR / f"{job_id}_speech.mp3")
        speech_mp3 = synthesize(
            flat_text,
            provider=req.tts_provider,
            output_path=speech_path,
            api_key=tts_key,
            voice=req.voice,
            model=req.tts_model,
        )
        speech_wav = convert_to_wav(speech_mp3)
        jobs[job_id]["progress"] = "Speech synthesized. Generating beat..."

        # 3. Beat
        estimated_speech_seconds = req.duration_minutes * 60 * 1.2
        beat_path = str(DEFAULT_OUTPUT_DIR / f"{job_id}_beat.wav")
        save_beat_to_wav(req.bpm, estimated_speech_seconds, beat_path)
        jobs[job_id]["progress"] = "Beat generated. Mixing audio..."

        # 4. Mix
        podcast_wav = str(DEFAULT_OUTPUT_DIR / f"{job_id}_podcast.wav")
        build_podcast_audio(
            speech_wav_path=speech_wav,
            beat_wav_path=beat_path,
            output_path=podcast_wav,
            bpm=req.bpm,
        )

        # 5. MP3 conversion
        podcast_mp3 = str(DEFAULT_OUTPUT_DIR / f"{job_id}_podcast.mp3")
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

        jobs[job_id].update({
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
        jobs[job_id].update({
            "status": "failed",
            "progress": "Failed.",
            "error": str(e),
            "completed_at": datetime.now(timezone.utc).isoformat(),
        })


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------


@app.get("/api/services/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        version="2.0.0",
        uptime=time.time() - _start_time,
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


@app.post("/api/services/generate", status_code=202)
async def generate_podcast(req: GenerateRequest):
    """Start full podcast generation asynchronously. Returns job_id for polling."""
    job_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()

    jobs[job_id] = {
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
    job = jobs.get(job_id)
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
    job = jobs.get(job_id)
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
    job = jobs.get(job_id)
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
async def generate_script_endpoint(req: ScriptRequest):
    """Generate a podcast script only. Returns the script JSON synchronously."""
    try:
        llm_key = req.api_key
        if req.provider == "openrouter":
            llm_key = req.openrouter_key or os.environ.get("OPENROUTER_API_KEY", "")

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
        tts_key = req.api_key
        if req.tts_provider == "openrouter":
            tts_key = req.openrouter_key or os.environ.get("OPENROUTER_API_KEY", "")

        # TTS
        speech_path = str(DEFAULT_OUTPUT_DIR / f"{job_id}_speech.mp3")
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
        beat_path = str(DEFAULT_OUTPUT_DIR / f"{job_id}_beat.wav")
        save_beat_to_wav(req.bpm, estimated_speech_seconds, beat_path)

        # Mix
        podcast_wav = str(DEFAULT_OUTPUT_DIR / f"{job_id}_podcast.wav")
        build_podcast_audio(
            speech_wav_path=speech_wav,
            beat_wav_path=beat_path,
            output_path=podcast_wav,
            bpm=req.bpm,
        )

        # MP3
        podcast_mp3 = str(DEFAULT_OUTPUT_DIR / f"{job_id}_podcast.mp3")
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
        beat_path = str(DEFAULT_OUTPUT_DIR / f"{job_id}_beat.wav")
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
        tts_key = req.api_key
        if req.tts_provider == "openrouter":
            tts_key = req.openrouter_key or os.environ.get("OPENROUTER_API_KEY", "")

        # TTS synthesis
        speech_path = str(DEFAULT_OUTPUT_DIR / f"{job_id}_speech.mp3")
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

        # Save uploaded files to disk
        speech_path = str(DEFAULT_OUTPUT_DIR / f"{job_id}_upload_speech.wav")
        beat_path = str(DEFAULT_OUTPUT_DIR / f"{job_id}_upload_beat.wav")

        speech_content = await speech_wav.read()
        beat_content = await beat_wav.read()

        with open(speech_path, "wb") as f:
            f.write(speech_content)
        with open(beat_path, "wb") as f:
            f.write(beat_content)

        # Overlay using build_podcast_audio (intro + ducked mix + outro)
        podcast_wav = str(DEFAULT_OUTPUT_DIR / f"{job_id}_podcast.wav")
        build_podcast_audio(
            speech_wav_path=speech_path,
            beat_wav_path=beat_path,
            output_path=podcast_wav,
            bpm=bpm,
            intro_seconds=intro_seconds,
            outro_seconds=outro_seconds,
        )

        # MP3 conversion
        podcast_mp3 = str(DEFAULT_OUTPUT_DIR / f"{job_id}_podcast.mp3")
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
# Cleanup old jobs (simple TTL-based eviction)
# ---------------------------------------------------------------------------


@app.on_event("startup")
async def startup():
    global _start_time
    _start_time = time.time()
    # Ensure output directory exists
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    print(f"📡 Podcast Worker Service starting on http://0.0.0.0:8100")
    print(f"   Output directory: {DEFAULT_OUTPUT_DIR}")
    uvicorn.run(app, host="0.0.0.0", port=8100, log_level="info")
