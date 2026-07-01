"""Async job endpoints for the Podcast Worker Service — full pipeline generation."""

import os
import threading
import time
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, JSONResponse

from podcast_worker.core.models import GenerateRequest
from podcast_worker.core.script_generator import generate_script, flatten_script
from podcast_worker.core.tts_engine import synthesize, convert_to_wav
from podcast_worker.core.beat_generator import save_beat_to_wav
from podcast_worker.core.audio_mixer import build_podcast_audio, convert_to_mp3
from podcast_worker.core.dependencies import resolve_llm_key
from podcast_worker.main import state

router = APIRouter(tags=["jobs"])


def _run_full_generation(job_id: str, req: GenerateRequest):
    """Run the full generation pipeline in a background thread."""
    state.jobs[job_id]["status"] = "running"
    try:
        state.jobs[job_id]["progress"] = "Generating script..."

        llm_key = resolve_llm_key(req, "llm_provider")
        script = generate_script(
            topic=req.topic,
            bpm=req.bpm,
            duration_minutes=req.duration_minutes,
            provider=req.llm_provider,
            api_key=llm_key,
            model=req.llm_model,
        )
        state.jobs[job_id]["progress"] = "Script generated. Synthesizing speech..."

        flat_text = flatten_script(script)
        tts_key = resolve_llm_key(req, "tts_provider")

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

        estimated_speech_seconds = req.duration_minutes * 60 * 1.2
        beat_path = str(state.output_dir / f"{job_id}_beat.wav")
        save_beat_to_wav(req.bpm, estimated_speech_seconds, beat_path)
        state.jobs[job_id]["progress"] = "Beat generated. Mixing audio..."

        podcast_wav = str(state.output_dir / f"{job_id}_podcast.wav")
        build_podcast_audio(
            speech_wav_path=speech_wav,
            beat_wav_path=beat_path,
            output_path=podcast_wav,
            bpm=req.bpm,
        )

        podcast_mp3 = str(state.output_dir / f"{job_id}_podcast.mp3")
        try:
            final_path = convert_to_mp3(podcast_wav, podcast_mp3)
        except Exception:
            final_path = podcast_wav

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


@router.post("/api/services/generate", status_code=202)
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


@router.get("/api/services/jobs/{job_id}")
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


@router.get("/api/services/jobs/{job_id}/result")
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


@router.get("/api/services/jobs/{job_id}/script")
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