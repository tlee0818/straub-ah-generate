"""Synchronous audio endpoint router — TTS, beat, speech, overlay."""

import os
import uuid

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse

from podcast_worker.core.models import AudioRequest, BeatRequest, SpeechRequest
from podcast_worker.core.tts_engine import synthesize, convert_to_wav
from podcast_worker.core.beat_generator import save_beat_to_wav
from podcast_worker.core.audio_mixer import build_podcast_audio, convert_to_mp3
from podcast_worker.core.dependencies import resolve_llm_key
from podcast_worker.main import state

router = APIRouter(tags=["audio"])


@router.post("/api/services/generate-audio")
async def generate_audio_endpoint(req: AudioRequest):
    """Generate audio from text: TTS + beat + mix. Returns paths synchronously."""
    try:
        job_id = str(uuid.uuid4())
        tts_key = resolve_llm_key(req, "tts_provider")

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

        estimated_speech_seconds = req.duration_minutes * 60 * 1.2
        beat_path = str(state.output_dir / f"{job_id}_beat.wav")
        save_beat_to_wav(req.bpm, estimated_speech_seconds, beat_path)

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

        return {
            "status": "ok",
            "job_id": job_id,
            "audio_path": final_path,
            "speech_wav": speech_wav,
            "beat_wav": beat_path,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/services/generate-beat")
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


@router.post("/api/services/generate-speech")
async def generate_speech_endpoint(req: SpeechRequest):
    """Generate speech audio from text (TTS only — no beat, no mix)."""
    try:
        job_id = str(uuid.uuid4())
        tts_key = resolve_llm_key(req, "tts_provider")

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


@router.post("/api/services/overlay-audio")
async def overlay_audio_endpoint(
    speech_wav: UploadFile = File(...),
    beat_wav: UploadFile = File(...),
    bpm: int = Form(...),
    duration_minutes: int = Form(default=5),
    intro_seconds: float = Form(default=4.0),
    outro_seconds: float = Form(default=6.0),
):
    """Overlay pre-generated speech and beat WAVs into a finished podcast.

    Accepts two WAV files as multipart upload with BPM/duration form fields.
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