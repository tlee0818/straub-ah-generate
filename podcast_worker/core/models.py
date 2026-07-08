"""Pydantic models for the Podcast Worker API."""

from typing import Optional

from pydantic import BaseModel, Field

from podcast_worker.core.config import settings


# ── Request Models ──────────────────────────────────────────────────────


class ScriptRequest(BaseModel):
    topic: str = Field(..., min_length=1, max_length=200)
    bpm: int = Field(..., ge=settings.min_bpm, le=settings.max_bpm)
    duration_minutes: int = Field(default=5, ge=1, le=30)
    provider: Optional[str] = None
    api_key: Optional[str] = None
    model: Optional[str] = None
    openrouter_key: Optional[str] = None


class AudioRequest(BaseModel):
    speech_text: str = Field(..., min_length=1, max_length=settings.max_text_chars)
    bpm: int = Field(..., ge=settings.min_bpm, le=settings.max_bpm)
    duration_minutes: int = Field(default=5, ge=1, le=30)
    tts_provider: Optional[str] = None
    voice: Optional[str] = None
    api_key: Optional[str] = None
    openrouter_key: Optional[str] = None
    tts_model: Optional[str] = None


class GenerateRequest(BaseModel):
    topic: str = Field(..., min_length=1, max_length=200)
    bpm: int = Field(..., ge=settings.min_bpm, le=settings.max_bpm)
    duration_minutes: int = Field(default=5, ge=1, le=30)
    llm_provider: Optional[str] = None
    tts_provider: Optional[str] = None
    voice: Optional[str] = None
    llm_model: Optional[str] = None
    tts_model: Optional[str] = None
    api_key: Optional[str] = None
    openrouter_key: Optional[str] = None


class BeatRequest(BaseModel):
    bpm: int = Field(..., ge=settings.min_bpm, le=settings.max_bpm)
    duration_seconds: float = Field(..., gt=0, le=1800)


class SpeechRequest(BaseModel):
    """Request for TTS-only generation (no beat, no mix)."""
    speech_text: str = Field(..., min_length=1, max_length=settings.max_text_chars)
    tts_provider: Optional[str] = None
    voice: Optional[str] = None
    api_key: Optional[str] = None
    openrouter_key: Optional[str] = None
    tts_model: Optional[str] = None


class OverlayRequest(BaseModel):
    """Request to overlay pre-generated speech and beat WAVs."""
    bpm: int = Field(..., ge=settings.min_bpm, le=settings.max_bpm)
    duration_minutes: int = Field(default=5, ge=1, le=30)
    intro_seconds: float = Field(default=4.0, ge=0, le=30)
    outro_seconds: float = Field(default=6.0, ge=0, le=30)


class ScriptStoreRequest(BaseModel):
    """Request to generate and store a script."""
    topic: str = Field(..., min_length=1, max_length=200)
    bpm: int = Field(..., ge=settings.min_bpm, le=settings.max_bpm)
    duration_minutes: int = Field(default=5, ge=1, le=30)
    provider: Optional[str] = None
    api_key: Optional[str] = None
    model: Optional[str] = None
    openrouter_key: Optional[str] = None


class FollowUpRequest(BaseModel):
    """Request to generate follow-up questions for an existing script."""
    topic: str = Field(..., min_length=1, max_length=200)
    provider: Optional[str] = None
    api_key: Optional[str] = None
    model: Optional[str] = None
    openrouter_key: Optional[str] = None


class SummaryRequest(BaseModel):
    """Request to generate a summary for an existing script."""
    provider: Optional[str] = None
    api_key: Optional[str] = None
    model: Optional[str] = None
    openrouter_key: Optional[str] = None


# ── Response Models ─────────────────────────────────────────────────────


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