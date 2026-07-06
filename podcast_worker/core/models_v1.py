"""Pydantic v2 schemas for the /api/v1/* product contract.

These models are the canonical request/response shapes shared with the iOS
client per API_SPEC.md.  Legacy /api/services/* models remain in core/models.py.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field

from podcast_worker.core.config import settings


# ═══════════════════════════════════════════════════════════════════════════
# Enums (as Literal string unions for OpenAPI compatibility)
# ═══════════════════════════════════════════════════════════════════════════

ProjectStatus = str   # queued | generating | partially_ready | ready | failed | deleted
SegmentStatus = str   # queued | scripting | validating | tts | mixing | ready | failed
ValidationStatus = str  # pending | validated | needs_review | failed
ArtifactKind = str     # segment_audio | final_mp3 | script_json
ArtifactStatus = str   # pending | ready | failed | deleted


# ═══════════════════════════════════════════════════════════════════════════
# Nested / shared models
# ═══════════════════════════════════════════════════════════════════════════


class ProvenanceMetadata(BaseModel):
    """Quality gate record for a segment.  Must be stored before TTS begins."""
    prompt_id: str = Field(..., description="Identifier for the generation prompt/template run")
    model: str = Field(..., description="Server-side model/profile actually used")
    source_refs: list[str] = Field(default_factory=list, description="Source references when available")
    claim_notes: list[str] = Field(default_factory=list, description="Notes for claims or rationale when sources are unavailable")
    validation_status: ValidationStatus = Field(default="pending", description="Validation state")
    validation_errors: list[str] = Field(default_factory=list, description="Empty when validation passes")
    validated_at: Optional[str] = Field(default=None, description="ISO-8601 timestamp when validation completed")


class SegmentError(BaseModel):
    code: str = Field(..., description="Machine-readable error code")
    message: str = Field(..., description="Human-readable error detail")
    retryable: bool = Field(default=False, description="Whether this error is retryable")


class Segment(BaseModel):
    """Sub-topic-level generation and playback unit."""
    segment_id: str = Field(..., description="Stable segment identifier")
    index: int = Field(..., description="Playback order within the project")
    subtopic: str = Field(..., description="Segment topic derived from project outline")
    title: Optional[str] = Field(default=None, description="User-facing segment title")
    status: SegmentStatus = Field(default="queued", description="Segment lifecycle status")
    duration_seconds: Optional[float] = Field(default=None, description="Known after audio artifact creation")
    text: Optional[str] = Field(default=None, description="Generated segment script text")
    provenance: Optional[ProvenanceMetadata] = Field(default=None, description="Required before tts/mixing/ready")
    artifact_ids: list[str] = Field(default_factory=list, description="Canonical artifacts produced for this segment")
    primary_audio_artifact_id: Optional[str] = Field(default=None, description="Main playable audio artifact when ready")
    error: Optional[SegmentError] = Field(default=None, description="Segment-level failure details")
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class Artifact(BaseModel):
    """Durable media or generation output metadata."""
    artifact_id: str = Field(..., description="Stable canonical artifact identifier")
    kind: ArtifactKind = Field(..., description="segment_audio | final_mp3 | script_json")
    segment_id: Optional[str] = Field(default=None, description="Present for segment artifacts")
    content_type: str = Field(..., description="MIME type, e.g. audio/mpeg")
    duration_seconds: Optional[float] = Field(default=None, description="Present for audio when known")
    size_bytes: Optional[int] = Field(default=None, description="Present when stored")
    checksum_sha256: Optional[str] = Field(default=None, description="Cache and integrity validator")
    status: ArtifactStatus = Field(default="pending", description="pending | ready | failed | deleted")
    download_url: str = Field(..., description="Authenticated API endpoint for this artifact")
    signed_transfer_url: Optional[str] = Field(default=None, description="Optional short-lived transfer URL")
    signed_transfer_expires_at: Optional[str] = Field(default=None, description="Required when signed URL is present")
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ProjectError(BaseModel):
    code: str
    message: str
    retryable: bool = False


# ═══════════════════════════════════════════════════════════════════════════
# Request models
# ═══════════════════════════════════════════════════════════════════════════


class ProjectCreateRequest(BaseModel):
    """POST /api/v1/projects — user intent only; provider secrets are server-side."""
    topic: str = Field(..., min_length=1, max_length=200, description="Podcast topic")
    bpm: int = Field(..., ge=settings.min_bpm, le=settings.max_bpm, description="Beats per minute")
    duration_minutes: int = Field(default=5, ge=1, le=30, description="Target duration in minutes")
    voice_id: Optional[str] = Field(default=None, description="Voice ID from /api/v1/config voices list")
    llm_profile_id: Optional[str] = Field(default="default", description="Server-defined safe LLM profile id")
    tts_profile_id: Optional[str] = Field(default="default", description="Server-defined safe TTS profile id")


class TransferUrlRequest(BaseModel):
    """POST /api/v1/artifacts/{id}/transfer-url — no body needed, but kept for future params."""
    pass


# ═══════════════════════════════════════════════════════════════════════════
# Response models
# ═══════════════════════════════════════════════════════════════════════════


class PodcastProjectResponse(BaseModel):
    """Full canonical PodcastProject manifest (GET /api/v1/projects/{id})."""
    project_id: str
    owner_id: str
    topic: str
    bpm: int
    duration_minutes: int
    status: ProjectStatus
    revision_token: str
    segments: list[Segment] = Field(default_factory=list)
    artifacts: list[Artifact] = Field(default_factory=list)
    errors: list[ProjectError] = Field(default_factory=list)
    final_download_ready: bool = False
    final_artifact_id: Optional[str] = None
    created_at: str
    updated_at: str
    deleted_at: Optional[str] = None


class ProjectCreateResponse(BaseModel):
    """Response for POST /api/v1/projects (202 Accepted)."""
    project: PodcastProjectResponse
    manifest_url: str


class ProjectSummary(BaseModel):
    """Lightweight project entry for GET /api/v1/projects list."""
    project_id: str
    topic: str
    bpm: int
    duration_minutes: int
    status: ProjectStatus
    revision_token: str
    segment_count: int
    ready_segment_count: int
    final_download_ready: bool
    created_at: str
    updated_at: str


class ProjectListResponse(BaseModel):
    """Response for GET /api/v1/projects."""
    projects: list[ProjectSummary]


class ProjectDeleteResponse(BaseModel):
    """Response for DELETE /api/v1/projects/{id}."""
    project_id: str
    status: str
    deleted_at: str


class TransferUrlResponse(BaseModel):
    """Response for POST /api/v1/artifacts/{id}/transfer-url."""
    artifact_id: str
    signed_transfer_url: str
    signed_transfer_expires_at: str


class HealthResponse(BaseModel):
    """GET /api/v1/health."""
    status: str
    version: str
    uptime_seconds: float


class VoiceProfile(BaseModel):
    id: str
    label: str


class ConfigResponse(BaseModel):
    """GET /api/v1/config — safe for iOS to display."""
    llm_profiles: list[VoiceProfile] = Field(default_factory=list)
    tts_profiles: list[VoiceProfile] = Field(default_factory=list)
    voices: list[VoiceProfile] = Field(default_factory=list)
    bpm_range: dict = Field(default_factory=lambda: {"min": settings.min_bpm, "max": settings.max_bpm})
    duration_minutes_range: dict = Field(default_factory=lambda: {"min": 1, "max": 30})


class ErrorResponse(BaseModel):
    """Standard v1 error envelope."""
    code: str
    message: str
    details: Optional[dict] = None


class ErrorEnvelope(BaseModel):
    error: ErrorResponse