"""Pydantic v2 schemas for the /api/v1/* product contract.

These models are the canonical request/response shapes shared with the iOS
client per API_SPEC.md.  Legacy /api/services/* models remain in core/models.py.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, Field, StrictInt, model_validator

from podcast_worker.core.config import settings


# ═══════════════════════════════════════════════════════════════════════════
# Enums (as Literal string unions for OpenAPI compatibility)
# ═══════════════════════════════════════════════════════════════════════════

ProjectStatus = str   # queued | generating | partially_ready | ready | failed | deleted
SegmentStatus = str   # queued | scripting | validating | tts | mixing | ready | failed
ValidationStatus = str  # pending | validated | needs_review | failed
ArtifactKind = str     # segment_audio | final_mp3 | script_json
ArtifactStatus = str   # pending | ready | failed | deleted

GenerationStageName = str  # research | text_generation | fact_checking | tts | mixing | finalizing
GenerationStageState = str  # pending | running | completed | failed | cancelled
GenerationDisposition = str  # active | cancellation_requested | terminal
TerminalOutcome = str  # ready | failed | cancelled


def required_segment_count(duration_minutes: int) -> int:
    """Return the server-owned, bounded segment count for a valid duration."""
    return min(max((duration_minutes + 1) // 2, 1), 12)


# ═══════════════════════════════════════════════════════════════════════════
# Nested / shared models
# ═══════════════════════════════════════════════════════════════════════════


class ProvenanceMetadata(BaseModel):
    """Quality gate record for a segment.  Must be stored before TTS begins."""
    prompt_id: str = Field(..., description="Identifier for the generation prompt/template run")
    model: Literal["server-managed"] = Field(default="server-managed", description="Fixed public sentinel; provider/model bindings are server-side")
    source_refs: list[str] = Field(default_factory=list, description="Source references when available")
    claim_notes: list[str] = Field(default_factory=list, description="Notes for claims or rationale when sources are unavailable")
    validation_status: ValidationStatus = Field(default="pending", description="Validation state")
    validation_errors: list[str] = Field(default_factory=list, description="Empty when validation passes")
    validated_at: Optional[str] = Field(default=None, description="ISO-8601 timestamp when validation completed")


class SegmentError(BaseModel):
    code: str = Field(..., description="Machine-readable error code")
    message: str = Field(..., description="Human-readable error detail")
    retryable: bool = Field(default=False, description="Whether this error is retryable")


class GenerationStageProgress(BaseModel):
    name: GenerationStageName
    state: GenerationStageState
    completed_units: int = Field(..., ge=0)
    total_units: int = Field(..., gt=0)
    current_segment_id: Optional[str] = None
    updated_at: str

    @model_validator(mode="after")
    def validate_counter(self) -> "GenerationStageProgress":
        if self.completed_units > self.total_units:
            raise ValueError("completed_units must not exceed total_units")
        return self


class GenerationActivity(BaseModel):
    kind: str  # stage | cancellation
    stage: Optional[GenerationStageName] = None
    segment_id: Optional[str] = None


class GenerationProgress(BaseModel):
    schema_version: int = 1
    progress_version: int = Field(..., ge=0)
    disposition: GenerationDisposition
    terminal_outcome: Optional[TerminalOutcome] = None
    is_terminal: bool
    planned_segment_count: int = Field(..., ge=1)
    cancellation_requested_at: Optional[str] = None
    terminal_at: Optional[str] = None
    last_transition_at: str
    current_activity: Optional[GenerationActivity] = None
    stages: list[GenerationStageProgress]

    @model_validator(mode="after")
    def validate_snapshot(self) -> "GenerationProgress":
        expected = {
            "research",
            "text_generation",
            "fact_checking",
            "tts",
            "mixing",
            "finalizing",
        }
        names = [stage.name for stage in self.stages]
        if len(names) != len(expected) or set(names) != expected:
            raise ValueError("generation progress must contain every canonical stage exactly once")
        if self.is_terminal != (self.disposition == "terminal"):
            raise ValueError("terminal disposition and is_terminal must agree")
        if self.is_terminal != (self.current_activity is None):
            raise ValueError("current_activity must be null exactly for terminal progress")
        return self


class CancellationState(BaseModel):
    state: str
    requested_at: str
    observed_at: Optional[str] = None


class Segment(BaseModel):
    """Sub-topic-level generation and playback unit."""
    segment_id: str = Field(..., description="Stable segment identifier")
    index: int = Field(..., ge=0, description="Playback order within the project")
    subtopic: str = Field(..., min_length=1, description="Segment topic derived from project outline")
    title: Optional[str] = Field(default=None, description="User-facing segment title")
    segment_type: str = Field(default="content", description="intro | content | outro")
    planned_duration_seconds: Optional[float] = Field(default=None, gt=0)
    status: SegmentStatus = Field(default="queued", description="Segment lifecycle status")
    duration_seconds: Optional[float] = Field(default=None, description="Known after audio artifact creation")
    text: Optional[str] = Field(default=None, description="Generated segment script text")
    provenance: Optional[ProvenanceMetadata] = Field(default=None, description="Required before tts/mixing/ready")
    artifact_ids: list[str] = Field(default_factory=list, description="Canonical artifacts produced for this segment")
    primary_audio_artifact_id: Optional[str] = Field(default=None, description="Main playable audio artifact when ready")
    error: Optional[SegmentError] = Field(default=None, description="Segment-level failure details")
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class OutlineSection(BaseModel):
    """One planned script section derived from project segments."""
    index: int = Field(..., ge=0, description="Playback order within the project")
    segment_id: Optional[str] = Field(default=None, description="Segment id when the section has been created")
    segment_type: str = Field(default="content", pattern="^(intro|content|outro)$", description="intro | content | outro")
    topic: str = Field(..., min_length=1, description="Planned section topic")
    title: str = Field(..., min_length=1, description="User-facing section title")
    approx_duration_seconds: float = Field(..., gt=0, description="Planned section duration")


class ProjectOutlineResponse(BaseModel):
    """Response for GET /api/v1/projects/{id}/outline."""
    project_id: str
    topic: str
    title: str
    sections: list[OutlineSection] = Field(default_factory=list)


class OutlinePreviewBinding(BaseModel):
    outline_preview_id: str
    llm_profile_id: str
    tts_profile_id: str
    llm_routing_revision: str
    tts_routing_revision: str
    expires_at: str
    request_hash: str
    outline_hash: str


class OutlinePreviewResponse(ProjectOutlineResponse):
    binding: OutlinePreviewBinding
    outline_preview_id: str
    llm_profile_id: str
    tts_profile_id: str
    routing_revision: str
    tts_routing_revision: str
    expires_at: str


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

class OutlinePreviewRequest(BaseModel):
    """POST /api/v1/projects/outline-preview — generate a reviewable outline first."""
    topic: str = Field(..., min_length=1, max_length=200, description="Podcast topic")
    bpm: int = Field(..., ge=settings.min_bpm, le=settings.max_bpm, description="Beats per minute")
    duration_minutes: StrictInt = Field(default=5, ge=1, le=30, description="Target duration in minutes")
    llm_profile_id: Optional[str] = Field(default="default", description="Server-defined safe LLM profile id")
    tts_profile_id: Optional[str] = Field(default=None, description="Server-defined safe TTS profile id")


class SpeakerProfile(BaseModel):
    """Dialogue speaker profile used by script generation prompts."""
    name: Optional[str] = Field(default=None, max_length=80, description="Display name or role label")
    voice_id: Optional[str] = Field(default=None, max_length=64, description="Opaque voice ID from /api/v1/config")
    voice: Optional[str] = Field(default=None, max_length=120, description="Legacy display-only voice description")
    tone: Optional[str] = Field(default=None, max_length=120, description="Speaking tone")
    humor: Optional[str] = Field(default=None, max_length=120, description="Humor level/style")
    style: Optional[str] = Field(default=None, max_length=160, description="Interviewing or expertise style")
    expertise: Optional[str] = Field(default=None, max_length=160, description="SME expertise framing")



class ProjectCreateRequest(BaseModel):
    """POST /api/v1/projects — user intent only; provider secrets are server-side."""
    topic: str = Field(..., min_length=1, max_length=200, description="Podcast topic")
    bpm: int = Field(..., ge=settings.min_bpm, le=settings.max_bpm, description="Beats per minute")
    duration_minutes: StrictInt = Field(default=5, ge=1, le=30, description="Target duration in minutes")
    voice_id: Optional[str] = Field(default=None, description="Voice ID from /api/v1/config voices list")
    llm_profile_id: Optional[str] = Field(default="default", description="Server-defined safe LLM profile id")
    tts_profile_id: Optional[str] = Field(default="default", description="Server-defined safe TTS profile id")
    outline_preview_id: Optional[str] = Field(default=None, max_length=80, description="Opaque approved outline preview binding")
    approved_outline: Optional[ProjectOutlineResponse] = Field(default=None, description="User-reviewed outline to use for script generation")
    interviewer_profile: Optional[SpeakerProfile] = Field(default=None, description="Interviewer voice, tone, humor, and style")
    sme_profile: Optional[SpeakerProfile] = Field(default=None, description="Subject matter expert guest voice, tone, humor, and expertise")
    outline_preview_id: Optional[str] = Field(
        default=None,
        description="One-time server binding returned with the approved outline preview",
    )

    @model_validator(mode="after")
    def validate_approved_outline(self) -> "ProjectCreateRequest":
        if self.approved_outline is None:
            return self
        sections = self.approved_outline.sections
        expected_count = required_segment_count(self.duration_minutes)
        if len(sections) != expected_count:
            raise ValueError(
                f"approved outline must contain exactly {expected_count} sections"
            )
        if [section.index for section in sections] != list(range(expected_count)):
            raise ValueError("approved outline section indices must be contiguous from zero")
        if self.approved_outline.topic.strip() != self.topic.strip():
            raise ValueError("approved outline topic must match the project topic")
        return self


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
    generation_progress: Optional[GenerationProgress] = None


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
    generation_progress: Optional[GenerationProgress] = None


class ProjectListResponse(BaseModel):
    """Response for GET /api/v1/projects."""
    projects: list[ProjectSummary]


class ProjectDeleteResponse(BaseModel):
    """Response for DELETE /api/v1/projects/{id}."""
    project_id: str
    status: str
    deleted_at: str


class ProjectCancellationResponse(BaseModel):
    project: PodcastProjectResponse
    cancellation: CancellationState


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


class LLMProfileSummary(BaseModel):
    id: str = Field(..., pattern=r"^[a-z0-9][a-z0-9-]{0,63}$")
    label: str
    description: Optional[str] = None


class TTSProfileSummary(BaseModel):
    id: str = Field(..., pattern=r"^[a-z0-9][a-z0-9-]{0,63}$")
    label: str
    description: Optional[str] = None


class VoiceProfile(BaseModel):
    id: str = Field(..., pattern=r"^[a-z0-9][a-z0-9-]{0,63}$")
    label: str
    roles: list[Literal["interviewer", "guest"]] = Field(default_factory=list)




class ConfigResponse(BaseModel):
    """GET /api/v1/config — safe for iOS to display."""
    llm_profiles: list[LLMProfileSummary] = Field(default_factory=list)
    tts_profiles: list[TTSProfileSummary] = Field(default_factory=list)
    voices: list[VoiceProfile] = Field(default_factory=list)
    default_llm_profile_id: Optional[str] = None
    default_tts_profile_id: Optional[str] = None
    bpm_range: dict = Field(default_factory=lambda: {"min": settings.min_bpm, "max": settings.max_bpm})
    duration_minutes_range: dict = Field(default_factory=lambda: {"min": 1, "max": 30})


class ErrorResponse(BaseModel):
    """Standard v1 error envelope."""
    code: str
    message: str
    details: Optional[dict] = None


class ErrorEnvelope(BaseModel):
    error: ErrorResponse