"""v1 project endpoints.

POST   /api/v1/projects         — Create project, start background generation
GET    /api/v1/projects         — List projects for authenticated owner
GET    /api/v1/projects/{id}    — Full canonical PodcastProject manifest
DELETE /api/v1/projects/{id}    — Delete project and artifacts
"""

from __future__ import annotations

import hashlib
import json
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse

from podcast_worker.core import persistence
from podcast_worker.core.auth import require_auth
from podcast_worker.core.config import (
    RoutingConfigurationError,
    resolve_llm_profile,
    resolve_tts_profile,
    settings,
    validate_profile_pair,
)
from podcast_worker.core.script_generator import generate_script_outline
from podcast_worker.core.observability import log_event, metrics
from podcast_worker.core.models_v1 import (
    ErrorEnvelope,
    ErrorResponse,
    PodcastProjectResponse,
    OutlinePreviewResponse,
    OutlinePreviewRequest,
    OutlinePreviewResponse,
    ProjectCreateRequest,
    ProjectCreateResponse,
    ProjectCancellationResponse,
    ProjectDeleteResponse,
    ProjectOutlineResponse,
    ProjectListResponse,
    ProjectSummary,
)
from podcast_worker.core.segment_pipeline import run_project_pipeline

router = APIRouter(prefix="/api/v1/projects", tags=["v1-projects"])
_generation_slots = threading.BoundedSemaphore(settings.max_concurrent_generations)


def _db_path() -> str:
    return settings.db_path


def _output_dir() -> str:
    configured = Path(settings.output_dir)
    if configured.is_absolute():
        return str(configured)
    project_root = Path(__file__).resolve().parent.parent.parent
    return str(project_root / configured)


def _routing_error(exc: RoutingConfigurationError, pair: bool = False) -> HTTPException:
    code = "incompatible_generation_profiles" if pair else str(exc)
    return HTTPException(status_code=422, detail=ErrorEnvelope(error=ErrorResponse(
        code=code, message="The selected generation profile is unavailable or incompatible."
    )).model_dump())


def _llm_snapshot_payload(snapshot) -> dict:
    return json.loads(snapshot.canonical_json())


def _tts_snapshot_payload(snapshot) -> dict:
    return {
        "profile_id": snapshot.profile_id, "revision": snapshot.revision,
        "provider": snapshot.provider, "strategy": snapshot.strategy, "model_id": snapshot.model_id,
        "output_format": snapshot.output_format, "max_scene_characters": snapshot.max_scene_characters,
        "max_scene_turns": snapshot.max_scene_turns, "max_fragment_characters": snapshot.max_fragment_characters,
        "voice_bindings": dict(snapshot.voice_bindings), "max_attempts": snapshot.max_attempts,
        "context_max_request_ids": snapshot.context_max_request_ids,
        "context_max_age_seconds": snapshot.context_max_age_seconds,
        "continuity_after_expiry": snapshot.continuity_after_expiry,
        "max_concurrent_requests": snapshot.max_concurrent_requests,
        "budget": {"mode": snapshot.budget.mode, "currency": snapshot.budget.currency,
                   "caps": dict(snapshot.budget.caps), "pricing": dict(snapshot.budget.pricing)},
    }
def _snapshot_record(snapshot, payload: dict, prefix: str) -> dict:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    digest = hashlib.sha256(canonical.encode()).hexdigest()
    return {
        "snapshot_id": f"{prefix}_{digest[:20]}",
        "profile_id": snapshot.profile_id,
        "revision": snapshot.revision,
        "payload": payload,
        "sha256": digest,
    }



def _short_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _run_project_pipeline_with_slot(slot, *args) -> None:
    try:
        run_project_pipeline(*args)
    finally:
        slot.release()


def _start_project_pipeline(args: tuple) -> None:
    thread = threading.Thread(
        target=_run_project_pipeline_with_slot,
        args=(_generation_slots, *args),
        daemon=True,
    )
    thread.start()


def _project_not_found(project_id: str) -> HTTPException:
    return HTTPException(
        status_code=404,
        detail=ErrorEnvelope(
            error=ErrorResponse(
                code="not_found",
                message="Project not found.",
                details={"project_id": project_id},
            ),
        ).model_dump(),
    )


async def _load_owned_project(project_id: str, owner_id: str) -> dict:
    project = await persistence.get_project(_db_path(), project_id)
    if project is None or project.get("owner_id") != owner_id:
        raise _project_not_found(project_id)
    return project


def _outline_response_from_generator(topic: str, outline: dict) -> ProjectOutlineResponse:
    sections = []
    for index, section in enumerate(outline.get("sections", [])):
        sections.append(
            {
                "index": index,
                "segment_id": None,
                "segment_type": section.get("segment_type", "content"),
                "topic": section.get("topic") or section.get("title") or f"Section {index + 1}",
                "title": section.get("title") or section.get("topic"),
                "approx_duration_seconds": section.get("approx_duration_seconds"),
            }
        )

    return ProjectOutlineResponse(
        project_id="preview",
        topic=topic,
        title=outline.get("title", topic),
        sections=sections,
    )


def _outline_response_to_generator(outline: ProjectOutlineResponse | None) -> dict | None:
    if outline is None:
        return None

    return {
        "title": outline.title,
        "sections": [
            {
                "index": section.index,
                "segment_type": section.segment_type,
                "topic": section.topic,
                "title": section.title,
                "approx_duration_seconds": section.approx_duration_seconds,
            }
            for section in outline.sections
        ],
    }




# ═══════════════════════════════════════════════════════════════════════════
# POST /api/v1/projects
# ═══════════════════════════════════════════════════════════════════════════

@router.post("/outline-preview", response_model=OutlinePreviewResponse)
async def preview_project_outline(
    req: OutlinePreviewRequest,
    owner_id: str = Depends(require_auth),
):
    """Generate and persist a one-time reviewable outline binding."""
    llm_snapshot = resolve_llm_profile(req.llm_profile_id)
    tts_snapshot = resolve_tts_profile(req.tts_profile_id)
    validate_profile_pair(llm_snapshot, tts_snapshot)
    provider = llm_snapshot.route_for("outline").provider
    model = llm_snapshot.route_for("outline").model
    outline = generate_script_outline(
        topic=req.topic,
        bpm=req.bpm,
        duration_minutes=req.duration_minutes,
        provider=provider,
        model=model,
    )
    response_outline = _outline_response_from_generator(req.topic, outline)
    binding = await persistence.create_outline_preview_binding(
        _db_path(),
        _short_id("opv"),
        owner_id,
        req.topic,
        req.bpm,
        req.duration_minutes,
        _outline_response_to_generator(response_outline),
        llm_snapshot.profile_id,
        tts_snapshot.profile_id,
        llm_snapshot.revision,
        tts_snapshot.revision,
        (datetime.now(timezone.utc) + timedelta(minutes=15)).isoformat(),
    )
    llm_payload = _llm_snapshot_payload(llm_snapshot)
    tts_payload = _tts_snapshot_payload(tts_snapshot)
    persistence._create_preview_binding(
        _db_path(),
        {
            "outline_preview_id": binding["outline_preview_id"],
            "owner_id": owner_id,
            "topic": req.topic,
            "bpm": req.bpm,
            "duration_minutes": req.duration_minutes,
            "outline": _outline_response_to_generator(response_outline),
            "llm_profile_id": llm_snapshot.profile_id,
            "tts_profile_id": tts_snapshot.profile_id,
            "routing_revision": llm_snapshot.revision,
            "tts_routing_revision": tts_snapshot.revision,
            "expires_at": binding["expires_at"],
        },
        _snapshot_record(llm_snapshot, llm_payload, "lsn"),
        _snapshot_record(tts_snapshot, tts_payload, "tsn"),
        {
            "ledger_id": _short_id("led"),
            "policy": {
                "mode": llm_snapshot.budget.mode,
                "caps": dict(llm_snapshot.budget.caps),
                "pricing": dict(llm_snapshot.budget.pricing),
            },
            "currency": llm_snapshot.budget.currency,
        },
    )
    return OutlinePreviewResponse(
        **response_outline.model_dump(),
        binding=binding,
        outline_preview_id=binding["outline_preview_id"],
        llm_profile_id=binding["llm_profile_id"],
        tts_profile_id=binding["tts_profile_id"],
        routing_revision=binding["llm_routing_revision"],
        tts_routing_revision=binding["tts_routing_revision"],
        expires_at=binding["expires_at"],
    )



@router.post("", status_code=202, response_model=ProjectCreateResponse)
async def create_project(
    req: ProjectCreateRequest,
    owner_id: str = Depends(require_auth),
):
    """Atomically create a durable project before scheduling generation."""
    project_id = _short_id("prj")
    db = _db_path()
    if not _generation_slots.acquire(blocking=False):
        raise HTTPException(status_code=503, detail="Generation capacity is full.")

    outline = _outline_response_to_generator(req.approved_outline)
    outline_preview_id = req.outline_preview_id
    if outline is None or outline_preview_id is None:
        _generation_slots.release()
        raise HTTPException(
            status_code=409,
            detail=ErrorEnvelope(
                error=ErrorResponse(
                    code="preview_binding_not_found",
                    message="A caller-reviewed outline preview binding is required.",
                )
            ).model_dump(),
        )

    interviewer_profile = req.interviewer_profile.model_dump(exclude_none=True) if req.interviewer_profile else None
    sme_profile = req.sme_profile.model_dump(exclude_none=True) if req.sme_profile else None
    llm_snapshot = resolve_llm_profile(req.llm_profile_id)
    tts_snapshot = resolve_tts_profile(req.tts_profile_id)
    validate_profile_pair(llm_snapshot, tts_snapshot)
    persisted_preview = persistence._get_preview_binding(db, outline_preview_id, owner_id)
    if persisted_preview is None:
        _generation_slots.release()
        raise HTTPException(
            status_code=409,
            detail=ErrorEnvelope(
                error=ErrorResponse(
                    code="preview_binding_not_found",
                    message="The persisted execution binding is unavailable.",
                )
            ).model_dump(),
        )
    try:
        project = await persistence.create_progress_project(
            db,
            project_id,
            owner_id,
            req.topic,
            req.bpm,
            req.duration_minutes,
            outline,
            interviewer_profile,
            sme_profile,
            persisted_preview["llm_snapshot_id"],
            persisted_preview["tts_snapshot_id"],
            outline_preview_id,
        )
    except ValueError as exc:
        _generation_slots.release()
        code = str(exc)
        if code.startswith("preview_binding_"):
            raise HTTPException(
                status_code=409,
                detail=ErrorEnvelope(
                    error=ErrorResponse(code=code, message="Outline preview binding is invalid.")
                ).model_dump(),
            ) from exc
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception:
        _generation_slots.release()
        raise

    work_id = f"work_{uuid.uuid5(uuid.NAMESPACE_URL, f'{project_id}:pipeline').hex}"
    lease_owner = _short_id("worker")
    claim = await persistence.claim_next_work(
        db,
        lease_owner,
        settings.work_lease_seconds,
        work_id,
    )
    if claim is None:
        _generation_slots.release()
        raise HTTPException(status_code=503, detail="Generation work could not be claimed.")
    _start_project_pipeline(
        (
            db,
            claim["work_id"],
            claim["lease_owner"],
            claim["lease_epoch"],
            _output_dir(),
        )
    )

    response_project = PodcastProjectResponse(**project)
    return ProjectCreateResponse(
        project=response_project,
        manifest_url=f"/api/v1/projects/{project_id}",
    )


# ═══════════════════════════════════════════════════════════════════════════
# GET /api/v1/projects
# ═══════════════════════════════════════════════════════════════════════════


@router.get("", response_model=ProjectListResponse)
async def list_projects(owner_id: str = Depends(require_auth)):
    """List durable project manifests for the authenticated owner."""
    projects = await persistence.list_projects(_db_path(), owner_id)
    return ProjectListResponse(
        projects=[ProjectSummary(**p) for p in projects],
    )


# ═══════════════════════════════════════════════════════════════════════════
# GET /api/v1/projects/{project_id}
# ═══════════════════════════════════════════════════════════════════════════


@router.get("/{project_id}", response_model=PodcastProjectResponse)
async def get_project(project_id: str, owner_id: str = Depends(require_auth)):
    """Return the canonical backend-owned PodcastProject manifest."""
    project = await _load_owned_project(project_id, owner_id)

    # Populate artifact_ids for each segment from the artifacts list
    artifacts = project.get("artifacts", [])
    seg_artifact_map: dict[str, list[str]] = {}
    for art in artifacts:
        sid = art.get("segment_id")
        if sid:
            seg_artifact_map.setdefault(sid, []).append(art["artifact_id"])

    for seg in project.get("segments", []):
        seg["artifact_ids"] = seg_artifact_map.get(seg["segment_id"], [])

    return PodcastProjectResponse(**project)


@router.get("/{project_id}/outline", response_model=ProjectOutlineResponse)
async def get_project_outline(project_id: str, owner_id: str = Depends(require_auth)):
    """Return the project's script outline without full segment text or artifacts."""
    project = await _load_owned_project(project_id, owner_id)
    sections = []
    for seg in project.get("segments", []):
        sections.append(
            {
                "index": seg["index"],
                "segment_id": seg["segment_id"],
                "segment_type": seg.get("segment_type", "content"),
                "topic": seg["subtopic"],
                "title": seg.get("title"),
                "approx_duration_seconds": seg.get("planned_duration_seconds"),
            }
        )

    title = sections[0]["title"] if sections and sections[0].get("title") else project["topic"]
    return ProjectOutlineResponse(
        project_id=project_id,
        topic=project["topic"],
        title=title,
        sections=sections,
    )


@router.post("/{project_id}/cancel", response_model=ProjectCancellationResponse)
async def cancel_project(project_id: str, owner_id: str = Depends(require_auth)):
    """Request two-phase cancellation and return the canonical project snapshot."""
    try:
        result = await persistence.request_project_cancellation(
            _db_path(), project_id, owner_id
        )
    except ValueError as exc:
        if str(exc) != "project_not_cancellable":
            raise
        raise HTTPException(
            status_code=409,
            detail=ErrorEnvelope(
                error=ErrorResponse(
                    code="project_not_cancellable",
                    message="Project is already terminal and cannot be cancelled.",
                    details={"project_id": project_id},
                )
            ).model_dump(),
        ) from exc
    if result is None:
        raise _project_not_found(project_id)

    metrics.increment(
        "podcast_cancel_requests_total",
        outcome=result["state"],
        route="project_cancel",
    )
    log_event(
        "cancel_request",
        project_id=project_id,
        outcome=result["state"],
        progress_version=result["project"]["generation_progress"]["progress_version"],
    )
    response = ProjectCancellationResponse(
        project=PodcastProjectResponse(**result["project"]),
        cancellation={
            "state": result["state"],
            "requested_at": result["requested_at"],
            "observed_at": result["observed_at"],
        },
    )
    return JSONResponse(
        status_code=result["http_status"],
        content=response.model_dump(mode="json"),
    )

# ═══════════════════════════════════════════════════════════════════════════
# DELETE /api/v1/projects/{project_id}
# ═══════════════════════════════════════════════════════════════════════════


@router.delete("/{project_id}", response_model=ProjectDeleteResponse)
async def delete_project(project_id: str, owner_id: str = Depends(require_auth)):
    """Delete the project and its retained artifacts."""
    # Verify ownership first
    project = await _load_owned_project(project_id, owner_id)

    result = await persistence.delete_project(_db_path(), project_id)
    if result is None:
        raise HTTPException(
            status_code=404,
            detail=ErrorEnvelope(
                error=ErrorResponse(
                    code="not_found",
                    message="Project not found.",
                    details={"project_id": project_id},
                ),
            ).model_dump(),
        )

    return ProjectDeleteResponse(**result)