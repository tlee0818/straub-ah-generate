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
from podcast_worker.core.models_v1 import (
    ErrorEnvelope,
    ErrorResponse,
    PodcastProjectResponse,
    OutlinePreviewResponse,
    OutlinePreviewRequest,
    ProjectCreateRequest,
    ProjectCreateResponse,
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

def _short_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _run_project_pipeline_with_slot(slot, *args) -> None:
    try:
        run_project_pipeline(*args)
    finally:
        slot.release()


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
    """Generate a reviewable outline with a durable paired safe-profile binding."""
    try:
        llm = resolve_llm_profile(req.llm_profile_id)
        tts = resolve_tts_profile(req.tts_profile_id)
        validate_profile_pair(llm, tts)
    except RoutingConfigurationError as exc:
        raise _routing_error(exc, str(exc) == "incompatible_generation_profiles") from exc
    llm_profile_id, tts_profile_id = llm.profile_id, tts.profile_id
    preview_id, ledger_id, attempt_id = _short_id("opv"), _short_id("led"), _short_id("att")
    routing_revision, tts_routing_revision = llm.revision, tts.revision
    llm_payload, tts_payload = _llm_snapshot_payload(llm), _tts_snapshot_payload(tts)
    llm_snapshot = {"snapshot_id": _short_id("lsn"), "profile_id": llm_profile_id, "revision": routing_revision,
                    "payload": llm_payload, "sha256": hashlib.sha256(json.dumps(llm_payload, sort_keys=True).encode()).hexdigest()}
    tts_snapshot = {"snapshot_id": _short_id("tsn"), "profile_id": tts_profile_id, "revision": tts_routing_revision,
                    "payload": tts_payload, "sha256": hashlib.sha256(json.dumps(tts_payload, sort_keys=True).encode()).hexdigest()}
    expires_at = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    policy = {"mode": llm.budget.mode, "caps": dict(llm.budget.caps), "pricing": dict(llm.budget.pricing)}
    await persistence.create_preview_binding(_db_path(), {
        "outline_preview_id": preview_id, "owner_id": owner_id, "topic": req.topic, "bpm": req.bpm,
        "duration_minutes": req.duration_minutes, "outline": {}, "llm_profile_id": llm_profile_id,
        "tts_profile_id": tts_profile_id, "routing_revision": routing_revision,
        "tts_routing_revision": tts_routing_revision, "expires_at": expires_at,
    }, llm_snapshot, tts_snapshot, {"ledger_id": ledger_id, "policy": policy, "currency": llm.budget.currency})
    reservation = {"entry_id": _short_id("ledent"), "ledger_id": ledger_id, "category": "llm",
                   "operation_type": "outline", "correlation_id": preview_id, "resource_unit": "request",
                   "amount": 1, "pricing": dict(llm.budget.pricing)}
    accepted = await persistence.reserve_ledger_operation(_db_path(), reservation, {
        "attempt_id": attempt_id, "snapshot_revision": routing_revision,
        "binding": {"purpose": "outline", "provider": llm.route_for("outline").provider, "model": llm.route_for("outline").model},
    })
    if not accepted:
        raise HTTPException(status_code=409, detail=ErrorEnvelope(error=ErrorResponse(
            code="generation_budget_exhausted", message="Generation budget is unavailable."
        )).model_dump())
    try:
        outline = generate_script_outline(topic=req.topic, bpm=req.bpm, duration_minutes=req.duration_minutes, snapshot=llm)
    except Exception as exc:
        await persistence.settle_ledger_operation(_db_path(), ledger_id, attempt_id, {**reservation, "entry_id": _short_id("ledent")},
                                                  "failed", error={"code": "outline_failed"})
        raise
    response = _outline_response_from_generator(req.topic, outline)
    await persistence.update_preview_outline(_db_path(), preview_id, response.model_dump())
    await persistence.settle_ledger_operation(_db_path(), ledger_id, attempt_id, {**reservation, "entry_id": _short_id("ledent")},
                                              "succeeded")
    return OutlinePreviewResponse(**response.model_dump(), outline_preview_id=preview_id,
                                  llm_profile_id=llm_profile_id, tts_profile_id=tts_profile_id,
                                  routing_revision=routing_revision, tts_routing_revision=tts_routing_revision,
                                  expires_at=expires_at)



@router.post("", status_code=202, response_model=ProjectCreateResponse)
async def create_project(
    req: ProjectCreateRequest,
    owner_id: str = Depends(require_auth),
):
    """Create a durable PodcastProject and start backend generation."""
    interviewer_voice_id = req.interviewer_profile.voice_id if req.interviewer_profile else None
    guest_voice_id = req.sme_profile.voice_id if req.sme_profile else None
    try:
        llm = resolve_llm_profile(req.llm_profile_id)
        tts = resolve_tts_profile(req.tts_profile_id, interviewer_voice_id, guest_voice_id)
        validate_profile_pair(llm, tts)
    except RoutingConfigurationError as exc:
        raise _routing_error(exc, str(exc) == "incompatible_generation_profiles") from exc
    llm_profile_id, tts_profile_id = llm.profile_id, tts.profile_id
    routing_revision, tts_routing_revision = llm.revision, tts.revision
    preview = None
    if req.outline_preview_id:
        preview = await persistence.get_preview_binding(_db_path(), req.outline_preview_id, owner_id)
        if preview is None or preview.get("consumed_at") is not None or preview.get("expires_at", "") <= datetime.now(timezone.utc).isoformat():
            raise HTTPException(status_code=409, detail=ErrorEnvelope(error=ErrorResponse(
                code="preview_expired" if preview else "preview_not_found",
                message="The outline preview is no longer available."
            )).model_dump())
        if not preview.get("tts_snapshot_id"):
            raise HTTPException(status_code=409, detail=ErrorEnvelope(error=ErrorResponse(
                code="preview_binding_required", message="This outline preview must be regenerated."
            )).model_dump())
        if (preview["topic"], preview["bpm"], preview["duration_minutes"], preview["llm_profile_id"],
            preview["tts_profile_id"], preview["routing_revision"], preview["tts_routing_revision"]) != (
                req.topic, req.bpm, req.duration_minutes, llm_profile_id, tts_profile_id,
                routing_revision, tts_routing_revision):
            raise HTTPException(status_code=409, detail=ErrorEnvelope(error=ErrorResponse(
                code="preview_profile_mismatch", message="The selected generation request differs from the preview."
            )).model_dump())
    project_id = _short_id("prj")
    db = _db_path()
    if not _generation_slots.acquire(blocking=False):
        raise HTTPException(status_code=503, detail="Generation capacity is full.")
    project = await persistence.create_project(db, project_id, owner_id, req.topic, req.bpm, req.duration_minutes)
    if preview:
        result = await persistence.consume_preview_binding(
            db, req.outline_preview_id, owner_id, project_id, req.topic, req.bpm, req.duration_minutes,
            llm_profile_id, tts_profile_id, routing_revision, tts_routing_revision,
        )
        if result != "ok":
            await persistence.delete_project(db, project_id)
            _generation_slots.release()
            raise HTTPException(status_code=409, detail=ErrorEnvelope(error=ErrorResponse(
                code=result, message="The outline preview could not be consumed."
            )).model_dump())
    if preview:
        ledger_id = preview["ledger_id"]
    else:
        llm_payload, tts_payload = _llm_snapshot_payload(llm), _tts_snapshot_payload(tts)
        llm_snapshot = {"snapshot_id": _short_id("lsn"), "profile_id": llm_profile_id,
                        "revision": routing_revision, "payload": llm_payload,
                        "sha256": hashlib.sha256(json.dumps(llm_payload, sort_keys=True).encode()).hexdigest()}
        tts_snapshot = {"snapshot_id": _short_id("tsn"), "profile_id": tts_profile_id,
                        "revision": tts_routing_revision, "payload": tts_payload,
                        "sha256": hashlib.sha256(json.dumps(tts_payload, sort_keys=True).encode()).hexdigest()}
        ledger_id = _short_id("led")
        await persistence.create_project_execution(db, project_id, owner_id, llm_snapshot, tts_snapshot, {
            "ledger_id": ledger_id,
            "policy": {"mode": llm.budget.mode, "caps": dict(llm.budget.caps), "pricing": dict(llm.budget.pricing)},
            "currency": llm.budget.currency,
        })
    await persistence.upsert_durable_record(db, "work_items", {
        "work_id": _short_id("wrk"), "project_id": project_id, "ledger_id": ledger_id,
        "kind": "project_pipeline", "state": "pending",
        "payload_json": {"llm_profile_id": llm_profile_id, "tts_profile_id": tts_profile_id,
                         "routing_revision": routing_revision, "tts_routing_revision": tts_routing_revision},
    })
    await persistence.claim_reconcilable_work(
        db, "work_items", f"request:{project_id}",
        (datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat(),
    )

    # Start background segment generation in a daemon thread
    outline = _outline_response_to_generator(req.approved_outline)
    interviewer_profile = req.interviewer_profile.model_dump(exclude_none=True) if req.interviewer_profile else None
    sme_profile = req.sme_profile.model_dump(exclude_none=True) if req.sme_profile else None
    slot = _generation_slots
    thread = threading.Thread(
        target=_run_project_pipeline_with_slot,
        args=(slot, db, project_id, _output_dir(), req.topic,
              req.bpm, req.duration_minutes, req.voice_id, outline,
              interviewer_profile, sme_profile),
        daemon=True,
    )
    thread.start()

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
                "segment_type": "content",
                "topic": seg["subtopic"],
                "title": seg.get("title"),
                "approx_duration_seconds": seg.get("duration_seconds"),
            }
        )

    title = sections[0]["title"] if sections and sections[0].get("title") else project["topic"]
    return ProjectOutlineResponse(
        project_id=project_id,
        topic=project["topic"],
        title=title,
        sections=sections,
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