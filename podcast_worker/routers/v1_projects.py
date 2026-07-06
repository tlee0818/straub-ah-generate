"""v1 project endpoints.

POST   /api/v1/projects         — Create project, start background generation
GET    /api/v1/projects         — List projects for authenticated owner
GET    /api/v1/projects/{id}    — Full canonical PodcastProject manifest
DELETE /api/v1/projects/{id}    — Delete project and artifacts
"""

from __future__ import annotations

import threading
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException

from podcast_worker.core import persistence
from podcast_worker.core.auth import require_auth
from podcast_worker.core.config import settings
from podcast_worker.core.script_generator import generate_script_outline
from podcast_worker.core.models_v1 import (
    ErrorEnvelope,
    ErrorResponse,
    PodcastProjectResponse,
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


def _db_path() -> str:
    return settings.db_path


def _output_dir() -> str:
    project_root = Path(__file__).resolve().parent.parent.parent
    return str(project_root / "output")


def _short_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


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

@router.post("/outline-preview", response_model=ProjectOutlineResponse)
async def preview_project_outline(
    req: OutlinePreviewRequest,
    owner_id: str = Depends(require_auth),
):
    """Generate a reviewable script outline before starting full project generation."""
    del owner_id
    provider = settings.llm_provider
    model = settings.openai_model if provider == "openai" else settings.openrouter_model
    outline = generate_script_outline(
        topic=req.topic,
        bpm=req.bpm,
        duration_minutes=req.duration_minutes,
        provider=provider,
        model=model,
    )
    return _outline_response_from_generator(req.topic, outline)



@router.post("", status_code=202, response_model=ProjectCreateResponse)
async def create_project(
    req: ProjectCreateRequest,
    owner_id: str = Depends(require_auth),
):
    """Create a durable PodcastProject and start backend generation."""
    project_id = _short_id("prj")
    db = _db_path()

    project = await persistence.create_project(
        db, project_id, owner_id, req.topic, req.bpm, req.duration_minutes,
    )

    # Start background segment generation in a daemon thread
    outline = _outline_response_to_generator(req.approved_outline)
    thread = threading.Thread(
        target=run_project_pipeline,
        args=(db, project_id, _output_dir(), req.topic,
              req.bpm, req.duration_minutes, req.voice_id, outline),
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