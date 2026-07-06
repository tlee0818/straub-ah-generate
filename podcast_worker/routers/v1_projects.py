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
from podcast_worker.core.models_v1 import (
    ErrorEnvelope,
    ErrorResponse,
    PodcastProjectResponse,
    ProjectCreateRequest,
    ProjectCreateResponse,
    ProjectDeleteResponse,
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


# ═══════════════════════════════════════════════════════════════════════════
# POST /api/v1/projects
# ═══════════════════════════════════════════════════════════════════════════


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
    thread = threading.Thread(
        target=run_project_pipeline,
        args=(db, project_id, _output_dir(), req.topic,
              req.bpm, req.duration_minutes, req.voice_id),
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
    project = await persistence.get_project(_db_path(), project_id)
    if project is None or project.get("owner_id") != owner_id:
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


# ═══════════════════════════════════════════════════════════════════════════
# DELETE /api/v1/projects/{project_id}
# ═══════════════════════════════════════════════════════════════════════════


@router.delete("/{project_id}", response_model=ProjectDeleteResponse)
async def delete_project(project_id: str, owner_id: str = Depends(require_auth)):
    """Delete the project and its retained artifacts."""
    # Verify ownership first
    project = await persistence.get_project(_db_path(), project_id)
    if project is None or project.get("owner_id") != owner_id:
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