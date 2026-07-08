"""v1 artifact endpoints.

GET  /api/v1/artifacts/{artifact_id}             — Stream/download an artifact
POST /api/v1/artifacts/{artifact_id}/transfer-url — Refresh signed transfer URL
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from podcast_worker.core import persistence
from podcast_worker.core.auth import require_auth
from podcast_worker.core.config import settings
from podcast_worker.core.models_v1 import (
    ErrorEnvelope,
    ErrorResponse,
    TransferUrlResponse,
)

router = APIRouter(prefix="/api/v1/artifacts", tags=["v1-artifacts"])


def _db_path() -> str:
    return settings.db_path


def _output_dir() -> str:
    configured = Path(settings.output_dir)
    if configured.is_absolute():
        return str(configured)
    project_root = Path(__file__).resolve().parent.parent.parent
    return str(project_root / configured)


def _safe_output_path(out_dir: Path, filename: str) -> Path | None:
    candidate = (out_dir / filename).resolve()
    try:
        candidate.relative_to(out_dir.resolve())
    except ValueError:
        return None
    return candidate


# ═══════════════════════════════════════════════════════════════════════════
# GET /api/v1/artifacts/{artifact_id}
# ═══════════════════════════════════════════════════════════════════════════


@router.get("/{artifact_id}")
async def download_artifact(artifact_id: str, owner_id: str = Depends(require_auth)):
    """Stream or download an artifact for the authenticated owner."""
    artifact = await persistence.get_artifact(_db_path(), artifact_id)
    if artifact is None:
        raise HTTPException(
            status_code=404,
            detail=ErrorEnvelope(
                error=ErrorResponse(
                    code="not_found",
                    message="Artifact not found.",
                    details={"artifact_id": artifact_id},
                ),
            ).model_dump(),
        )

    if artifact.get("status") != "ready":
        raise HTTPException(
            status_code=404,
            detail=ErrorEnvelope(
                error=ErrorResponse(
                    code="not_ready",
                    message="Artifact is not ready for download.",
                    details={"artifact_id": artifact_id, "status": artifact["status"]},
                ),
            ).model_dump(),
        )

    # Verify the owning project belongs to the authenticated owner
    project_id = artifact.get("project_id")
    if project_id:
        project = await persistence.get_project(_db_path(), project_id)
        if project is None or project.get("owner_id") != owner_id:
            raise HTTPException(
                status_code=403,
                detail=ErrorEnvelope(
                    error=ErrorResponse(
                        code="forbidden",
                        message="Cannot access this artifact.",
                    ),
                ).model_dump(),
            )

    # Locate the file on disk without allowing DB-contaminated ids to escape output_dir.
    out_dir = Path(_output_dir())
    content_type = artifact.get("content_type", "audio/mpeg")
    media_type = content_type
    candidate_names = [
        f"mixed_{artifact.get('segment_id', '')}.mp3",
        f"final_{project_id}.mp3",
        f"speech_{artifact.get('segment_id', '')}.mp3",
    ]

    for name in candidate_names:
        candidate = _safe_output_path(out_dir, name)
        if candidate and candidate.exists() and candidate.is_file():
            return FileResponse(
                path=str(candidate),
                media_type=media_type,
                filename=f"{artifact_id}.mp3",
            )

    raise HTTPException(
        status_code=404,
        detail=ErrorEnvelope(
            error=ErrorResponse(
                code="not_found",
                message="Artifact file not found on disk.",
                details={"artifact_id": artifact_id},
            ),
        ).model_dump(),
    )


# ═══════════════════════════════════════════════════════════════════════════
# POST /api/v1/artifacts/{artifact_id}/transfer-url
# ═══════════════════════════════════════════════════════════════════════════


@router.post("/{artifact_id}/transfer-url", response_model=TransferUrlResponse)
async def refresh_transfer_url(artifact_id: str, owner_id: str = Depends(require_auth)):
    """Refresh a short-lived signed media transfer URL.

    In v1 with local file storage, signed URLs are identity — we return the
    authenticated download URL as the transfer URL.  When object storage
    offload is added, this endpoint will generate real signed URLs.
    """
    artifact = await persistence.get_artifact(_db_path(), artifact_id)
    if artifact is None:
        raise HTTPException(
            status_code=404,
            detail=ErrorEnvelope(
                error=ErrorResponse(
                    code="not_found",
                    message="Artifact not found.",
                    details={"artifact_id": artifact_id},
                ),
            ).model_dump(),
        )

    # Verify ownership
    project_id = artifact.get("project_id")
    if project_id:
        project = await persistence.get_project(_db_path(), project_id)
        if project is None or project.get("owner_id") != owner_id:
            raise HTTPException(
                status_code=403,
                detail=ErrorEnvelope(
                    error=ErrorResponse(
                        code="forbidden",
                        message="Cannot access this artifact.",
                    ),
                ).model_dump(),
            )

    from datetime import datetime, timedelta, timezone

    expires = (datetime.now(timezone.utc) + timedelta(minutes=15)).isoformat()
    download_url = f"/api/v1/artifacts/{artifact_id}"

    return TransferUrlResponse(
        artifact_id=artifact_id,
        signed_transfer_url=download_url,
        signed_transfer_expires_at=expires,
    )