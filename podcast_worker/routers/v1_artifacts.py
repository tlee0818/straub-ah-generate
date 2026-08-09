"""v1 artifact endpoints.

GET  /api/v1/artifacts/{artifact_id}             — Stream/download an artifact
POST /api/v1/artifacts/{artifact_id}/transfer-url — Refresh signed transfer URL
"""

from __future__ import annotations

import hashlib
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
_PUBLIC_ARTIFACT_KINDS = {"segment_audio", "final_mp3", "script_json"}


def _deny_internal_artifact(artifact: dict) -> None:
    if artifact.get("kind") not in _PUBLIC_ARTIFACT_KINDS:
        raise HTTPException(
            status_code=404,
            detail=ErrorEnvelope(error=ErrorResponse(
                code="not_found", message="Artifact not found."
            )).model_dump(),
        )


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
def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()



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
    _deny_internal_artifact(artifact)

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

    # Serve only the artifact's persisted canonical object key.
    object_key = artifact.get("object_key")
    if not object_key:
        raise HTTPException(
            status_code=404,
            detail=ErrorEnvelope(error=ErrorResponse(
                code="not_found", message="Artifact file not found on disk.",
                details={"artifact_id": artifact_id},
            )).model_dump(),
        )
    out_dir = Path(_output_dir())
    candidate = _safe_output_path(out_dir, object_key)
    if candidate is None or not candidate.is_file():
        raise HTTPException(
            status_code=404,
            detail=ErrorEnvelope(error=ErrorResponse(
                code="not_found", message="Artifact file not found on disk.",
                details={"artifact_id": artifact_id},
            )).model_dump(),
        )

    expected_size = artifact.get("size_bytes")
    expected_checksum = artifact.get("checksum_sha256")
    if (
        expected_size is None
        or expected_checksum is None
        or candidate.stat().st_size != expected_size
        or _sha256_file(candidate) != expected_checksum
    ):
        raise HTTPException(
            status_code=409,
            detail=ErrorEnvelope(error=ErrorResponse(
                code="artifact_integrity_failed",
                message="Artifact bytes failed integrity verification.",
                details={"artifact_id": artifact_id},
            )).model_dump(),
        )

    return FileResponse(
        path=str(candidate),
        media_type=artifact.get("content_type", "audio/mpeg"),
        filename=f"{artifact_id}{candidate.suffix}",
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
    _deny_internal_artifact(artifact)

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