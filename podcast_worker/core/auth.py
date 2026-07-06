"""Auth dependency for /api/v1/* endpoints.

Requires an Authorization: Bearer <token> header on every v1 product endpoint.
For the hosted single-user v1 deployment the token identifies the single
configured owner.  Future multi-user support may map the same field to a
real user identity.
"""

from __future__ import annotations

from typing import Optional

from fastapi import Header, HTTPException, Request

from podcast_worker.core.config import settings
from podcast_worker.core.models_v1 import ErrorEnvelope, ErrorResponse


async def require_auth(
    authorization: Optional[str] = Header(default=None, alias="Authorization"),
) -> str:
    """Validate the Bearer token and return the owner_id (always 'single-user' in v1)."""
    if not settings.auth_token:
        # No token configured — auth is effectively disabled for dev.
        return "single-user"

    if not authorization or not authorization.startswith("Bearer "):
        raise _unauthorized("Authentication is required.")

    token = authorization[7:].strip()
    if token != settings.auth_token:
        raise _unauthorized("Invalid authentication token.")

    return "single-user"


async def optional_auth(request: Request) -> str | None:
    """Optional auth for health endpoints. Returns owner_id or None."""
    auth_header = request.headers.get("Authorization", "")
    if not auth_header:
        if settings.auth_token:
            return None  # auth required but not provided
        return "single-user"

    if not auth_header.startswith("Bearer "):
        return None

    token = auth_header[7:].strip()
    if settings.auth_token and token != settings.auth_token:
        return None

    return "single-user"


def _unauthorized(message: str) -> HTTPException:
    return HTTPException(
        status_code=401,
        detail=ErrorEnvelope(
            error=ErrorResponse(code="unauthorized", message=message),
        ).model_dump(),
    )