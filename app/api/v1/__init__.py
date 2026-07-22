"""Versioned API surface.

Every route mounted on this router requires the service token. Attaching the
dependency at the router — rather than per-endpoint — means a new endpoint is
authenticated by default and cannot be forgotten.

``/health`` and ``/ready`` are deliberately NOT mounted here: probes must work
without credentials.
"""

from fastapi import APIRouter, Depends

from app.api.deps import require_service_token

router = APIRouter(
    prefix="/v1",
    dependencies=[Depends(require_service_token)],
    responses={
        401: {"description": "Missing or invalid service credentials"},
        503: {"description": "Service credentials are not configured"},
    },
)
