"""Liveness and readiness probes.

``/health``  — is the process up? No dependency calls, so a database blip never
               causes a restart loop.
``/ready``   — can it actually serve traffic? Checks each dependency and returns
               503 if any required one is down.
"""

import logging
from typing import Any, Literal

from fastapi import APIRouter, Depends, Response
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.db import get_session

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict[str, Literal["ok"]]:
    return {"status": "ok"}


def _check_database(session: Session) -> dict[str, Any]:
    try:
        session.execute(text("SELECT 1"))
    except Exception as exc:
        # Log the detail; the response says only that it failed.
        logger.warning("readiness_check_failed", extra={"dependency": "database"}, exc_info=exc)
        return {"status": "down"}
    return {"status": "up"}


@router.get("/ready")
def ready(
    response: Response,
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    # S3 and SQS probes are added in step 4, when those are configured.
    checks: dict[str, Any] = {"database": _check_database(session)}

    ok = all(check["status"] == "up" for check in checks.values())
    if not ok:
        response.status_code = 503
    return {
        "status": "ready" if ok else "not_ready",
        "checks": checks,
        "aws_mode": "localstack" if settings.uses_local_aws else "aws",
    }
