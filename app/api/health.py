"""Liveness and readiness probes.

``/health``  — is the process up? No dependency calls, so a database blip never
               causes a restart loop.
``/ready``   — can it actually serve traffic? Checks each dependency and returns
               503 if any required one is down.

Probes report only up/down. Failure detail goes to the log, never to the response —
these endpoints are unauthenticated so orchestrators can reach them.
"""

import logging
from typing import TYPE_CHECKING, Annotated, Any, Literal

from fastapi import APIRouter, Depends, Response
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.api.deps import s3_client, sqs_client
from app.core.config import Settings, get_settings
from app.core.db import get_session

if TYPE_CHECKING:  # pragma: no cover - typing only
    from mypy_boto3_s3.client import S3Client
    from mypy_boto3_sqs.client import SQSClient

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])

_UP: dict[str, Any] = {"status": "up"}
_DOWN: dict[str, Any] = {"status": "down"}
_NOT_CONFIGURED: dict[str, Any] = {"status": "not_configured"}


@router.get("/health")
def health() -> dict[str, Literal["ok"]]:
    return {"status": "ok"}


def _check(name: str, probe: Any) -> dict[str, Any]:
    try:
        probe()
    except Exception as exc:
        logger.warning("readiness_check_failed", extra={"dependency": name}, exc_info=exc)
        return _DOWN
    return _UP


@router.get("/ready")
def ready(
    response: Response,
    session: Annotated[Session, Depends(get_session)],
    settings: Annotated[Settings, Depends(get_settings)],
    s3: Annotated["S3Client", Depends(s3_client)],
    sqs: Annotated["SQSClient", Depends(sqs_client)],
) -> dict[str, Any]:
    checks: dict[str, Any] = {
        "database": _check("database", lambda: session.execute(text("SELECT 1"))),
        "s3": (
            _check("s3", lambda: s3.head_bucket(Bucket=settings.s3_bucket))
            if settings.s3_bucket
            else _NOT_CONFIGURED
        ),
        "sqs": (
            _check(
                "sqs",
                lambda: sqs.get_queue_attributes(
                    QueueUrl=settings.sqs_queue_url, AttributeNames=["QueueArn"]
                ),
            )
            if settings.sqs_queue_url
            else _NOT_CONFIGURED
        ),
    }

    # `not_configured` is not a failure: it keeps the service startable in
    # environments where a dependency is genuinely absent, while still being visible.
    ok = all(check["status"] != "down" for check in checks.values())
    if not ok:
        response.status_code = 503

    return {
        "status": "ready" if ok else "not_ready",
        "checks": checks,
        "aws_mode": "localstack" if settings.uses_local_aws else "aws",
    }
