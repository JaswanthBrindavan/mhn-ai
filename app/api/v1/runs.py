"""Report-processing run endpoints.

Handlers stay thin: parse, delegate, return. Business logic lives in
``app.services.runs``.

Authentication is applied by the parent router (see ``app/api/v1/__init__.py``), which
authenticates Spring as a service. These handlers perform no user-level authorization.
"""

import uuid
from typing import TYPE_CHECKING, Annotated

from fastapi import APIRouter, Depends, Header, Response, status
from sqlalchemy.orm import Session

from app.api.deps import s3_client, sqs_client
from app.core.config import Settings, get_settings
from app.core.db import get_session

if TYPE_CHECKING:  # pragma: no cover - typing only
    from mypy_boto3_s3.client import S3Client
    from mypy_boto3_sqs.client import SQSClient
from app.schemas.runs import (
    CancelRunResponse,
    CreateRunRequest,
    CreateRunResponse,
    RunResponse,
)
from app.services import runs as runs_service

router = APIRouter(tags=["report-processing-runs"])


@router.post(
    "/report-processing-runs",
    response_model=CreateRunResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit reports for AI processing",
)
def create_run(
    payload: CreateRunRequest,
    session: Annotated[Session, Depends(get_session)],
    s3: Annotated["S3Client", Depends(s3_client)],
    sqs: Annotated["SQSClient", Depends(sqs_client)],
    settings: Annotated[Settings, Depends(get_settings)],
    x_request_id: Annotated[str | None, Header(alias="X-Request-Id")] = None,
) -> CreateRunResponse:
    """Persist the run, publish to the queue, and return immediately.

    202 Accepted: no AI work happens inline. Callers poll the run endpoint.
    """
    return runs_service.create_run(
        session, payload, x_request_id, s3=s3, sqs=sqs, settings=settings
    )


@router.get(
    "/report-processing-runs/{run_id}",
    response_model=RunResponse,
    summary="Batch progress and per-report stage",
)
def get_run(
    run_id: uuid.UUID,
    session: Annotated[Session, Depends(get_session)],
) -> RunResponse:
    return runs_service.get_run(session, run_id)


@router.delete(
    "/report-processing-runs/{run_id}",
    response_model=CancelRunResponse,
    summary="Cancel unfinished items in a run",
)
def cancel_run(
    run_id: uuid.UUID,
    session: Annotated[Session, Depends(get_session)],
    response: Response,
) -> CancelRunResponse:
    result = runs_service.cancel_run(session, run_id)
    if not result.cancelled_item_ids:
        # Nothing was in flight. Still a success, but say so distinctly.
        response.status_code = status.HTTP_200_OK
    return result
