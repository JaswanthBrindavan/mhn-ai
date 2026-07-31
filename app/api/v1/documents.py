"""Per-document AI-result endpoints.

The unit is an uploaded document (an ``unclassified_files`` id): read its AI result, or
retry it if it did not complete. Handlers stay thin; logic lives in
``app.services.results``. Authentication is applied by the parent router.

Both routes name the document type — ``/documents/reports/{id}/ai-result`` — and answer
only when the document really was classified as that type, so a caller reading an insurance
policy can never be handed a lab report's values by a wrong id.

The caller learns which type to use from ``document_type`` on the run it already polls
(``GET /v1/document-processing-runs/{run_id}``); there is no untyped variant, so there is
exactly one way to read a result and exactly one way to retry.

Submission stays type-agnostic, because the type is something this service *detects* — see
``DocumentType``.
"""

from typing import TYPE_CHECKING, Annotated, Any

from fastapi import APIRouter, Depends, Header, status
from sqlalchemy.orm import Session

from app.api.deps import s3_client, sqs_client
from app.core.config import Settings, get_settings
from app.core.db import get_session
from app.schemas.results import DocumentAiResult, DocumentType, RetryResponse
from app.services import results as results_service

if TYPE_CHECKING:  # pragma: no cover - typing only
    from mypy_boto3_s3.client import S3Client
    from mypy_boto3_sqs.client import SQSClient

router = APIRouter(tags=["documents"])

_NOT_FOUND: dict[int | str, dict[str, Any]] = {
    404: {"description": "No AI result exists for this document"}
}
_WRONG_TYPE: dict[int | str, dict[str, Any]] = {
    409: {"description": "Classified as a different type, or not classified yet"}
}


@router.get(
    "/documents/{document_type}/{document_id}/ai-result",
    response_model=DocumentAiResult,
    summary="AI result for a document of a known type",
    responses={**_NOT_FOUND, **_WRONG_TYPE},
)
def get_typed_ai_result(
    document_type: DocumentType,
    document_id: int,
    session: Annotated[Session, Depends(get_session)],
) -> DocumentAiResult:
    return results_service.get_document_ai_result(session, document_id, document_type=document_type)


@router.post(
    "/documents/{document_type}/{document_id}/ai-result:retry",
    response_model=RetryResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Retry a document of a known type that did not complete",
    responses={
        **_NOT_FOUND,
        409: {"description": "Wrong type, not classified yet, completed, or in progress"},
    },
)
def retry_typed_ai_result(
    document_type: DocumentType,
    document_id: int,
    session: Annotated[Session, Depends(get_session)],
    s3: Annotated["S3Client", Depends(s3_client)],
    sqs: Annotated["SQSClient", Depends(sqs_client)],
    settings: Annotated[Settings, Depends(get_settings)],
    x_request_id: Annotated[str | None, Header(alias="X-Request-Id")] = None,
) -> RetryResponse:
    return results_service.retry_document(
        session,
        document_id,
        x_request_id,
        s3=s3,
        sqs=sqs,
        settings=settings,
        document_type=document_type,
    )
