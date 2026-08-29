"""Per-document AI-result endpoints.

The unit is an uploaded document (an ``unclassified_files`` id): read its AI result, or
retry it if it did not complete. Handlers stay thin; logic lives in
``app.services.results``. Authentication is applied by the parent router.

Both routes name the document type — ``/documents/reports/{id}/ai-result`` — and answer
only when the document really was classified as that type, so a caller reading an insurance
policy can never be handed a lab report's values by a wrong id.

The caller learns which type to use either from ``GET /v1/documents/{id}/status`` or from
``document_type`` on the run. There is exactly one way to read a result and one way to
retry — both typed. The status route is the sole untyped one, and may be: it returns
lifecycle state only, never anything extracted.

Submission stays type-agnostic, because the type is something this service *detects* — see
``DocumentType``.
"""

from typing import TYPE_CHECKING, Annotated, Any

from fastapi import APIRouter, Depends, Header, status
from sqlalchemy.orm import Session

from app.api.deps import s3_client, sqs_client
from app.core.config import Settings, get_settings
from app.core.db import get_session
from app.schemas.results import (
    DocumentAiResult,
    DocumentStatusResponse,
    DocumentType,
    NameCandidatesRequest,
    NameCandidatesResponse,
    NameChecksRequest,
    NameChecksResponse,
    RetryResponse,
)
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
    "/documents/{document_id}/status",
    response_model=DocumentStatusResponse,
    summary="Where a document has got to, and the type its result is readable under",
    responses=_NOT_FOUND,
)
def get_document_status(
    document_id: int,
    session: Annotated[Session, Depends(get_session)],
) -> DocumentStatusResponse:
    """The one untyped route. Two path segments, so it cannot shadow the three-segment
    result routes below."""
    return results_service.get_document_status(session, document_id)


@router.post(
    "/documents/name-checks",
    response_model=NameChecksResponse,
    summary="Identity verdicts for a set of documents, in one call",
)
def get_name_checks(
    request: NameChecksRequest,
    session: Annotated[Session, Depends(get_session)],
) -> NameChecksResponse:
    """Answers a LIST screen: which of these documents are waiting on their owner.

    A plural route rather than N calls to ``/status``, because a wallet list holds several
    intake rows and the alternative turns the most-hit screen in the app into an N+1 across
    a service boundary.

    Deliberately narrower than ``/status``: verdicts only, **no printed names**. A list
    needs to know a decision is waiting, not who the document names -- and that name is the
    most identifying field a document has, so it stays on the one screen that asks the
    question. Documents with no verdict are omitted rather than returned null.

    Two path segments, and every other document route has three, so "name-checks" cannot
    be read as a document id whatever the declaration order.
    """
    return results_service.get_name_checks(session, request)


@router.post(
    "/documents/{document_id}/refile",
    response_model=RetryResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Move a mismatched document to the section it was classified as, and process it",
    responses={
        **_NOT_FOUND,
        409: {"description": "Not filed, not classified, already there, or not filable"},
    },
)
def refile_document(
    document_id: int,
    session: Annotated[Session, Depends(get_session)],
    s3: Annotated["S3Client", Depends(s3_client)],
    sqs: Annotated["SQSClient", Depends(sqs_client)],
    settings: Annotated[Settings, Depends(get_settings)],
    x_request_id: Annotated[str | None, Header(alias="X-Request-Id")] = None,
) -> RetryResponse:
    """The one action offered on a document flagged ``section_mismatch``.

    Untyped for the same reason ``/status`` is, and more so: the caller is acting *on* a
    disagreement about the type, so it cannot be made to name one first. It returns nothing
    extracted — an item id and a status — so leaving it untyped discloses nothing.
    """
    return results_service.refile_document(
        session, document_id, x_request_id, s3=s3, sqs=sqs, settings=settings
    )


@router.post(
    "/documents/{document_id}/analyze",
    response_model=RetryResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Run the AI stages on a document that was filed and left unread",
    responses={
        **_NOT_FOUND,
        409: {"description": "Not filed, already analysed, or not waiting to be"},
    },
)
def analyze_document(
    document_id: int,
    session: Annotated[Session, Depends(get_session)],
    s3: Annotated["S3Client", Depends(s3_client)],
    sqs: Annotated["SQSClient", Depends(sqs_client)],
    settings: Annotated[Settings, Depends(get_settings)],
    x_request_id: Annotated[str | None, Header(alias="X-Request-Id")] = None,
) -> RetryResponse:
    """ "Read this one now" — the answer to a document filed but deliberately left unread.

    Untyped for the same reason ``/refile`` and ``/confirm-identity`` are: the caller is
    acting *on* a document rather than naming a type, and nothing extracted comes back —
    an item id and a status.
    """
    return results_service.analyze_document(
        session, document_id, x_request_id, s3=s3, sqs=sqs, settings=settings
    )


@router.post(
    "/documents/{document_id}/confirm-identity",
    response_model=RetryResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Accept a name-mismatched document as the account holder's own, and process it",
    responses={
        **_NOT_FOUND,
        409: {"description": "Not waiting on an identity decision, or not classified yet"},
    },
)
def confirm_identity(
    document_id: int,
    session: Annotated[Session, Depends(get_session)],
    s3: Annotated["S3Client", Depends(s3_client)],
    sqs: Annotated["SQSClient", Depends(sqs_client)],
    settings: Annotated[Settings, Depends(get_settings)],
    x_request_id: Annotated[str | None, Header(alias="X-Request-Id")] = None,
) -> RetryResponse:
    """The "yes, this document is mine" answer to a name mismatch.

    Untyped for the same reason ``/refile`` is: the caller is acting *on* a question about
    the document, so it cannot be made to name a type first, and nothing extracted is
    returned — an item id and a status.
    """
    return results_service.confirm_identity(
        session, document_id, x_request_id, s3=s3, sqs=sqs, settings=settings
    )


@router.post(
    "/documents/{document_id}/name-candidates",
    response_model=NameCandidatesResponse,
    summary="Which of the supplied people the name on this document matches",
    responses=_NOT_FOUND,
)
def name_candidates(
    document_id: int,
    payload: NameCandidatesRequest,
    session: Annotated[Session, Depends(get_session)],
) -> NameCandidatesResponse:
    """The "which of my family is this?" answer, by comparing strings and nothing more.

    The candidate list comes from Spring, already filtered to people the caller may write
    to; this service reads no family table and makes no access decision. Reads only.
    """
    return results_service.name_candidates(session, document_id, payload)


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
    "/documents/{document_type}/{document_id}/ai-result/retry",
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
