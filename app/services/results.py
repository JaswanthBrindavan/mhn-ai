"""Read a document's AI result, and retry a document that did not complete.

The result is keyed by the source ``unclassified_files`` id (``document_id``): find the
latest processing item for it, then gather the per-stage results (all keyed by that
item's id). Retry re-submits a single not-completed document through the same idempotent
submission path, so there is one code path for creating and publishing work.
"""

import uuid
from typing import TYPE_CHECKING

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.errors import ApiError
from app.models.ai_results import (
    AiReportClassification,
    AiReportExtraction,
    AiReportInsight,
    AiSectionExtraction,
)
from app.models.enums import ACTIVE_STATUSES, RunItemStatus
from app.models.processing import AiProcessingRunItem
from app.schemas.results import (
    ClassificationResult,
    DocumentAiResult,
    DocumentType,
    RetryResponse,
)
from app.schemas.runs import CreateRunRequest
from app.services import runs as runs_service
from app.services.classification import SECTION_BY_DOCUMENT_TYPE

if TYPE_CHECKING:  # pragma: no cover - typing only
    from mypy_boto3_s3.client import S3Client
    from mypy_boto3_sqs.client import SQSClient

    from app.core.config import Settings

_ACTIVE = {status.value for status in ACTIVE_STATUSES}


def _latest_item(session: Session, document_id: int) -> AiProcessingRunItem | None:
    return session.execute(
        select(AiProcessingRunItem)
        .where(AiProcessingRunItem.document_id == document_id)
        .order_by(AiProcessingRunItem.created_at.desc())
        .limit(1)
    ).scalar_one_or_none()


def _classification(session: Session, item_id: uuid.UUID) -> AiReportClassification | None:
    return session.execute(
        select(AiReportClassification).where(AiReportClassification.run_item_id == item_id)
    ).scalar_one_or_none()


def _require_type(
    item: AiProcessingRunItem,
    clf: AiReportClassification | None,
    document_type: DocumentType,
    *,
    unclassified_ok: bool = False,
) -> None:
    """Check the type in the URL against the section the document was classified as.

    A mismatch is always refused. Both refusals are 409 rather than 404: the document and
    its result exist — it is the *type* in the path that is wrong or not yet known, and
    Spring must not read either case as "no such document".

    ``unclassified_ok`` covers the case where the document has no classification yet. The
    two routes want opposite answers, deliberately:

    * **Reading** a result under a typed URL asserts the document *is* that type, so
      answering with an unverified type would be the disclosure this route exists to
      prevent. Refused.
    * **Retrying** returns no document data; it re-queues work. Refusing there would block
      the commonest retry of all — a document that failed *during* classification, and so
      has no section precisely because it needs retrying. Allowed, which also keeps retry a
      single endpoint rather than sending that one case somewhere else.
    """
    if clf is None:
        if unclassified_ok:
            return
        raise ApiError(
            409,
            "not_classified_yet",
            "This document has not been classified yet, so its type cannot be confirmed",
            {"status": item.status},
        )
    expected = SECTION_BY_DOCUMENT_TYPE[document_type]
    if clf.section != expected.value:
        raise ApiError(
            409,
            "section_mismatch",
            f"This document was classified as '{clf.section}', not '{document_type.value}'",
            {"detected_section": clf.section, "requested_type": document_type.value},
        )


def get_document_ai_result(
    session: Session, document_id: int, *, document_type: DocumentType
) -> DocumentAiResult:
    item = _latest_item(session, document_id)
    if item is None:
        raise ApiError(404, "no_ai_result", "No AI result exists for this document")

    clf = _classification(session, item.id)
    _require_type(item, clf, document_type)

    extraction_data = session.execute(
        select(AiReportExtraction.data).where(AiReportExtraction.run_item_id == item.id)
    ).scalar_one_or_none()
    insights = session.execute(
        select(AiReportInsight.data).where(AiReportInsight.run_item_id == item.id)
    ).scalar_one_or_none()
    # A non-report section writes here instead of ai_report_extractions — different shape,
    # so it gets its own field rather than being squeezed into `extraction`.
    section_extraction = session.execute(
        select(AiSectionExtraction.data).where(AiSectionExtraction.run_item_id == item.id)
    ).scalar_one_or_none()

    classification = (
        ClassificationResult(
            section=clf.section,
            title=clf.title,
            confidence=float(clf.confidence),
            reasoning=clf.reasoning,
        )
        if clf is not None
        else None
    )

    return DocumentAiResult(
        document_id=document_id,
        item_id=item.id,
        run_id=item.run_id,
        status=item.status,
        reports_id=item.reports_id,
        last_error_code=item.last_error_code,
        classification=classification,
        extraction=extraction_data,
        insights=insights,
        section_extraction=section_extraction,
    )


def retry_document(
    session: Session,
    document_id: int,
    request_id: str | None,
    *,
    s3: "S3Client",
    sqs: "SQSClient",
    settings: "Settings",
    document_type: DocumentType,
) -> RetryResponse:
    """Re-run a document that did not complete (failed / rejected / cancelled).

    A completed document is left alone — its result is final and its source has already
    been moved into ``reports``, so there is nothing to reprocess. An in-flight document
    is already being worked on. Everything else is re-submitted through ``create_run``,
    which validates the source afresh, creates a new item, and publishes it.
    """
    item = _latest_item(session, document_id)
    if item is None:
        raise ApiError(404, "no_ai_result", "No AI result exists for this document to retry")

    # The type is checked before the status: a wrong type in the path is the caller
    # addressing the wrong document, which is worth saying plainly even when the document
    # also happens to be completed or in flight.
    _require_type(item, _classification(session, item.id), document_type, unclassified_ok=True)

    if item.status in _ACTIVE:
        raise ApiError(
            409,
            "already_in_progress",
            "Document is already being processed",
            {"item_id": str(item.id), "status": item.status},
        )
    if item.status == RunItemStatus.COMPLETED.value:
        raise ApiError(
            409,
            "already_completed",
            "Document already completed; its result is final and the source has been moved",
            {"reports_id": item.reports_id},
        )

    result = runs_service.create_run(
        session,
        CreateRunRequest(document_ids=[document_id]),
        request_id,
        s3=s3,
        sqs=sqs,
        settings=settings,
    )
    submitted = result.items[0]
    return RetryResponse(
        document_id=document_id,
        item_id=submitted.item_id,
        run_id=result.run_id,
        status=submitted.status,
    )
