"""Read a document's AI result, and retry a document that did not complete.

The result is keyed by the source ``unclassified_files`` id (``document_id``): find the
latest processing item for it, then gather the per-stage results (all keyed by that
item's id). Retry re-submits a single not-completed document through the same idempotent
submission path, so there is one code path for creating and publishing work.
"""

from typing import TYPE_CHECKING

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.errors import ApiError
from app.models.ai_results import (
    AiReportClassification,
    AiReportExtraction,
    AiReportInsight,
)
from app.models.enums import ACTIVE_STATUSES, RunItemStatus
from app.models.processing import AiProcessingRunItem
from app.schemas.results import ClassificationResult, DocumentAiResult, RetryResponse
from app.schemas.runs import CreateRunRequest
from app.services import runs as runs_service

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


def get_document_ai_result(session: Session, document_id: int) -> DocumentAiResult:
    item = _latest_item(session, document_id)
    if item is None:
        raise ApiError(404, "no_ai_result", "No AI result exists for this document")

    clf = session.execute(
        select(AiReportClassification).where(AiReportClassification.run_item_id == item.id)
    ).scalar_one_or_none()
    extraction = session.execute(
        select(AiReportExtraction.data).where(AiReportExtraction.run_item_id == item.id)
    ).scalar_one_or_none()
    insights = session.execute(
        select(AiReportInsight.data).where(AiReportInsight.run_item_id == item.id)
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
        extraction=extraction,
        insights=insights,
    )


def retry_document(
    session: Session,
    document_id: int,
    request_id: str | None,
    *,
    s3: "S3Client",
    sqs: "SQSClient",
    settings: "Settings",
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
