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
    DocumentStatusResponse,
    DocumentType,
    NameCandidatesRequest,
    NameCandidatesResponse,
    NameCheck,
    RetryResponse,
)
from app.schemas.runs import CreateRunRequest, SubmittedDocument
from app.services import filing, identity, names
from app.services import runs as runs_service
from app.services.classification import (
    DOCUMENT_TYPE_BY_SECTION,
    SECTION_BY_DOCUMENT_TYPE,
    DocumentSection,
)

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


def get_document_status(session: Session, document_id: int) -> DocumentStatusResponse:
    """Where a document has got to, and the type its result will be readable under.

    Untyped on purpose, and the one route that is: a caller cannot know the type before
    classification decides it, so requiring it here would make the endpoint unusable for
    the state it exists to report. Safe to leave untyped because nothing extracted is
    returned bar the name verdict, which a mismatched document has nowhere else to be read
    from — see ``DocumentStatusResponse``.
    """
    item = _latest_item(session, document_id)
    if item is None:
        raise ApiError(404, "no_ai_result", "No AI result exists for this document")

    clf = _classification(session, item.id)
    return DocumentStatusResponse(
        document_id=document_id,
        item_id=item.id,
        run_id=item.run_id,
        status=item.status,
        # Absent from the map for a section with no addressable type, and null before
        # classification has run at all. Both mean "no result URL to build yet".
        document_type=DOCUMENT_TYPE_BY_SECTION.get(clf.section) if clf is not None else None,
        last_error_code=item.last_error_code,
        section_row_id=item.section_row_id,
        filed_section=item.filed_section,
        # Null while no verdict exists — distinct from a verdict of `unknown`. A mismatched
        # document is never filed, so its `content` row does not exist and this is the only
        # place the app can read the printed name from.
        name_check=(
            NameCheck(
                verdict=clf.name_match,
                document_name=clf.patient_name,
                confirmed=clf.identity_confirmed_at is not None,
            )
            if clf is not None and clf.name_match is not None
            else None
        ),
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
        section_row_id=item.section_row_id,
        last_error_code=item.last_error_code,
        intended_section=item.intended_section,
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

    A completed document is left alone — its result is final. An in-flight document is
    already being worked on. Everything else is re-submitted through ``create_run``, which
    validates the source afresh, creates a new item, and publishes it.

    Whether the document was already filed into its section table makes no difference:
    filing deletes the intake row, but the item's ``source_key`` still points at the
    object, so ``create_run`` resolves it either way. A document that failed *after* being
    filed is precisely the one a retry is for.
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
            "Document already completed; its result is final",
            {"section_row_id": item.section_row_id},
        )

    result = runs_service.create_run(
        session,
        CreateRunRequest(
            documents=[
                SubmittedDocument(
                    document_id=document_id,
                    # Preserve the user's original choice: without it a retry of a
                    # mismatched document would be processed as if uploaded globally.
                    intended_section=(
                        DocumentSection(item.intended_section) if item.intended_section else None
                    ),
                )
            ]
        ),
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


def refile_document(
    session: Session,
    document_id: int,
    request_id: str | None,
    *,
    s3: "S3Client",
    sqs: "SQSClient",
    settings: "Settings",
) -> RetryResponse:
    """Accept this service's classification: move the document there and process it.

    The one action offered on a document flagged ``section_mismatch`` — filed where the
    user put it, with nothing read from it. There is no general section-to-section mover
    and this is not one: the destination is always the section we detected, because that is
    the only one we have an opinion about.

    Refused for anything else. A document that was actually processed is not movable at
    all: its stored results describe the section it is in, and moving it would leave a lab
    report's insights attached to a row in Insurance.

    The move and the reprocessing are separate steps on purpose. Once the row has moved and
    the run item points at it, an ordinary submission does the rest — ``_source_keys``
    resolves the new key from that item, and ``filing._adopt_prior_filing`` finds a prior
    filing whose section now matches, so the stages update the moved row rather than filing
    a second copy. ``intended_section`` is deliberately not carried over: the user has just
    accepted our reading, so there is no competing intent left to compare against.
    """
    item = _latest_item(session, document_id)
    if item is None:
        raise ApiError(404, "no_ai_result", "No AI result exists for this document")

    if item.section_row_id is None or item.filed_section is None:
        raise ApiError(
            409,
            "not_filed",
            "This document has not been filed into a section",
            {"status": item.status},
        )

    clf = _classification(session, item.id)
    if clf is None:
        raise ApiError(409, "not_classified_yet", "This document has not been classified yet")

    if clf.section == item.filed_section:
        raise ApiError(
            409,
            "already_in_detected_section",
            "This document is already in the section it was classified as",
            {"filed_section": item.filed_section},
        )

    detected = DocumentSection(clf.section)
    if detected not in filing.SECTION_TABLES:
        # `medical_condition`, `unknown`: we never file these, so there is nowhere to move
        # it TO. The app does not offer the action for them either.
        raise ApiError(
            409,
            "section_not_filable",
            f"Documents classified as {clf.section} are not filed by this service",
            {"detected_section": clf.section},
        )

    filing.refile_to_detected(
        session,
        s3,
        item_id=item.id,
        section_row_id=item.section_row_id,
        from_section=DocumentSection(item.filed_section),
        to_section=detected,
        bucket=settings.s3_bucket,
    )

    result = runs_service.create_run(
        session,
        CreateRunRequest(documents=[SubmittedDocument(document_id=document_id)]),
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


def confirm_identity(
    session: Session,
    document_id: int,
    request_id: str | None,
    *,
    s3: "S3Client",
    sqs: "SQSClient",
    settings: "Settings",
) -> RetryResponse:
    """Accept a document the user says is theirs despite the name printed on it.

    Mirrors ``refile_document``: record the decision, then re-submit through the ordinary
    path. ``identity.settled_verdict`` sees the confirmation on the next pass and the gate
    lets the document through, so nothing here needs to know how the gate works.

    Offered only on a document actually waiting on that question — the gate leaves
    ``name_mismatch`` as the item's error code, and that flag is the permission, exactly as
    ``section_mismatch`` is for refiling.
    """
    item = _latest_item(session, document_id)
    if item is None:
        raise ApiError(404, "no_ai_result", "No AI result exists for this document")

    if item.last_error_code != "name_mismatch":
        raise ApiError(
            409,
            "not_awaiting_identity",
            "This document is not waiting on an identity decision",
            {"status": item.status, "last_error_code": item.last_error_code},
        )

    if not identity.confirm_identity(session, document_id):
        raise ApiError(409, "not_classified_yet", "This document has not been classified yet")

    result = runs_service.create_run(
        session,
        CreateRunRequest(
            documents=[
                SubmittedDocument(
                    document_id=document_id,
                    # Carried over for the same reason retry does: the user answered a
                    # question about WHOSE the document is, not about where it belongs.
                    # Dropping it would silently re-process a sectioned upload as a global
                    # one.
                    intended_section=(
                        DocumentSection(item.intended_section) if item.intended_section else None
                    ),
                )
            ]
        ),
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


def name_candidates(
    session: Session, document_id: int, payload: NameCandidatesRequest
) -> NameCandidatesResponse:
    """Which of these people the name on the document matches.

    **A pure string comparison over a list the caller supplies.** This service does not
    read ``family_connect`` or any other family table, and makes no access decision. Spring
    has already filtered the list to people the caller may write to, and re-checks that on
    the move itself — which is why this endpoint takes a list of candidates rather than a
    user id. Do not "helpfully" add a family query here: a second implementation of the
    access rules would drift from Spring's, and a drift bug leaks one family member's
    records to another.

    Reads only. An unreadable or absent document name matches nobody, rather than fanning
    an unknown name out across a family.
    """
    item = _latest_item(session, document_id)
    if item is None:
        raise ApiError(404, "no_ai_result", "No AI result exists for this document")

    clf = _classification(session, item.id)
    return NameCandidatesResponse(
        matches=names.matches_any(
            clf.patient_name if clf is not None else None,
            {candidate.user_id: candidate.name for candidate in payload.candidates},
        )
    )
