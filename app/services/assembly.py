"""Assemble the ``content`` payload of a filed document from the per-stage results.

Each stage persisted its own result to an ``ai_*`` table; this reads them back and builds
the user-facing JSON written into the section row's ``content``. It is built more than once
over a document's life — at filing, before any extraction exists, and again when the
pipeline ends — so the payload carries an explicit ``state`` saying which moment it is. The
AI payload lives under a dedicated ``ai`` key so any keys Spring also writes on that row are
never clobbered.

Pure read + shape: no writes here. The transactional filing move (INSERT the section row,
record ``section_row_id``, DELETE ``unclassified_files``) and the later ``content`` update
both live in ``app.services.filing``.
"""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.ai_results import (
    AiReportClassification,
    AiReportExtraction,
    AiReportInsight,
    AiSectionExtraction,
)
from app.models.processing import AiProcessingRunItem


class ContentState(StrEnum):
    """How far the pipeline has got, as recorded in ``content.ai.state``.

    Mandatory rather than inferred: ``insights`` is permanently null for a non-report
    section, so the app cannot tell "still working" from "nothing to show" by looking for
    null fields.
    """

    #: Filed, classification recorded, nothing extracted yet.
    CLASSIFIED = "classified"
    #: The pipeline finished.
    COMPLETE = "complete"
    #: Filed, but the pipeline did not finish (failed, gave up, or was cancelled).
    FAILED = "failed"


#: Version of the content["ai"] shape, so consumers can migrate on change.
#:
#: 2.1 — insights became multi-part rather than a single `body`. Later gained `document_id`,
#:       the intake id, so the app can act on a filed document (`/refile`) after the intake
#:       row it came from has been deleted. The shape moved again
#:       afterwards, to the app's six fields (heading + risk_patterns render Risk Patterns,
#:       suggestion_heading + suggestions render Suggestions, what_it_is / why_it_varies the
#:       explanatory body) WITHOUT another bump, because no consumer keys off this value.
#: 2.0 — the payload is now written at filing time, before extraction has run, and gained
#:       `state` and `section_extraction`. 1.1 was report-only and written once, at the move.
#:
#: Nothing branches on it today: mhn-react types it as an opaque string and Spring never
#: reads it. That is why the drift above was harmless — and why it is worth knowing before
#: anyone starts treating it as a contract.
CONTENT_SCHEMA_VERSION = "2.1"


def build_content(session: Session, item_id: UUID, *, state: ContentState) -> dict[str, Any]:
    """The ``content`` payload for a filed document, at whatever stage it has reached.

    Called three times over a document's life: at filing (``classified``), on completion
    (``complete``), and when a filed document ends without finishing (``failed``). Whichever
    result rows exist populate their key; the section comes from the classification row, so
    this does not need telling which kind of document it is.

    ``state`` is a required argument and deliberately not inferred from which keys are
    populated: ``insights`` is *permanently* null for a non-report section (insurance,
    scans and vaccinations are transcription-only and have no insights stage), so a filed
    vaccination record that is finished looks identical to one still mid-pipeline. Without
    the explicit state the app would spin forever waiting for a field that will never
    arrive — do not "simplify" it away.

    ``extraction`` and ``section_extraction`` are mutually exclusive by construction: a
    report writes ``ai_report_extractions``, a non-report section writes
    ``ai_section_extractions``, never both. They stay separate keys because they carry
    different shapes and a consumer must never read an insurance policy's fields as a lab
    result set.
    """
    clf = session.execute(
        select(
            AiReportClassification.section,
            AiReportClassification.title,
            AiReportClassification.confidence,
            AiReportClassification.patient_name,
            AiReportClassification.name_match,
            AiReportClassification.identity_confirmed_at,
        ).where(AiReportClassification.run_item_id == item_id)
    ).one_or_none()

    extraction = session.execute(
        select(AiReportExtraction.data).where(AiReportExtraction.run_item_id == item_id)
    ).scalar_one_or_none()

    section_extraction = session.execute(
        select(AiSectionExtraction.data).where(AiSectionExtraction.run_item_id == item_id)
    ).scalar_one_or_none()

    insights = session.execute(
        select(AiReportInsight.data).where(AiReportInsight.run_item_id == item_id)
    ).scalar_one_or_none()

    # The intake id, carried on the payload because filing DELETES the intake row: from
    # then on nothing Spring holds addresses this document in our contract. `/refile` and
    # `/status` are both keyed on it, and the app reaches a filed document only through
    # its section row. Read it here rather than making the caller pass it — every caller
    # would have to thread the same value that is already one column away.
    document_id = session.execute(
        select(AiProcessingRunItem.document_id).where(AiProcessingRunItem.id == item_id)
    ).scalar_one_or_none()

    classification = (
        {"section": clf.section, "title": clf.title, "confidence": float(clf.confidence)}
        if clf is not None
        else None
    )

    # Null rather than a dict of nulls when no verdict exists: a document classified before
    # the name check ran, and one never classified at all, both mean "we have not looked" —
    # which the app must be able to tell apart from a verdict of `unknown`, meaning we
    # looked and the document printed no name. The account holder's own name is left out on
    # purpose; the client knows who is logged in and this payload travels further.
    name_check = (
        {
            "verdict": clf.name_match,
            "document_name": clf.patient_name,
            "confirmed": clf.identity_confirmed_at is not None,
        }
        if clf is not None and clf.name_match is not None
        else None
    )

    return {
        "ai": {
            "schema_version": CONTENT_SCHEMA_VERSION,
            "state": state.value,
            "document_id": document_id,
            "classification": classification,
            "name_check": name_check,
            "extraction": extraction,
            "section_extraction": section_extraction,
            "insights": insights,
            "generated_at": datetime.now(UTC).isoformat(),
        }
    }
