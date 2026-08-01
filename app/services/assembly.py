"""Assemble the ``content`` payload of a filed document from the per-stage results.

Each stage persisted its own result to an ``ai_*`` table; this reads them back and builds
the user-facing JSON written into the section row's ``content``. It is built more than once
over a document's life — at filing, before any extraction exists, and again when the
pipeline ends — so the payload carries an explicit ``state`` saying which moment it is. The
AI payload lives under a dedicated ``ai`` key so any keys Spring also writes on that row are
never clobbered.

Pure read + shape: no writes here. The transactional move (INSERT ``reports``, record
``section_row_id``, DELETE ``unclassified_files``, mark completed) is in
``app.services.processing.move_and_complete``.
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
#: 2.0 — the payload is now written at filing time, before extraction has run, and gained
#: `state` and `section_extraction`. 1.1 was report-only and written once, at the move.
CONTENT_SCHEMA_VERSION = "2.0"


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

    classification = (
        {"section": clf.section, "title": clf.title, "confidence": float(clf.confidence)}
        if clf is not None
        else None
    )

    return {
        "ai": {
            "schema_version": CONTENT_SCHEMA_VERSION,
            "state": state.value,
            "classification": classification,
            "extraction": extraction,
            "section_extraction": section_extraction,
            "insights": insights,
            "generated_at": datetime.now(UTC).isoformat(),
        }
    }
