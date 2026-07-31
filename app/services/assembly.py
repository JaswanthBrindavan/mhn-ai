"""Assemble the final ``reports.content`` payload from the per-stage results.

Each stage persisted its own result to an ``ai_*`` table; this reads them back and
builds the single user-facing JSON that gets written into ``reports.content`` when the
document is moved. The AI payload lives under a dedicated ``ai`` key so any keys Spring
also writes on that row are never clobbered.

Pure read + shape: no writes here. The transactional move (INSERT ``reports``, record
``reports_id``, DELETE ``unclassified_files``, mark completed) is in
``app.services.processing.move_and_complete``.
"""

from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.ai_results import (
    AiReportClassification,
    AiReportExtraction,
    AiReportInsight,
)

#: Version of the reports.content["ai"] shape, so consumers can migrate on change.
#: 1.1 — extraction results gained range_source/matched_parameter/matched_group and the
#: extraction payload gained patient_age/patient_gender (approved-THP ideal-range override).
#: The field set is unchanged since; matched_group now carries the THP age bracket ("18-60")
#: rather than a demographic group name, which no stored payload ever used — the override
#: has never run outside tests.
CONTENT_SCHEMA_VERSION = "1.1"


def build_content(session: Session, item_id: UUID) -> dict[str, Any]:
    """The ``reports.content`` payload for a completed report item."""
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
            "classification": classification,
            "extraction": extraction,
            "insights": insights,
            "generated_at": datetime.now(UTC).isoformat(),
        }
    }
