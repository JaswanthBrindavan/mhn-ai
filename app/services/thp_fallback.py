"""Persist the R&D worklist: tests that fell back to the report's own reference range.

The single seam for the fallback log. Delete-then-insert per run item keeps it idempotent
under SQS redelivery (mirrors ``_persist_extraction``'s upsert). If the worklist ever moves
to a REST push, only this module changes.
"""

from dataclasses import dataclass

from sqlalchemy import delete, insert

from app.models.ai_results import AiThpFallback
from app.workers.stagetypes import StageContext


@dataclass(frozen=True)
class FallbackEntry:
    """One extracted test for which no approved-THP ideal range applied."""

    test_name: str
    reason: str  # unmatched | unapproved | no_ideal_range | unit_mismatch
    matched_parameter: str | None = None
    group_attempted: str | None = None
    patient_age: str | None = None
    patient_gender: str | None = None
    report_reference_range: str | None = None
    #: What the report printed the value in. The fix for a unit_mismatch row is to add this
    #: to the parameter's alternate units, so it has to be in the worklist.
    report_unit: str | None = None


def record_fallbacks(ctx: StageContext, entries: list[FallbackEntry]) -> None:
    """Replace this item's worklist rows. Always clears first, so a redelivery that
    resolves differently (or produces no fallbacks) leaves no stale rows."""
    ctx.session.execute(delete(AiThpFallback).where(AiThpFallback.run_item_id == ctx.item_id))
    if entries:
        ctx.session.execute(
            insert(AiThpFallback),
            [
                {
                    "run_item_id": ctx.item_id,
                    "document_id": ctx.document_id,
                    "test_name": e.test_name,
                    "matched_parameter": e.matched_parameter,
                    "group_attempted": e.group_attempted,
                    "reason": e.reason,
                    "patient_age": e.patient_age,
                    "patient_gender": e.patient_gender,
                    "report_reference_range": e.report_reference_range,
                    "report_unit": e.report_unit,
                }
                for e in entries
            ],
        )
    ctx.session.commit()
