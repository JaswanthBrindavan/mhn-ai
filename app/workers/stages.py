"""The report processing pipeline: classify -> extract -> generate insights.

Step 6 fills in classification. Extraction and insight generation are still stubs:

* classify          -> implemented (app.services.classification)
* extract/normalise -> step 7
* generate insights -> step 8

Shared types (``StageContext``, ``TransientStageError``, ``RejectStageError``) live in
``app.workers.stagetypes`` and are re-exported here for existing importers. Stage
bodies, when filled in, must stay idempotent — a redelivered message re-runs the whole
sequence, so a stage upserts its results rather than appending.
"""

from app.models.enums import RunItemStatus
from app.services.classification import classify_report
from app.workers.stagetypes import (
    RejectStageError,
    Stage,
    StageContext,
    TransientStageError,
)

__all__ = [
    "STAGE_SEQUENCE",
    "RejectStageError",
    "Stage",
    "StageContext",
    "TransientStageError",
]


def _extract(ctx: StageContext) -> None:
    """Structured lab data + deterministic normalisation. Body added in step 7."""


def _generate_insights(ctx: StageContext) -> None:
    """Informational insights. Body added in step 8."""


#: (status to move into before running, stage callable). The order IS the pipeline.
STAGE_SEQUENCE: list[tuple[RunItemStatus, Stage]] = [
    (RunItemStatus.CLASSIFYING, classify_report),
    (RunItemStatus.EXTRACTING, _extract),
    (RunItemStatus.GENERATING_INSIGHTS, _generate_insights),
]
