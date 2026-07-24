"""The report processing pipeline: classify -> extract -> generate insights.

* classify          -> implemented (app.services.classification)
* extract/normalise -> implemented (app.services.extraction)
* generate insights -> implemented (app.services.insights)

Shared types (``StageContext``, ``TransientStageError``, ``RejectStageError``) live in
``app.workers.stagetypes`` and are re-exported here for existing importers. Stage
bodies, when filled in, must stay idempotent — a redelivered message re-runs the whole
sequence, so a stage upserts its results rather than appending.
"""

from app.models.enums import RunItemStatus
from app.services.classification import classify_report
from app.services.extraction import extract_report
from app.services.insights import generate_insights
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


#: (status to move into before running, stage callable). The order IS the pipeline.
STAGE_SEQUENCE: list[tuple[RunItemStatus, Stage]] = [
    (RunItemStatus.CLASSIFYING, classify_report),
    (RunItemStatus.EXTRACTING, extract_report),
    (RunItemStatus.GENERATING_INSIGHTS, generate_insights),
]
