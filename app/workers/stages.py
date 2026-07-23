"""The report processing pipeline: classify -> extract -> generate insights.

Step 5 wires up the durable execution harness around these stages; the stage
*bodies* are filled in later:

* classify         -> step 6
* extract/normalise -> step 7
* generate insights -> step 8

Each stage is a plain callable taking a :class:`StageContext`. Contract for the
bodies added later:

* **Idempotent.** A redelivered message re-runs the whole sequence, so a stage must
  upsert its results, never append.
* **Raise for transient trouble** (S3 blip, AI provider 5xx/429): raise
  :class:`TransientStageError` so the item is left for redelivery and retried.
* **Raise for a wrong document type**: raise :class:`RejectStageError` so the item becomes
  terminally ``rejected`` with a reason, instead of being retried.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy.orm import Session

from app.core.config import Settings
from app.models.enums import RunItemStatus

if TYPE_CHECKING:  # pragma: no cover - typing only
    from mypy_boto3_s3.client import S3Client

logger = logging.getLogger(__name__)


class TransientStageError(Exception):
    """Recoverable failure: leave the item for redelivery and retry."""


class RejectStageError(Exception):
    """The document is not a processable report. Terminal, not retried."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass
class StageContext:
    item_id: UUID
    run_id: UUID
    report_id: int
    session: Session
    s3: "S3Client"
    settings: Settings


def _classify(ctx: StageContext) -> None:
    """Report vs not-a-report, title, confidence. Body added in step 6."""


def _extract(ctx: StageContext) -> None:
    """Structured lab data + deterministic normalisation. Body added in step 7."""


def _generate_insights(ctx: StageContext) -> None:
    """Informational insights. Body added in step 8."""


Stage = Callable[[StageContext], None]

#: (status to move into before running, stage callable). The order IS the pipeline.
STAGE_SEQUENCE: list[tuple[RunItemStatus, Stage]] = [
    (RunItemStatus.CLASSIFYING, _classify),
    (RunItemStatus.EXTRACTING, _extract),
    (RunItemStatus.GENERATING_INSIGHTS, _generate_insights),
]
