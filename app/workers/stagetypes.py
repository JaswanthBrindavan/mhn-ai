"""Shared types for the processing pipeline.

Split out from ``stages`` so a stage implementation (e.g. the classification service)
can import ``StageContext`` and the stage exceptions without importing ``stages``,
which in turn imports the implementations — that would be a cycle.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy.orm import Session

from app.core.config import Settings
from app.integrations.ai.base import AIProvider

if TYPE_CHECKING:  # pragma: no cover - typing only
    from mypy_boto3_s3.client import S3Client


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
    document_id: int
    #: The document's current S3 key. Read from the run item, NOT from unclassified_files:
    #: filing deletes that row mid-pipeline and later stages still need the object.
    source_key: str
    #: The processing attempt this run belongs to, for per-attempt cost logging.
    attempt: int
    session: Session
    s3: "S3Client"
    ai: AIProvider
    settings: Settings


Stage = Callable[[StageContext], None]
