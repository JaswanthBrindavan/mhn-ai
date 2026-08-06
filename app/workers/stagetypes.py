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


class PermanentStageError(Exception):
    """A genuine processing failure that retrying cannot fix. Terminal, ends ``failed``.

    Distinct from ``RejectStageError`` because the two mean opposite things to the caller:
    a rejection is *routing* — Spring is explicitly told not to surface it as an error —
    while this is a failure a person may need to act on.

    Distinct from ``TransientStageError`` because some failures are deterministic. A
    response cut off at the token ceiling fails validation identically on every attempt,
    so treating it as transient burns the full attempt cap at full price to arrive at the
    same place.
    """

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
