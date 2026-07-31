"""Request and response models for document-processing runs.

The submitted unit of work is an uploaded document (an ``unclassified_files`` id).
``reports_id`` on an item is set only once a document is classified as a report and
moved into the ``reports`` table.
"""

import uuid
from datetime import datetime
from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.results import DocumentType

MAX_DOCUMENTS_PER_RUN = 500


class SubmitOutcome(StrEnum):
    """What happened to each document id in a submission."""

    CREATED = "created"
    #: An in-flight item already existed; it was reused rather than duplicated.
    REUSED = "reused"
    #: Already completed and force_reprocess was not set, so nothing was re-run.
    ALREADY_COMPLETED = "already_completed"


class CreateRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_ids: Annotated[
        list[int],
        Field(
            min_length=1,
            max_length=MAX_DOCUMENTS_PER_RUN,
            description="unclassified_files ids to classify and process",
        ),
    ]

    # AUDIT ONLY. This service performs no user-level authorization; Spring has
    # already made that decision. See app/api/deps.py.
    requested_by_user_id: uuid.UUID | None = Field(
        default=None,
        description="Recorded for audit. Never used for access control.",
    )

    force_reprocess: bool = Field(
        default=False,
        description="Re-run documents that already completed. Ignored for in-flight items.",
    )


class RunItemResponse(BaseModel):
    # populate_by_name lets the field be built either from an ORM object (whose
    # attribute is `id`) or from a dict keyed by the public name.
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    #: Exposed as `item_id`; read from the model's `id` attribute.
    item_id: uuid.UUID = Field(validation_alias="id")
    document_id: int
    #: The reports row created if this document was moved into the reports section.
    reports_id: int | None = None
    #: Which `/v1/documents/{document_type}/...` route reads this document's result.
    #: Null until it is classified, and for a section with no addressable type
    #: (`bills`, `medical_condition`, `unknown` — none of which produce an AI result).
    document_type: DocumentType | None = None
    status: str
    attempt_count: int
    last_error_code: str | None = None
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None


class SubmittedItem(BaseModel):
    document_id: int
    item_id: uuid.UUID
    #: `queued` once the message is on the queue. Stays `pending` if publishing
    #: failed — the item is durable and the stale-item sweep will retry it.
    status: str
    outcome: SubmitOutcome
    #: Set when the item was rejected at submit, e.g. unsupported_content_type.
    error_code: str | None = None


class CreateRunResponse(BaseModel):
    run_id: uuid.UUID
    created_at: datetime
    items: list[SubmittedItem]


class RunProgress(BaseModel):
    total: int
    pending: int = 0
    queued: int = 0
    processing: int = 0
    classifying: int = 0
    extracting: int = 0
    generating_insights: int = 0
    completed: int = 0
    failed: int = 0
    rejected: int = 0
    cancelled: int = 0


class RunResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    run_id: uuid.UUID
    caller: str
    requested_by_user_id: uuid.UUID | None
    force_reprocess: bool
    created_at: datetime
    updated_at: datetime
    #: True when no item is still in flight.
    finished: bool
    progress: RunProgress
    items: list[RunItemResponse]


class CancelRunResponse(BaseModel):
    run_id: uuid.UUID
    cancelled_item_ids: list[uuid.UUID]
    #: Items already terminal, which cancellation left untouched.
    unaffected_item_ids: list[uuid.UUID]
