"""Processing run and run-item tables.

A **run** is one submission from Spring (possibly many documents). A **run item** is the
per-document unit of work and carries all lifecycle state. Its identity is the source
``unclassified_files`` id (``document_id``); once a document is classified as a report and
moved into ``reports``, that created row's id is recorded in ``reports_id``.

The run has no denormalised status column: progress is derived by counting item
statuses at read time. A stored aggregate would need updating from every worker on
every transition, and would drift the first time one of those updates was missed.
"""

import uuid
from datetime import datetime

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.db import Base
from app.models.enums import ACTIVE_STATUSES, RunItemStatus

_STATUS_VALUES = ", ".join(f"'{status.value}'" for status in RunItemStatus)
_ACTIVE_VALUES = ", ".join(f"'{status.value}'" for status in sorted(ACTIVE_STATUSES))

#: The exact predicate of `uq_ai_run_items_active_report`. ON CONFLICT must repeat an
#: index's predicate verbatim to target a partial index, so both are built from here
#: rather than written out twice and allowed to drift.
ACTIVE_STATUS_PREDICATE = f"status IN ({_ACTIVE_VALUES})"


class AiProcessingRun(Base):
    """One submission from Spring."""

    __tablename__ = "ai_processing_runs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
        default=uuid.uuid4,
    )

    # AUDIT ONLY. This service performs no user-level authorization -- Spring has
    # already decided. Never compare this to reports.user_id: that column is the
    # report's subject, while family-connect lets a relative be the uploader.
    # See app/api/deps.py for the full reasoning.
    requested_by_user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), nullable=True
    )
    #: Which service submitted this run. Only "spring" today.
    caller: Mapped[str] = mapped_column(String(64), nullable=False, default="spring")
    #: Correlation id supplied by the caller, for tracing across services.
    request_id: Mapped[str | None] = mapped_column(String(128), nullable=True)

    force_reprocess: Mapped[bool] = mapped_column(
        nullable=False, server_default=text("false"), default=False
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )

    items: Mapped[list["AiProcessingRunItem"]] = relationship(
        back_populates="run", cascade="all, delete-orphan", lazy="selectin"
    )

    __table_args__ = (Index("ix_ai_processing_runs_created_at", "created_at"),)


class AiProcessingRunItem(Base):
    """Per-document unit of work. Holds all lifecycle state."""

    __tablename__ = "ai_processing_run_items"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
        default=uuid.uuid4,
    )
    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("ai_processing_runs.id", ondelete="CASCADE"),
        nullable=False,
    )

    # The source document: an `unclassified_files` id. Integer, no foreign key -- those
    # tables are Spring-owned and a constraint from our table would couple their
    # migrations to ours.
    document_id: Mapped[int] = mapped_column(Integer, nullable=False)
    #: The `reports` row created when this document was moved into the reports section.
    #: Null until the move happens (only for documents classified as reports).
    reports_id: Mapped[int | None] = mapped_column(Integer, nullable=True)

    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default=RunItemStatus.PENDING.value
    )

    attempt_count: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0"), default=0
    )

    #: Checksum of the source object, filled by the worker after download. Combined
    #: with document_id this identifies "this exact file already processed".
    content_hash: Mapped[str | None] = mapped_column(String(128), nullable=True)

    #: Stable, machine-readable failure reason. Never free-form model output.
    last_error_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    #: Sanitised detail. Must never contain report contents, prompts, or credentials.
    last_error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    run: Mapped[AiProcessingRun] = relationship(back_populates="items")

    __table_args__ = (
        CheckConstraint(
            f"status IN ({_STATUS_VALUES})",
            name="ck_ai_processing_run_items_status",
        ),
        # THE idempotency guarantee: at most one in-flight item per document, enforced
        # by the database rather than by application checks that races can slip past.
        Index(
            "uq_ai_run_items_active_document",
            "document_id",
            unique=True,
            postgresql_where=text(ACTIVE_STATUS_PREDICATE),
        ),
        Index("ix_ai_run_items_run_id", "run_id"),
        Index("ix_ai_run_items_document_id", "document_id"),
        Index("ix_ai_run_items_status", "status"),
    )
