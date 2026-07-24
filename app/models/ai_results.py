"""Result and audit tables written by the worker's AI stages.

``ai_report_classifications`` holds the classification result (one per run item).
``ai_process_logs`` records every model call — provider, model, prompt/schema version,
tokens, estimated cost, duration, outcome, and sanitized failure data — keyed by
(run_item_id, stage, attempt) so a retry never double-logs a single attempt's cost.
"""

import uuid
from datetime import datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.core.db import Base


class AiReportClassification(Base):
    """Classification result for one run item: the detected section, title, confidence."""

    __tablename__ = "ai_report_classifications"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
        default=uuid.uuid4,
    )
    # One classification per item; a re-run upserts this row rather than appending.
    run_item_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("ai_processing_run_items.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )
    #: The source document (an unclassified_files id).
    document_id: Mapped[int] = mapped_column(Integer, nullable=False)

    #: The MyHealthNotion section the model placed this document in (a resource_type
    #: value, or "unknown"). This is what drives routing.
    section: Mapped[str] = mapped_column(String(32), nullable=False)
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    confidence: Mapped[float] = mapped_column(Numeric(4, 3), nullable=False)
    #: Short model justification, kept for audit. Not user-facing and not a diagnosis.
    reasoning: Mapped[str | None] = mapped_column(Text, nullable=True)

    prompt_version: Mapped[str] = mapped_column(String(32), nullable=False)
    schema_version: Mapped[str] = mapped_column(String(32), nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )


class AiReportExtraction(Base):
    """Structured extraction result for one run item.

    The validated lab data (test name, value, unit, reference range, observed date,
    source context) plus the deterministic normalisation computed in Python (numeric
    value, abnormal/out-of-range flag, converted value/unit) is stored as a single JSONB
    ``data`` payload. One row per item; a re-run upserts it. Kept separate from the final
    ``reports.content`` so extraction and insights can be assembled independently.
    """

    __tablename__ = "ai_report_extractions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
        default=uuid.uuid4,
    )
    run_item_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("ai_processing_run_items.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )
    #: The source document (an unclassified_files id).
    document_id: Mapped[int] = mapped_column(Integer, nullable=False)

    #: {"results": [ {test_name, value, unit, reference_range, observed_date,
    #: source_context, value_numeric, abnormal_flag, normalized_value, normalized_unit,
    #: normalized}, ... ], "report_date": ...}
    data: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)

    prompt_version: Mapped[str] = mapped_column(String(32), nullable=False)
    schema_version: Mapped[str] = mapped_column(String(32), nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )


class AiProcessLog(Base):
    """One model call's provenance and cost. Never stores report contents or prompts."""

    __tablename__ = "ai_process_logs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
        default=uuid.uuid4,
    )
    run_item_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("ai_processing_run_items.id", ondelete="CASCADE"),
        nullable=False,
    )
    #: The source document (an unclassified_files id).
    document_id: Mapped[int] = mapped_column(Integer, nullable=False)

    stage: Mapped[str] = mapped_column(String(32), nullable=False)
    #: Which processing attempt produced this call. Each retry is a real, separately
    #: billed call, so each gets its own row — the unique constraint only prevents
    #: double-logging within one attempt.
    attempt: Mapped[int] = mapped_column(Integer, nullable=False)

    provider: Mapped[str] = mapped_column(String(64), nullable=False)
    model: Mapped[str] = mapped_column(String(128), nullable=False)
    prompt_version: Mapped[str] = mapped_column(String(32), nullable=False)
    schema_version: Mapped[str] = mapped_column(String(32), nullable=False)

    input_tokens: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    output_tokens: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    cache_read_input_tokens: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )
    cache_creation_input_tokens: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )
    estimated_cost_usd: Mapped[Decimal] = mapped_column(
        Numeric(12, 6), nullable=False, server_default=text("0")
    )
    duration_ms: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))

    #: succeeded | rejected | validation_failed | refused | error
    outcome: Mapped[str] = mapped_column(String(32), nullable=False)
    error_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    #: Sanitized detail only — field names/messages, never model output or report text.
    error_detail: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        UniqueConstraint(
            "run_item_id", "stage", "attempt", name="uq_ai_process_logs_item_stage_attempt"
        ),
    )
