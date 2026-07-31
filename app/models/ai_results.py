"""Result and audit tables written by the worker's AI stages.

``ai_report_classifications`` holds the classification result (one per run item).
``ai_section_extractions`` holds the fields transcribed from a non-report section
(insurance, scans/imaging, vaccinations), one per run item.
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
    Index,
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
    #: source_context, value_numeric, abnormal_flag, range_source, matched_parameter,
    #: matched_group, normalized_value, normalized_unit, normalized}, ... ],
    #: "report_date": ..., "patient_age": ..., "patient_gender": ...}
    #: range_source is "ideal_range" when an approved-THP age-group range drove the flag,
    #: else "report_range" (the report's own printed range).
    data: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)

    prompt_version: Mapped[str] = mapped_column(String(32), nullable=False)
    schema_version: Mapped[str] = mapped_column(String(32), nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )


class AiReportInsight(Base):
    """Informational insights generated for one run item.

    Built from the validated extraction (not the raw file), so insights cannot introduce
    values that bypassed extraction. Informational only — never a diagnosis, emergency
    instruction, or medical certainty (enforced by the prompt and validated shape). One
    row per item, upserted; the ``data`` JSONB is assembled into ``reports.content``.
    """

    __tablename__ = "ai_report_insights"

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

    #: {"insights": [{heading, body, related_tests}, ...], "summary": ..., "disclaimer": ...}
    data: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)

    prompt_version: Mapped[str] = mapped_column(String(32), nullable=False)
    schema_version: Mapped[str] = mapped_column(String(32), nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )


class AiSectionExtraction(Base):
    """Structured extraction for a NON-report section (insurance, scans, vaccinations).

    Kept separate from ``ai_report_extractions`` because the shape differs per section
    and none of it is lab results: that table's ``data`` is a normalised result set with
    abnormal flags, this one holds a section's own fields. ``section`` records which
    shape ``data`` carries, so a reader never has to infer it.

    ``data`` is {"section": ..., "fields": {...}, "flags": [...]} — the validated fields
    with dates normalised to ISO in Python, plus any data-quality flags (e.g. a policy
    end date preceding its start). One row per item; a re-run upserts it.
    """

    __tablename__ = "ai_section_extractions"

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

    #: The resource_type value this extraction is for: insurance | scans_imaging |
    #: vaccinations. Determines the shape of ``data["fields"]``.
    section: Mapped[str] = mapped_column(String(32), nullable=False)

    data: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)

    prompt_version: Mapped[str] = mapped_column(String(32), nullable=False)
    schema_version: Mapped[str] = mapped_column(String(32), nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (Index("ix_ai_section_extractions_section", "section"),)


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


class AiThpFallback(Base):
    """R&D worklist: one row per extracted test where we fell back to the report's own
    reference range because no approved-THP ideal range applied.

    ``reason`` tells R&D what to fix — ``unmatched`` (the test is not a known parameter),
    ``unapproved`` (the parameter exists but isn't doctor-approved), or ``no_ideal_range``
    (approved, but no ideal range for the patient's age group). Many rows per run item;
    a re-run replaces this item's rows (delete-then-insert), so there are no duplicates.
    Never surfaced in ``reports.content`` — internal curation signal only.
    """

    __tablename__ = "ai_thp_fallbacks"

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

    test_name: Mapped[str] = mapped_column(String(256), nullable=False)
    #: The approved/known parameter this matched, if any (null when reason is unmatched).
    matched_parameter: Mapped[str | None] = mapped_column(String(256), nullable=True)
    #: The most-specific age-group key we tried (null when there was no demographic to try).
    group_attempted: Mapped[str | None] = mapped_column(String(64), nullable=True)
    #: unmatched | unapproved | no_ideal_range
    reason: Mapped[str] = mapped_column(String(32), nullable=False)

    #: Demographics the report gave us (context for R&D; not identifiers).
    patient_age: Mapped[str | None] = mapped_column(String(32), nullable=True)
    patient_gender: Mapped[str | None] = mapped_column(String(32), nullable=True)
    #: The report's own printed range that we fell back to (for R&D to sanity-check).
    report_reference_range: Mapped[str | None] = mapped_column(String(128), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        Index("ix_ai_thp_fallbacks_reason", "reason"),
        Index("ix_ai_thp_fallbacks_matched_parameter", "matched_parameter"),
        Index("ix_ai_thp_fallbacks_run_item_id", "run_item_id"),
    )
