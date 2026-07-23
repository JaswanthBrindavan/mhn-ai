"""Document classification — the first real pipeline stage.

Flow: download the source object from ``unclassified_files``, ask the model which
MyHealthNotion section it belongs to under a fixed schema, validate the JSON with
Pydantic (never repair it), persist the classification and a process log, then either
continue the pipeline (it is a report) or reject the item with the detected section as
the reason.

The actual move into the ``reports`` table (INSERT the row, write ``reports.content``,
DELETE the ``unclassified_files`` row) happens in the assembly stage after extraction and
insights, so a document is only moved once its content is ready — not here.

Idempotent: the classification and the process log are upserted, so a redelivery that
re-runs the stage overwrites its own prior attempt rather than duplicating rows.
"""

import logging
import time
from decimal import Decimal
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, ValidationError, field_validator
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.integrations.ai.base import (
    AIProviderError,
    DocumentPayload,
    StructuredResponse,
)
from app.integrations.ai.pricing import estimate_cost_usd
from app.integrations.s3 import (
    SourceObjectMissingError,
    SourceObjectUnavailableError,
    get_object,
)
from app.models.ai_results import AiProcessLog, AiReportClassification
from app.models.spring import unclassified_files
from app.services.source_validation import resolve_content_type
from app.workers.stagetypes import RejectStageError, StageContext, TransientStageError

logger = logging.getLogger(__name__)

PROMPT_VERSION = "clf-2026-07-23"
SCHEMA_VERSION = "clf-2"
PROVIDER_NAME = "anthropic"
STAGE_NAME = "classifying"
#: Classification output is small (a section, a title, a short reason). Kept tight to
#: bound cost; the JSON structured-output format keeps responses compact.
CLASSIFY_MAX_TOKENS = 2048


class DocumentSection(StrEnum):
    """A MyHealthNotion section (the ``resource_type_enum`` values), or unknown.

    This is what drives Spring's routing. Only ``REPORTS`` is deep-processed this sprint.
    """

    REPORTS = "reports"
    SCANS_IMAGING = "scans_imaging"  # MRI, X-ray, CT, ultrasound, radiology reports
    PRESCRIPTIONS = "prescriptions"
    INSURANCE = "insurance"
    BILLS = "bills"
    VACCINATIONS = "vaccinations"
    MEDICAL_CONDITION = "medical_condition"
    UNKNOWN = "unknown"  # cannot confidently place -> stays in unclassified_files


#: Only the reports section is moved and deep-processed this sprint. Everything else is
#: recognised, recorded, and left in unclassified_files for future sprints to route.
PROCESSABLE_SECTIONS: frozenset[DocumentSection] = frozenset({DocumentSection.REPORTS})


class DocumentClassification(BaseModel):
    """Validated model output. Written to the DB only after this parses cleanly."""

    section: DocumentSection
    title: str = Field(min_length=1, max_length=512)
    confidence: float
    reasoning: str = Field(default="", max_length=2000)

    @field_validator("confidence")
    @classmethod
    def _clamp_confidence(cls, value: float) -> float:
        # Confidence is an advisory signal, not a medical fact — clamp a stray 1.02
        # into range rather than failing the whole classification over it.
        return max(0.0, min(1.0, value))


#: Structured-output schema. Hand-written to stay within what json_schema supports
#: (no numeric/length constraints, additionalProperties false, every field required).
CLASSIFICATION_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "section": {
            "type": "string",
            "enum": [member.value for member in DocumentSection],
        },
        "title": {"type": "string"},
        "confidence": {"type": "number"},
        "reasoning": {"type": "string"},
    },
    "required": ["section", "title", "confidence", "reasoning"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = (
    "You are a document classifier in a medical-records intake pipeline. You receive a "
    "single uploaded document and decide which section of the app it belongs to. You do "
    "not diagnose, interpret results, or give medical advice — you only classify.\n\n"
    "Choose exactly one section:\n"
    "- reports: a diagnostic report a clinician files as a result — laboratory report, "
    "pathology report, or a clinical/discharge summary.\n"
    "- scans_imaging: imaging and its radiology report — MRI, X-ray, CT, ultrasound, and "
    "the radiologist's read of them.\n"
    "- prescriptions: a prescription or medication order.\n"
    "- insurance: insurance cards, policies, claims, or coverage letters.\n"
    "- bills: invoices, receipts, or billing statements.\n"
    "- vaccinations: immunisation or vaccination records.\n"
    "- medical_condition: a record describing a diagnosed condition or its history.\n"
    "- unknown: use ONLY when the document is unreadable or you cannot confidently place "
    "it in any section.\n\n"
    "Also return:\n"
    "- title: a short, human-readable label, at most a few words (for example "
    "'Complete Blood Count' or 'Chest X-Ray'). Do not invent details; do not include "
    "long patient identifiers.\n"
    "- confidence: your calibrated confidence between 0 and 1.\n"
    "- reasoning: one concise sentence citing what drove the decision. Do not restate "
    "patient data or clinical values.\n\n"
    "Be conservative: if the document is unreadable or genuinely ambiguous, choose "
    "'unknown' rather than guessing a section."
)

INSTRUCTION = "Classify the attached document into one section."


def classify_report(ctx: StageContext) -> None:
    """Stage entrypoint: classify the document, persist, and gate the pipeline."""
    filepath = _document_filepath(ctx)
    document = _load_document(ctx, filepath)

    started = time.perf_counter()
    try:
        response = ctx.ai.analyze_document(
            document=document,
            system=SYSTEM_PROMPT,
            instruction=INSTRUCTION,
            json_schema=CLASSIFICATION_JSON_SCHEMA,
            max_tokens=CLASSIFY_MAX_TOKENS,
        )
    except AIProviderError as exc:
        _log(
            ctx,
            outcome="error",
            error_code="ai_provider_error",
            detail=str(exc),
            duration_ms=_elapsed_ms(started),
        )
        raise TransientStageError(f"classification provider error: {exc}") from exc

    duration_ms = _elapsed_ms(started)

    if response.refused:
        _log(
            ctx,
            outcome="refused",
            error_code="model_refusal",
            response=response,
            duration_ms=duration_ms,
        )
        raise TransientStageError("classification refused by safety classifier")

    try:
        result = DocumentClassification.model_validate_json(response.text)
    except ValidationError as exc:
        # Never repair invalid model output — record the failure and let it retry.
        _log(
            ctx,
            outcome="validation_failed",
            error_code="invalid_model_output",
            detail=_sanitize_validation_error(exc),
            response=response,
            duration_ms=duration_ms,
        )
        raise TransientStageError("classification output failed validation") from exc

    _persist_classification(ctx, result)

    if result.section not in PROCESSABLE_SECTIONS:
        # Correctly classified, just not a report this sprint routes/processes. The
        # detected section is the reason; the document stays in unclassified_files.
        reason = result.section.value
        _log(ctx, outcome="rejected", error_code=reason, response=response, duration_ms=duration_ms)
        raise RejectStageError(reason, f"Document classified as {reason}, not a report")

    _log(ctx, outcome="succeeded", response=response, duration_ms=duration_ms)


# --- helpers ----------------------------------------------------------------


def _document_filepath(ctx: StageContext) -> str:
    filepath = ctx.session.execute(
        select(unclassified_files.c.filepath).where(unclassified_files.c.id == ctx.document_id)
    ).scalar_one_or_none()
    if not filepath:
        # The item should not exist without a source document, but be defensive.
        raise RejectStageError("source_document_missing", "Source document no longer exists")
    return str(filepath)


def _load_document(ctx: StageContext, filepath: str) -> DocumentPayload:
    try:
        content = get_object(ctx.s3, ctx.settings.s3_bucket, filepath)
    except SourceObjectMissingError as exc:
        raise RejectStageError("source_object_missing", "Source file was not found") from exc
    except SourceObjectUnavailableError as exc:
        raise TransientStageError(f"source storage unavailable: {exc}") from exc

    content_type = resolve_content_type(content.metadata)
    if content_type is None or content_type not in ctx.settings.allowed_content_type_set:
        # Validated at submit, but the object could have changed underneath us.
        raise RejectStageError("unsupported_content_type", "Source file type is not supported")

    return DocumentPayload(data=content.data, content_type=content_type, filename=filepath)


def _persist_classification(ctx: StageContext, result: DocumentClassification) -> None:
    stmt = (
        pg_insert(AiReportClassification)
        .values(
            run_item_id=ctx.item_id,
            document_id=ctx.document_id,
            section=result.section.value,
            title=result.title,
            confidence=result.confidence,
            reasoning=result.reasoning or None,
            prompt_version=PROMPT_VERSION,
            schema_version=SCHEMA_VERSION,
        )
        .on_conflict_do_update(
            index_elements=[AiReportClassification.run_item_id],
            set_={
                "document_id": ctx.document_id,
                "section": result.section.value,
                "title": result.title,
                "confidence": result.confidence,
                "reasoning": result.reasoning or None,
                "prompt_version": PROMPT_VERSION,
                "schema_version": SCHEMA_VERSION,
            },
        )
    )
    ctx.session.execute(stmt)
    ctx.session.commit()


def _log(
    ctx: StageContext,
    *,
    outcome: str,
    duration_ms: int,
    response: StructuredResponse | None = None,
    error_code: str | None = None,
    detail: str | None = None,
) -> None:
    usage = response.usage if response is not None else None
    model = response.model if response is not None else (ctx.settings.ai_model or "unknown")
    cost = estimate_cost_usd(model, usage) if usage is not None else Decimal("0")

    stmt = (
        pg_insert(AiProcessLog)
        .values(
            run_item_id=ctx.item_id,
            document_id=ctx.document_id,
            stage=STAGE_NAME,
            attempt=ctx.attempt,
            provider=PROVIDER_NAME,
            model=model,
            prompt_version=PROMPT_VERSION,
            schema_version=SCHEMA_VERSION,
            input_tokens=usage.input_tokens if usage else 0,
            output_tokens=usage.output_tokens if usage else 0,
            cache_read_input_tokens=usage.cache_read_input_tokens if usage else 0,
            cache_creation_input_tokens=usage.cache_creation_input_tokens if usage else 0,
            estimated_cost_usd=cost,
            duration_ms=duration_ms,
            outcome=outcome,
            error_code=error_code,
            error_detail=detail,
        )
        # One row per (item, stage, attempt): a re-run of the SAME attempt updates it,
        # so a single attempt's cost is never logged twice.
        .on_conflict_do_update(
            constraint="uq_ai_process_logs_item_stage_attempt",
            set_={
                "model": model,
                "input_tokens": usage.input_tokens if usage else 0,
                "output_tokens": usage.output_tokens if usage else 0,
                "cache_read_input_tokens": usage.cache_read_input_tokens if usage else 0,
                "cache_creation_input_tokens": usage.cache_creation_input_tokens if usage else 0,
                "estimated_cost_usd": cost,
                "duration_ms": duration_ms,
                "outcome": outcome,
                "error_code": error_code,
                "error_detail": detail,
            },
        )
    )
    ctx.session.execute(stmt)
    ctx.session.commit()


def _sanitize_validation_error(exc: ValidationError) -> str:
    # Field locations and messages only — never the offending model output/value,
    # which could echo report contents.
    parts = [f"{'.'.join(str(p) for p in err['loc'])}: {err['type']}" for err in exc.errors()]
    return "; ".join(parts)[:2000]


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)
