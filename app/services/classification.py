"""Report auto-classification — the first real pipeline stage.

Flow: download the source object, ask the model to classify it under a fixed schema,
validate the JSON with Pydantic (never repair it), persist the result and a process
log, then either continue the pipeline (it is a processable report) or reject the item
with a clear reason (it is not).

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

from app.integrations.ai.base import AIProviderError, DocumentPayload, StructuredResponse
from app.integrations.ai.pricing import estimate_cost_usd
from app.integrations.s3 import (
    SourceObjectMissingError,
    SourceObjectUnavailableError,
    get_object,
)
from app.models.ai_results import AiProcessLog, AiReportClassification
from app.models.spring import reports
from app.services.source_validation import resolve_content_type
from app.workers.stagetypes import RejectStageError, StageContext, TransientStageError

logger = logging.getLogger(__name__)

PROMPT_VERSION = "clf-2026-07-23"
SCHEMA_VERSION = "clf-1"
PROVIDER_NAME = "anthropic"
STAGE_NAME = "classifying"
#: Classification output is small (a label, a title, a short reason). Kept tight to
#: bound cost; the JSON structured-output format keeps responses compact.
CLASSIFY_MAX_TOKENS = 2048


class DocumentType(StrEnum):
    LAB_REPORT = "lab_report"
    PATHOLOGY_REPORT = "pathology_report"
    RADIOLOGY_REPORT = "radiology_report"
    DISCHARGE_SUMMARY = "discharge_summary"
    PRESCRIPTION = "prescription"
    MEDICAL_INVOICE = "medical_invoice"
    INSURANCE_DOCUMENT = "insurance_document"
    OTHER_MEDICAL = "other_medical"
    NON_MEDICAL = "non_medical"
    UNKNOWN = "unknown"


#: The document types this sprint actually processes (textual diagnostic reports).
#: Everything else is a wrong document type and gets rejected with a reason.
PROCESSABLE_TYPES: frozenset[DocumentType] = frozenset(
    {
        DocumentType.LAB_REPORT,
        DocumentType.PATHOLOGY_REPORT,
        DocumentType.RADIOLOGY_REPORT,
        DocumentType.DISCHARGE_SUMMARY,
    }
)


class ReportClassification(BaseModel):
    """Validated model output. Written to the DB only after this parses cleanly."""

    is_report: bool
    document_type: DocumentType
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
        "is_report": {"type": "boolean"},
        "document_type": {
            "type": "string",
            "enum": [member.value for member in DocumentType],
        },
        "title": {"type": "string"},
        "confidence": {"type": "number"},
        "reasoning": {"type": "string"},
    },
    "required": ["is_report", "document_type", "title", "confidence", "reasoning"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = (
    "You are a document classifier in a medical-records intake pipeline. You receive a "
    "single uploaded document and decide what kind of document it is. You do not "
    "diagnose, interpret results, or give medical advice — you only classify.\n\n"
    "Return your answer in the required structured format:\n"
    "- is_report: true only if the document is a medical diagnostic report a clinician "
    "would file as a result — for example a laboratory report, pathology report, "
    "radiology/imaging report, or discharge summary. It is false for prescriptions, "
    "bills or invoices, insurance paperwork, appointment letters, marketing, blank "
    "forms, or anything non-medical.\n"
    "- document_type: the single best-fitting category from the allowed list. Use "
    "'unknown' only when the document is too unclear or unreadable to categorise.\n"
    "- title: a short, human-readable label for the document, at most a few words "
    "(for example 'Complete Blood Count' or 'Lipid Panel Report'). Do not invent "
    "details that are not present; do not include long patient identifiers.\n"
    "- confidence: your calibrated confidence between 0 and 1.\n"
    "- reasoning: one concise sentence citing what in the document drove the decision. "
    "Do not restate patient data or clinical values.\n\n"
    "Be conservative: if the document is unreadable or you cannot tell what it is, set "
    "is_report to false and document_type to 'unknown' rather than guessing."
)

INSTRUCTION = "Classify the attached document."


def classify_report(ctx: StageContext) -> None:
    """Stage entrypoint: classify the report, persist, and gate the pipeline."""
    filepath = _report_filepath(ctx)
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
        duration_ms = _elapsed_ms(started)
        _log(
            ctx,
            outcome="error",
            error_code="ai_provider_error",
            detail=str(exc),
            duration_ms=duration_ms,
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
        result = ReportClassification.model_validate_json(response.text)
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

    processable = result.is_report and result.document_type in PROCESSABLE_TYPES
    if not processable:
        reason = _rejection_reason(result)
        _log(ctx, outcome="rejected", error_code=reason, response=response, duration_ms=duration_ms)
        raise RejectStageError(reason, _rejection_message(result))

    _log(ctx, outcome="succeeded", response=response, duration_ms=duration_ms)


# --- helpers ----------------------------------------------------------------


def _report_filepath(ctx: StageContext) -> str:
    filepath = ctx.session.execute(
        select(reports.c.filepath).where(reports.c.id == ctx.report_id)
    ).scalar_one_or_none()
    if not filepath:
        # The item should not exist without a report, but be defensive.
        raise RejectStageError("source_report_missing", "Report record no longer exists")
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


def _persist_classification(ctx: StageContext, result: ReportClassification) -> None:
    stmt = (
        pg_insert(AiReportClassification)
        .values(
            run_item_id=ctx.item_id,
            report_id=ctx.report_id,
            is_report=result.is_report,
            document_type=result.document_type.value,
            title=result.title,
            confidence=result.confidence,
            reasoning=result.reasoning or None,
            prompt_version=PROMPT_VERSION,
            schema_version=SCHEMA_VERSION,
        )
        .on_conflict_do_update(
            index_elements=[AiReportClassification.run_item_id],
            set_={
                "report_id": ctx.report_id,
                "is_report": result.is_report,
                "document_type": result.document_type.value,
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
            report_id=ctx.report_id,
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


def _rejection_reason(result: ReportClassification) -> str:
    if not result.is_report:
        return "not_a_report"
    return "wrong_document_type"


def _rejection_message(result: ReportClassification) -> str:
    return f"Document classified as {result.document_type.value}, not a processable report"


def _sanitize_validation_error(exc: ValidationError) -> str:
    # Field locations and messages only — never the offending model output/value,
    # which could echo report contents.
    parts = [f"{'.'.join(str(p) for p in err['loc'])}: {err['type']}" for err in exc.errors()]
    return "; ".join(parts)[:2000]


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)
