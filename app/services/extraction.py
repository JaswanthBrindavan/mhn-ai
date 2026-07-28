"""Structured extraction — the second pipeline stage (runs after a report classifies).

Flow: reload the source object, ask the model for the report's lab results under a fixed
schema, validate the JSON with Pydantic (never repair it), apply deterministic
normalisation in Python (abnormal-range flags and curated unit conversion — never the
model), then persist the enriched result to ``ai_report_extractions`` and a process log.

Idempotent: the extraction row and the process log are upserted, so a redelivery that
re-runs the stage overwrites its own prior attempt rather than duplicating rows.
"""

import logging
import time
from typing import Any

from pydantic import BaseModel, Field, ValidationError
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.integrations.ai.base import AIProviderError
from app.models.ai_results import AiReportExtraction
from app.services import ideal_ranges, normalization
from app.services.ai_logging import elapsed_ms, log_process, sanitize_validation_error
from app.services.source_loading import load_source_document
from app.services.thp_fallback import FallbackEntry, record_fallbacks
from app.workers.stagetypes import StageContext, TransientStageError

logger = logging.getLogger(__name__)

PROMPT_VERSION = "ext-2026-07-27"
SCHEMA_VERSION = "ext-2"
STAGE_NAME = "extracting"
#: Reports can carry many analytes; give the model room but keep it bounded.
EXTRACT_MAX_TOKENS = 8192


class ExtractedLabResult(BaseModel):
    """One extracted result, exactly as read from the document. Values stay as text —
    Python parses and flags them; the model never does arithmetic."""

    test_name: str = Field(min_length=1, max_length=256)
    value: str | None = Field(default=None, max_length=128)
    unit: str | None = Field(default=None, max_length=64)
    reference_range: str | None = Field(default=None, max_length=128)
    observed_date: str | None = Field(default=None, max_length=64)
    source_context: str | None = Field(default=None, max_length=512)


class DocumentExtraction(BaseModel):
    """Validated model output. An empty ``results`` list is valid (e.g. a summary)."""

    results: list[ExtractedLabResult]
    report_date: str | None = Field(default=None, max_length=64)
    #: Patient demographics as printed on the report (free text: "23", "6 months", "M").
    #: Drive the age-group ideal-range lookup; parsed in Python, never by the model.
    patient_age: str | None = Field(default=None, max_length=32)
    patient_gender: str | None = Field(default=None, max_length=32)


#: Structured-output schema. Same constraints as classification: no numeric/length
#: constraints, additionalProperties false, every field required (nullable via a union).
_NULLABLE_STR = {"type": ["string", "null"]}
EXTRACTION_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "test_name": {"type": "string"},
                    "value": _NULLABLE_STR,
                    "unit": _NULLABLE_STR,
                    "reference_range": _NULLABLE_STR,
                    "observed_date": _NULLABLE_STR,
                    "source_context": _NULLABLE_STR,
                },
                "required": [
                    "test_name",
                    "value",
                    "unit",
                    "reference_range",
                    "observed_date",
                    "source_context",
                ],
                "additionalProperties": False,
            },
        },
        "report_date": _NULLABLE_STR,
        "patient_age": _NULLABLE_STR,
        "patient_gender": _NULLABLE_STR,
    },
    "required": ["results", "report_date", "patient_age", "patient_gender"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = (
    "You extract structured data from a medical laboratory or diagnostic report. You do "
    "not diagnose, interpret, or judge whether a value is normal — you only transcribe "
    "what the document states.\n\n"
    "For every test result in the document, return an object with:\n"
    "- test_name: the analyte or test name exactly as printed (for example "
    "'Fasting Glucose', 'Hemoglobin').\n"
    "- value: the measured result as printed, as a string. Copy it verbatim including any "
    "'<' or '>' — do NOT convert, round, or compute anything.\n"
    "- unit: the unit as printed (for example 'mg/dL', 'g/dL'), or null.\n"
    "- reference_range: the reference/normal range as printed (for example '3.5-5.0', "
    "'< 200'), or null. Do NOT decide whether the value is in range.\n"
    "- observed_date: the date this specimen/result is dated, if shown, or null.\n"
    "- source_context: a short snippet of surrounding label text that identifies the row, "
    "or null. Do not copy patient identifiers.\n\n"
    "Also return report_date: the report's overall date if shown, else null.\n"
    "Also return patient_age and patient_gender exactly as printed (for example '23', "
    "'6 months', 'M', 'Female'), or null if not shown. These demographics select the "
    "correct reference range; they are not patient identifiers, so returning them is "
    "expected. Do not infer or compute them.\n\n"
    "Transcribe only what is present. If the document has no tabular results (for example "
    "a discharge summary), return an empty results list. Never invent values or ranges."
)

INSTRUCTION = "Extract every lab/test result from the attached report as structured data."


def extract_report(ctx: StageContext) -> None:
    """Stage entrypoint: extract, normalise deterministically, and persist."""
    document = load_source_document(ctx)

    started = time.perf_counter()
    try:
        response = ctx.ai.analyze_document(
            document=document,
            system=SYSTEM_PROMPT,
            instruction=INSTRUCTION,
            json_schema=EXTRACTION_JSON_SCHEMA,
            max_tokens=EXTRACT_MAX_TOKENS,
        )
    except AIProviderError as exc:
        _log(
            ctx,
            outcome="error",
            error_code="ai_provider_error",
            detail=str(exc),
            duration_ms=elapsed_ms(started),
        )
        raise TransientStageError(f"extraction provider error: {exc}") from exc

    duration_ms = elapsed_ms(started)

    if response.refused:
        _log(
            ctx,
            outcome="refused",
            error_code="model_refusal",
            response=response,
            duration_ms=duration_ms,
        )
        raise TransientStageError("extraction refused by safety classifier")

    try:
        result = DocumentExtraction.model_validate_json(response.text)
    except ValidationError as exc:
        # Never repair invalid model output — record the failure and let it retry.
        _log(
            ctx,
            outcome="validation_failed",
            error_code="invalid_model_output",
            detail=sanitize_validation_error(exc),
            response=response,
            duration_ms=duration_ms,
        )
        raise TransientStageError("extraction output failed validation") from exc

    # Resolve approved-THP ideal ranges (age-group) when enabled; otherwise behaviour is
    # exactly as before (report's own reference range drives the flag).
    if ctx.settings.ideal_ranges_enabled:
        lookup: ideal_ranges.Lookup | None = ideal_ranges.load_lookup(ctx.session)
        ladder = ideal_ranges.build_group_ladder(result.patient_age, result.patient_gender)
        demographics = ideal_ranges.has_demographics(result.patient_age, result.patient_gender)
    else:
        lookup, ladder, demographics = None, [], False

    payload, fallbacks = _normalize(result, lookup, ladder, demographics)
    _persist_extraction(ctx, payload)
    if lookup is not None:
        # Always call (even with no fallbacks) to clear a prior attempt's worklist rows.
        record_fallbacks(ctx, fallbacks)
    _log(ctx, outcome="succeeded", response=response, duration_ms=duration_ms)


# --- helpers ----------------------------------------------------------------


def _normalize(
    result: DocumentExtraction,
    lookup: ideal_ranges.Lookup | None,
    ladder: list[str],
    demographics_present: bool,
) -> tuple[dict[str, Any], list[FallbackEntry]]:
    """Enrich each result deterministically, applying the approved ideal-range override
    when one resolves. Returns the persisted payload and any R&D worklist fallbacks."""
    enriched: list[dict[str, Any]] = []
    fallbacks: list[FallbackEntry] = []

    for r in result.results:
        data = r.model_dump()
        if lookup is None:  # feature off — unchanged behaviour, no worklist
            enriched.append(normalization.enrich_result(data, gender=result.patient_gender))
            continue

        res = ideal_ranges.resolve(data["test_name"], lookup, ladder)
        if res.bounds is not None:
            enriched.append(
                normalization.enrich_result(
                    data,
                    override_bounds=res.bounds,
                    matched_parameter=res.matched_parameter,
                    matched_group=res.matched_group,
                    gender=result.patient_gender,
                )
            )
            continue

        enriched.append(normalization.enrich_result(data, gender=result.patient_gender))
        # Log real curation gaps (unmatched/unapproved always; no_ideal_range only when the
        # report actually gave demographics — otherwise the gap is the report's, not R&D's).
        if res.reason != "no_ideal_range" or demographics_present:
            fallbacks.append(
                FallbackEntry(
                    test_name=data["test_name"],
                    reason=res.reason or "no_ideal_range",
                    matched_parameter=res.matched_parameter,
                    group_attempted=ladder[0] if ladder else None,
                    patient_age=result.patient_age,
                    patient_gender=result.patient_gender,
                    report_reference_range=data.get("reference_range"),
                )
            )

    payload = {
        "results": enriched,
        "report_date": result.report_date,
        "patient_age": result.patient_age,
        "patient_gender": result.patient_gender,
    }
    return payload, fallbacks


def _persist_extraction(ctx: StageContext, payload: dict[str, Any]) -> None:
    stmt = (
        pg_insert(AiReportExtraction)
        .values(
            run_item_id=ctx.item_id,
            document_id=ctx.document_id,
            data=payload,
            prompt_version=PROMPT_VERSION,
            schema_version=SCHEMA_VERSION,
        )
        .on_conflict_do_update(
            index_elements=[AiReportExtraction.run_item_id],
            set_={
                "document_id": ctx.document_id,
                "data": payload,
                "prompt_version": PROMPT_VERSION,
                "schema_version": SCHEMA_VERSION,
            },
        )
    )
    ctx.session.execute(stmt)
    ctx.session.commit()


def _log(ctx: StageContext, **kwargs: Any) -> None:
    log_process(
        ctx,
        stage=STAGE_NAME,
        prompt_version=PROMPT_VERSION,
        schema_version=SCHEMA_VERSION,
        **kwargs,
    )
