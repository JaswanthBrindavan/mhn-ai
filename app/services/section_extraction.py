"""Section extraction — the stage that handles non-report documents.

The report pipeline is classify -> extract -> insights. This is the whole post-filing
pipeline for the three sections that are transcribed rather than interpreted — insurance,
scans/imaging and vaccinations — writing their fields to ``ai_section_extractions`` and
stopping. There is no insights stage for them: there is nothing clinical to interpret.
``prescriptions`` has its own stage rather than a spec here — ``app.services.prescriptions``
sends the document itself rather than its OCR text, because the dosing sits in a column
beside the medicine and flattening that puts a dose on the wrong row. The sections that
remain rejected are those that are manual-upload-only by product decision (``bills``,
``medical_condition``).

Flow: read this item's classification to learn the section, reload the source object,
extract its TEXT (embedded layer first, Tesseract OCR for image-only pages), ask the
model for that section's fields under a fixed schema, validate with Pydantic (never
repaired), normalise dates in Python (never the model), then persist and log.

The model is given the extracted text, not the file — the opposite of what the report
pipeline does. ``app.services.ocr`` carries that argument and what it costs; the part
that matters here is that OCR provenance is stored in the payload, so a missing field
can be traced to a bad scan rather than blamed on the model.

Dates are normalised here rather than trusted from the model, for the same reason
extraction computes abnormal flags in Python: a deterministic rule beats a prompt. An
unreadable date becomes null rather than a guess, and a section whose dates are
inverted (an end before its start) is recorded as a data-quality flag rather than
silently stored as fact.

Idempotent: the extraction row and the process log are upserted, so a redelivery that
re-runs the stage overwrites its own prior attempt rather than duplicating rows.

Live since the auto-filing work. ``SECTION_PIPELINES`` routes insurance, scans/imaging and
vaccinations here — the branch it once needed — and each runs this stage and stops. The
supported set is derived from ``SECTION_SPECS``, so adding a section is an entry there and
nothing else.
"""

import logging
import time
from typing import Any

from pydantic import BaseModel, ValidationError
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.integrations.ai.base import AIProviderError
from app.models.ai_results import AiReportClassification, AiSectionExtraction
from app.services.ai_logging import elapsed_ms, log_process, sanitize_validation_error
from app.services.classification import DocumentSection
from app.services.dates import in_order, iso_date
from app.services.ocr import ExtractedText, TextExtractionError, extract_text
from app.services.section_specs import INSTRUCTION_PREFIX, SectionSpec, spec_for
from app.services.source_loading import load_source_document
from app.workers.stagetypes import RejectStageError, StageContext, TransientStageError

logger = logging.getLogger(__name__)

PROMPT_VERSION = "sec-2026-07-27"
SCHEMA_VERSION = "sec-1"
STAGE_NAME = "extracting_section"


def extract_section(ctx: StageContext) -> None:
    """Stage entrypoint: transcribe a non-report document's fields for its section."""
    section = _classified_section(ctx)
    spec = _spec_or_reject(section)

    document = load_source_document(ctx)

    try:
        extracted = extract_text(document)
    except TextExtractionError as exc:
        # Unreadable by both the text layer and OCR — no model call is worth making.
        _log(
            ctx,
            outcome="rejected",
            error_code="text_extraction_failed",
            detail=str(exc),
            duration_ms=0,
        )
        raise RejectStageError("text_extraction_failed", str(exc)) from exc

    if not extracted.text.strip():
        # A blank read is not a model failure; say so rather than paying to be told.
        _log(ctx, outcome="rejected", error_code="no_text_extracted", duration_ms=0)
        raise RejectStageError("no_text_extracted", "No readable text in the document")

    started = time.perf_counter()
    try:
        response = ctx.ai.generate_structured(
            system=spec.system_prompt,
            instruction=INSTRUCTION_PREFIX + extracted.text,
            json_schema=spec.json_schema,
            max_tokens=spec.max_tokens,
        )
    except AIProviderError as exc:
        _log(
            ctx,
            outcome="error",
            error_code="ai_provider_error",
            detail=str(exc),
            duration_ms=elapsed_ms(started),
        )
        raise TransientStageError(f"section extraction provider error: {exc}") from exc

    duration_ms = elapsed_ms(started)

    if response.refused:
        _log(
            ctx,
            outcome="refused",
            error_code="model_refusal",
            response=response,
            duration_ms=duration_ms,
        )
        raise TransientStageError("section extraction refused by safety classifier")

    try:
        result = spec.model.model_validate_json(response.text)
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
        raise TransientStageError("section extraction output failed validation") from exc

    payload = build_payload(spec, result, extracted)
    _persist(ctx, section, payload)
    _log(ctx, outcome="succeeded", response=response, duration_ms=duration_ms)


def build_payload(
    spec: SectionSpec, result: BaseModel, extracted: ExtractedText | None = None
) -> dict[str, Any]:
    """The stored shape: validated fields, ISO dates, data-quality flags, OCR provenance.

    Separated from the stage so it can be exercised without a database or a model call.
    """
    fields = result.model_dump()
    for name in spec.date_fields:
        fields[name] = iso_date(fields.get(name))

    payload: dict[str, Any] = {
        "section": spec.section.value,
        "fields": fields,
        "flags": _date_flags(spec, fields),
    }
    if extracted is not None:
        payload["source"] = extracted.as_metadata()
        payload["flags"].extend(_ocr_flags(extracted))
    return payload


# --- helpers ----------------------------------------------------------------


def _date_flags(spec: SectionSpec, fields: dict[str, Any]) -> list[dict[str, str]]:
    """Record inverted date pairs instead of storing them as fact.

    A policy whose end precedes its start is a bad read or a bad document; either way
    the downstream 'is this still active' question cannot be answered from it. Flagging
    keeps the values visible for a human without asserting they are correct.
    """
    flags: list[dict[str, str]] = []
    for earlier, later in spec.date_order:
        if not in_order(fields.get(earlier), fields.get(later)):
            flags.append(
                {
                    "code": "dates_out_of_order",
                    "field": later,
                    "detail": f"{later} precedes {earlier}",
                }
            )
    return flags


#: Below this mean Tesseract confidence the read is unreliable enough that a missing
#: field is more likely a bad scan than a bad model. Flagged, not rejected — a poor scan
#: still yields usable fields, and discarding them helps nobody.
LOW_CONFIDENCE_THRESHOLD = 0.60


def _ocr_flags(extracted: ExtractedText) -> list[dict[str, str]]:
    """Surface a weak read so a reviewer can tell OCR apart from model error."""
    flags: list[dict[str, str]] = []
    confidence = extracted.mean_confidence
    if confidence is not None and confidence < LOW_CONFIDENCE_THRESHOLD:
        flags.append(
            {
                "code": "low_ocr_confidence",
                "field": "",
                "detail": f"mean OCR confidence {confidence:.2f}",
            }
        )
    skipped = [p for p in extracted.pages if p.method == "skipped"]
    if skipped:
        flags.append(
            {
                "code": "pages_not_read",
                "field": "",
                "detail": f"{len(skipped)} page(s) past the OCR cap were not read",
            }
        )
    return flags


def _classified_section(ctx: StageContext) -> DocumentSection:
    """The section this item's classification stage recorded."""
    value = ctx.session.execute(
        select(AiReportClassification.section).where(
            AiReportClassification.run_item_id == ctx.item_id
        )
    ).scalar_one_or_none()
    if value is None:
        # Classification always runs first; a missing row means the pipeline was
        # interrupted. Retry restarts from the top and re-classifies.
        raise TransientStageError("classification result missing for section extraction")
    try:
        return DocumentSection(value)
    except ValueError as exc:
        raise RejectStageError(
            "unknown_section", f"Classification recorded an unknown section {value!r}"
        ) from exc


def _spec_or_reject(section: DocumentSection) -> SectionSpec:
    """This package handles three sections; anything else is terminal, not retried."""
    try:
        return spec_for(section)
    except KeyError as exc:
        raise RejectStageError(section.value, f"No section extractor for {section.value}") from exc


def _persist(ctx: StageContext, section: DocumentSection, payload: dict[str, Any]) -> None:
    stmt = (
        pg_insert(AiSectionExtraction)
        .values(
            run_item_id=ctx.item_id,
            document_id=ctx.document_id,
            section=section.value,
            data=payload,
            prompt_version=PROMPT_VERSION,
            schema_version=SCHEMA_VERSION,
        )
        .on_conflict_do_update(
            index_elements=[AiSectionExtraction.run_item_id],
            set_={
                "document_id": ctx.document_id,
                "section": section.value,
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
