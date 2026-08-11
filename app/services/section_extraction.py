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
silently stored as fact. Money goes the same way (``app.services.money``): amounts are
reduced to a bare decimal string and the currency to an ISO code, because a symbol
printed in a column header rarely survives text extraction intact.

Idempotent: the extraction row and the process log are upserted, so a redelivery that
re-runs the stage overwrites its own prior attempt rather than duplicating rows.

Live since the auto-filing work. ``SECTION_PIPELINES`` routes insurance, scans/imaging and
vaccinations here — the branch it once needed — and each runs this stage and stops. The
supported set is derived from ``SECTION_SPECS``, so adding a section is an entry there and
nothing else.
"""

import logging
import time
from functools import partial
from typing import Any

from pydantic import BaseModel, ValidationError
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.integrations.ai.base import AIProviderError
from app.models.ai_results import AiReportClassification, AiSectionExtraction
from app.services.ai_logging import (
    check_response,
    elapsed_ms,
    log_process,
    sanitize_validation_error,
)
from app.services.classification import DocumentSection
from app.services.dates import add_interval, in_order, iso_date
from app.services.money import normalise_amount, normalise_currency
from app.services.ocr import ExtractedText, TextExtractionError, extract_text
from app.services.section_specs import INSTRUCTION_PREFIX, SectionSpec, spec_for
from app.services.source_loading import load_source_document
from app.workers.stagetypes import RejectStageError, StageContext, TransientStageError

logger = logging.getLogger(__name__)

PROMPT_VERSION = "sec-2026-08-11"
#: sec-3 added ``next_due_interval``: a vaccination record stating "due after 4 weeks"
#: rather than a date. The model used to compute that date; Python does now.
SCHEMA_VERSION = "sec-3"
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

    # Refusal (transient) and truncation (permanent) are the same check for every
    # stage, so it lives in one place; partial binds this stage's own log helper.
    check_response(
        response,
        log=partial(_log, ctx),
        duration_ms=duration_ms,
        what="section extraction",
    )

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


def record_section_mismatch(
    ctx: StageContext, filed_section: DocumentSection, detected_section: DocumentSection
) -> None:
    """Record that a document was filed where the USER put it, not where we placed it.

    The document goes to the section the user chose — an upload category or a move out of
    Unclassified is an explicit instruction, and refusing it left the document stranded in
    Unclassified with no working recovery at all. What is withheld is the pipeline: we do
    not transcribe an insurance policy with the scan extractor because someone filed it
    under Scans.

    The flag is what makes that state legible, and it is also the permission: the app shows
    a "Move to <detected>" action only on a document carrying this, so a correctly filed
    one cannot be shuffled around. Written as an ``ai_section_extractions`` row with no
    fields, exactly as ``prescriptions.record_handwritten`` does for a page we deliberately
    did not read — which means the existing content envelope and the app's existing flag
    rendering both work with no change.

    No model call is made, so the process log records ``"skipped"`` for provider and model
    rather than claiming one that never happened.
    """
    payload: dict[str, Any] = {
        "section": filed_section.value,
        "fields": {},
        "flags": [
            {
                "code": "section_mismatch",
                "field": "",
                "detail": (
                    f"This looks like {_article(detected_section)}, not "
                    f"{_article(filed_section)}. It has been saved here because that is "
                    f"where you filed it, and nothing was read from it."
                ),
            }
        ],
    }
    _persist(ctx, filed_section, payload)
    _log(ctx, outcome="succeeded", duration_ms=0)
    logger.info(
        "document_filed_against_classification",
        extra={
            "item_id": str(ctx.item_id),
            "filed_section": filed_section.value,
            "detected_section": detected_section.value,
        },
    )


#: Section names as a person would say them, for the sentence above.
_SECTION_LABEL = {
    DocumentSection.REPORTS: "a lab report",
    DocumentSection.SCANS_IMAGING: "a scan report",
    DocumentSection.INSURANCE: "an insurance document",
    DocumentSection.VACCINATIONS: "a vaccination record",
    DocumentSection.PRESCRIPTIONS: "a prescription",
    DocumentSection.BILLS: "a bill",
    DocumentSection.MEDICAL_CONDITION: "a medical condition record",
    DocumentSection.UNKNOWN: "something we could not identify",
}


def _article(section: DocumentSection) -> str:
    return _SECTION_LABEL.get(section, section.value.replace("_", " "))


def build_payload(
    spec: SectionSpec, result: BaseModel, extracted: ExtractedText | None = None
) -> dict[str, Any]:
    """The stored shape: validated fields, ISO dates, data-quality flags, OCR provenance.

    Separated from the stage so it can be exercised without a database or a model call.
    """
    fields = result.model_dump()
    for name in spec.date_fields:
        fields[name] = iso_date(fields.get(name))
    for name in spec.amount_fields:
        fields[name] = normalise_amount(fields.get(name))
    for name in spec.currency_fields:
        fields[name] = normalise_currency(fields.get(name))
    for target, start, interval in spec.derived_from_interval:
        # A printed date wins; this only fills the gap the model was told to leave.
        if not fields.get(target):
            fields[target] = add_interval(fields.get(start), fields.get(interval))

    flags = _date_flags(spec, fields)
    flags.extend(_drop_unsourced_summary(spec, fields))

    payload: dict[str, Any] = {
        "section": spec.section.value,
        "fields": fields,
        "flags": flags,
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


def _drop_unsourced_summary(spec: SectionSpec, fields: dict[str, Any]) -> list[dict[str, str]]:
    """Delete a patient-facing summary that has nothing behind it, and say so.

    A scan report's ``summary`` is written ABOUT the radiologist's impression and
    findings, not transcribed from the document. When neither is present there is no read
    to put into plain words — and asked for three to six sentences anyway, a model fills
    the gap. Measured on this prompt with a bare X-ray image's burned-in header:

        "The radiologist reviewed the pictures and found no broken bones, no problems
         with the heart or lungs, and no other abnormalities. Everything looked normal."

    Nothing in that document says a radiologist saw it, or mentions heart or lungs. It is
    a false all-clear on a chest X-ray, produced from two stamped words, and it is the
    worst output this service can generate.

    So Python decides, exactly as it decides abnormal flags, dates and money: a summary
    survives only when something it could have been written from survives. The factual
    fields are untouched — scan type, body part, date and facility are transcription, and
    the user is still shown "Chest X-Ray, 12 March" for an image with no report.

    The flag fires whenever there is no read, whether or not a summary had to be deleted —
    with the prompt tightened the model usually returns null on its own, and an empty card
    with no explanation reads as "the AI failed" rather than "there was nothing here to
    read". That is the same absence-versus-failure confusion the pending-document note
    had. Saying it plainly is the point of the feature, not a side effect of the guard.
    """
    if not spec.summary_field or not spec.summary_sources:
        return []
    if any(fields.get(name) for name in spec.summary_sources):
        return []

    # Nothing to summarise. Delete any summary the model wrote anyway, and say why the
    # card is empty either way.
    fields[spec.summary_field] = None
    return [
        {
            "code": "no_radiologist_read",
            "field": spec.summary_field,
            "detail": (
                "No radiologist's report was found in this document — only the scan "
                "itself. It is saved and you can open it any time."
            ),
        }
    ]


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
