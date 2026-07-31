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
from dataclasses import replace
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, ValidationError, field_validator
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.integrations.ai.base import AIProviderError
from app.integrations.ai.factory import get_stage_provider
from app.models.ai_results import AiReportClassification
from app.services.ai_logging import elapsed_ms, log_process, sanitize_validation_error
from app.services.pdf_pages import limit_pdf_pages
from app.services.source_loading import load_source_document
from app.workers.stagetypes import RejectStageError, StageContext, TransientStageError

logger = logging.getLogger(__name__)

PROMPT_VERSION = "clf-2026-07-23"
SCHEMA_VERSION = "clf-2"
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
    document = load_source_document(ctx)
    # The document type is evident from the first pages; send only those to the
    # classifier. Extraction still reads the whole document.
    document = replace(
        document, data=limit_pdf_pages(document.data, ctx.settings.classify_max_pages)
    )

    # Picking one label off two pages does not need a frontier model; the stage can be
    # pointed at a cheaper provider without touching the rest of the pipeline.
    provider = get_stage_provider(ctx.settings, ctx.ai, stage=STAGE_NAME)

    started = time.perf_counter()
    try:
        response = provider.analyze_document(
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
            duration_ms=elapsed_ms(started),
        )
        raise TransientStageError(f"classification provider error: {exc}") from exc

    duration_ms = elapsed_ms(started)

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
            detail=sanitize_validation_error(exc),
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


def _log(ctx: StageContext, **kwargs: Any) -> None:
    log_process(
        ctx,
        stage=STAGE_NAME,
        prompt_version=PROMPT_VERSION,
        schema_version=SCHEMA_VERSION,
        **kwargs,
    )
