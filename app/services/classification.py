"""Document classification — the first real pipeline stage.

Flow: download the source object from ``unclassified_files``, ask the model which
MyHealthNotion section it belongs to under a fixed schema, validate the JSON with
Pydantic (never repair it), then persist the classification and a process log.

**This stage classifies; it does not route.** Whether the detected section has a pipeline
is decided by ``app.workers.processor``, which reads the section this stage recorded and
looks it up in ``SECTION_PIPELINES``. Rejecting here would mean importing that table,
which imports this module — and it would also misreport a correct classification of an
unhandled section as a failure of this stage.

Filing — INSERT the section row, write its ``content``, DELETE the ``unclassified_files``
row, move the S3 object — happens in ``app.services.filing`` immediately AFTER this stage,
so the document reaches its section within seconds and the later stages update the row in
place. (This note used to say filing waited until the end of the pipeline; it has not
since 2026-08-01.) Not here either way: this stage classifies and nothing else.

Idempotent: the classification and the process log are upserted, so a redelivery that
re-runs the stage overwrites its own prior attempt rather than duplicating rows.
"""

import logging
import time
from dataclasses import replace
from datetime import date
from enum import StrEnum
from functools import partial
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field, ValidationError, field_validator
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from app.integrations.ai.base import AIProviderError
from app.integrations.ai.factory import get_stage_provider
from app.models.ai_results import AiReportClassification
from app.schemas.results import DocumentType
from app.services import document_date
from app.services.ai_logging import (
    check_response,
    elapsed_ms,
    log_process,
    sanitize_validation_error,
)
from app.services.pdf_pages import limit_pdf_pages
from app.services.source_loading import load_source_document
from app.workers.stagetypes import StageContext, TransientStageError

logger = logging.getLogger(__name__)

#: clf-2026-08-07 moved the prescriptions/reports boundary. "reports" claimed "a clinical
#: or discharge summary", which swallowed every Indian consultation note that ends in a
#: medicines table — the commonest shape a prescription actually arrives in. Two real
#: prescriptions were filed as lab reports, ran the report pipeline, found no results, and
#: showed the patient an empty analysis while four prescribed medicines went unread.
#: The rule is now precedence-based: medicines listed anywhere make it a prescription.
PROMPT_VERSION = "clf-2026-08-21"
SCHEMA_VERSION = "clf-3"
STAGE_NAME = "classifying"
#: Classification output is small (a section, a title, a short reason). Kept tight to
#: bound cost; the JSON structured-output format keeps responses compact.
CLASSIFY_MAX_TOKENS = 2048


class DocumentSection(StrEnum):
    """A MyHealthNotion section (the ``resource_type_enum`` values), or unknown.

    This is what drives routing. ``SECTION_PIPELINES`` in app/workers/stages.py decides
    which of these the service actually processes; the rest are recorded and rejected.
    """

    REPORTS = "reports"
    SCANS_IMAGING = "scans_imaging"  # MRI, X-ray, CT, ultrasound, radiology reports
    PRESCRIPTIONS = "prescriptions"
    INSURANCE = "insurance"
    BILLS = "bills"
    VACCINATIONS = "vaccinations"
    MEDICAL_CONDITION = "medical_condition"
    UNKNOWN = "unknown"  # cannot confidently place -> stays in unclassified_files


#: URL type -> the section a document must have been classified as to be read under it.
#:
#: Lives here, beside ``DocumentSection``, because both ``results`` and ``runs`` need it and
#: ``results`` already imports ``runs`` — putting it in either would make that a cycle.
#:
#: The sections with no addressable type are deliberate, not an oversight:
#: ``medical_condition`` is entered by hand and ``unknown`` is by definition unclassified,
#: so neither produces an AI result to read. ``bills`` was in that list until it gained a
#: section spec — it is now extracted like any other section and addressable like one.
SECTION_BY_DOCUMENT_TYPE: dict[DocumentType, DocumentSection] = {
    DocumentType.REPORTS: DocumentSection.REPORTS,
    DocumentType.SCANS: DocumentSection.SCANS_IMAGING,
    DocumentType.INSURANCE: DocumentSection.INSURANCE,
    DocumentType.VACCINATIONS: DocumentSection.VACCINATIONS,
    DocumentType.PRESCRIPTIONS: DocumentSection.PRESCRIPTIONS,
    DocumentType.BILLS: DocumentSection.BILLS,
}

#: The same mapping keyed by the section string as stored, for going from a classification
#: back to the URL that reads it.
DOCUMENT_TYPE_BY_SECTION: dict[str, DocumentType] = {
    section.value: document_type for document_type, section in SECTION_BY_DOCUMENT_TYPE.items()
}


#: How much of a document's substance is handwritten. Ordered least to most.
HANDWRITING_LEVELS = ("none", "some", "mostly")


class LabelledDate(BaseModel):
    """One date printed on the document, with whatever label sat beside it."""

    #: "Sample Collected", "Study Date", "Invoice Date" — or empty for a bare date in a
    #: header, which is common and must not be discarded.
    label: str = Field(default="", max_length=64)
    value: str = Field(max_length=64)

    @field_validator("label", mode="before")
    @classmethod
    def _null_label_is_blank(cls, value: object) -> object:
        # The schema allows null because a date is often printed bare in a header, and
        # absent and blank must not become two states downstream.
        return "" if value is None else value


class DocumentClassification(BaseModel):
    """Validated model output. Written to the DB only after this parses cleanly."""

    section: DocumentSection
    title: str = Field(min_length=1, max_length=512)
    #: Routing input for prescriptions only: a mostly-handwritten one is filed but never
    #: extracted, because nothing could check what the model read off it. Deliberately NOT
    #: persisted — a retry re-runs classification from the top, so this is recomputed rather
    #: than stored, which keeps the whole feature clear of a schema change.
    handwriting: str = Field(default="none", max_length=32)
    confidence: float
    reasoning: str = Field(default="", max_length=2000)
    #: The patient's name exactly as printed, or None when the document prints none.
    #: Read here rather than in extraction because this stage runs for EVERY document
    #: and runs before filing — the gate has to answer "is this yours?" before the
    #: document enters a wallet, not after.
    patient_name: str | None = Field(default=None, max_length=255)
    #: Every date printed on the document, verbatim, with its printed label. Read here for
    #: the same reason `patient_name` is: this is the only stage every document runs, and
    #: the only one that finishes before the user is asked to confirm what they uploaded.
    #: WHICH of these is the document's date is decided by app/services/document_date.py —
    #: the model transcribes, Python chooses. Defaulted rather than required so every
    #: classification written before this field existed still parses.
    dates: list[LabelledDate] = Field(default_factory=list, max_length=20)

    @field_validator("dates", mode="before")
    @classmethod
    def _usable_dates(cls, value: object) -> object:
        # Degrades to [] rather than failing, like the two validators below: an advisory
        # field must not cost a document its section, its title and its patient name.
        # An entry with no value is dropped rather than kept with an empty one — empty
        # parses to None downstream and would read as a date we considered and rejected
        # rather than one that was never there.
        if not isinstance(value, list):
            return []
        return [
            item
            for item in value[:20]
            if isinstance(item, dict) and str(item.get("value") or "").strip()
        ]

    @field_validator("patient_name", mode="before")
    @classmethod
    def _blank_is_none(cls, value: object) -> object:
        # A model asked for a name it cannot find sometimes returns "" or "   ".
        # Absent and blank mean the same thing and must not be two states downstream.
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("handwriting", mode="before")
    @classmethod
    def _known_level(cls, value: object) -> str:
        # Out-of-vocabulary becomes "none" rather than failing the document. Getting this
        # wrong in the safe direction means we extract a document we might have skipped —
        # the same answer as before this field existed. Failing validation instead would
        # lose the whole classification over an advisory field.
        if isinstance(value, str) and value.strip().lower() in HANDWRITING_LEVELS:
            return value.strip().lower()
        if value not in (None, ""):
            logger.warning("model returned handwriting %r, which is not one of the three", value)
        return "none"

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
        #: A plain nullable string rather than a JSON-Schema enum, for the reason
        #: PRESCRIPTION_JSON_SCHEMA gives for `form`: an enum cannot express "one of these
        #: three, or nothing", and the validator below is the guarantee regardless.
        "handwriting": {"type": ["string", "null"]},
        "confidence": {"type": "number"},
        "reasoning": {"type": "string"},
        "patient_name": {"type": ["string", "null"]},
        #: A list rather than one date, because choosing between them is not the model's
        #: job — see app/services/document_date.py. An empty list is a real answer.
        "dates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": ["string", "null"]},
                    "value": {"type": "string"},
                },
                "required": ["label", "value"],
                "additionalProperties": False,
            },
        },
    },
    "required": [
        "section",
        "title",
        "handwriting",
        "confidence",
        "reasoning",
        "patient_name",
        "dates",
    ],
    "additionalProperties": False,
}

SYSTEM_PROMPT = (
    "You are a document classifier in a medical-records intake pipeline. You receive a "
    "single uploaded document and decide which section of the app it belongs to. You do "
    "not diagnose, interpret results, or give medical advice — you only classify.\n\n"
    "Choose exactly one section:\n"
    "- reports: a diagnostic report a clinician files as a RESULT — laboratory report, "
    "pathology report, or a test panel. What makes it a report is measured values with "
    "reference ranges.\n"
    "- scans_imaging: imaging and its radiology report — MRI, X-ray, CT, ultrasound, and "
    "the radiologist's read of them.\n"
    "- prescriptions: any document that ORDERS OR ITEMISES MEDICINES — a prescription slip, "
    "a doctor's consultation note whose medicines are listed at the end, a discharge "
    "medication list, or an itemised pharmacy bill naming the drugs dispensed. The tell is "
    "a list of medicines with any of: dose, strength, frequency, duration, or intake "
    "instruction.\n"
    "- insurance: insurance cards, policies, claims, or coverage letters.\n"
    "- bills: invoices, receipts and billing statements that do NOT itemise medicines — "
    "consultation fees, room charges, procedure or test charges.\n"
    "- vaccinations: immunisation or vaccination records.\n"
    "- medical_condition: a record describing a diagnosed condition or its history.\n"
    "- unknown: use ONLY when the document is unreadable or you cannot confidently place "
    "it in any section.\n\n"
    "PRECEDENCE, because these overlap in practice: if the document orders or itemises "
    "medicines, it is 'prescriptions' — even when it also carries consultation notes, "
    "complaints, a diagnosis, a bill total, or is headed 'Consultation Summary', 'OPD "
    "Summary' or 'Discharge Summary'. A document is only 'reports' when its substance is "
    "measured results. A document is only 'bills' when nothing on it is a medicine.\n\n"
    "Also return:\n"
    "- title: a short, human-readable label, at most a few words (for example "
    "'Complete Blood Count' or 'Chest X-Ray'). Do not invent details; do not include "
    "long patient identifiers.\n"
    "- handwriting: how much of the document's SUBSTANCE is handwritten rather than "
    "printed — 'none', 'some', or 'mostly'. Judge the medicines, values and dates, not "
    "the letterhead: a printed form whose drug names are written in by hand is 'mostly', "
    "because the part that matters is handwritten. A printed document carrying only a "
    "handwritten signature or stamp is 'none'.\n"
    "- confidence: your calibrated confidence between 0 and 1.\n"
    "- reasoning: one concise sentence citing what drove the decision. Do not restate "
    "patient data or clinical values.\n"
    "- patient_name: the name of the PERSON THE DOCUMENT IS ABOUT, exactly as printed. "
    "Copy it verbatim including any title. If the document names a doctor, a hospital, a "
    "policyholder and a patient, return the PATIENT. Return null if no patient name is "
    "printed — many scans, bills and vaccination cards carry none. Never infer a name "
    "from context, a file name, or an email address, and never guess.\n"
    "- dates: every date printed on the document, each with the label printed beside it, "
    "copied verbatim in the document's own format. Include ALL of them — a report prints a "
    "collection date, a received date and a release date, and we choose between them "
    "ourselves. Use the label exactly as printed ('Sample Collected', 'Study Date', "
    "'Invoice Date'), or an empty label for a date printed with none. Return an empty list "
    "if the document prints no date. Never infer, compute or guess a date.\n\n"
    "Be conservative: if the document is unreadable or genuinely ambiguous, choose "
    "'unknown' rather than guessing a section."
)

INSTRUCTION = "Classify the attached document into one section, and transcribe the dates it prints."


def classify_report(ctx: StageContext) -> None:
    """Stage entrypoint: classify the document and persist the result. Routing is the
    caller's job — see the module docstring."""
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

    # Refusal (transient) and truncation (permanent) are the same check for every
    # stage, so it lives in one place; partial binds this stage's own log helper.
    check_response(
        response,
        log=partial(_log, ctx),
        duration_ms=duration_ms,
        what="classification",
    )

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
    # Handed to the router in memory rather than stored: see StageContext.handwriting.
    ctx.handwriting = result.handwriting
    # Whether we can *process* this section is the router's decision, not this stage's:
    # classification succeeded either way, and a stage that rejected here would have to
    # know the pipeline table — which imports this module. See workers/stages.py.
    _log(ctx, outcome="succeeded", response=response, duration_ms=duration_ms)


# --- helpers ----------------------------------------------------------------


def adopt_prior(session: Session, *, item_id: UUID, document_id: int) -> bool:
    """Copy the newest classification for this document onto THIS item. True if it did.

    A resumed or retried document is a NEW run item, and the classification row is keyed
    on ``run_item_id`` — so without this the pipeline would have to read the document
    again. Paying twice is the smaller half of the problem. A second reading can land on a
    different section, and ``filing._adopt_prior_filing`` then refuses with
    ``section_changed_on_retry``: terminal, deliberately not re-filed, and reached by a
    user who did nothing but press a button. Adopting makes that disagreement impossible
    rather than unlikely.

    The mirror of ``identity._carry_settled``, on the same table and for the same reason:
    what an earlier pass established about a document has to survive the item that
    established it.

    ``prompt_version`` and ``schema_version`` are copied, not stamped fresh. No model call
    was made under this version, and claiming one would put a lie in the audit trail — the
    rule that makes a skipped insights stage record ``"skipped"`` rather than the
    configured model.
    """
    prior = session.execute(
        select(AiReportClassification)
        .where(
            AiReportClassification.document_id == document_id,
            AiReportClassification.run_item_id != item_id,
        )
        .order_by(AiReportClassification.created_at.desc())
        .limit(1)
    ).scalar_one_or_none()
    if prior is None:
        return False

    session.execute(
        pg_insert(AiReportClassification)
        .values(
            run_item_id=item_id,
            document_id=document_id,
            section=prior.section,
            title=prior.title,
            confidence=prior.confidence,
            reasoning=prior.reasoning,
            patient_name=prior.patient_name,
            name_match=prior.name_match,
            identity_confirmed_at=prior.identity_confirmed_at,
            document_date=prior.document_date,
            document_date_label=prior.document_date_label,
            prompt_version=prior.prompt_version,
            schema_version=prior.schema_version,
        )
        # A redelivery of the resumed message re-runs this against an item that already
        # has its row. Nothing has changed, so there is nothing to update.
        .on_conflict_do_nothing(index_elements=[AiReportClassification.run_item_id])
    )
    session.commit()
    logger.info(
        "classification_adopted",
        extra={"item_id": str(item_id), "document_id": document_id, "section": prior.section},
    )
    return True


def chosen_date(result: DocumentClassification) -> tuple[date | None, str | None]:
    """The one date this document is about, and the label it was printed under.

    Split out from persistence so the rule can be exercised against a parsed
    classification without a database, and so the model's raw transcription stays
    visibly separate from the choice made over it.
    """
    return document_date.pick(result.section.value, [(d.label, d.value) for d in result.dates])


def _persist_classification(ctx: StageContext, result: DocumentClassification) -> None:
    picked, picked_label = chosen_date(result)
    stmt = (
        pg_insert(AiReportClassification)
        .values(
            run_item_id=ctx.item_id,
            document_id=ctx.document_id,
            section=result.section.value,
            title=result.title,
            confidence=result.confidence,
            reasoning=result.reasoning or None,
            patient_name=result.patient_name,
            document_date=picked,
            document_date_label=picked_label,
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
                "patient_name": result.patient_name,
                # In the update as well as the insert: a redelivered message re-runs
                # this stage against the same item, and omitting these here would
                # leave the first pass's date under the second pass's reading.
                "document_date": picked,
                "document_date_label": picked_label,
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
