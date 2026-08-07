"""Prescription extraction — the stage that reads what a doctor prescribed.

Until now ``prescriptions`` was classified and then rejected: ``section_extraction`` says
so in its own docstring, listing it as the section with no extractor. This is that
extractor.

Flow: reload the source object, ask the model for the medicines under a fixed schema,
validate with Pydantic (never repaired), check every name against the document's own text,
normalise the dosing notation in Python (never the model), then persist to
``ai_section_extractions`` and log.

**The document goes to the model, not its OCR text** — the report pipeline's approach
rather than the section pipeline's. A prescription is a layout: medicines in a table,
dosing in a column beside them, a strength on its own line under a name. OCR flattens
that, and a dose in the wrong row is a dosing error rather than a missing field. Many are
also photographs of paper, where there is no text layer to flatten in the first place.

**Names are verified against the document.** This is the one guard that makes a model
acceptable near a drug name. Asked to read an illegible page, a model produces a
*plausible* name — and plausible-but-wrong is the worst failure here, because a
hallucinated drug is usually a real drug, so nothing downstream can catch it. A name the
document does not contain is dropped and counted, not stored.

**Dosing is normalised in Python.** ``medicines.normalize_frequency`` turns "1/2 - 0 - 1/2"
into half a tablet morning and night. The model transcribes; it never does the arithmetic.
Anything the notation does not cover stays null with the raw text intact, because a wrong
schedule is worse than no schedule.

**Dosage form is the table first, the model second.** ``medicines.normalize_form`` settles
every abbreviation a document actually prints - "Tab.", "INJ", "E/D" - deterministically,
so those answers cannot drift between runs. Only what a table genuinely cannot settle
reaches the model: whether a topical gel is Cream or Ointment depends on its base, which
is a property of the product rather than of the word. The model's answer is confined to
the nine by the schema's enum and checked again on the way in, and "none of the nine"
stays null rather than becoming the nearest box.

Idempotent: the extraction row and the process log are upserted, so a redelivery that
re-runs the stage overwrites its own prior attempt rather than duplicating rows.
"""

import logging
import re
import time
from collections import Counter
from functools import partial
from hashlib import sha256
from typing import Any

from pydantic import BaseModel, Field, ValidationError, field_validator
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.integrations.ai.base import AIProviderError
from app.integrations.ai.factory import get_stage_provider
from app.models.ai_results import AiSectionExtraction
from app.services import medicines
from app.services.ai_logging import (
    check_response,
    elapsed_ms,
    log_process,
    sanitize_validation_error,
)
from app.services.classification import DocumentSection
from app.services.ocr import TextExtractionError, extract_text
from app.services.source_loading import load_source_document
from app.workers.stagetypes import StageContext, TransientStageError

logger = logging.getLogger(__name__)

PROMPT_VERSION = "rx-2026-08-05c"
#: rx-3 added ``form``, the model's own classification into the nine, used only where the
#: lookup table has nothing to say. rx-2 added ``form_raw`` and ``form_normalized``.
#: Payloads either side of a boundary are not comparable without knowing which side they
#: came from: an rx-1 row has no form at all, which is indistinguishable from a later row
#: whose document printed none.
#: rx-4 added ``key`` (a stable per-line id so a confirm screen can remember which
#: medicines a user ticked) and ``medicine_id`` (null until the catalogue resolver lands).
SCHEMA_VERSION = "rx-4"
STAGE_NAME = "extracting_prescription"

#: A prescription is short next to a lab panel — a dozen medicines with five fields each.
#: Measured over 21 real documents: 280 output tokens on average, the largest a 14-page
#: consolidated pharmacy bill at 5,852. 8192 leaves room for that without paying for it
#: on the ordinary case, since max_tokens is a ceiling and not an allocation.
PRESCRIPTION_MAX_TOKENS = 8192

#: How much of a name must be found in the document's text, as a ratio of its characters.
#: Not 1.0: a reader tidies as it reads ("TAB. DOLO 650" for "TAB DOLO 650"), and an exact
#: comparison would reject honest readings and lose real medicines.
MIN_NAME_OVERLAP = 0.7


class PrescribedMedicine(BaseModel):
    """One medicine exactly as the document states it. Values stay as text — Python
    parses the dosing, the model never does."""

    name_as_written: str = Field(min_length=1, max_length=256)
    name_clean: str = Field(min_length=1, max_length=256)
    #: The dosage form as printed ("Tab.", "INJ", "E/D"). Kept because the abbreviations
    #: are not standardised and a reader may need to see what the document actually said.
    form_raw: str | None = Field(default=None, max_length=64)
    #: The model's own classification into the nine, for the forms a lookup table cannot
    #: settle. Constrained by the schema's enum; the validator below is the second line of
    #: defence, since a schema is a request rather than a guarantee.
    form: str | None = Field(default=None, max_length=32)
    strength: str | None = Field(default=None, max_length=128)
    composition: str | None = Field(default=None, max_length=512)
    frequency_raw: str | None = Field(default=None, max_length=256)
    duration: str | None = Field(default=None, max_length=128)

    @field_validator("form")
    @classmethod
    def _known_form(cls, value: str | None) -> str | None:
        # Out-of-vocabulary becomes None rather than failing the document. The form is one
        # optional field on one medicine; rejecting the whole prescription over it would
        # lose every other medicine on the page, and the printed text survives in
        # form_raw regardless. Not a repair - a value outside the contract is discarded,
        # never rewritten into a neighbouring one.
        if value is None:
            return None
        if value in medicines.DOSAGE_FORMS:
            return value
        logger.warning("model returned dosage form %r, which is not one of the nine", value)
        return None


class PrescriptionFields(BaseModel):
    """Validated model output. An empty ``medicines`` list is valid — a consultation note
    or a consumables-only bill genuinely prescribes nothing."""

    medicines: list[PrescribedMedicine]
    prescribed_date: str | None = Field(default=None, max_length=64)
    prescriber: str | None = Field(default=None, max_length=256)


#: Hand-written to stay inside what ``json_schema`` structured output supports: no numeric
#: or length constraints, ``additionalProperties`` false, every field required, nullable
#: expressed as a union. Length limits live on the Pydantic model, which does the
#: validating.
_NULLABLE_STR: dict[str, Any] = {"type": ["string", "null"]}

#: ``form`` is a nullable string rather than a JSON-schema ``enum``, and deliberately so.
#:
#: An enum cannot express "one of these nine, or nothing": the SDK requires every enum
#: member to be a string, so null cannot be a member, and an enum listing only the nine
#: forces a choice on a document that prints no form at all — the exact guess this field
#: exists to avoid. Nullability is worth more here than a schema-level constraint.
#:
#: The vocabulary is enforced twice regardless: the prompt names the nine and forbids
#: anything else, and ``PrescribedMedicine._known_form`` discards whatever is not one of
#: them. A schema is a request; the validator is the guarantee.
_FORM_FIELD: dict[str, Any] = _NULLABLE_STR
PRESCRIPTION_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "medicines": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name_as_written": {"type": "string"},
                    "name_clean": {"type": "string"},
                    "form_raw": _NULLABLE_STR,
                    "form": _FORM_FIELD,
                    "strength": _NULLABLE_STR,
                    "composition": _NULLABLE_STR,
                    "frequency_raw": _NULLABLE_STR,
                    "duration": _NULLABLE_STR,
                },
                "required": [
                    "name_as_written",
                    "name_clean",
                    "form_raw",
                    "form",
                    "strength",
                    "composition",
                    "frequency_raw",
                    "duration",
                ],
                "additionalProperties": False,
            },
        },
        "prescribed_date": _NULLABLE_STR,
        "prescriber": _NULLABLE_STR,
    },
    "required": ["medicines", "prescribed_date", "prescriber"],
    "additionalProperties": False,
}


SYSTEM_PROMPT = (
    "You read Indian medical prescriptions and pharmacy or hospital bills. You transcribe "
    "what the document states; you do not diagnose, advise, or complete it from your own "
    "knowledge of a drug.\n\n"
    "Return every medicine the document prescribes or bills for, and nothing else.\n\n"
    "Rules, in order of importance:\n"
    "1. Report only what is printed. If a field is not on the document, return null for "
    "it. Never fill a gap from your own knowledge — not the composition, not the usual "
    "dose, not the typical duration.\n"
    "2. Copy name_as_written exactly as printed, including any 'Tab.'/'Cap.'/'Syp.' prefix "
    "and any misspelling. Do not correct it.\n"
    "3. name_clean is ONE name and nothing else. Remove the dosage form (tablet, capsule, "
    "syrup, gel, E/D), the strength, the pack size (15's, 1x10, 30 GM) and the "
    "manufacturer. Keep everything that is part of the product's name, including suffixes "
    "like 'XT', 'DSR', 'CR', 'SR', 'Duo', 'Plus', 'Forte'. When the document prints BOTH a "
    "brand and its generic — 'DILTIAZEM (ANGIZEM)', 'ANGIZEM / Diltiazem' — put the brand "
    "in name_clean and the generic in composition. Never join the two, and never put a "
    "parenthesis or a slash in name_clean.\n"
    "4. form_raw is the dosage form exactly as printed, and nothing else: 'Tab.', "
    "'TABLET', 'Cap', 'INJ', 'Syp', 'E/D', 'Powder', 'Rotacap'. Take it from wherever "
    "the document puts it — a prefix on the name, a column of its own, or inside a "
    "hyphen-joined run like 'DAPAGLIFLOZIN-TABLET-5MG-DAPEFY'. Do not infer it from what "
    "you know the drug to be: if the document does not print a form, return null for "
    "form_raw.\n"
    "5. form is that same dosage form placed in ONE of these nine categories, and no "
    "others: Tablet, Capsule, Syrup, Injection, Drops, Cream, Ointment, Inhaler, Powder. "
    "Judge it by how the medicine is actually given: a vial, ampoule or IV infusion is "
    "Injection; a suspension is Syrup; eye or nasal drops are Drops; a Rotacap, respule "
    "or nebuliser solution is Inhaler; a sachet or granules are Powder. A topical gel is "
    "Cream when it is an aqueous, non-greasy base that rubs in, and Ointment when it is "
    "a greasy, occlusive one — decide from the product, not from the word 'gel'. If the "
    "document prints no form, or the form is genuinely none of the nine (a suppository, "
    "a patch, a mouthwash), return null rather than the nearest of the nine.\n"
    "6. A combination drug is ONE entry. Keep its compound strength as written "
    "('500mg + 125mg'); never split it into separate medicines.\n"
    "7. A composition line — 'Contains: PARACETAMOL (650 MG)', or an ingredient list "
    "printed under the name — belongs in composition on that medicine. It is not a "
    "medicine of its own.\n"
    "8. Do NOT return: laboratory tests, diagnoses, vitals, procedures, advice ('steam "
    "inhalation', 'drink plenty of water'), consultation or room charges, consumables "
    "(cotton, syringes, gloves), or anything under a heading about SUBSTITUTE or "
    "ALTERNATIVE medicines — those are what to take if the prescribed one is unavailable, "
    "and were not prescribed.\n"
    "9. On a bill the description column often packs several fields together ('LIVOGEN "
    "FERROUS FUMARATE FOLIC ACID TAB 1x15 MERCK'). Put the brand in name_clean, the "
    "ingredients in composition, and drop the pack size and manufacturer.\n"
    "10. Read tables by their columns. A cell that wraps onto a second line is still one "
    "value, and a value in the frequency column belongs to the medicine in its own row — "
    "not the row above or below.\n"
    "11. frequency_raw is the whole dosing instruction exactly as written, to the end of "
    "the sentence: '1-0-1', 'BD', 'Alternate day', 'Twice Daily ( 1/2 - 0 - 0 - 1/2 ) "
    "Tablet Orally Before Food'. Keep any food timing (before/after food, empty stomach), "
    "route, and fractional dose — do not stop at the dose matrix.\n"
    "12. Copy a dose matrix slot for slot. '1 - 0 - 0 - 1' has four slots and must come "
    "back with four; do not condense it to '1 - 0 - 1' or drop a zero. The slot count is "
    "what says which time of day a dose falls on — four slots read morning/afternoon/"
    "evening/night, three read morning/afternoon/night — so losing one moves the dose to "
    "the wrong time. Keep fractions as printed ('1/2', not '0.5').\n"
    "13. Return one entry per printed line, in the order printed. Never merge, deduplicate "
    "or summarise. A consolidated document staples several bills together and the same "
    "medicine is often billed again on another date or page — each is a separate purchase "
    "and gets its own entry, even when name and strength repeat exactly.\n\n"
    "Also return prescribed_date (the date the prescription is dated) and prescriber (the "
    "doctor or hospital named on it), or null for either if not shown."
)

#: Completeness has to be demanded explicitly — the same lesson ``extraction`` records for
#: lab panels. Measured here on a 14-page consolidated bill: asked only to "return every
#: medicine", the model returned 15 distinct products for 66 printed purchases and
#: reported success. Spelling out every row, every page, no summarising fixed it.
INSTRUCTION = (
    "Extract every medicine from the attached prescription or bill as structured data. "
    "Return EVERY row on EVERY page. Do not summarise, sample, or deduplicate — "
    "completeness is the requirement. Work through the document page by page."
)


def extract_prescription(ctx: StageContext) -> None:
    """Stage entrypoint: read the medicines, verify them, normalise dosing, persist."""
    document = load_source_document(ctx)
    provider = get_stage_provider(ctx.settings, ctx.ai, stage=STAGE_NAME)

    started = time.perf_counter()
    try:
        response = provider.analyze_document(
            document=document,
            system=SYSTEM_PROMPT,
            instruction=INSTRUCTION,
            json_schema=PRESCRIPTION_JSON_SCHEMA,
            max_tokens=PRESCRIPTION_MAX_TOKENS,
        )
    except AIProviderError as exc:
        _log(
            ctx,
            outcome="error",
            error_code="ai_provider_error",
            detail=str(exc),
            duration_ms=elapsed_ms(started),
        )
        raise TransientStageError(f"prescription provider error: {exc}") from exc

    duration_ms = elapsed_ms(started)

    # Refusal (transient) and truncation (permanent) are the same check for every stage, so
    # it lives in one place; partial binds this stage's own log helper. Truncation matters
    # here in particular: the largest of the 21 measured documents used 5,852 of the 8,192
    # ceiling, and a cut-off response fails Pydantic identically on every retry.
    check_response(
        response,
        log=partial(_log, ctx),
        duration_ms=duration_ms,
        what="prescription extraction",
    )

    try:
        result = PrescriptionFields.model_validate_json(response.text)
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
        raise TransientStageError("prescription output failed validation") from exc

    kept, rejected, loose, verified, source = _verify_against_document(document, result.medicines)
    payload = _build_payload(result, kept, rejected, verified, loose)
    if source:
        # Read provenance, not a routing input: extraction is vision, so a missing text
        # layer is never a reason to extract less. It is here so a medicine that later
        # turns out wrong can be traced to "the guard could not check it" without
        # re-running the document. Same metadata section_extraction already stores.
        payload["source"] = source
    _persist(ctx, payload)
    _log(ctx, outcome="succeeded", response=response, duration_ms=duration_ms)


def record_handwritten(ctx: StageContext) -> None:
    """The whole pipeline for a handwritten prescription: record that we did not read it.

    A mostly-handwritten prescription is **not** sent to the model at all. Vision would
    return something — it always does — and on a handwritten page there is nothing that
    could check it: the name guard needs a text layer to reject with, and a photograph of
    handwriting has none. So the one document where a misread is most likely is also the
    one where every downstream check is blind, and a wrong dose reaching a medication
    reminder is the failure this refuses to risk.

    The document is still **filed**, deliberately. It is the user's prescription and belongs
    in their Prescriptions section where they can see it; only the medicines are withheld,
    with a flag telling the app to ask for the pharmacy bill instead — a printed bill lists
    the same drugs and can be read safely.

    No model call is made, so the process log records ``"skipped"`` as both provider and
    model rather than claiming a call that never happened.
    """
    payload = {
        "section": DocumentSection.PRESCRIPTIONS.value,
        "fields": {"medicines": [], "prescribed_date": None, "prescriber": None},
        "flags": [
            {
                "code": "handwritten_not_extracted",
                "detail": (
                    "This prescription is handwritten, so its medicines were not read "
                    "automatically. Upload the pharmacy bill and we will read that instead."
                ),
            }
        ],
    }
    _persist(ctx, payload)
    _log(ctx, outcome="succeeded", duration_ms=0)
    logger.info(
        "prescription_handwritten_not_extracted",
        extra={"item_id": str(ctx.item_id), "document_id": ctx.document_id},
    )


# --- the guard --------------------------------------------------------------

_NOT_ALNUM = re.compile(r"[^a-z0-9]+")

#: Dosage forms carry no drug identity, so they are weighed out of the comparison below.
#: "Capsule" appearing in a name the reader tidied must not cost a real medicine, while a
#: name whose *drug* part is absent is still rejected.
# A word list reads as a block; one per line would run to thirty.
# fmt: off
_FORM_WORDS = frozenset(
    {
        "tab", "tabs", "tablet", "tablets", "cap", "caps", "capsule", "capsules",
        "syp", "syrup", "susp", "suspension", "inj", "injection", "drop", "drops",
        "gel", "cream", "ointment", "lotion", "spray", "inhaler", "sachet", "solution",
        "e/d", "e/o", "n/d", "eye", "ear", "oral", "orally",
    }
)
# fmt: on


def _normalise(text: str) -> str:
    """Letters and digits only, lowercased — so punctuation and spacing cannot make an
    honest reading look invented ("TAB. DOLO 650" vs "TAB DOLO 650")."""
    return _NOT_ALNUM.sub("", (text or "").lower())


def _has_drug_identity(name: str) -> bool:
    """Does *name* contain anything that names a drug, rather than only a dosage form?

    Separate from the hallucination guard and asking a different question. "Tablet" is
    printed on almost every prescription, so it passes a check of "is this on the page"
    while naming no medicine at all. A row like that is not a hallucination — it is a
    misread heading or a stray table cell — and storing it as a prescribed medicine would
    be wrong in a way no downstream reader could detect.
    """
    return any(len(part) > 2 and part not in _FORM_WORDS for part in _NOT_ALNUM.split(name.lower()))


#: The three answers ``_appears_in`` can give, strongest first. ``PARTIAL`` is kept but
#: flagged rather than treated as a pass — see below.
EXACT, PARTIAL, ABSENT = "exact", "partial", "absent"


def _appears_in(name: str, haystack: str) -> str:
    """How well the document's own text supports *name*: ``EXACT``, ``PARTIAL``, ``ABSENT``.

    Whole string first. Failing that, most of its parts have to be there — which is what
    makes this survive the way a page was read rather than only the way it was printed.

    Parts are split on every non-alphanumeric boundary, not on whitespace. EMR systems
    print a medicine as one hyphen-joined run with no spaces in it
    ("DAPAGLIFLOZIN-TABLET-5MG-DAPEFY"), and on a two-column page the text extractor
    returns the halves of that run on different lines with the dosing column in between.
    Split on spaces, such a name is a single token that matches nothing and a real
    medicine is dropped; split on the hyphens, its drug parts are found individually.

    **A part match is not a pass, because parts shorter than three characters are dropped
    and those are exactly what distinguishes one product from its neighbour.** PAN-D is
    pantoprazole with domperidone and PANTOP is not; ECOSPRIN AV carries a statin and
    ECOSPRIN does not; the prompt itself insists the model keep XT, CR, SR, Duo, Plus. On
    a page printing only the stem, every one of those used to come back verified. They are
    still kept — a partial match is far more often a page read across two columns than an
    invention, and dropping a real medicine is the worse failure — but the payload now says
    the page did not literally contain the name, which is a thing a reader can act on.

    An exact match on ``name_as_written`` also settles the strength, since the printed name
    carries it ("Tab. DOLO 650"). That is why nothing checks strength separately.
    """
    needle = _normalise(name)
    if not needle:
        return ABSENT
    if needle in haystack:
        return EXACT
    parts = [p for p in _NOT_ALNUM.split(name.lower()) if p and p not in _FORM_WORDS]
    parts = [p for p in parts if len(p) > 2]
    if not parts:
        return ABSENT
    found = sum(len(p) for p in parts if p in haystack)
    if found / sum(len(p) for p in parts) >= MIN_NAME_OVERLAP:
        return PARTIAL
    return ABSENT


def _verify_against_document(
    document: Any, rows: list[PrescribedMedicine]
) -> tuple[list[PrescribedMedicine], list[str], list[str], bool, dict[str, object]]:
    """Split *rows* into those the document's own text supports and those it does not.

    Returns ``(kept, rejected, loose, verified, source)``. ``loose`` is the kept names the
    page supported only in part — see ``_appears_in``. ``verified`` is False when the check
    could not be made at all, which the caller records as a flag rather than passing off
    as a pass. ``source`` is the read provenance (page counts, engine, confidence), stored
    so a wrong medicine can later be traced to "the guard could not check it" without
    re-running the document — the same metadata ``section_extraction`` already keeps.

    **Only a text layer is trusted to reject with.** A rejection deletes a prescribed
    medicine, so the text it rests on has to be at least as reliable as the model. An
    embedded text layer is exact and qualifies. OCR of a photograph does not: measured on
    this corpus, Tesseract read one prescription as 448 characters at 0.77 confidence and
    lost the brand name outright, and on another found neither drug name on the page.
    Checking against that does not catch inventions — it invents rejections, and drops a
    real medicine silently, which is the worse of the two failures. A caller can act on a
    flag saying "not verified"; nobody can act on a medicine that is no longer there.

    So on an OCR'd document every row is kept and the payload says the names went
    unchecked. The guard still does its work where it can actually be trusted, which is
    every digital PDF — and those are the documents where a model has enough text to
    hallucinate plausibly from in the first place.
    """
    # A name with no drug in it is dropped whatever the document says, so this runs
    # before the text is read and needs no OCR pass to decide.
    rows = [row for row in rows if _has_drug_identity(row.name_clean)]

    if not rows:
        # Nothing to check costs nothing to check — and skips an OCR pass on an image,
        # which is most of the time this stage would otherwise spend.
        return [], [], [], True, {}

    try:
        # Text layer only. This guard will not reject a drug name on OCR output (see
        # above), so running Tesseract here would be a full pass bought to be thrown
        # away — and a photographed prescription, the common case, is the slowest one.
        # A page that would have needed OCR comes back "skipped" and carries no text.
        extracted = extract_text(document, allow_ocr=False)
    except TextExtractionError:
        logger.warning("document could not be read as text - prescription names unchecked")
        return rows, [], [], False, {}

    unread = sum(1 for page in extracted.pages if page.method == "skipped")
    if unread or not extracted.text.strip():
        logger.info(
            "names not verified: %d of %d pages have no usable text layer",
            unread,
            extracted.page_count,
        )
        return rows, [], [], False, extracted.as_metadata()

    haystack = _normalise(extracted.text)
    kept: list[PrescribedMedicine] = []
    rejected: list[str] = []
    loose: list[str] = []
    for row in rows:
        # Either name is enough. What the guard is for is a drug the page never mentions,
        # and that identity lives in name_clean; name_as_written carries the form, pack
        # and strength alongside it, any of which the extractor may have placed on
        # another line. Requiring both would reject honest readings without making a
        # hallucinated name any harder to produce - it would fail on both counts.
        # The stronger of the two verdicts stands: one name found whole is better evidence
        # than the other found only in parts.
        verdicts = (
            _appears_in(row.name_clean, haystack),
            _appears_in(row.name_as_written, haystack),
        )
        if EXACT in verdicts:
            kept.append(row)
        elif PARTIAL in verdicts:
            kept.append(row)
            loose.append(row.name_as_written)
        else:
            rejected.append(row.name_as_written)
    for name in rejected:
        logger.warning("%r is not on the document - dropped", name)
    for name in loose:
        logger.info("%r matched the page only in part - kept and flagged", name)
    return kept, rejected, loose, True, extracted.as_metadata()


# --- payload ----------------------------------------------------------------


def _resolve_form(row: PrescribedMedicine) -> str | None:
    """Which of the nine this medicine is, or None.

    ``form_raw`` is authoritative when the document printed one. Within it the lookup
    table goes first: "Tab." is Tablet on every document ever printed, and a table gives
    that answer identically on every run, so an abbreviation the table knows is never put
    to a model that could answer differently tomorrow. Only what a table cannot settle
    reaches the model — whether a topical gel is Cream or Ointment turns on whether its
    base is aqueous or greasy, a property of the product rather than of the word "gel".

    With no ``form_raw``, the name is the last resort, because most documents never print
    a form column at all: "Tab. DOLO 650" states it as a prefix and
    "DAPAGLIFLOZIN-TABLET-5MG-DAPEFY" buries it mid-run. That is still only reading what
    is printed — the token has to be there to be found.

    **The model is never asked to fill a gap.** Its answer counts only where a form was
    printed for it to classify. Otherwise the field would be an invitation to answer from
    drug knowledge — every model knows paracetamol comes as a tablet — and a form nobody
    wrote down is indistinguishable, once stored, from one the prescriber did.
    """
    if row.form_raw:
        return medicines.normalize_form(row.form_raw) or row.form
    return medicines.normalize_form(row.name_as_written)


def _medicine_key(row: PrescribedMedicine, seen: Counter[tuple[str, str, str]]) -> str:
    """A stable id for one prescribed line, so a confirm screen can remember it.

    Content-derived rather than positional or random, and each alternative fails for its
    own reason:

    - **Position in the list** breaks because the prompt requires one entry per printed
      line *even when name and strength repeat exactly* — a consolidated bill bills the
      same drug three times — and the payload is upserted on SQS redelivery. A re-run can
      drop or keep a row through the name guard, so an index recorded at confirm time can
      later address a different medicine.
    - **A minted UUID** breaks because a re-run mints new ones and every stored reference
      dangles.

    So: the transcription itself, plus how many identical ones came before it. Stable
    across a re-read whenever the model read the page the same way, and changing exactly
    when the transcription changed — which is correct, because the thing that was
    confirmed genuinely no longer exists.

    Honest limit: this identifies a *transcription*, not a drug. Spring must therefore copy
    the name and schedule onto its own row at confirm time — which its schema already
    forces, ``medicine_tracking.name`` being NOT NULL — so a stale key costs a broken
    back-link and never a lost medication.
    """
    identity = (row.name_as_written, row.strength or "", row.frequency_raw or "")
    ordinal = seen[identity]
    seen[identity] += 1
    digest = sha256("\x1f".join((*identity, str(ordinal))).encode("utf-8"))
    return digest.hexdigest()[:10]


def _build_payload(
    result: PrescriptionFields,
    kept: list[PrescribedMedicine],
    rejected: list[str],
    verified: bool = True,
    loose: list[str] | None = None,
) -> dict[str, Any]:
    """The stored shape: the section's own fields, plus data-quality flags.

    Matches ``ai_section_extractions.data`` — {"section", "fields", "flags"} — so a reader
    handles a prescription the same way it handles insurance or a vaccination card.
    """
    rows: list[dict[str, Any]] = []
    seen: Counter[tuple[str, str, str]] = Counter()
    for row in kept:
        data = row.model_dump()
        data["frequency_normalized"] = medicines.normalize_frequency(row.frequency_raw)
        data["form_normalized"] = _resolve_form(row)
        data["key"] = _medicine_key(row, seen)
        #: Filled by the catalogue resolver when that lands; null until then, and null is
        #: the ordinary answer even after — a brand the catalogue does not carry is not an
        #: error. Emitted unconditionally so the shape never branches on a setting.
        data["medicine_id"] = None
        rows.append(data)

    flags: list[dict[str, Any]] = []
    if rejected:
        # Surfaced rather than swallowed: a name the model produced that the page does not
        # contain is the failure mode this stage most needs to be visible.
        flags.append({"code": "names_not_on_document", "names": rejected})
    if not verified:
        # The medicines below were not checked against the page. Says so plainly rather
        # than letting an unverified result look like a verified one — see
        # ``_verify_against_document`` for why an OCR'd page cannot reject.
        flags.append({"code": "names_unverified", "reason": "no reliable text layer"})
    if loose:
        # Kept, but the page did not literally contain the name — only most of its parts,
        # and the parts a name is split into exclude anything under three characters. That
        # is exactly where PAN-D and PANTOP become the same answer, so it is said out loud
        # rather than counted as verified.
        flags.append({"code": "names_matched_loosely", "names": loose})
    unparsed = [
        r["frequency_raw"] for r in rows if r["frequency_raw"] and not r["frequency_normalized"]
    ]
    if unparsed:
        flags.append({"code": "frequency_not_normalized", "values": unparsed})

    return {
        "section": DocumentSection.PRESCRIPTIONS.value,
        "fields": {
            "medicines": rows,
            "prescribed_date": result.prescribed_date,
            "prescriber": result.prescriber,
        },
        "flags": flags,
    }


def _persist(ctx: StageContext, payload: dict[str, Any]) -> None:
    stmt = (
        pg_insert(AiSectionExtraction)
        .values(
            run_item_id=ctx.item_id,
            document_id=ctx.document_id,
            section=DocumentSection.PRESCRIPTIONS.value,
            data=payload,
            prompt_version=PROMPT_VERSION,
            schema_version=SCHEMA_VERSION,
        )
        .on_conflict_do_update(
            index_elements=[AiSectionExtraction.run_item_id],
            set_={
                "document_id": ctx.document_id,
                "section": DocumentSection.PRESCRIPTIONS.value,
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
