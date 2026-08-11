"""What each non-report section extracts: prompt, schema, and validated shape.

One ``SectionSpec`` per section. The stage in ``app.services.section_extraction`` is
generic and reads this, so adding a section is an entry in ``SECTION_SPECS`` plus a
Pydantic model, a hand-written schema and a prompt — no new stage code, and nothing to
change in the persistence, the logging or the worker.

Every schema follows the same rules as classification and extraction: hand-written to
stay inside what ``json_schema`` structured output supports — no numeric or length
constraints, ``additionalProperties`` false, every field required, nullable expressed as
a union. Length limits live on the Pydantic model, which is what actually validates.

Dates are requested as ``DD/MM/YYYY`` but never trusted: ``app.services.dates``
normalises whatever comes back before it is stored.
"""

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from app.services.classification import DocumentSection

_NULLABLE_STR: dict[str, Any] = {"type": ["string", "null"]}

#: Shared date instruction. Repeated verbatim in each prompt so a section's rules read
#: as one block rather than pointing elsewhere.
_DATE_RULE = (
    "- Dates: return DD/MM/YYYY, digits only. Never a month name, never a clock time. "
    '"1st October 2019" -> "01/10/2019". "31/08/2021 18:11:28" -> "31/08/2021". '
    "Use null when the document does not state the date.\n"
)

_NO_INVENTION_RULE = (
    "- Do NOT invent values. Use null for anything the document does not state, and an "
    "empty list where there is nothing to list.\n"
)


# --------------------------------------------------------------------------- #
# insurance                                                                    #
# --------------------------------------------------------------------------- #
class CoveredCondition(BaseModel):
    name: str = Field(min_length=1, max_length=256)
    #: The stated limit or sum for this condition, if the policy prints one.
    cap: str | None = Field(default=None, max_length=128)


class Exclusion(BaseModel):
    title: str = Field(min_length=1, max_length=256)


class InsuranceFields(BaseModel):
    """Validated insurance policy fields. Empty lists are valid (a receipt or ID card
    carries dates and an insurer but no benefit schedule).

    ``currency`` is separate from the amounts on purpose: the symbol is usually printed
    once in a column header rather than beside each number, and often survives text
    extraction as a stray character. See ``app.services.money``, which normalises all
    three after validation.

    The money caps are generous because a cap the model is not told about is not a limit
    — it is a paid failure. Over-long output fails validation, which this stage treats as
    transient, so it retries the whole call. The prompt asks for digits only; the caps sit
    well above that so a wordier answer is normalised rather than retried.
    """

    insurer: str | None = Field(default=None, max_length=256)
    policy_name: str | None = Field(default=None, max_length=256)
    policy_type: str | None = Field(default=None, max_length=128)
    currency: str | None = Field(default=None, max_length=32)
    sum_insured: str | None = Field(default=None, max_length=64)
    premium_amount: str | None = Field(default=None, max_length=64)
    co_pay: str | None = Field(default=None, max_length=128)
    start_date: str | None = Field(default=None, max_length=64)
    end_date: str | None = Field(default=None, max_length=64)
    covered_conditions: list[CoveredCondition]
    exclusions: list[Exclusion]


_INSURANCE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "insurer": _NULLABLE_STR,
        "policy_name": _NULLABLE_STR,
        "policy_type": _NULLABLE_STR,
        "currency": _NULLABLE_STR,
        "sum_insured": _NULLABLE_STR,
        "premium_amount": _NULLABLE_STR,
        "co_pay": _NULLABLE_STR,
        "start_date": _NULLABLE_STR,
        "end_date": _NULLABLE_STR,
        "covered_conditions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"name": {"type": "string"}, "cap": _NULLABLE_STR},
                "required": ["name", "cap"],
                "additionalProperties": False,
            },
        },
        "exclusions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"title": {"type": "string"}},
                "required": ["title"],
                "additionalProperties": False,
            },
        },
    },
    "required": [
        "insurer",
        "policy_name",
        "policy_type",
        "currency",
        "sum_insured",
        "premium_amount",
        "co_pay",
        "start_date",
        "end_date",
        "covered_conditions",
        "exclusions",
    ],
    "additionalProperties": False,
}

_INSURANCE_PROMPT = (
    "You transcribe structured fields from a health-insurance document. You do not "
    "advise on cover, judge whether a policy is adequate, or interpret terms — you only "
    "record what the document states.\n\n"
    "Return:\n"
    "- insurer: the insurance company or scheme name.\n"
    "- policy_name: the product name, if printed.\n"
    "- policy_type: a short type, e.g. 'Health Insurance (Family Floater)'.\n"
    "- currency: the ISO-4217 code of the money on this document — 'INR' for Indian "
    "rupees, three letters, nothing else. The symbol is often printed ONCE in a column "
    "header ('Sum Insured (₹)') rather than beside each figure, and may reach you as a "
    "stray character such as ` or ? because the text layer could not resolve it. Words "
    "count as evidence too: 'Rs', 'Rs.', or an amount spelled out ('RUPEES FIFTY-TWO "
    "THOUSAND ... ONLY'). Null only if the document shows no currency anywhere.\n"
    "- sum_insured: the total sum insured for the policy. DIGITS AS PRINTED and nothing "
    "else — '300,000.00', never 'Rs 300,000', never words. Not the cumulative bonus "
    "('CB Amount'), not a per-benefit sub-limit, not one member's share.\n"
    "- premium_amount: the TOTAL premium payable, including tax — the figure the "
    "customer actually paid. Where the document prints a breakdown (basic premium, "
    "loadings, service tax, total), return the total, not a component. Digits as "
    "printed.\n"
    "- co_pay: the co-payment the insured bears per claim, as printed ('20%', 'Rs 1,000 "
    "per claim'). Null if the document does not mention a co-payment at all.\n"
    "- start_date / end_date: the policy period.\n"
    "- covered_conditions: EVERY medical condition, treatment or benefit this document "
    "states is covered — NOT the insured people. In the order the document prints them, "
    "with no ranking and no limit on how many. Copy each name VERBATIM from the "
    "document; do not paraphrase it, tidy it, or translate it into a condition name of "
    "your own. cap is the stated limit or sum for that item, as printed, or null.\n"
    "- exclusions: EVERY item this document states is NOT covered, under the same "
    "rules.\n\n"
    "Rules:\n"
    + _NO_INVENTION_RULE
    + _DATE_RULE
    + "- A benefit table is often printed in two columns and read across them, so one "
    "line can splice two unrelated entries together. Emit only entries that read as a "
    "whole benefit. A stray fragment left over from the splice — a bare number, or a "
    "phrase like 'Claims free' — is not a benefit and must not be listed as one.\n"
    "- NEVER add an item to covered_conditions or exclusions because the product name, "
    "the insurer, or a similar policy you have seen implies it. Only what THIS document "
    "prints. A person will be told they are covered for exactly what you list, and being "
    "told they are covered for something they are not is the worst thing you can do "
    "here.\n"
    "- Many schedules print the benefits but leave the exclusions to a separate policy "
    "wording document. When this document defers to one ('refer the Policy Wordings'), "
    "return an empty list. Do not supply the missing side from general knowledge.\n"
    "- A schedule, receipt, or ID card usually has no benefit list. Return empty "
    "lists rather than inferring cover from the product name.\n"
)


# --------------------------------------------------------------------------- #
# scans / imaging                                                              #
# --------------------------------------------------------------------------- #
class ScanFields(BaseModel):
    """Validated imaging-report fields.

    ``summary`` is the patient-facing one: a radiology report is written for another
    clinician, so without it the reader is left with `impression`, which is close to
    unreadable if you have no medical training. It is allowed real room — several
    sentences — because explaining a term in everyday words costs more words than using
    it. The radiologist's exact wording is preserved verbatim in ``impression``, so
    nothing is lost by making ``summary`` plain.
    """

    scan_type: str | None = Field(default=None, max_length=64)
    body_part: str | None = Field(default=None, max_length=128)
    scan_date: str | None = Field(default=None, max_length=64)
    facility: str | None = Field(default=None, max_length=256)
    summary: str | None = Field(default=None, max_length=1200)
    impression: str | None = Field(default=None, max_length=2000)
    findings: list[str]


_SCAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "scan_type": _NULLABLE_STR,
        "body_part": _NULLABLE_STR,
        "scan_date": _NULLABLE_STR,
        "facility": _NULLABLE_STR,
        "summary": _NULLABLE_STR,
        "impression": _NULLABLE_STR,
        "findings": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "scan_type",
        "body_part",
        "scan_date",
        "facility",
        "summary",
        "impression",
        "findings",
    ],
    "additionalProperties": False,
}

_SCAN_PROMPT = (
    "You transcribe structured fields from a medical imaging or radiology report. You "
    "are NOT a doctor: you do not diagnose, do not give advice, and do not state "
    "anything with medical certainty.\n\n"
    "Return:\n"
    "- scan_type: e.g. CT, MRI, X-Ray, Ultrasound, PET, Mammography, ECG.\n"
    "- body_part: the region examined, e.g. Brain, Chest, Right Knee.\n"
    "- scan_date: the date of the study.\n"
    "- facility: the hospital or imaging centre NAME only. Null if only an address "
    "appears.\n"
    "- summary: THREE TO SIX short sentences explaining, in the simplest possible "
    "English, what this scan looked at and what it showed. Write for an adult with no "
    "medical background and limited reading confidence:\n"
    "    * Say what part of the body was scanned and, in a few words, what that part "
    "does — 'the lower back, which carries your weight and protects the nerves running "
    "down to your legs'.\n"
    "    * NEVER use a clinical term on its own. Either replace it with everyday words "
    "or explain it immediately: not 'posterocentral disc protrusion', but 'one of the "
    "soft cushions between the bones is bulging out towards the back'.\n"
    "    * Spell out abbreviations and bone or joint labels the same way. 'L4-5' is "
    "'between the fourth and fifth bones of the lower back'.\n"
    "    * Use short sentences. One idea each. Prefer common words: 'swelling' over "
    "'oedema', 'narrowing' over 'stenosis', 'pressing on' over 'impinging'.\n"
    "    * Say plainly when something looks normal, and say plainly what was found when "
    "it does not. Describe ONLY what the report states.\n"
    "  You are still not a doctor: no diagnosis, no advice, no reassurance or alarm "
    "beyond what the report itself says. The radiologist's exact wording is preserved "
    "in impression, so nothing is lost by keeping this plain.\n"
    "  For example, where the report states the knee is normal: 'This was an X-ray of "
    "the left knee. The knee is the joint in the middle of your leg. The report says the "
    "pictures did not show any problems.'\n"
    "    * The summary RESTATES impression and findings in plainer words. It may not add "
    "anything they do not contain. If the impression is one short phrase, the summary is "
    "one or two plain sentences — do not expand it. NEVER name an organ, a structure or a "
    "condition the document does not mention: an impression reading 'Normal study' "
    "becomes 'The report says this scan looked normal', NOT a list of the parts that were "
    "checked and found healthy.\n"
    "    * NEVER state or imply that a radiologist reviewed, reported on or cleared the "
    "images unless the document says so. A scan image carrying only a header, a "
    "technologist's note or a stamp has not been reported on.\n"
    "    * A technologist's working note ('repeat done, patient moved', exposure "
    "settings) is not a finding and does not belong in the summary at all.\n"
    "- impression: the radiologist's impression or conclusion, as printed.\n"
    "- findings: at most 5 short key findings, most important first.\n\n"
    "**Many uploads are the IMAGE alone — an X-ray or scan with a burned-in header and no "
    "radiologist's report anywhere in the text.** That is expected and is not a failure. "
    "Transcribe the factual fields you can read (scan_type, body_part, scan_date, "
    "facility) and return null for summary and impression and an empty findings list. Do "
    "not describe what the picture might show: you are reading TEXT, you cannot see the "
    "image, and a reassuring sentence about a scan nobody reported on is the most harmful "
    "thing you could write here.\n\n"
    "Rules:\n" + _NO_INVENTION_RULE + _DATE_RULE
)


# --------------------------------------------------------------------------- #
# vaccinations                                                                 #
# --------------------------------------------------------------------------- #
class VaccinationFields(BaseModel):
    """Validated vaccination-record fields. No free-text summary: a vaccination record
    is a set of facts, and every one of them is a field here."""

    title: str | None = Field(default=None, max_length=256)
    vaccine_name: str | None = Field(default=None, max_length=256)
    dose_info: str | None = Field(default=None, max_length=128)
    date_given: str | None = Field(default=None, max_length=64)
    next_due_date: str | None = Field(default=None, max_length=64)
    facility: str | None = Field(default=None, max_length=256)


_VACCINATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "title": _NULLABLE_STR,
        "vaccine_name": _NULLABLE_STR,
        "dose_info": _NULLABLE_STR,
        "date_given": _NULLABLE_STR,
        "next_due_date": _NULLABLE_STR,
        "facility": _NULLABLE_STR,
    },
    "required": [
        "title",
        "vaccine_name",
        "dose_info",
        "date_given",
        "next_due_date",
        "facility",
    ],
    "additionalProperties": False,
}

_VACCINATION_PROMPT = (
    "You transcribe structured fields from a vaccination record or certificate. You do "
    "not advise on schedules, eligibility, or whether a dose is due — you only record "
    "what the document states.\n\n"
    "Return:\n"
    "- title: a short human label for what this vaccination is, including the dose if "
    "stated, e.g. 'COVID-19 (Covishield) - Dose 2'.\n"
    "- vaccine_name: the vaccine itself, e.g. 'Covishield', 'Tetanus Toxoid'.\n"
    "- dose_info: e.g. 'Dose 2 of 2', 'Booster', '1st dose'.\n"
    "- date_given: the date this dose was administered.\n"
    "- next_due_date: the NEXT scheduled dose only. Null if the record shows none or "
    "the series is complete.\n"
    "  If the record gives a WINDOW rather than a single day — 'Between 31 Jan 2022 and "
    "14 Feb 2022', 'due after 4 weeks', '31/01/2022 - 14/02/2022' — return the START of "
    "that window, the day the dose first becomes due. Do not return null just because the "
    "record states a range: a stated next dose is the whole point of this field.\n"
    "- facility: the vaccination centre or hospital NAME only, not its address.\n\n"
    "Rules:\n" + _NO_INVENTION_RULE + _DATE_RULE
)


# --------------------------------------------------------------------------- #
# registry                                                                     #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SectionSpec:
    """Everything the generic extraction stage needs to handle one section."""

    section: DocumentSection
    model: type[BaseModel]
    json_schema: dict[str, Any]
    system_prompt: str
    #: Bounded per section: a policy wording carries more than a vaccination card.
    max_tokens: int
    #: Date fields on ``model``, normalised to ISO before the payload is stored.
    date_fields: tuple[str, ...]
    #: (earlier, later) pairs where the later date may not precede the earlier one.
    date_order: tuple[tuple[str, str], ...] = ()
    #: Money fields, reduced to a bare decimal string before storage — same reason as
    #: dates: the model transcribes, Python decides.
    amount_fields: tuple[str, ...] = ()
    #: Fields holding a currency, resolved to an ISO-4217 code. Listed explicitly rather
    #: than found by name so the rule is visible in the spec, like every other field.
    currency_fields: tuple[str, ...] = ()
    #: A patient-facing field written ABOUT other fields rather than transcribed from the
    #: document, and the fields it must be written from. When none of those carries
    #: anything, the summary has no source and is dropped — see ``interpretation_flag``.
    #:
    #: Only scans have one, and it is the field that made this necessary: asked for three
    #: to six sentences, a model handed a bare X-ray image's burned-in header produced
    #: "The radiologist reviewed the pictures and found no problems with the heart or
    #: lungs. Everything looked normal." No radiologist read that document. Python decides
    #: whether there was anything to summarise, for the same reason it decides abnormal
    #: flags and dates: a prompt is a request, and this one is a false all-clear.
    summary_field: str | None = None
    #: What the summary must be written from. Non-empty in any of these means there is a
    #: read to put into plain words.
    summary_sources: tuple[str, ...] = ()


INSTRUCTION_PREFIX = (
    "Below is the text extracted from one document. Some of it may come from OCR of a "
    "scan, so it can contain broken words, stray characters, and lost table alignment. "
    "Read through that: transcribe what the document states, and use null for anything "
    "you cannot read with confidence rather than guessing at it.\n\n"
)

SECTION_SPECS: dict[DocumentSection, SectionSpec] = {
    DocumentSection.INSURANCE: SectionSpec(
        section=DocumentSection.INSURANCE,
        model=InsuranceFields,
        json_schema=_INSURANCE_SCHEMA,
        system_prompt=_INSURANCE_PROMPT,
        # Raised with the benefit lists: they are no longer capped at five, and a group
        # policy can print a long schedule. Truncation is not free — a cut-off response
        # fails validation, which this stage treats as transient, so it is retried at
        # full price and fails identically each time.
        max_tokens=8192,
        date_fields=("start_date", "end_date"),
        date_order=(("start_date", "end_date"),),
        amount_fields=("sum_insured", "premium_amount"),
        currency_fields=("currency",),
    ),
    DocumentSection.SCANS_IMAGING: SectionSpec(
        section=DocumentSection.SCANS_IMAGING,
        model=ScanFields,
        json_schema=_SCAN_SCHEMA,
        system_prompt=_SCAN_PROMPT,
        max_tokens=4096,
        date_fields=("scan_date",),
        summary_field="summary",
        summary_sources=("impression", "findings"),
    ),
    DocumentSection.VACCINATIONS: SectionSpec(
        section=DocumentSection.VACCINATIONS,
        model=VaccinationFields,
        json_schema=_VACCINATION_SCHEMA,
        system_prompt=_VACCINATION_PROMPT,
        max_tokens=2048,
        date_fields=("date_given", "next_due_date"),
        date_order=(("date_given", "next_due_date"),),
    ),
}

#: Sections this package can extract. The classification stage gates on this.
SUPPORTED_SECTIONS: frozenset[DocumentSection] = frozenset(SECTION_SPECS)


def spec_for(section: DocumentSection) -> SectionSpec:
    """The spec for a section. Raises ``KeyError`` naming the supported sections."""
    try:
        return SECTION_SPECS[section]
    except KeyError:
        supported = sorted(s.value for s in SUPPORTED_SECTIONS)
        raise KeyError(
            f"no extraction spec for section {section.value!r}; have {supported}"
        ) from None
