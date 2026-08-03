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
    carries dates and an insurer but no benefit schedule)."""

    insurer: str | None = Field(default=None, max_length=256)
    policy_name: str | None = Field(default=None, max_length=256)
    policy_type: str | None = Field(default=None, max_length=128)
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
    "- co_pay: the co-payment the insured bears per claim, as printed. Null if none.\n"
    "- start_date / end_date: the policy period.\n"
    "- covered_conditions: at most 5, most important first. These are the medical "
    "conditions, treatments, or benefits the policy covers — NOT the insured people. "
    "cap is the stated limit or sum for that condition, or null.\n"
    "- exclusions: at most 5, most important first. What the policy does not cover.\n\n"
    "Rules:\n"
    + _NO_INVENTION_RULE
    + _DATE_RULE
    + "- A schedule, receipt, or ID card usually has no benefit list. Return empty "
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
    "  For example: 'This was an X-ray of the left knee. The knee is the joint in the "
    "middle of your leg. The pictures did not show any broken bones or other problems. "
    "Everything looked normal.'\n"
    "- impression: the radiologist's impression or conclusion, as printed.\n"
    "- findings: at most 5 short key findings, most important first.\n\n"
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
        max_tokens=4096,
        date_fields=("start_date", "end_date"),
        date_order=(("start_date", "end_date"),),
    ),
    DocumentSection.SCANS_IMAGING: SectionSpec(
        section=DocumentSection.SCANS_IMAGING,
        model=ScanFields,
        json_schema=_SCAN_SCHEMA,
        system_prompt=_SCAN_PROMPT,
        max_tokens=4096,
        date_fields=("scan_date",),
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
