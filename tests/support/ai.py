"""A fake AIProvider for tests.

The real provider talks to Anthropic (costs money, non-deterministic), so every test
exercises the logic around it against this fake, which returns a canned structured
response and records the document it was handed.
"""

import json
from typing import Any

from app.integrations.ai.base import (
    AIUsage,
    DocumentPayload,
    StructuredResponse,
)
from app.services.classification import DocumentSection
from app.services.section_specs import SECTION_SPECS

DEFAULT_USAGE = AIUsage(input_tokens=1200, output_tokens=40)


def structured_response(
    payload: dict[str, Any] | str,
    *,
    model: str = "claude-opus-4-8",
    stop_reason: str | None = "end_turn",
    usage: AIUsage = DEFAULT_USAGE,
) -> StructuredResponse:
    text = payload if isinstance(payload, str) else json.dumps(payload)
    return StructuredResponse(text=text, model=model, stop_reason=stop_reason, usage=usage)


def classification_payload(**overrides: Any) -> dict[str, Any]:
    payload = {
        "section": "reports",
        "title": "Complete Blood Count",
        "confidence": 0.96,
        "reasoning": "Structured lab result values with reference ranges.",
    }
    payload.update(overrides)
    return payload


def extraction_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "results": [
            {
                "test_name": "Fasting Glucose",
                "value": "126",
                "unit": "mg/dL",
                "reference_range": "70-99",
                "observed_date": "2026-07-20",
                "source_context": "Glucose, Fasting",
            }
        ],
        "report_date": "2026-07-20",
        "patient_age": "45",
        "patient_gender": "Male",
    }
    payload.update(overrides)
    return payload


def insights_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "insights": [
            {
                "heading": "Fasting glucose above the typical range",
                "body": "Your fasting glucose is above the listed reference range. "
                "Consider discussing this with a healthcare professional.",
                "related_tests": ["Fasting Glucose"],
            }
        ],
        "summary": "One result is outside its reference range.",
    }
    payload.update(overrides)
    return payload


def section_payload(section: DocumentSection | str, **overrides: Any) -> dict[str, Any]:
    """A minimal valid payload **built from that section's own schema**.

    Derived rather than hand-written on purpose: the field names live in exactly one place
    (``SECTION_SPECS``), so renaming one cannot leave a stale copy here. A hand-written
    version already drifted once — it used ``vaccine``/``dose`` where the spec says
    ``vaccine_name``/``dose_info`` — and the mismatch was invisible because every field is
    optional, so the wrong payload still validated.

    A test that cares about *content* passes its own payload; this is for tests that only
    need the stage to succeed.
    """
    spec = SECTION_SPECS[DocumentSection(section)]
    payload: dict[str, Any] = {
        name: [] if prop.get("type") == "array" else None
        for name, prop in spec.json_schema["properties"].items()
    }
    payload.update(overrides)
    return payload


#: Discriminating property -> canned payload, for the pipeline's own three schemas. The
#: section schemas are NOT listed here: they are matched against ``SECTION_SPECS`` itself,
#: so a new section needs no entry anywhere in this file.
_PAYLOAD_BY_PROPERTY: dict[str, Any] = {
    "section": classification_payload,
    "results": extraction_payload,
    "insights": insights_payload,
}


def _default_for(json_schema: dict[str, Any]) -> StructuredResponse:
    """Pick a canned payload from the requested schema, so a full pipeline that classifies,
    then extracts (a report or a section), then generates insights on one fake gets a valid
    response at each stage.

    **Unrecognised schemas raise.** This used to fall through to the classification
    payload, which does not merely produce a confusing error — it can *validate*: a
    vaccination record whose fields are all optional accepted the classification payload
    and came back titled "Complete Blood Count". A loud failure naming the properties is
    the only safe default when a new schema is added.
    """
    props = json_schema.get("properties", {})
    for marker, build in _PAYLOAD_BY_PROPERTY.items():
        if marker in props:
            return structured_response(build())
    # Matched against the registry rather than a hand-listed property, so adding a
    # SectionSpec is genuinely the only step.
    for section, spec in SECTION_SPECS.items():
        if spec.json_schema == json_schema:
            return structured_response(section_payload(section))
    raise AssertionError(
        f"FakeAIProvider has no canned payload for a schema with properties {sorted(props)}; "
        "add a SectionSpec, an entry in _PAYLOAD_BY_PROPERTY, or pass an explicit response."
    )


class FakeAIProvider:
    """Returns a fixed response (or raises), recording each call for assertions.

    Without an explicit ``response`` it picks a canned payload from the requested
    schema's shape, so a full pipeline that classifies then extracts on one fake gets a
    valid response for each stage.
    """

    def __init__(
        self,
        *,
        response: StructuredResponse | None = None,
        error: Exception | None = None,
    ) -> None:
        self._response = response
        self._error = error
        self.calls: list[dict[str, Any]] = []

    def set_response(self, response: StructuredResponse) -> None:
        self._response = response
        self._error = None

    def set_error(self, error: Exception) -> None:
        self._error = error

    def analyze_document(
        self,
        *,
        document: DocumentPayload,
        system: str,
        instruction: str,
        json_schema: dict[str, Any],
        max_tokens: int,
    ) -> StructuredResponse:
        self.calls.append(
            {
                "document": document,
                "system": system,
                "instruction": instruction,
                "json_schema": json_schema,
                "max_tokens": max_tokens,
            }
        )
        if self._error is not None:
            raise self._error
        if self._response is not None:
            return self._response
        return _default_for(json_schema)

    def generate_structured(
        self,
        *,
        system: str,
        instruction: str,
        json_schema: dict[str, Any],
        max_tokens: int,
        model: str | None = None,
    ) -> StructuredResponse:
        self.calls.append(
            {
                "document": None,
                "system": system,
                "instruction": instruction,
                "json_schema": json_schema,
                "max_tokens": max_tokens,
                "model": model,
            }
        )
        if self._error is not None:
            raise self._error
        if self._response is not None:
            return self._response
        return _default_for(json_schema)

    @property
    def last_document(self) -> DocumentPayload:
        return self.calls[-1]["document"]

    @property
    def last_instruction(self) -> str:
        return str(self.calls[-1]["instruction"])
