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


def _default_for(json_schema: dict[str, Any]) -> StructuredResponse:
    """Pick a canned payload from the requested schema's shape, so a full pipeline that
    classifies, extracts, then generates insights on one fake gets a valid response each."""
    props = json_schema.get("properties", {})
    if "insights" in props:
        return structured_response(insights_payload())
    if "results" in props:
        return structured_response(extraction_payload())
    return structured_response(classification_payload())


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
