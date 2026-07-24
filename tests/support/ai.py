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


class FakeAIProvider:
    """Returns a fixed response (or raises), recording each call for assertions."""

    def __init__(
        self,
        *,
        response: StructuredResponse | None = None,
        error: Exception | None = None,
    ) -> None:
        self._response = response or structured_response(classification_payload())
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
        return self._response

    @property
    def last_document(self) -> DocumentPayload:
        return self.calls[-1]["document"]
