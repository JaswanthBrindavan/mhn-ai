"""Provider-agnostic types and the AIProvider interface.

The provider returns raw model output (the JSON text) plus usage and stop reason. It
deliberately does NOT parse or validate — validation with Pydantic happens in the
service layer, so the "never write unvalidated model output" rule lives in one place
and the provider stays a thin transport.
"""

from dataclasses import dataclass
from typing import Any, Protocol


class AIProviderError(Exception):
    """A transport/API failure talking to the model. Treat as transient."""


@dataclass(frozen=True)
class AIUsage:
    """Token accounting from one model call, used for cost logging."""

    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


@dataclass(frozen=True)
class DocumentPayload:
    """A source document to analyse.

    ``content_type`` is the resolved media type (``application/pdf`` / ``image/png`` /
    ``image/jpeg``) — the provider uses it to choose the right content block.
    """

    data: bytes
    content_type: str
    filename: str


@dataclass(frozen=True)
class StructuredResponse:
    """Raw structured-output result. ``text`` is the model's JSON, still unvalidated."""

    text: str
    model: str
    stop_reason: str | None
    usage: AIUsage

    @property
    def refused(self) -> bool:
        return self.stop_reason == "refusal"

    @property
    def truncated(self) -> bool:
        return self.stop_reason == "max_tokens"


class AIProvider(Protocol):
    """Analyse a document and return structured JSON constrained to ``json_schema``."""

    def analyze_document(
        self,
        *,
        document: DocumentPayload,
        system: str,
        instruction: str,
        json_schema: dict[str, Any],
        max_tokens: int,
    ) -> StructuredResponse: ...
