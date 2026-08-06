"""Gemini implementation of ``AIProvider``.

Exists for the **extraction** stage: transcribing a report is the most expensive stage on
Claude (the document goes as rendered pages, ~2,300 tokens each) and the cheapest thing a
capable vision model can do. Gemini bills a page at ~520 tokens, so the same document costs
roughly a fifth as much. Measured at 80% cheaper with identical transcribed values — see
``docs/ai-provider-comparison.md``.

Like the Anthropic provider this is a thin transport: it returns raw JSON text plus usage,
and never parses or validates. Validation stays in the service layer.

Two behaviours are deliberately pinned rather than left to defaults:

* ``temperature=0`` — extraction is transcription; there is nothing to be creative about,
  and a deterministic stage is one that can be compared run to run.
* ``thinking_budget=0`` — reasoning tokens bill as output and buy nothing when copying
  numbers out of a table.

The completeness problem this provider surfaced is handled in the prompt and in
``extraction._dedupe_results``, not here: asked only to "extract every result", the model
returns a representative sample of a long report. ``INSTRUCTION`` demands every row
explicitly. Do not soften it.
"""

import logging
from typing import TYPE_CHECKING, Any

from app.integrations.ai.base import AIProviderError, AIUsage, DocumentPayload, StructuredResponse

if TYPE_CHECKING:  # pragma: no cover - typing only
    from google import genai

logger = logging.getLogger(__name__)

#: The vendor, not the model family: this column answers "who received this document",
#: and PHI to Google is a compliance question about Google rather than about Gemini.
PROVIDER_NAME = "google"

DEFAULT_MODEL = "gemini-3.1-flash-lite"
#: Generous, so a 120-result panel is never cut short. Truncation is surfaced by
#: ``StructuredResponse.truncated`` rather than silently returning a short list.
MAX_OUTPUT_TOKENS = 16_000


def to_gemini_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Translate our provider-agnostic JSON Schema into Gemini's dialect.

    Gemini accepts an OpenAPI-3 subset, which differs from JSON Schema in two ways that
    matter to us:

    * ``"type": ["string", "null"]`` is rejected — a union list is not a valid type. The
      equivalent is a single type plus ``nullable``.
    * ``additionalProperties`` is not part of the subset.

    Everything else passes through. Keeping this in the provider is the point: the schema in
    ``extraction.py`` stays vendor-neutral and each provider adapts it, rather than the
    service layer knowing who it is talking to.
    """
    out: dict[str, Any] = {}
    for key, value in schema.items():
        if key == "additionalProperties":
            continue
        if key == "type" and isinstance(value, list):
            concrete = [t for t in value if t != "null"]
            out["type"] = concrete[0] if concrete else "string"
            if len(concrete) != len(value):
                out["nullable"] = True
            continue
        if isinstance(value, dict):
            out[key] = to_gemini_schema(value)
        elif isinstance(value, list):
            out[key] = [to_gemini_schema(v) if isinstance(v, dict) else v for v in value]
        else:
            out[key] = value
    return out


class GeminiProvider:
    def __init__(self, client: "genai.Client", model: str) -> None:
        self._client = client
        self._model = model

    def analyze_document(
        self,
        *,
        document: DocumentPayload,
        system: str,
        instruction: str,
        json_schema: dict[str, Any],
        max_tokens: int,
    ) -> StructuredResponse:
        """Send the document itself. PDFs go inline; Gemini reads pages natively."""
        from google.genai import types

        return self._generate(
            contents=[
                types.Part.from_bytes(data=document.data, mime_type=document.content_type),
                instruction,
            ],
            system=system,
            json_schema=json_schema,
            max_tokens=max_tokens,
        )

    def generate_structured(
        self,
        *,
        system: str,
        instruction: str,
        json_schema: dict[str, Any],
        max_tokens: int,
        model: str | None = None,
    ) -> StructuredResponse:
        """Text-only structured generation, with no document attached."""
        return self._generate(
            contents=[instruction],
            system=system,
            json_schema=json_schema,
            max_tokens=max_tokens,
            model=model,
        )

    def _generate(
        self,
        *,
        contents: list[Any],
        system: str,
        json_schema: dict[str, Any],
        max_tokens: int,
        model: str | None = None,
    ) -> StructuredResponse:
        from google.genai import types

        used = model or self._model
        # Translated OUTSIDE the try: a schema our translator cannot handle is a programming
        # error, not a transport failure, and must not be dressed up as transient and retried.
        gemini_schema = to_gemini_schema(json_schema)
        try:
            response = self._client.models.generate_content(
                model=used,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    response_mime_type="application/json",
                    response_schema=gemini_schema,
                    temperature=0.0,
                    max_output_tokens=max(max_tokens, MAX_OUTPUT_TOKENS),
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )
        except Exception as exc:
            # Same contract as the Anthropic provider: transport failures are transient.
            raise AIProviderError(f"gemini request failed: {type(exc).__name__}: {exc}") from exc

        return _to_structured_response(response, used)


def _to_structured_response(response: Any, model: str) -> StructuredResponse:
    """Map Gemini's response onto our provider-agnostic shape.

    ``finish_reason`` is normalised to the vocabulary ``StructuredResponse`` already
    understands, so ``refused`` and ``truncated`` mean the same thing whichever provider
    produced the response.
    """
    usage = getattr(response, "usage_metadata", None)
    text = getattr(response, "text", None)
    if not text:
        raise AIProviderError("gemini returned an empty response")

    candidates = getattr(response, "candidates", None) or []
    raw_finish = str(getattr(candidates[0], "finish_reason", "")) if candidates else ""
    stop_reason = _STOP_REASONS.get(raw_finish.rsplit(".", 1)[-1].upper())

    return StructuredResponse(
        text=text,
        provider=PROVIDER_NAME,
        model=model,
        stop_reason=stop_reason,
        usage=AIUsage(
            input_tokens=getattr(usage, "prompt_token_count", 0) or 0,
            output_tokens=getattr(usage, "candidates_token_count", 0) or 0,
        ),
    )


#: Gemini finish reasons -> the vocabulary StructuredResponse checks for. Anything not
#: listed maps to None, i.e. "completed normally".
_STOP_REASONS = {
    "MAX_TOKENS": "max_tokens",
    "SAFETY": "refusal",
    "PROHIBITED_CONTENT": "refusal",
    "BLOCKLIST": "refusal",
    "SPII": "refusal",
    "RECITATION": "refusal",
}
