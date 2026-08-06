"""Anthropic implementation of AIProvider.

This is the one place that talks to the model. It cannot be exercised in CI (a real
call costs money and is non-deterministic), so it is kept small and its request shape
is asserted with a stub client in unit tests.

Documents go through the Files API rather than inline base64: our 25 MB size limit
would inflate to ~34 MB of base64 and exceed the 32 MB request cap, and a stored file
can be reused across stages (classification now, extraction next).
"""

import io
import logging
import re
from typing import TYPE_CHECKING, Any

from app.integrations.ai.base import (
    AIProviderError,
    AIUsage,
    DocumentPayload,
    StructuredResponse,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from anthropic import Anthropic

logger = logging.getLogger(__name__)

#: Written to every ``ai_process_logs`` row this provider produces. That column is the
#: audit trail for which vendor received a document, so it is set here — beside the call
#: that actually sends it — rather than assumed by the logger.
PROVIDER_NAME = "anthropic"

DEFAULT_MODEL = "claude-opus-4-8"
_FILES_BETA = "files-api-2025-04-14"

# The Files API rejects filenames with forbidden characters (path separators, etc.).
# Our filenames are S3 keys like "uploads/demo/report.pdf" — always slash-bearing — so
# reduce to a safe basename before upload. The name is cosmetic: we reference the file
# by the returned file_id and delete it after the call.
_UNSAFE_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9._-]")


def _safe_upload_filename(name: str) -> str:
    base = name.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    cleaned = _UNSAFE_FILENAME_CHARS.sub("_", base).strip("._")
    return (cleaned or "document")[:255]


class AnthropicProvider:
    def __init__(self, client: "Anthropic", model: str) -> None:
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
        file_id = self._upload(document)
        # Typed as Any: these are documented request dicts the SDK accepts at runtime,
        # but their nested TypedDicts are impractical to satisfy statically.
        messages: Any = [
            {
                "role": "user",
                "content": [
                    _document_block(document, file_id),
                    {"type": "text", "text": instruction},
                ],
            }
        ]
        # Constrain the output shape; the JSON is still validated with Pydantic in the
        # service before anything is written.
        output_config: Any = {"format": {"type": "json_schema", "schema": json_schema}}
        try:
            response = self._client.beta.messages.create(
                model=self._model,
                max_tokens=max_tokens,
                betas=[_FILES_BETA],
                system=system,
                messages=messages,
                output_config=output_config,
            )
        except Exception as exc:
            raise AIProviderError(type(exc).__name__) from exc
        finally:
            self._delete(file_id)

        return _to_structured_response(response, self._model)

    def generate_structured(
        self,
        *,
        system: str,
        instruction: str,
        json_schema: dict[str, Any],
        max_tokens: int,
        model: str | None = None,
    ) -> StructuredResponse:
        chosen_model = model or self._model
        messages: Any = [{"role": "user", "content": [{"type": "text", "text": instruction}]}]
        output_config: Any = {"format": {"type": "json_schema", "schema": json_schema}}
        try:
            response = self._client.beta.messages.create(
                model=chosen_model,
                max_tokens=max_tokens,
                betas=[_FILES_BETA],
                system=system,
                messages=messages,
                output_config=output_config,
            )
        except Exception as exc:
            raise AIProviderError(type(exc).__name__) from exc

        return _to_structured_response(response, chosen_model)

    def _upload(self, document: DocumentPayload) -> str:
        try:
            uploaded = self._client.beta.files.upload(
                file=(
                    _safe_upload_filename(document.filename),
                    io.BytesIO(document.data),
                    document.content_type,
                ),
            )
        except Exception as exc:
            raise AIProviderError(type(exc).__name__) from exc
        return str(uploaded.id)

    def _delete(self, file_id: str) -> None:
        # Best effort: a stranded file is a minor storage cost, not a processing error.
        try:
            self._client.beta.files.delete(file_id)
        except Exception:
            logger.warning("ai_file_delete_failed", extra={"file_id": file_id})


def _document_block(document: DocumentPayload, file_id: str) -> dict[str, Any]:
    source = {"type": "file", "file_id": file_id}
    # PDFs and text are "document" blocks; images are "image" blocks.
    if document.content_type.startswith("image/"):
        return {"type": "image", "source": source}
    return {"type": "document", "source": source}


def _to_structured_response(response: Any, model: str) -> StructuredResponse:
    text = next(
        (block.text for block in response.content if getattr(block, "type", None) == "text"),
        "",
    )
    usage = response.usage
    return StructuredResponse(
        text=text,
        provider=PROVIDER_NAME,
        model=getattr(response, "model", model),
        stop_reason=getattr(response, "stop_reason", None),
        usage=AIUsage(
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            cache_read_input_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
            cache_creation_input_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
        ),
    )
