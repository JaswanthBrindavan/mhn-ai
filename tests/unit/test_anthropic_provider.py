"""The one piece that talks to Anthropic — verified against a stub client.

No network: a stub records the request so we can assert the beta flag, the structured-
output config, the document-vs-image block choice, and that the uploaded file is always
deleted. The live call itself is out of scope for CI.
"""

from types import SimpleNamespace

from app.integrations.ai.anthropic_provider import AnthropicProvider
from app.integrations.ai.base import DocumentPayload

_SCHEMA = {"type": "object", "properties": {}, "additionalProperties": False}


class _StubFiles:
    def __init__(self) -> None:
        self.uploaded: list = []
        self.deleted: list = []

    def upload(self, *, file):
        self.uploaded.append(file)
        return SimpleNamespace(id="file_123")

    def delete(self, file_id):
        self.deleted.append(file_id)


class _StubMessages:
    def __init__(self, response) -> None:
        self._response = response
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._response


class _StubClient:
    def __init__(self, response) -> None:
        self.beta = SimpleNamespace(files=_StubFiles(), messages=_StubMessages(response))


def _response(text: str = '{"ok": true}', stop_reason: str = "end_turn"):
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        usage=SimpleNamespace(
            input_tokens=11,
            output_tokens=7,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        ),
        model="claude-opus-4-8",
        stop_reason=stop_reason,
    )


def _analyze(client, document):
    return AnthropicProvider(client, "claude-opus-4-8").analyze_document(
        document=document,
        system="sys",
        instruction="do it",
        json_schema=_SCHEMA,
        max_tokens=2048,
    )


def test_generate_structured_sends_text_only_and_uploads_nothing():
    client = _StubClient(_response())

    result = AnthropicProvider(client, "claude-opus-4-8").generate_structured(
        system="sys", instruction="over this data", json_schema=_SCHEMA, max_tokens=1024
    )

    call = client.beta.messages.calls[0]
    assert call["output_config"]["format"]["schema"] is _SCHEMA
    content = call["messages"][0]["content"]
    assert content == [{"type": "text", "text": "over this data"}]
    # No document went through the Files API.
    assert client.beta.files.uploaded == []
    assert client.beta.files.deleted == []
    assert result.usage.output_tokens == 7


def test_pdf_uses_a_document_block_with_files_beta():
    client = _StubClient(_response())
    doc = DocumentPayload(data=b"%PDF-1.4", content_type="application/pdf", filename="r.pdf")

    result = _analyze(client, doc)

    call = client.beta.messages.calls[0]
    assert call["betas"] == ["files-api-2025-04-14"]
    assert call["output_config"]["format"]["type"] == "json_schema"
    assert call["output_config"]["format"]["schema"] is _SCHEMA
    block = call["messages"][0]["content"][0]
    assert block["type"] == "document"
    assert block["source"] == {"type": "file", "file_id": "file_123"}
    assert result.usage.input_tokens == 11
    assert result.stop_reason == "end_turn"


def test_image_uses_an_image_block():
    client = _StubClient(_response())
    doc = DocumentPayload(data=b"\x89PNG", content_type="image/png", filename="r.png")

    _analyze(client, doc)

    block = client.beta.messages.calls[0]["messages"][0]["content"][0]
    assert block["type"] == "image"


def test_upload_filename_is_sanitized_from_the_s3_key():
    # Filenames are S3 keys ("uploads/demo/report.pdf"); the Files API rejects the '/'
    # (forbidden character -> 400). The upload must send a safe basename.
    client = _StubClient(_response())
    doc = DocumentPayload(
        data=b"%PDF-1.4",
        content_type="application/pdf",
        filename="uploads/demo/sample report (1).pdf",
    )

    _analyze(client, doc)

    sent_filename = client.beta.files.uploaded[0][0]
    assert "/" not in sent_filename
    assert set(sent_filename) <= set(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-"
    )
    assert sent_filename.endswith(".pdf")


def test_uploaded_file_is_always_deleted():
    client = _StubClient(_response())
    doc = DocumentPayload(data=b"%PDF", content_type="application/pdf", filename="r.pdf")

    _analyze(client, doc)

    assert client.beta.files.deleted == ["file_123"]


def test_delete_runs_even_when_the_model_call_raises():
    client = _StubClient(_response())

    def _boom(**_kwargs):
        raise RuntimeError("api down")

    client.beta.messages.create = _boom  # type: ignore[method-assign]
    doc = DocumentPayload(data=b"%PDF", content_type="application/pdf", filename="r.pdf")

    import pytest

    from app.integrations.ai.base import AIProviderError

    with pytest.raises(AIProviderError):
        _analyze(client, doc)
    # The upload happened, so cleanup must still have run.
    assert client.beta.files.deleted == ["file_123"]
