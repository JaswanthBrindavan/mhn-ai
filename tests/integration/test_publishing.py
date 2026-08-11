"""Queue publishing, source validation at submit, and the commit-before-publish rule."""

import json

import pytest
from sqlalchemy import text

from app.models.enums import RunItemStatus
from tests.integration.conftest import BUCKET

pytestmark = pytest.mark.integration


def _drain(sqs, queue_url) -> list[dict]:
    messages: list[dict] = []
    while True:
        batch = sqs.receive_message(QueueUrl=queue_url, MaxNumberOfMessages=10).get("Messages", [])
        if not batch:
            return messages
        messages.extend(batch)


# --- publishing -------------------------------------------------------------


def test_submitted_item_is_queued_and_published(api, make_document, aws):
    _, sqs, queue_url, _ = aws
    document_id = make_document()

    body = api.post(
        "/v1/document-processing-runs", json={"documents": [{"document_id": document_id}]}
    ).json()

    assert body["items"][0]["status"] == RunItemStatus.QUEUED.value

    messages = _drain(sqs, queue_url)
    assert len(messages) == 1
    payload = json.loads(messages[0]["Body"])
    assert payload["item_id"] == body["items"][0]["item_id"]
    assert payload["document_id"] == document_id


def test_message_contains_no_s3_key_or_report_data(api, make_document, aws):
    """The queue must never become a second copy of patient data."""
    _, sqs, queue_url, _ = aws
    api.post("/v1/document-processing-runs", json={"documents": [{"document_id": make_document()}]})

    payload = json.loads(_drain(sqs, queue_url)[0]["Body"])

    assert set(payload) == {"schema_version", "item_id", "run_id", "document_id", "attempt"}


def test_each_report_gets_exactly_one_message(api, make_document, aws):
    _, sqs, queue_url, _ = aws
    document_ids = [make_document(), make_document(), make_document()]

    api.post(
        "/v1/document-processing-runs",
        json={"documents": [{"document_id": d} for d in document_ids]},
    )

    assert len(_drain(sqs, queue_url)) == 3


def test_reused_item_is_not_published_twice(api, make_document, aws):
    """An in-flight item already has a message; re-submitting must not add another."""
    _, sqs, queue_url, _ = aws
    document_id = make_document()

    api.post("/v1/document-processing-runs", json={"documents": [{"document_id": document_id}]})
    second = api.post(
        "/v1/document-processing-runs", json={"documents": [{"document_id": document_id}]}
    ).json()

    assert second["items"][0]["outcome"] == "reused"
    assert len(_drain(sqs, queue_url)) == 1


def test_rejected_item_is_never_published(api, make_document, aws):
    _, sqs, queue_url, _ = aws
    bad = make_document(content_type="application/msword", suffix=".docx")

    body = api.post(
        "/v1/document-processing-runs", json={"documents": [{"document_id": bad}]}
    ).json()

    assert body["items"][0]["status"] == RunItemStatus.REJECTED.value
    assert _drain(sqs, queue_url) == []


# --- source validation at submit --------------------------------------------


def test_missing_s3_object_is_rejected_with_a_reason(api, make_document):
    document_id = make_document(upload=False)  # row exists, object does not

    body = api.post(
        "/v1/document-processing-runs", json={"documents": [{"document_id": document_id}]}
    ).json()

    item = body["items"][0]
    assert item["status"] == RunItemStatus.REJECTED.value
    assert item["error_code"] == "source_object_missing"


def test_oversized_file_is_rejected(api, make_document, test_settings):
    big = make_document(body=b"x" * (test_settings.max_file_bytes + 1))

    body = api.post(
        "/v1/document-processing-runs", json={"documents": [{"document_id": big}]}
    ).json()

    assert body["items"][0]["error_code"] == "file_too_large"


def test_unsupported_type_is_rejected(api, make_document):
    bad = make_document(content_type="application/msword", suffix=".docx")

    body = api.post(
        "/v1/document-processing-runs", json={"documents": [{"document_id": bad}]}
    ).json()

    assert body["items"][0]["error_code"] == "unsupported_content_type"


def test_octet_stream_pdf_is_accepted(api, make_document, aws):
    """Uploads often lose their content type; the extension must save them."""
    document_id = make_document(content_type="application/octet-stream", suffix=".pdf")

    body = api.post(
        "/v1/document-processing-runs", json={"documents": [{"document_id": document_id}]}
    ).json()

    assert body["items"][0]["status"] == RunItemStatus.QUEUED.value


def test_one_bad_report_does_not_block_the_rest_of_the_batch(api, make_document, aws):
    _, sqs, queue_url, _ = aws
    good_a, bad, good_b = make_document(), make_document(upload=False), make_document()

    body = api.post(
        "/v1/document-processing-runs",
        json={
            "documents": [
                {"document_id": good_a},
                {"document_id": bad},
                {"document_id": good_b},
            ]
        },
    ).json()

    statuses = {item["document_id"]: item["status"] for item in body["items"]}
    assert statuses[good_a] == RunItemStatus.QUEUED.value
    assert statuses[good_b] == RunItemStatus.QUEUED.value
    assert statuses[bad] == RunItemStatus.REJECTED.value
    assert len(_drain(sqs, queue_url)) == 2


def test_rejected_report_can_be_resubmitted_after_the_file_is_fixed(
    api, make_document, aws, db_session
):
    """Rejected is terminal, so the unique index does not block a later attempt."""
    s3 = aws[0]
    document_id = make_document(upload=False)
    first = api.post(
        "/v1/document-processing-runs", json={"documents": [{"document_id": document_id}]}
    ).json()
    assert first["items"][0]["status"] == RunItemStatus.REJECTED.value

    # The object shows up afterwards, e.g. an upload that had not finished.
    key = db_session.execute(
        text("SELECT filepath FROM unclassified_files WHERE id = :id"), {"id": document_id}
    ).scalar_one()
    s3.put_object(Bucket=BUCKET, Key=key, Body=b"%PDF-1.4 now here", ContentType="application/pdf")

    again = api.post(
        "/v1/document-processing-runs", json={"documents": [{"document_id": document_id}]}
    ).json()
    assert again["items"][0]["status"] == RunItemStatus.QUEUED.value


# --- failure handling -------------------------------------------------------


def test_publish_failure_fails_the_item_rather_than_leaving_it_pending(
    api, make_document, aws, db_session
):
    """An item with no message has no consumer, so it must not look accepted.

    It used to stay ``pending`` on the reasoning that a stale-item sweep would retry it.
    No sweep was ever built, so the document stopped for ever: no message, no error, and
    a 202 saying it had been taken. ``failed`` is visible on the run, on the status
    endpoint, and — unlike ``pending`` — is retryable through the retry endpoint.
    """
    _, sqs, queue_url, _ = aws
    document_id = make_document()
    sqs.delete_queue(QueueUrl=queue_url)  # publishing will now fail

    body = api.post(
        "/v1/document-processing-runs", json={"documents": [{"document_id": document_id}]}
    ).json()

    item = body["items"][0]
    # Still recorded and still 202: the run exists and the caller is told, per document,
    # exactly what happened to each one.
    assert item["status"] == RunItemStatus.FAILED.value
    assert item["error_code"] == "publish_failed"

    # And the state is durable, not just a field in the response.
    status = api.get(f"/v1/documents/{document_id}/status").json()
    assert status["status"] == RunItemStatus.FAILED.value
    assert status["last_error_code"] == "publish_failed"
