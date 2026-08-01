"""S3 and SQS integration behaviour, against moto rather than real AWS."""

import json
import uuid

import boto3
import pytest
from moto import mock_aws

from app.integrations.s3 import (
    SourceObjectMissingError,
    SourceObjectUnavailableError,
    copy_object,
    delete_object,
    head_object,
    object_exists,
)
from app.integrations.sqs import (
    MESSAGE_SCHEMA_VERSION,
    PublishError,
    build_message,
    publish_processing_item,
)

REGION = "ap-south-1"
BUCKET = "mhn-reports-test"


@pytest.fixture
def s3():
    with mock_aws():
        client = boto3.client("s3", region_name=REGION)
        client.create_bucket(
            Bucket=BUCKET, CreateBucketConfiguration={"LocationConstraint": REGION}
        )
        yield client


@pytest.fixture
def sqs():
    with mock_aws():
        client = boto3.client("sqs", region_name=REGION)
        url = client.create_queue(QueueName="report-processing")["QueueUrl"]
        yield client, url


# --- S3 ---------------------------------------------------------------------


def test_head_object_returns_metadata(s3):
    s3.put_object(Bucket=BUCKET, Key="reports/a.pdf", Body=b"x" * 42, ContentType="application/pdf")

    meta = head_object(s3, BUCKET, "reports/a.pdf")

    assert meta.size_bytes == 42
    assert meta.content_type == "application/pdf"
    assert meta.key == "reports/a.pdf"
    assert meta.etag and '"' not in meta.etag  # quotes stripped for use as a hash


def test_missing_object_raises_source_object_missing(s3):
    with pytest.raises(SourceObjectMissingError):
        head_object(s3, BUCKET, "reports/nope.pdf")


def test_missing_bucket_is_not_treated_as_transient(s3):
    """A wrong bucket is a permanent misconfiguration, surfaced as missing."""
    with pytest.raises(SourceObjectMissingError):
        head_object(s3, "bucket-that-does-not-exist", "a.pdf")


def test_connection_failure_is_transient(s3):
    """A transient failure must NOT permanently reject a valid report."""

    class Boom:
        def head_object(self, **_kwargs):
            raise OSError("connection reset")

    with pytest.raises(SourceObjectUnavailableError):
        head_object(Boom(), BUCKET, "reports/a.pdf")  # type: ignore[arg-type]


def test_copy_object_duplicates_the_object(s3):
    s3.put_object(Bucket=BUCKET, Key="unclassified/a1b2", Body=b"pdf-bytes")

    copy_object(s3, BUCKET, "unclassified/a1b2", "reports/a1b2")

    assert s3.get_object(Bucket=BUCKET, Key="reports/a1b2")["Body"].read() == b"pdf-bytes"
    # The source is untouched: deleting it is a separate, later step.
    assert s3.get_object(Bucket=BUCKET, Key="unclassified/a1b2")["Body"].read() == b"pdf-bytes"


def test_copy_object_missing_source_is_permanent(s3):
    with pytest.raises(SourceObjectMissingError):
        copy_object(s3, BUCKET, "unclassified/gone", "reports/gone")


def test_copy_object_is_idempotent(s3):
    """A redelivered filing re-copies over the same key rather than failing."""
    s3.put_object(Bucket=BUCKET, Key="unclassified/a1b2", Body=b"pdf-bytes")
    copy_object(s3, BUCKET, "unclassified/a1b2", "reports/a1b2")
    copy_object(s3, BUCKET, "unclassified/a1b2", "reports/a1b2")
    assert s3.get_object(Bucket=BUCKET, Key="reports/a1b2")["Body"].read() == b"pdf-bytes"


def test_object_exists(s3):
    s3.put_object(Bucket=BUCKET, Key="unclassified_preview/a1b2", Body=b"png")
    assert object_exists(s3, BUCKET, "unclassified_preview/a1b2") is True
    assert object_exists(s3, BUCKET, "unclassified_preview/nope") is False


def test_delete_object_removes_it_and_tolerates_absence(s3):
    s3.put_object(Bucket=BUCKET, Key="unclassified/a1b2", Body=b"pdf-bytes")
    delete_object(s3, BUCKET, "unclassified/a1b2")
    assert object_exists(s3, BUCKET, "unclassified/a1b2") is False
    # S3 delete is a no-op on a missing key; a redelivered filing must not fail here.
    delete_object(s3, BUCKET, "unclassified/a1b2")


# --- SQS --------------------------------------------------------------------


def test_message_body_carries_identifiers_only():
    """No report contents, no S3 keys, no patient data on the queue."""
    body = build_message(item_id=uuid.uuid4(), run_id=uuid.uuid4(), document_id=7, attempt=0)
    assert set(body) == {"schema_version", "item_id", "run_id", "document_id", "attempt"}
    assert body["schema_version"] == MESSAGE_SCHEMA_VERSION


def test_publish_puts_a_readable_message_on_the_queue(sqs):
    client, url = sqs
    item_id, run_id = uuid.uuid4(), uuid.uuid4()

    message_id = publish_processing_item(
        client, url, item_id=item_id, run_id=run_id, document_id=42
    )

    assert message_id
    received = client.receive_message(QueueUrl=url, MaxNumberOfMessages=1)["Messages"][0]
    body = json.loads(received["Body"])
    assert body["item_id"] == str(item_id)
    assert body["document_id"] == 42
    # The key must not travel with the message.
    assert "filepath" not in body


def test_publish_failure_raises_publish_failed(sqs):
    client, _ = sqs
    with pytest.raises(PublishError):
        publish_processing_item(
            client,
            "https://sqs.ap-south-1.amazonaws.com/000000000000/does-not-exist",
            item_id=uuid.uuid4(),
            run_id=uuid.uuid4(),
            document_id=1,
        )
