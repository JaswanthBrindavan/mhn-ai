"""Publishing work onto the report-processing queue.

One standard queue for all report jobs, with a dead-letter queue behind a redrive
policy. Standard queues are at-least-once, so every worker action must be idempotent —
the message is a pointer to work, never the work itself.

The message body carries **identifiers only**: no report contents, no S3 keys, no
patient data. A queue is a copy of your data in another system with its own retention
and access rules; the worker re-reads what it needs from the database instead.
"""

import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from uuid import UUID

from botocore.exceptions import ClientError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from mypy_boto3_sqs.client import SQSClient

logger = logging.getLogger(__name__)

#: Bumped when the message shape changes, so a worker can reject what it cannot read.
MESSAGE_SCHEMA_VERSION = 1


class PublishError(Exception):
    """The message could not be enqueued."""


@dataclass(frozen=True)
class ReceivedMessage:
    """A parsed, schema-checked message ready to process.

    ``receipt_handle`` is the token used to delete the message or extend its
    visibility — it is specific to this receipt, not the message, so it must come
    from this exact receive call.
    """

    receipt_handle: str
    item_id: UUID
    run_id: UUID
    document_id: int
    #: SQS's count of how many times this message has been delivered. The redrive
    #: policy sends it to the DLQ once this exceeds maxReceiveCount.
    approx_receive_count: int


def build_message(*, item_id: UUID, run_id: UUID, document_id: int, attempt: int) -> dict[str, Any]:
    return {
        "schema_version": MESSAGE_SCHEMA_VERSION,
        "item_id": str(item_id),
        "run_id": str(run_id),
        "document_id": document_id,
        "attempt": attempt,
    }


def publish_processing_item(
    sqs: "SQSClient",
    queue_url: str,
    *,
    item_id: UUID,
    run_id: UUID,
    document_id: int,
    attempt: int = 0,
) -> str:
    """Enqueue one report for processing. Returns the SQS message id."""
    body = build_message(item_id=item_id, run_id=run_id, document_id=document_id, attempt=attempt)
    try:
        response = sqs.send_message(
            QueueUrl=queue_url,
            MessageBody=json.dumps(body, separators=(",", ":")),
            MessageAttributes={
                # Cheap to filter on without parsing the body.
                "item_id": {"StringValue": str(item_id), "DataType": "String"},
                "schema_version": {
                    "StringValue": str(MESSAGE_SCHEMA_VERSION),
                    "DataType": "Number",
                },
            },
        )
    except ClientError as exc:
        code = str(exc.response.get("Error", {}).get("Code", ""))
        raise PublishError(code) from exc
    except Exception as exc:
        raise PublishError(type(exc).__name__) from exc

    return response["MessageId"]


def receive_messages(
    sqs: "SQSClient",
    queue_url: str,
    *,
    max_messages: int,
    wait_seconds: int,
    visibility_timeout: int,
) -> list[ReceivedMessage]:
    """Long-poll for up to ``max_messages`` messages.

    A message whose body is unparseable or whose schema version we do not recognise
    is skipped WITHOUT deleting it: it stays on the queue and, after the redrive
    policy's maxReceiveCount, moves to the DLQ. Deleting it here would silently drop
    work — during a rolling deploy a newer producer's message may just be unreadable
    by an older worker, not truly poison.
    """
    response = sqs.receive_message(
        QueueUrl=queue_url,
        MaxNumberOfMessages=max(1, min(10, max_messages)),
        WaitTimeSeconds=wait_seconds,
        VisibilityTimeout=visibility_timeout,
        MessageSystemAttributeNames=["ApproximateReceiveCount"],
    )

    parsed: list[ReceivedMessage] = []
    for raw in response.get("Messages", []):
        message = _parse(raw)
        if message is not None:
            parsed.append(message)
    return parsed


def _parse(raw: Any) -> ReceivedMessage | None:
    receipt = raw.get("ReceiptHandle", "")
    try:
        body = json.loads(raw.get("Body", ""))
    except (ValueError, TypeError):
        logger.warning("sqs_message_unparseable", extra={"message_id": raw.get("MessageId")})
        return None

    version = body.get("schema_version")
    if version != MESSAGE_SCHEMA_VERSION:
        # Unknown/newer shape: leave it for a worker that understands it, or the DLQ.
        logger.warning(
            "sqs_message_schema_mismatch",
            extra={"message_id": raw.get("MessageId"), "schema_version": version},
        )
        return None

    try:
        return ReceivedMessage(
            receipt_handle=receipt,
            item_id=UUID(str(body["item_id"])),
            run_id=UUID(str(body["run_id"])),
            document_id=int(body["document_id"]),
            approx_receive_count=int(raw.get("Attributes", {}).get("ApproximateReceiveCount", 1)),
        )
    except (KeyError, ValueError, TypeError):
        logger.warning("sqs_message_missing_fields", extra={"message_id": raw.get("MessageId")})
        return None


def delete_message(sqs: "SQSClient", queue_url: str, receipt_handle: str) -> None:
    """Acknowledge a message so it is not redelivered. Best effort.

    A failed delete is not fatal: the message becomes visible again and is
    reprocessed, which is safe because every worker action is idempotent. Never let
    a delete failure crash the item that already finished its work.
    """
    try:
        sqs.delete_message(QueueUrl=queue_url, ReceiptHandle=receipt_handle)
    except ClientError as exc:
        code = str(exc.response.get("Error", {}).get("Code", ""))
        logger.warning("sqs_delete_failed", extra={"error_code": code})


def extend_visibility(
    sqs: "SQSClient", queue_url: str, receipt_handle: str, visibility_timeout: int
) -> bool:
    """Push out the message's visibility deadline. Returns False on failure.

    Used by the heartbeat while a long stage runs, so the message is not redelivered
    to another worker mid-flight.
    """
    try:
        sqs.change_message_visibility(
            QueueUrl=queue_url,
            ReceiptHandle=receipt_handle,
            VisibilityTimeout=visibility_timeout,
        )
    except ClientError as exc:
        code = str(exc.response.get("Error", {}).get("Code", ""))
        logger.warning("sqs_extend_visibility_failed", extra={"error_code": code})
        return False
    return True
