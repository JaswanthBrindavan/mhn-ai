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


def build_message(*, item_id: UUID, run_id: UUID, report_id: int, attempt: int) -> dict[str, Any]:
    return {
        "schema_version": MESSAGE_SCHEMA_VERSION,
        "item_id": str(item_id),
        "run_id": str(run_id),
        "report_id": report_id,
        "attempt": attempt,
    }


def publish_processing_item(
    sqs: "SQSClient",
    queue_url: str,
    *,
    item_id: UUID,
    run_id: UUID,
    report_id: int,
    attempt: int = 0,
) -> str:
    """Enqueue one report for processing. Returns the SQS message id."""
    body = build_message(item_id=item_id, run_id=run_id, report_id=report_id, attempt=attempt)
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
