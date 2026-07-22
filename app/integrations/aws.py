"""The single place boto3 clients are constructed.

Switching between LocalStack and real AWS is an environment change only:

* ``AWS_ENDPOINT_URL`` set   -> LocalStack
* ``AWS_ENDPOINT_URL`` empty -> real AWS, endpoints resolved by boto3

Credentials always come from the default boto3 chain, so a local dummy key pair and
a production IAM role take the same code path.

Rules enforced here, not by convention:
  * No other module may call ``boto3.client``.
  * Queue and bucket are addressed only by configured URL/name — never assembled
    from account id or region, which would work on LocalStack and break on AWS.
"""

from functools import lru_cache
from typing import TYPE_CHECKING, Any

import boto3
from botocore.config import Config

from app.core.config import get_settings

if TYPE_CHECKING:  # pragma: no cover - typing only
    from mypy_boto3_s3.client import S3Client
    from mypy_boto3_sqs.client import SQSClient


def _client_kwargs() -> dict[str, Any]:
    settings = get_settings()
    kwargs: dict[str, Any] = {"region_name": settings.aws_region}
    if settings.aws_endpoint_url:
        kwargs["endpoint_url"] = settings.aws_endpoint_url
    return kwargs


@lru_cache
def get_s3_client() -> "S3Client":
    settings = get_settings()
    config = Config(
        # LocalStack needs path-style addressing; real AWS accepts it too, so this
        # stays a config toggle rather than a branch in application code.
        s3={"addressing_style": "path" if settings.s3_force_path_style else "auto"},
        retries={"max_attempts": 3, "mode": "standard"},
    )
    return boto3.client("s3", config=config, **_client_kwargs())


@lru_cache
def get_sqs_client() -> "SQSClient":
    config = Config(retries={"max_attempts": 3, "mode": "standard"})
    return boto3.client("sqs", config=config, **_client_kwargs())
