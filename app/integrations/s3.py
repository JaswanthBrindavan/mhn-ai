"""Read-only access to source report files in S3.

This service needs ``s3:GetObject`` and ``s3:HeadObject`` and nothing more. It never
writes, deletes, or changes ACLs, and never makes an object public. Buckets and keys are
internal detail: they must not appear in API responses.
"""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from botocore.exceptions import ClientError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from mypy_boto3_s3.client import S3Client

logger = logging.getLogger(__name__)

#: Error codes S3 returns for "not there" and "not allowed to look". A 403 is treated
#: as absent on purpose: with ListBucket denied, S3 returns 403 for missing keys, and
#: distinguishing them would require a permission we deliberately do not hold.
_MISSING_CODES = frozenset({"404", "NoSuchKey", "NotFound"})
_DENIED_CODES = frozenset({"403", "AccessDenied", "Forbidden"})


class SourceObjectMissingError(Exception):
    """The object does not exist, or is not readable with our credentials."""


class SourceObjectUnavailableError(Exception):
    """S3 could not be reached, or failed transiently. Retry later."""


@dataclass(frozen=True)
class ObjectMetadata:
    key: str
    size_bytes: int
    content_type: str | None
    #: S3's checksum. Used later as part of the "this exact file" idempotency key.
    etag: str | None


def head_object(client: "S3Client", bucket: str, key: str) -> ObjectMetadata:
    """Fetch object metadata without downloading it.

    Raises ``SourceObjectMissingError`` for a permanent problem and
    ``SourceObjectUnavailableError`` for a transient one, so callers can reject the first
    and retry the second.
    """
    try:
        response = client.head_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        code = str(exc.response.get("Error", {}).get("Code", ""))
        status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        if code in _MISSING_CODES or code in _DENIED_CODES or status in (403, 404):
            raise SourceObjectMissingError(key) from exc
        # Throttling, 5xx, network problems: the object may well be fine.
        logger.warning("s3_head_object_failed", extra={"error_code": code, "http_status": status})
        raise SourceObjectUnavailableError(code) from exc
    except Exception as exc:  # connection errors, DNS, timeouts
        raise SourceObjectUnavailableError(str(type(exc).__name__)) from exc

    etag = response.get("ETag")
    return ObjectMetadata(
        key=key,
        size_bytes=int(response.get("ContentLength", 0)),
        content_type=response.get("ContentType"),
        etag=etag.strip('"') if etag else None,
    )
