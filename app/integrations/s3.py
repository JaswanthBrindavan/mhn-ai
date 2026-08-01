"""Access to source document files in S3.

Read: ``s3:GetObject`` and ``s3:HeadObject``. Write: ``s3:PutObject`` and
``s3:DeleteObject``, used **only** by the filing step, which relocates a classified
document from the ``unclassified/`` prefix into its section's prefix (copy, then delete the
original after the database commit — see ``app.services.filing``).

It still never changes ACLs and never makes an object public. Buckets and keys are internal
detail: they must not appear in API responses.
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


@dataclass(frozen=True)
class ObjectContent:
    """A downloaded object: its bytes plus the metadata from the same GET."""

    metadata: ObjectMetadata
    data: bytes


def get_object(client: "S3Client", bucket: str, key: str) -> ObjectContent:
    """Download an object's bytes. Same missing/transient split as ``head_object``."""
    try:
        response = client.get_object(Bucket=bucket, Key=key)
        data = response["Body"].read()
    except ClientError as exc:
        code = str(exc.response.get("Error", {}).get("Code", ""))
        status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        if code in _MISSING_CODES or code in _DENIED_CODES or status in (403, 404):
            raise SourceObjectMissingError(key) from exc
        logger.warning("s3_get_object_failed", extra={"error_code": code, "http_status": status})
        raise SourceObjectUnavailableError(code) from exc
    except Exception as exc:
        raise SourceObjectUnavailableError(str(type(exc).__name__)) from exc

    etag = response.get("ETag")
    metadata = ObjectMetadata(
        key=key,
        size_bytes=len(data),
        content_type=response.get("ContentType"),
        etag=etag.strip('"') if etag else None,
    )
    return ObjectContent(metadata=metadata, data=data)


def copy_object(client: "S3Client", bucket: str, from_key: str, to_key: str) -> None:
    """Server-side copy within the bucket. Idempotent: re-copying overwrites the same key.

    Same missing/transient split as ``get_object``, so a redelivered filing retries a blip
    and permanently rejects a document whose source has genuinely gone.
    """
    try:
        client.copy_object(
            Bucket=bucket, Key=to_key, CopySource={"Bucket": bucket, "Key": from_key}
        )
    except ClientError as exc:
        code = str(exc.response.get("Error", {}).get("Code", ""))
        status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        if code in _MISSING_CODES or code in _DENIED_CODES or status in (403, 404):
            raise SourceObjectMissingError(from_key) from exc
        logger.warning("s3_copy_object_failed", extra={"error_code": code, "http_status": status})
        raise SourceObjectUnavailableError(code) from exc
    except Exception as exc:
        raise SourceObjectUnavailableError(str(type(exc).__name__)) from exc


def delete_object(client: "S3Client", bucket: str, key: str) -> None:
    """Delete an object. A missing key is not an error — S3 treats DELETE as idempotent."""
    try:
        client.delete_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        code = str(exc.response.get("Error", {}).get("Code", ""))
        logger.warning("s3_delete_object_failed", extra={"error_code": code})
        raise SourceObjectUnavailableError(code) from exc
    except Exception as exc:
        raise SourceObjectUnavailableError(str(type(exc).__name__)) from exc


def object_exists(client: "S3Client", bucket: str, key: str) -> bool:
    """Whether an object is there, without raising — used to treat a preview as optional."""
    try:
        head_object(client, bucket, key)
    except SourceObjectMissingError:
        return False
    return True
