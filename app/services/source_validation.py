"""Validation of the source file before any AI work is spent on it.

Checks existence, content type, and size. This is a *sanity* layer, not access
control — see ``app/api/deps.py``.
"""

import logging
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

from app.core.config import Settings
from app.integrations.s3 import (
    ObjectMetadata,
    SourceObjectMissingError,
    SourceObjectUnavailableError,
    head_object,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from mypy_boto3_s3.client import S3Client

logger = logging.getLogger(__name__)

#: Content types that carry no information, so the key's extension is more reliable.
_UNINFORMATIVE_TYPES = frozenset(
    {"", "application/octet-stream", "binary/octet-stream", "application/x-www-form-urlencoded"}
)

_EXTENSION_TYPES = {
    ".pdf": "application/pdf",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
}


@dataclass(frozen=True)
class ValidationFailure:
    """A permanent reason this report cannot be processed."""

    code: str
    message: str


def resolve_content_type(meta: ObjectMetadata) -> str | None:
    """Best available content type for an object.

    Uploads frequently arrive as ``application/octet-stream`` — a browser or SDK that
    did not set the type. Rejecting those would refuse perfectly good PDFs, so fall
    back to the key's extension when the declared type carries no information.
    """
    declared = (meta.content_type or "").split(";")[0].strip().lower()
    if declared and declared not in _UNINFORMATIVE_TYPES:
        return declared

    suffix = PurePosixPath(meta.key).suffix.lower()
    return _EXTENSION_TYPES.get(suffix)


def validate_metadata(meta: ObjectMetadata, settings: Settings) -> ValidationFailure | None:
    """Return a failure, or None when the object is acceptable."""
    if meta.size_bytes <= 0:
        return ValidationFailure("empty_file", "Source file is empty")

    if meta.size_bytes > settings.max_file_bytes:
        return ValidationFailure(
            "file_too_large",
            f"Source file exceeds the {settings.max_file_bytes} byte limit",
        )

    content_type = resolve_content_type(meta)
    if content_type is None:
        return ValidationFailure(
            "unknown_content_type",
            "Could not determine the file type from metadata or extension",
        )
    if content_type not in settings.allowed_content_type_set:
        return ValidationFailure(
            "unsupported_content_type",
            f"Content type {content_type} is not supported",
        )

    return None


def validate_source_object(
    s3: "S3Client", settings: Settings, filepath: str
) -> tuple[ObjectMetadata | None, ValidationFailure | None]:
    """Head the object and validate it.

    ``SourceObjectUnavailableError`` is deliberately NOT converted into a failure: a
    transient S3 problem must not permanently reject a valid report. It propagates so
    the caller can answer 503 and let the client retry.
    """
    if not filepath:
        return None, ValidationFailure("missing_filepath", "Report has no source file")

    try:
        meta = head_object(s3, settings.s3_bucket, filepath)
    except SourceObjectMissingError:
        return None, ValidationFailure("source_object_missing", "Source file was not found")

    return meta, validate_metadata(meta, settings)


__all__ = [
    "SourceObjectUnavailableError",
    "ValidationFailure",
    "resolve_content_type",
    "validate_metadata",
    "validate_source_object",
]
