"""Source-file validation: size, content type, and the octet-stream fallback."""

import pytest

from app.core.config import Settings
from app.integrations.s3 import ObjectMetadata
from app.services.source_validation import (
    resolve_content_type,
    validate_metadata,
)

SETTINGS = Settings(
    database_url="postgresql+psycopg://u:p@h:5432/d",
    max_file_bytes=1000,
    allowed_content_types="application/pdf,image/jpeg,image/png",
)


def meta(**kwargs) -> ObjectMetadata:
    defaults = {
        "key": "reports/x.pdf",
        "size_bytes": 100,
        "content_type": "application/pdf",
        "etag": "abc",
    }
    return ObjectMetadata(**{**defaults, "key": kwargs.pop("key", defaults["key"]), **kwargs})


# --- content type resolution ------------------------------------------------


def test_declared_content_type_wins():
    assert resolve_content_type(meta(content_type="application/pdf")) == "application/pdf"


def test_charset_parameter_is_stripped():
    assert resolve_content_type(meta(content_type="application/pdf; charset=binary")) == (
        "application/pdf"
    )


def test_content_type_is_normalised_to_lowercase():
    assert resolve_content_type(meta(content_type="APPLICATION/PDF")) == "application/pdf"


@pytest.mark.parametrize("declared", ["application/octet-stream", "", None])
def test_uninformative_type_falls_back_to_extension(declared):
    """Uploads often arrive as octet-stream. Rejecting those would refuse good PDFs."""
    resolved = resolve_content_type(meta(key="reports/scan.pdf", content_type=declared))
    assert resolved == "application/pdf"


def test_extension_fallback_handles_images():
    assert resolve_content_type(meta(key="a/b/c.JPEG", content_type=None)) == "image/jpeg"


def test_unknown_extension_and_type_resolves_to_none():
    assert resolve_content_type(meta(key="reports/file.xyz", content_type=None)) is None


# --- validation -------------------------------------------------------------


def test_valid_object_passes():
    assert validate_metadata(meta(), SETTINGS) is None


def test_empty_file_is_rejected():
    failure = validate_metadata(meta(size_bytes=0), SETTINGS)
    assert failure is not None
    assert failure.code == "empty_file"


def test_oversized_file_is_rejected():
    failure = validate_metadata(meta(size_bytes=1001), SETTINGS)
    assert failure is not None
    assert failure.code == "file_too_large"


def test_file_exactly_at_the_limit_is_accepted():
    assert validate_metadata(meta(size_bytes=1000), SETTINGS) is None


def test_unsupported_type_is_rejected():
    failure = validate_metadata(meta(key="a.docx", content_type="application/msword"), SETTINGS)
    assert failure is not None
    assert failure.code == "unsupported_content_type"


def test_undeterminable_type_is_rejected():
    failure = validate_metadata(meta(key="a.xyz", content_type=None), SETTINGS)
    assert failure is not None
    assert failure.code == "unknown_content_type"
