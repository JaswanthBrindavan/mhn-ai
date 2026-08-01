"""Ports of Spring's keyForType / previewKeyFor. Behaviour must match exactly:
a divergence puts AI-filed objects under a different prefix from hand-filed ones."""

import pytest

from app.services.s3_keys import key_for_section, preview_key_for


@pytest.mark.parametrize(
    ("key", "section", "expected"),
    [
        ("unclassified/a1b2-c3d4", "reports", "reports/a1b2-c3d4"),
        ("unclassified/a1b2-c3d4", "scans_imaging", "scans_imaging/a1b2-c3d4"),
        # No slash: the whole key is the object name.
        ("a1b2-c3d4", "reports", "reports/a1b2-c3d4"),
        # Only the FIRST slash separates prefix from name.
        ("unclassified/nested/name", "insurance", "insurance/nested/name"),
    ],
)
def test_key_for_section(key: str, section: str, expected: str) -> None:
    assert key_for_section(key, section) == expected


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("unclassified/a1b2", "unclassified_preview/a1b2"),
        ("reports/a1b2", "reports_preview/a1b2"),
        ("a1b2", "a1b2_preview"),
    ],
)
def test_preview_key_for(key: str, expected: str) -> None:
    assert preview_key_for(key) == expected


def test_filing_a_document_moves_both_keys_together() -> None:
    """The pair a filing does: object and preview end up under the same new prefix."""
    source = "unclassified/a1b2"
    filed = key_for_section(source, "vaccinations")
    assert filed == "vaccinations/a1b2"
    assert preview_key_for(filed) == "vaccinations_preview/a1b2"
