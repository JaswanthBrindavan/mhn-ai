"""The reports.content / <section>.content payload.

`state` is mandatory and not inferable: `insights` is permanently null for a non-report
section, so its absence cannot mean "still working".
"""

from app.services.assembly import CONTENT_SCHEMA_VERSION, ContentState


def test_schema_version_is_current() -> None:
    """2.0 moved the payload to filing time and added `state`/`section_extraction`;
    2.1 split each insight into four explanatory parts."""
    assert CONTENT_SCHEMA_VERSION == "2.1"


def test_content_states() -> None:
    assert ContentState.CLASSIFIED.value == "classified"
    assert ContentState.COMPLETE.value == "complete"
    assert ContentState.FAILED.value == "failed"
