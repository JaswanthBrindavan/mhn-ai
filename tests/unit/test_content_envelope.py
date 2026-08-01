"""The reports.content / <section>.content payload.

`state` is mandatory and not inferable: `insights` is permanently null for a non-report
section, so its absence cannot mean "still working".
"""

from app.services.assembly import CONTENT_SCHEMA_VERSION, ContentState


def test_schema_version_is_two() -> None:
    """Bumped because the payload now appears before extraction has run, and gained
    `state` and `section_extraction`."""
    assert CONTENT_SCHEMA_VERSION == "2.0"


def test_content_states() -> None:
    assert ContentState.CLASSIFIED.value == "classified"
    assert ContentState.COMPLETE.value == "complete"
    assert ContentState.FAILED.value == "failed"
