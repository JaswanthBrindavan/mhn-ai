"""The reports.content / <section>.content payload.

`state` is mandatory and not inferable: `insights` is permanently null for a non-report
section, so its absence cannot mean "still working".
"""

from app.schemas.results import NameCheck
from app.services.assembly import CONTENT_SCHEMA_VERSION, ContentState


def test_schema_version_is_current() -> None:
    """2.0 moved the payload to filing time and added `state`/`section_extraction`;
    2.1 split each insight into four explanatory parts."""
    assert CONTENT_SCHEMA_VERSION == "2.1"


def test_content_states() -> None:
    assert ContentState.CLASSIFIED.value == "classified"
    assert ContentState.COMPLETE.value == "complete"
    assert ContentState.FAILED.value == "failed"


def test_name_check_carries_three_fields_and_not_the_account_holder() -> None:
    """The wire shape, guarded: the account holder's own name must never appear here —
    the client knows who is logged in, and this payload travels further than the dialog."""
    assert set(NameCheck(verdict="unknown").model_dump()) == {
        "verdict",
        "document_name",
        "confirmed",
    }
    assert NameCheck(verdict="unknown").document_name is None
    assert NameCheck(verdict="unknown").confirmed is False
