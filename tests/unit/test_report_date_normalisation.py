"""The report's own date, stored in one shape rather than the lab's.

`section_extraction` has always run its dates through `iso_date`; a report's
`report_date` was written straight through from the model, and the prompt asks
for "the report's overall date if shown" without naming a format. So the stored
value was whatever the lab printed — production rows hold "02 Sep 2026" — and
every consumer had to guess. The ones that assumed ISO read the date as absent
and fell back to the upload time, which is how a March scan came to list as
having happened the night somebody uploaded it.
"""

from __future__ import annotations

import pytest

from app.services.dates import iso_date


@pytest.mark.parametrize(
    ("printed", "stored"),
    [
        # The shape production is actually holding.
        ("02 Sep 2026", "2026-09-02"),
        # Already ISO: normalising must be a no-op, not a reformat.
        ("2026-09-02", "2026-09-02"),
        # Common on Indian lab reports.
        ("18-Mar-2026", "2026-03-18"),
        ("02/09/2026", "2026-09-02"),
        ("28th July 2026", "2026-07-28"),
        # DICOM StudyDate, off imaging reports.
        ("20260902", "2026-09-02"),
    ],
)
def test_a_printed_date_is_stored_as_iso(printed: str, stored: str):
    assert iso_date(printed) == stored


@pytest.mark.parametrize("unreadable", ["", "   ", "last Tuesday", "not shown", None])
def test_an_unreadable_date_is_stored_as_absent(unreadable):
    """The same answer as a document that printed no date, and the right one.

    A date nobody can parse is not a date. Keeping the raw string would only
    move the guessing downstream, to consumers with less context than this.
    """
    assert iso_date(unreadable) is None


def test_the_extraction_payload_normalises_on_the_way_out():
    """The payload builder is where this has to happen — it is the one place
    every report's extraction passes through before it is stored."""
    import inspect

    from app.services import extraction

    source = inspect.getsource(extraction)
    assert '"report_date": iso_date(result.report_date)' in source, (
        "report_date must be normalised where the payload is built, or rows go "
        "back to holding whatever the lab printed"
    )
