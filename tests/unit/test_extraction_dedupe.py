"""Deduplication of repeated test rows.

`INSTRUCTION` demands every row of every table, because without that a model returns a
representative sample of a long report and reports success. The price of demanding it is
that a model may emit its summary pass *and* its full pass, repeating some tests. That is
cleaned up deterministically here rather than by prompting harder.
"""

import pytest

from app.services.extraction import ExtractedLabResult, _dedupe_results


def _row(name: str, value=None, unit=None, reference_range=None) -> ExtractedLabResult:
    return ExtractedLabResult(
        test_name=name, value=value, unit=unit, reference_range=reference_range
    )


def test_identical_duplicates_collapse_to_one():
    rows = [_row("Hemoglobin", "15.0", "g/dL", "13 - 17")] * 2
    assert len(_dedupe_results(rows)) == 1


def test_the_more_informative_duplicate_wins():
    """Observed live: one copy carries the reference range, the other lost it."""
    without = _row("Average Blood Glucose", "166", "mg/dL", None)
    with_range = _row("Average Blood Glucose", "166", "mg/dL", "90-120")
    assert _dedupe_results([without, with_range])[0].reference_range == "90-120"
    # Order of arrival must not change the outcome.
    assert _dedupe_results([with_range, without])[0].reference_range == "90-120"


def test_original_order_is_preserved():
    rows = [_row("C"), _row("A"), _row("B"), _row("A")]
    assert [r.test_name for r in _dedupe_results(rows)] == ["C", "A", "B"]


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("BILIRUBIN - DIRECT", "BILIRUBIN -DIRECT"),  # whitespace differs
        ("Hemoglobin", "HEMOGLOBIN"),  # case differs
        ("Total  RBC", "Total RBC"),  # collapsed spaces
    ],
)
def test_names_matching_only_after_normalisation_are_the_same_test(first, second):
    assert len(_dedupe_results([_row(first, "1"), _row(second, "1")])) == 1


def test_distinct_tests_are_kept():
    rows = [_row("Hemoglobin"), _row("Hematocrit"), _row("Total RBC")]
    assert len(_dedupe_results(rows)) == 3


def test_empty_input_is_fine():
    assert _dedupe_results([]) == []


def _dated(name: str, value: str, observed_date: str | None) -> ExtractedLabResult:
    return ExtractedLabResult(test_name=name, value=value, observed_date=observed_date)


def test_the_same_test_on_two_dates_is_two_results() -> None:
    """A cumulative report's whole point is the trend, and a name-only key ate it.

    Indian labs print serial reports: creatinine in January and again in June, one row
    each. Keyed on the name alone these collapsed to whichever copy scored higher on
    `_informativeness` — not even the more recent one — so a rising creatinine was stored
    as a single value with nothing to compare it to, and nothing downstream could tell.
    """
    rows = [_dated("Creatinine", "0.9", "2026-01-14"), _dated("Creatinine", "1.6", "2026-06-02")]

    kept = _dedupe_results(rows)

    assert [r.value for r in kept] == ["0.9", "1.6"]


def test_the_double_pass_still_collapses_when_the_dates_agree() -> None:
    """The case dedupe exists for is untouched: same test, same date, one row kept."""
    rows = [_dated("Creatinine", "0.9", "2026-01-14"), _dated("Creatinine", "0.9", "2026-01-14")]

    assert len(_dedupe_results(rows)) == 1


def test_a_duplicate_where_only_one_copy_is_dated_survives_as_two() -> None:
    """Deliberate, and the safe direction.

    We cannot tell "the model dated one copy and not the other" from "two genuine
    observations, one undated". A visible duplicate is something a reader can see and a
    curator can fix; a dropped observation is invisible to both. Same trade `INSTRUCTION`
    already makes by asking for completeness and cleaning up afterwards.
    """
    rows = [_dated("Creatinine", "0.9", None), _dated("Creatinine", "0.9", "2026-01-14")]

    assert len(_dedupe_results(rows)) == 2
