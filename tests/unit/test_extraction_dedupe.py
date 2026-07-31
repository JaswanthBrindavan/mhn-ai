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
