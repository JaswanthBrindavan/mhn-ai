"""Deduplication of repeated test rows.

`INSTRUCTION` demands every row of every table, because without that a model returns a
representative sample of a long report and reports success. The price of demanding it is
that a model may emit its summary pass *and* its full pass, repeating some tests. That is
cleaned up deterministically here rather than by prompting harder.
"""

import pytest

from app.services.extraction import (
    EXTRACT_MAX_TOKENS,
    ExtractedLabResult,
    _dedupe_results,
    _inherit_ranges,
    mark_superseded,
)


def _row(
    name: str, value=None, unit=None, reference_range=None, observed_date=None
) -> ExtractedLabResult:
    return ExtractedLabResult(
        test_name=name,
        value=value,
        unit=unit,
        reference_range=reference_range,
        observed_date=observed_date,
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


def test_an_abnormality_marker_does_not_split_one_reading_in_two():
    """A cumulative report prints the same reading twice, and marks only one copy.

    MedPlus flags an out-of-range value with a trailing asterisk — its legend reads
    "Abnormal * Critical" — so the current-visit column gave
    `Cholesterol - LDL (Direct) *` WITH a reference range and the comparison table gave
    `Cholesterol - LDL (Direct)` with none. Different keys, so both survived, and the
    reader saw one LDL flagged high beside a second identical LDL reported as impossible
    to check.
    """
    marked = _row("Cholesterol - LDL (Direct) *", "123", None, "0-100", "18-Mar-26")
    plain = _row("Cholesterol - LDL (Direct)", "123", None, None, "18-Mar-26")

    kept = _dedupe_results([marked, plain])

    assert len(kept) == 1
    # The copy carrying the range wins, and the name is stored exactly as printed: the
    # marker is ignored for MATCHING, never edited out of the transcription.
    assert kept[0].reference_range == "0-100"
    assert kept[0].test_name == "Cholesterol - LDL (Direct) *"


def test_the_same_test_on_two_dates_still_survives():
    """The marker fix must not undo what the date in the key is there to protect."""
    rows = [
        _row("Triglycerides", "139", None, None, "18-Mar-26"),
        _row("Triglycerides", "219", None, None, "25-Nov-24"),
    ]
    assert len(_dedupe_results(rows)) == 2


def test_a_previous_visit_reading_is_marked_superseded():
    """The flag every consumer reads instead of re-deriving "which one is current".

    Computed once here for the same reason abnormal flags, dates and money are: the
    alternative is a second implementation in the app, and two answers to "which value is
    today's" is strictly worse than none.
    """
    rows = mark_superseded(
        [
            {"test_name": "Triglycerides", "observed_date": "18-Mar-26"},
            {"test_name": "Triglycerides", "observed_date": "25-Nov-24"},
        ]
    )
    assert [r["superseded"] for r in rows] == [False, True]


def test_a_marker_on_the_current_copy_does_not_hide_the_supersession():
    """The current reading is the flagged one, so the names differ by an asterisk."""
    rows = mark_superseded(
        [
            {"test_name": "Cholesterol - LDL (Direct) *", "observed_date": "18-Mar-26"},
            {"test_name": "Cholesterol - LDL (Direct)", "observed_date": "25-Nov-24"},
        ]
    )
    assert [r["superseded"] for r in rows] == [False, True]


def test_nothing_is_superseded_without_a_later_date():
    """Undated rows both stand, a dated row never displaces an undated one, and an
    ordinary single-visit report is untouched — every row False."""
    undated = mark_superseded(
        [{"test_name": "Ferritin"}, {"test_name": "Ferritin", "observed_date": None}]
    )
    assert [r["superseded"] for r in undated] == [False, False]

    mixed = mark_superseded(
        [
            {"test_name": "Ferritin", "observed_date": "18-Mar-26"},
            {"test_name": "Ferritin", "observed_date": None},
        ]
    )
    assert [r["superseded"] for r in mixed] == [False, False]

    one_visit = mark_superseded(
        [
            {"test_name": "Triglycerides", "observed_date": "18-Mar-26"},
            {"test_name": "Cholesterol - HDL", "observed_date": "18-Mar-26"},
        ]
    )
    assert [r["superseded"] for r in one_visit] == [False, False]


def test_a_previous_reading_inherits_the_interval_printed_once():
    """A comparison table prints its reference interval once, for both columns.

    On document 114 the current-visit table carries BRI and the previous-visit table
    below it has no such column, so 142 arrived with no range, could not be flagged, and
    rendered as "Not checked" — directly beside its own current reading showing
    "Range 0-100".
    """
    rows = _inherit_ranges(
        [
            _row("Cholesterol - LDL (Direct) *", "123", None, "0-100", "18-Mar-26"),
            _row("Cholesterol - LDL (Direct)", "142", None, None, "25-Nov-24"),
        ]
    )

    # Matched through name_key, so the marker on the current copy is no obstacle.
    assert [r.reference_range for r in rows] == ["0-100", "0-100"]
    # Only the range travels.
    assert [r.value for r in rows] == ["123", "142"]
    assert [r.observed_date for r in rows] == ["18-Mar-26", "25-Nov-24"]


def test_a_printed_interval_is_never_replaced():
    """A row stating its own interval keeps it — including a sex-split pair where the
    two genuinely differ and inheriting would flag against the wrong sex."""
    rows = _inherit_ranges(
        [
            _row("Haemoglobin", "13.0", None, "13 - 17"),
            _row("Haemoglobin", "12.0", None, "12 - 15"),
        ]
    )

    assert [r.reference_range for r in rows] == ["13 - 17", "12 - 15"]


def test_a_test_with_no_interval_anywhere_stays_unchecked():
    """Nothing is invented: an interval has to be printed somewhere on the document."""
    rows = _inherit_ranges([_row("Novel Marker", "5"), _row("Novel Marker", "7")])

    assert [r.reference_range for r in rows] == [None, None]


# --- the output ceiling ------------------------------------------------------


def test_the_output_ceiling_clears_the_largest_document_measured() -> None:
    """Documents 219/220 (2026-09-07) were a 43-page hospital investigation bundle. At the
    then-current 24000 the response stopped at 23,985 tokens and the item failed
    ``response_truncated``, which is PERMANENT — so the document extracted nothing at all
    and did not retry. Re-run with room to finish, it needed **45,926 tokens for 553
    results**.

    Pinned because the failure is invisible from the code: a ceiling that is too low looks
    exactly like a ceiling that is fine until someone uploads a big enough report.
    """
    assert EXTRACT_MAX_TOKENS >= 46_000


def test_the_output_ceiling_stays_inside_the_models_that_run_this_stage() -> None:
    """The other side of the same number, and the worse failure. Past a model's own limit
    the API rejects the REQUEST, so every document fails rather than the rare enormous one.

    ``gemini-3.1-flash-lite`` allows 65,536 and ``claude-haiku-4-5`` 64,000; the stage can
    be pointed at either (``_STAGE_OVERRIDES["extracting"]``), so the smaller one governs.
    """
    assert EXTRACT_MAX_TOKENS <= 64_000
