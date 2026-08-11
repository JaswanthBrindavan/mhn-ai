"""What a result was actually checked against, and why anything quoting a limit needs it.

``abnormal_flag`` says a value is high. The number it crossed to earn that is a separate
question, and until now the only range in the stored payload was the one the report
PRINTED — which is the right answer only while no approved ideal range is in play.

With ``IDEAL_RANGES_ENABLED`` on they diverge, and the divergence is user-visible: the
insights stage is told to open with "the value and the limit it crossed". Given the
printed range it would write a limit the value never crossed, next to a verdict computed
from a different one, on a report the reader is holding.
"""

from app.services.normalization import enrich_result, render_bounds


def test_bounds_are_rendered_as_the_comparison_actually_behaves() -> None:
    """``<=``, not ``<``, on a one-sided upper range.

    ``abnormal_flag`` flags high only when ``value > high``, so a range parsed from
    "< 200" treats exactly 200 as in range. Rendering it "< 200" would describe a rule
    the code does not apply.
    """
    assert render_bounds((13.0, 17.0)) == "13 - 17"
    assert render_bounds((None, 200.0)) == "<= 200"
    assert render_bounds((90.0, None)) == ">= 90"
    assert render_bounds(None) is None
    # Trailing ".0" reads as false precision beside a patient's own result.
    assert render_bounds((8.0, 8.6)) == "8 - 8.6"


def test_the_printed_range_is_what_is_quoted_when_it_is_what_was_used() -> None:
    row = enrich_result(
        {"test_name": "Uric Acid", "value": "8.6", "unit": "mg/dL", "reference_range": "3.5 - 8.0"}
    )
    assert row["abnormal_flag"] == "high"
    assert row["range_source"] == "report_range"
    assert row["flagged_against"] == "3.5 - 8"


def test_an_ideal_range_override_is_quoted_instead_of_the_printed_one() -> None:
    """The case that made this field necessary.

    The lab prints 3.5-8.0 and would call 7.5 normal. R&D's approved ideal range for this
    patient's age bracket tops out at 6.0, so the flag is `high` — and a card quoting
    "above the normal top of 8.0" would be quoting a limit this value is comfortably
    under, contradicting itself in front of the reader.
    """
    row = enrich_result(
        {"test_name": "Uric Acid", "value": "7.5", "unit": "mg/dL", "reference_range": "3.5 - 8.0"},
        override_bounds=(3.5, 6.0),
        matched_parameter="Uric Acid",
    )
    assert row["abnormal_flag"] == "high"
    assert row["range_source"] == "ideal_range"
    assert row["flagged_against"] == "3.5 - 6"
    # The printed range is still stored — nothing the document said is thrown away.
    assert row["reference_range"] == "3.5 - 8.0"


def test_no_printed_range_is_distinguishable_from_one_that_could_not_be_decided() -> None:
    """Two very different problems that used to look identical in the payload.

    A result with no range at all is a curation gap — nothing could ever have checked it.
    A result whose range was present but undecidable is a parser gap. Both stored
    ``range_source: "report_range"`` with a null flag, so neither could be counted, which
    is precisely the question docs/FUTURE.md's third-tier decision turns on.
    """
    nothing_to_check = enrich_result(
        {"test_name": "Vitamin D", "value": "45", "unit": "ng/mL", "reference_range": None}
    )
    assert nothing_to_check["abnormal_flag"] is None
    assert nothing_to_check["range_source"] == "none"
    assert nothing_to_check["flagged_against"] is None

    # A sex-split range with no gender to select by: present, and correctly refused.
    undecidable = enrich_result(
        {
            "test_name": "Iron",
            "value": "31",
            "unit": "ug/dL",
            "reference_range": "Male : 65 - 175 Female : 50 - 170",
        }
    )
    assert undecidable["abnormal_flag"] is None
    assert undecidable["range_source"] == "report_range"
    assert undecidable["flagged_against"] is None


def test_an_empty_string_range_counts_as_no_range() -> None:
    row = enrich_result({"test_name": "X", "value": "1", "unit": None, "reference_range": "   "})
    assert row["range_source"] == "none"
