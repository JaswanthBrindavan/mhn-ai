"""What the insights stage puts in front of the model.

This is the only place a number reaches patient-facing prose, so what goes in decides
what can come out. A mutation check found nothing pinned it at all: swapping
``flagged_against`` back to ``reference_range`` broke no test, while changing exactly
which limit the reader is told their result crossed.
"""

import json

from app.services.insights import SYSTEM_PROMPT, _context_json, _needs_interpretation


def _rows(extraction: dict) -> list[dict]:
    return json.loads(_context_json(extraction))["results"]


def _context(extraction: dict) -> dict:
    return json.loads(_context_json(extraction))


def _result(name: str, flag: str | None) -> dict:
    return {
        "test_name": name,
        "value": "1.0",
        "unit": "mg/dL",
        "reference_range": "0 - 2",
        "flagged_against": "0 - 2",
        "abnormal_flag": flag,
    }


def test_the_model_is_given_the_limit_the_flag_was_computed_against() -> None:
    """The divergence case, end to end through the payload the stage actually builds.

    The lab printed 3.5-8.0 and would call 7.5 normal. The approved ideal range for this
    patient's age bracket tops out at 6.0, so the system flagged it high. If the model is
    handed the printed range it will write "7.5, above the normal top of 8.0" — a
    sentence that is arithmetically false, beside a number the reader can see on their
    own report.
    """
    extraction = {
        "results": [
            {
                "test_name": "Uric Acid",
                "value": "7.5",
                "unit": "mg/dL",
                "reference_range": "3.5 - 8.0",
                "flagged_against": "3.5 - 6",
                "range_source": "ideal_range",
                "abnormal_flag": "high",
            }
        ]
    }

    row = _rows(extraction)[0]

    assert row["flagged_against"] == "3.5 - 6"
    # The printed range must NOT be there. Given both, the model has to choose which
    # limit to quote, and choosing is the judgement this stage keeps away from it.
    assert "reference_range" not in row


def test_the_prompt_names_the_field_it_must_quote_from() -> None:
    """A field the model is sent but never told to prefer is a field it may ignore."""
    assert "flagged_against" in SYSTEM_PROMPT


def test_only_the_five_agreed_keys_reach_the_model() -> None:
    """Deliberately narrow. Everything else in a result row is operational — matched
    parameter, conversion output, our own numeric parse — and sending it invites the
    model to reason over machinery rather than over the report."""
    extraction = {
        "results": [
            {
                "test_name": "Hb",
                "value": "9.1",
                "unit": "g/dL",
                "reference_range": "13 - 17",
                "flagged_against": "13 - 17",
                "abnormal_flag": "low",
                "value_numeric": 9.1,
                "matched_parameter": "Haemoglobin",
                "normalized_value": 91.0,
                "range_source": "report_range",
            }
        ]
    }

    assert set(_rows(extraction)[0]) == {
        "test_name",
        "value",
        "unit",
        "flagged_against",
        "abnormal_flag",
    }


def test_an_unchecked_result_still_reaches_the_model_but_carries_no_limit() -> None:
    """A result nothing could check must not be silently dropped from the input — the
    prompt requires it to be named in the summary as unchecked. It carries a null
    ``flagged_against``, so there is no limit available to quote about it."""
    extraction = {
        "results": [
            {
                "test_name": "Vitamin D",
                "value": "45",
                "unit": "ng/mL",
                "reference_range": None,
                "flagged_against": None,
                "abnormal_flag": None,
            }
        ]
    }

    row = _rows(extraction)[0]
    assert row["flagged_against"] is None
    # And it is worth paying the model for: undetermined is not "in range".
    assert _needs_interpretation(extraction["results"]) is True


def test_normal_results_are_not_sent_at_all() -> None:
    """The 2026-09-01 change. A normal result is a settled question — the flag was computed
    in Python, against the R&D ideal range where one resolved — and the prompt has always
    forbidden giving one its own insight. It is still extracted, stored and displayed; it
    is simply not part of what the model is asked about."""
    extraction = {
        "results": [
            _result("Uric Acid", "high"),
            _result("Sodium", "normal"),
            _result("Potassium", "normal"),
            _result("Vitamin D", None),
        ]
    }

    names = [row["test_name"] for row in _rows(extraction)]

    assert names == ["Uric Acid", "Vitamin D"]
    assert "Sodium" not in json.dumps(_context(extraction))


def test_the_normals_are_replaced_by_a_count() -> None:
    """The summary must still say how many came back in range, and the model cannot be
    left to infer or invent that. A count is the whole of what that sentence needs, and
    unlike a row it cannot be misquoted as a value."""
    extraction = {
        "results": [
            _result("Uric Acid", "high"),
            _result("Sodium", "normal"),
            _result("Potassium", "normal"),
            _result("Chloride", "normal"),
            _result("Vitamin D", None),
        ]
    }

    assert _context(extraction)["normal_count"] == 3


def test_a_panel_with_nothing_normal_still_reports_a_count() -> None:
    """Zero is a real answer and has to be present, not absent: the prompt reads the key
    unconditionally, and a missing one invites the model to fill the gap."""
    assert _context({"results": [_result("Uric Acid", "high")]})["normal_count"] == 0


def test_the_prompt_tells_the_model_the_normals_are_missing() -> None:
    """Without this the model is silently working from a partial panel and does not know
    it — which is exactly when it starts describing results it was never given."""
    assert "normal_count" in SYSTEM_PROMPT
    assert "are not given to you" in SYSTEM_PROMPT


def _dated(name: str, value: str, flag: str | None, observed: str | None) -> dict:
    return {
        "test_name": name,
        "value": value,
        "unit": "mg/dL",
        "reference_range": "0 - 150",
        "flagged_against": "0 - 150",
        "abnormal_flag": flag,
        "observed_date": observed,
    }


def test_a_superseded_result_is_not_interpreted() -> None:
    """Document 32: a cumulative report with "YOUR CURRENT VISIT" beside "FROM YOUR
    PREVIOUS 3 VISITS".

    Both readings are STORED on purpose — `_dedupe_results` keeps the trend, and its
    docstring says losing half of it is silent. But `_context_json` sends no
    `observed_date`, so the model cannot tell them apart and writes about whichever it is
    handed. Here the current triglycerides is 139 and in range; the 2024 column's 219 is
    not. Without this the reader is told their triglycerides are high, with a real
    number, from a value sixteen months stale.

    The normals filter made that certain rather than likely: 139 is `normal` and is
    dropped, so 219 was the ONLY triglycerides row left in the payload.
    """
    extraction = {
        "results": [
            _dated("Triglycerides", "139", "normal", "18-Mar-26"),
            _dated("Triglycerides", "219", "high", "25-Nov-24"),
        ]
    }

    assert _rows(extraction) == []
    # And it counts as the normal it is, rather than vanishing from the tally.
    assert _context(extraction)["normal_count"] == 1


def test_the_marker_on_one_copy_does_not_hide_the_supersession() -> None:
    """The current copy is the one the lab marked abnormal, so the names differ by an
    asterisk. Same test, and the older reading must still be dropped."""
    extraction = {
        "results": [
            _dated("Cholesterol - LDL (Direct) *", "123", "high", "18-Mar-26"),
            _dated("Cholesterol - LDL (Direct)", "142", "high", "25-Nov-24"),
        ]
    }

    values = [r["value"] for r in _rows(extraction)]
    assert values == ["123"]


def test_two_undated_results_both_survive() -> None:
    """Only a strictly LATER date supersedes. With no dates there is nothing to order
    by, and dropping one would be a guess about which reading is current."""
    extraction = {
        "results": [
            _dated("Ferritin", "30", "low", None),
            _dated("Ferritin", "45", "high", None),
        ]
    }

    assert len(_rows(extraction)) == 2


def test_a_dated_result_does_not_displace_an_undated_one() -> None:
    """ "No date" is not evidence of being older, so it is never treated as superseded."""
    extraction = {
        "results": [
            _dated("Ferritin", "30", "low", "18-Mar-26"),
            _dated("Ferritin", "45", "high", None),
        ]
    }

    assert len(_rows(extraction)) == 2


def test_an_ordinary_single_visit_report_is_untouched() -> None:
    """Nothing in it has a later twin, so the filter is a no-op — which is what keeps
    this change confined to the cumulative case that motivated it."""
    extraction = {
        "results": [
            _dated("Triglycerides", "219", "high", "18-Mar-26"),
            _dated("Cholesterol - HDL", "30", "low", "18-Mar-26"),
        ]
    }

    assert len(_rows(extraction)) == 2
