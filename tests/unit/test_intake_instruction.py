"""Food timing that a document prints in its own column.

The bug this pins: a real teleconsultation prescription lays its medicines out as
`Medicine Name | Dosage | Duration | Intake Instruction`, with "Post Meal" in the last
column. The prompt only ever asked for food timing *inside* `frequency_raw`, and rule 10
tells the model to read a table by its columns — so it correctly took the dosing column,
correctly left the other one alone, and every "Post Meal" on that document was lost.

The normaliser was never the problem: it reads "Post Meal" perfectly well once the text
reaches it. The fix is a field for the value to arrive in.
"""

from typing import Any

from app.services.medicines import normalize_frequency
from app.services.prescriptions import PrescribedMedicine, PrescriptionFields, _build_payload


def medicine(**kwargs: Any) -> PrescribedMedicine:
    base: dict[str, Any] = {
        "name_as_written": "Tab. Azithromycin 500mg",
        "name_clean": "Azithromycin",
        "frequency_raw": "Once a day",
        "intake_instruction": "Post Meal",
        "duration": "3 Days",
    }
    base.update(kwargs)
    return PrescribedMedicine(**base)


def stored(row: PrescribedMedicine) -> dict[str, Any]:
    result = PrescriptionFields(medicines=[row], prescribed_date=None, prescriber=None)
    return _build_payload(result, [row], [])["fields"]["medicines"][0]


def test_a_separate_intake_column_reaches_with_food() -> None:
    """The whole point: dosing in one column, food timing in another, one schedule out."""
    assert stored(medicine())["frequency_normalized"]["with_food"] is True


def test_both_fields_are_kept_as_printed() -> None:
    # Joined only for parsing. A reader still sees exactly what each column said.
    row = stored(medicine())
    assert row["frequency_raw"] == "Once a day"
    assert row["intake_instruction"] == "Post Meal"


def test_before_food_and_empty_stomach_are_read_too() -> None:
    assert (
        stored(medicine(intake_instruction="Before Food"))["frequency_normalized"]["with_food"]
        is False
    )
    assert (
        stored(medicine(intake_instruction="Empty Stomach"))["frequency_normalized"]["with_food"]
        is False
    )


def test_food_timing_inside_the_dosing_still_works() -> None:
    # The shape that already worked, which must not regress: one cell carrying both.
    row = medicine(frequency_raw="1-0-1 after food", intake_instruction=None)
    assert stored(row)["frequency_normalized"]["with_food"] is True


def test_no_intake_instruction_leaves_food_unknown() -> None:
    # Null, not False. "The document did not say" is not "take it before food".
    row = medicine(frequency_raw="1-0-1", intake_instruction=None)
    assert stored(row)["frequency_normalized"]["with_food"] is None


def test_an_intake_instruction_alone_does_not_invent_a_schedule() -> None:
    """A medicine with food timing and no dosing has no schedule, and must not gain one.

    `normalize_frequency` returns None for a modifier with nothing to modify, and joining
    the two fields must not change that — otherwise "Post Meal" alone would produce a
    schedule of zero doses that reads as a real instruction.
    """
    row = medicine(frequency_raw=None, intake_instruction="Post Meal")
    assert stored(row)["frequency_normalized"] is None
    assert normalize_frequency("Post Meal") is None
