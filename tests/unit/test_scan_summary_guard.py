"""A scan summary may not exist without a radiologist's read behind it.

Most uploads to this section are the IMAGE — an X-ray with a burned-in header and no
report anywhere in the text. ``summary`` is the one field written *about* the document
rather than transcribed from it, and asked for three to six sentences with nothing to
work from, a model fills the gap.

Measured against the real prompt on synthetic burn-in text, both of these came back from
Haiku:

    header + a technologist's scribble
      -> "...The scan was repeated because the patient moved during the first attempt.
          The report does not state what the pictures showed."
         (impression null, findings empty — a summary with no source, and a
          technologist's working note rendered as patient-facing prose)

    header + a stamped "NORMAL STUDY"
      -> "The radiologist reviewed the pictures and found no broken bones, no problems
          with the heart or lungs, and no other abnormalities. Everything looked normal."

Nothing in the second document says a radiologist saw it, or mentions heart or lungs. It
is a false all-clear on a chest X-ray built from two stamped words, and it is the worst
output this service can produce.

The prompt was tightened for the second case. The first is closed here in Python, for the
reason abnormal flags and dates are: a prompt is a request, and this one is a false
all-clear.
"""

from app.services.classification import DocumentSection
from app.services.section_extraction import build_payload
from app.services.section_specs import ScanFields, spec_for

SCANS = spec_for(DocumentSection.SCANS_IMAGING)


def _payload(**overrides: object) -> dict:
    base = {
        "scan_type": "X-Ray",
        "body_part": "Left Knee",
        "scan_date": "04/02/2026",
        "facility": "SUNRISE IMAGING",
        "summary": None,
        "impression": None,
        "findings": [],
    }
    base.update(overrides)
    return build_payload(SCANS, ScanFields(**base))  # type: ignore[arg-type]


def _flag_codes(payload: dict) -> list[str]:
    return [f["code"] for f in payload["flags"]]


def test_a_summary_with_no_impression_and_no_findings_is_dropped() -> None:
    """The measured case: an image, a header, and three sentences of invented prose."""
    payload = _payload(
        summary=(
            "This was an X-ray of the left knee taken from two angles. The scan was "
            "repeated because the patient moved. The report does not state what the "
            "pictures showed."
        )
    )

    assert payload["fields"]["summary"] is None
    assert "no_radiologist_read" in _flag_codes(payload)


def test_the_factual_fields_survive_an_image_with_no_report() -> None:
    """Only the interpretation goes. Scan type, body part, date and facility are
    transcription, and the user should still see "X-Ray, Left Knee, 4 Feb" for a scan
    that was never reported on — the document is theirs and it is filed."""
    payload = _payload(summary="Invented prose about the knee.")
    fields = payload["fields"]

    assert fields["scan_type"] == "X-Ray"
    assert fields["body_part"] == "Left Knee"
    assert fields["scan_date"] == "2026-02-04"
    assert fields["facility"] == "SUNRISE IMAGING"


def test_an_impression_is_enough_to_keep_the_summary() -> None:
    """A real report keeps its plain-English summary — that is the whole feature."""
    payload = _payload(
        impression="Mild degenerative change in the medial compartment.",
        summary="The report says there is some wear in the inner part of the knee joint.",
    )

    assert payload["fields"]["summary"] is not None
    assert "no_radiologist_read" not in _flag_codes(payload)


def test_findings_alone_are_enough_to_keep_the_summary() -> None:
    """Some reports list findings without a separate impression line."""
    payload = _payload(
        findings=["Joint space narrowing", "No acute fracture"],
        summary="The report notes narrowing of the joint and no broken bones.",
    )

    assert payload["fields"]["summary"] is not None
    assert "no_radiologist_read" not in _flag_codes(payload)


def test_an_image_with_no_report_says_so_even_when_nothing_was_deleted() -> None:
    """The common case once the prompt was tightened: the model returns null by itself.

    The flag still fires, and that is the feature rather than a side effect of the guard.
    An empty summary card with no explanation reads as "the AI failed" — the same
    absence-versus-failure confusion the pending-document note had. A scan uploaded
    without its report should say that is what happened.
    """
    payload = _payload()

    assert payload["fields"]["summary"] is None
    assert _flag_codes(payload) == ["no_radiologist_read"]


def test_the_guard_is_configured_on_scans_and_nowhere_else() -> None:
    """It applies to the one field written about a document rather than from it.

    Insurance and vaccinations transcribe every field they hold, so there is nothing for
    this rule to protect and turning it on there would only invent a way to lose data.
    """
    assert SCANS.summary_field == "summary"
    assert SCANS.summary_sources == ("impression", "findings")
    for section in (DocumentSection.INSURANCE, DocumentSection.VACCINATIONS):
        assert spec_for(section).summary_field is None
