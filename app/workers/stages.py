"""The processing pipeline: classify first, then whatever that section needs.

Classification is the shared first stage — every document is classified before anything
else is decided, because the section is what decides the rest. After it, the pipeline's
*shape* depends on the answer:

* ``reports``            -> extract lab results, generate insights
* ``insurance`` / ``scans_imaging`` / ``vaccinations``
                         -> transcribe the section's fields, and stop
* anything else          -> rejected; the document stays in ``unclassified_files``

That is why this is a table rather than a list: a report and an insurance policy do not
run the same stages, and a flat sequence cannot express "stop here for this kind".

Filing the document into its section table and updating that row's ``content`` are **not**
stages: they bracket the table above, in ``processor._run_pipeline``. Filing happens
straight after classification so the document appears in its section within seconds, and
the content update happens once the section's stages have finished.

Shared types (``StageContext``, ``TransientStageError``, ``RejectStageError``) live in
``app.workers.stagetypes`` and are re-exported here for existing importers. Stages must
stay idempotent — a redelivered message re-runs the whole pipeline, so a stage upserts its
results rather than appending.
"""

from app.models.enums import RunItemStatus
from app.services.classification import DocumentSection, classify_report
from app.services.extraction import extract_report
from app.services.insights import generate_insights
from app.services.prescriptions import extract_prescription, record_handwritten
from app.services.section_extraction import extract_section
from app.services.section_specs import SUPPORTED_SECTIONS
from app.workers.stagetypes import (
    PermanentStageError,
    RejectStageError,
    Stage,
    StageContext,
    TransientStageError,
)

__all__ = [
    "CLASSIFY_STAGE",
    "HANDWRITTEN_PRESCRIPTION_PIPELINE",
    "SECTION_PIPELINES",
    "PermanentStageError",
    "RejectStageError",
    "Stage",
    "StageContext",
    "StageStep",
    "TransientStageError",
]

#: (status to move into before running, stage callable).
StageStep = tuple[RunItemStatus, Stage]

#: Runs for every document, whatever it turns out to be.
CLASSIFY_STAGE: StageStep = (RunItemStatus.CLASSIFYING, classify_report)

#: Detected section -> the stages that follow classification. **A section absent from this
#: table is not processed**: the router rejects it and the document stays in
#: ``unclassified_files``. That is routing, not failure.
#:
#: The section entries are derived from ``SECTION_SPECS`` rather than listed here, so
#: adding a section is one entry in ``section_specs.py`` and nothing else — the two cannot
#: drift into disagreeing about which sections are supported.
SECTION_PIPELINES: dict[DocumentSection, list[StageStep]] = {
    DocumentSection.REPORTS: [
        (RunItemStatus.EXTRACTING, extract_report),
        (RunItemStatus.GENERATING_INSIGHTS, generate_insights),
    ],
    **{
        section: [(RunItemStatus.EXTRACTING, extract_section)]
        for section in sorted(SUPPORTED_SECTIONS, key=lambda s: s.value)
    },
    #: Prescriptions have their own stage rather than a ``SECTION_SPECS`` entry. The
    #: generic stage sends OCR text, and a prescription is a layout: the dose sits in a
    #: column beside the medicine, and flattening that puts a dose on the wrong row.
    #: ``extract_prescription`` sends the document itself, checks every name against the
    #: page, and normalises the dosing notation in Python.
    #:
    #: Listed after the expansion above so this wins outright if a ``prescriptions`` spec
    #: is ever added there, rather than the two silently disagreeing about the stage.
    DocumentSection.PRESCRIPTIONS: [
        (RunItemStatus.EXTRACTING, extract_prescription),
    ],
}

#: What a mostly-handwritten prescription runs instead of ``SECTION_PIPELINES``: it is
#: filed like any other, and then the only "extraction" is a record saying we did not read
#: it and why. A pipeline rather than a special case in the router, so it goes through the
#: same guarded stage runner — cancellation, logging and content assembly all behave
#: identically to every other document. See ``prescriptions.record_handwritten``.
HANDWRITTEN_PRESCRIPTION_PIPELINE: list[StageStep] = [
    (RunItemStatus.EXTRACTING, record_handwritten),
]
