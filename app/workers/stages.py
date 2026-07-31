"""The processing pipeline: classify first, then whatever that section needs.

Classification is the shared first stage — every document is classified before anything
else is decided, because the section is what decides the rest. After it, the pipeline's
*shape* depends on the answer:

* ``reports``            -> extract lab results, generate insights, then move into ``reports``
* ``insurance`` / ``scans_imaging`` / ``vaccinations``
                         -> transcribe the section's fields, and stop
* anything else          -> rejected; the document stays in ``unclassified_files``

That is why this is a table rather than a list: a report and an insurance policy do not
run the same stages, and a flat sequence cannot express "stop here for this kind".

Shared types (``StageContext``, ``TransientStageError``, ``RejectStageError``) live in
``app.workers.stagetypes`` and are re-exported here for existing importers. Stages must
stay idempotent — a redelivered message re-runs the whole pipeline, so a stage upserts its
results rather than appending.
"""

from app.models.enums import RunItemStatus
from app.services.classification import DocumentSection, classify_report
from app.services.extraction import extract_report
from app.services.insights import generate_insights
from app.services.section_extraction import extract_section
from app.services.section_specs import SUPPORTED_SECTIONS
from app.workers.stagetypes import (
    RejectStageError,
    Stage,
    StageContext,
    TransientStageError,
)

__all__ = [
    "CLASSIFY_STAGE",
    "SECTION_PIPELINES",
    "RejectStageError",
    "Stage",
    "StageContext",
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
}
