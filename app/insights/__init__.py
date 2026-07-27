"""Extraction for the non-report sections: insurance, scans/imaging, vaccinations.

The report pipeline (classify -> extract -> insights) deep-processes only ``reports``;
every other section the classifier recognises is currently rejected. This package adds
one generic stage, ``extract_section``, that transcribes three of those sections into
``ai_section_extractions``.

Adding a section is an entry in ``app.insights.sections`` — a Pydantic model, a JSON
schema, and a prompt. The stage, the persistence, and the worker are unchanged.

See ``app/insights/README.md`` for how to wire the stage into the pipeline.
"""

from app.insights.extraction import extract_section
from app.insights.sections import SECTION_SPECS, SUPPORTED_SECTIONS, SectionSpec, spec_for

__all__ = [
    "SECTION_SPECS",
    "SUPPORTED_SECTIONS",
    "SectionSpec",
    "extract_section",
    "spec_for",
]
