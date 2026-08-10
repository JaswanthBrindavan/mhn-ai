"""``RunProgress`` hand-mirrors ``RunItemStatus``, so something has to hold them together.

``get_run`` builds the progress object with ``RunProgress(total=..., **dict(counts))``,
where ``counts`` is a ``Counter`` over the statuses actually present on the run's items.
A status with no matching field is an unexpected keyword argument, and Pydantic raises —
so the failure is not a missing number in the response, it is **every**
``GET /v1/document-processing-runs/{id}`` covering an item in that state returning a 500.

That is a live risk rather than a theoretical one: ``docs/HANDOVER.md`` records a wanted
change to give section extraction its own status, and the note there mentions the CHECK
constraint it would need and not this. Adding the enum member without the field would
ship a broken read endpoint.

The same drift class as the dosage-form vocabulary, and guarded the same way: assert the
two sets are equal, rather than keeping a third hand-written copy of the list.
"""

from app.models.enums import RunItemStatus
from app.schemas.runs import RunProgress


def test_every_status_has_a_progress_field() -> None:
    fields = set(RunProgress.model_fields) - {"total"}
    assert fields == {status.value for status in RunItemStatus}


def test_progress_accepts_a_count_for_every_status() -> None:
    """The constructor call `get_run` actually makes, with every status present at once.

    Asserting the field names match is not quite enough on its own — this is the shape
    that breaks, so it is worth exercising it.
    """
    counts = {status.value: 1 for status in RunItemStatus}
    progress = RunProgress(total=len(counts), **counts)
    assert progress.total == len(counts)
    assert progress.completed == 1
