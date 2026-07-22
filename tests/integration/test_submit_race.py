"""What happens to the rest of a batch when one report loses the idempotency race.

The race window is real: two submissions can both pass the "is there an active item?"
check before either inserts. The loser hits the unique index. The question this file
answers is whether the *other* reports in the same batch survive that.
"""

import pytest
from sqlalchemy import text

from app.models.enums import RunItemStatus
from app.services import runs as runs_service

pytestmark = pytest.mark.integration


@pytest.fixture
def blind_first_check(monkeypatch):
    """Make the pre-insert check miss an existing active item exactly once.

    That is precisely what a concurrent submission looks like from inside this
    request: the SELECT ran before the other transaction committed.
    """
    real = runs_service._active_item_for_report
    missed: set[int] = set()

    def _patched(session, report_id):
        if report_id not in missed:
            missed.add(report_id)
            return None
        return real(session, report_id)

    monkeypatch.setattr(runs_service, "_active_item_for_report", _patched)


def test_losing_the_race_on_one_report_keeps_the_rest_of_the_batch(
    api, make_report, db_session, blind_first_check
):
    first, contended = make_report(), make_report()

    # Someone else already has `contended` in flight.
    other_run = db_session.execute(
        text("INSERT INTO ai_processing_runs (caller) VALUES ('other') RETURNING id")
    ).scalar_one()
    db_session.execute(
        text(
            "INSERT INTO ai_processing_run_items (run_id, report_id, status) "
            "VALUES (:run_id, :report_id, 'processing')"
        ),
        {"run_id": other_run, "report_id": contended},
    )
    db_session.flush()

    response = api.post("/v1/report-processing-runs", json={"report_ids": [first, contended]})

    assert response.status_code == 202
    body = response.json()
    outcomes = {item["report_id"]: item for item in body["items"]}

    # The contended report reuses the in-flight item rather than duplicating work.
    assert outcomes[contended]["outcome"] == "reused"

    # And the uncontended report must still have been created. If the race handler
    # rolled back the whole transaction, this item would have vanished.
    assert outcomes[first]["outcome"] == "created"
    surviving = db_session.execute(
        text("SELECT count(*) FROM ai_processing_run_items WHERE report_id = :r"),
        {"r": first},
    ).scalar_one()
    assert surviving == 1, "the uncontended report's item was lost by the race handler"


def test_the_run_itself_survives_the_race(api, make_report, db_session, blind_first_check):
    first, contended = make_report(), make_report()
    other_run = db_session.execute(
        text("INSERT INTO ai_processing_runs (caller) VALUES ('other') RETURNING id")
    ).scalar_one()
    db_session.execute(
        text(
            "INSERT INTO ai_processing_run_items (run_id, report_id, status) "
            "VALUES (:run_id, :report_id, 'queued')"
        ),
        {"run_id": other_run, "report_id": contended},
    )
    db_session.flush()

    run_id = api.post("/v1/report-processing-runs", json={"report_ids": [first, contended]}).json()[
        "run_id"
    ]

    fetched = api.get(f"/v1/report-processing-runs/{run_id}")
    assert fetched.status_code == 200
    statuses = [item["status"] for item in fetched.json()["items"]]
    assert RunItemStatus.QUEUED.value in statuses
