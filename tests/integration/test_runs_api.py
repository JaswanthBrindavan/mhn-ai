"""Run submission, progress, and cancellation against the live database."""

import uuid

import pytest

from app.models.enums import RunItemStatus

pytestmark = pytest.mark.integration


# --- authentication ---------------------------------------------------------


def test_v1_requires_the_service_token(api):
    response = api.post(
        "/v1/report-processing-runs",
        json={"report_ids": [1]},
        headers={"Authorization": "Bearer wrong"},
    )
    assert response.status_code == 401


# --- submission -------------------------------------------------------------


def test_submit_creates_a_run_and_items(api, make_report):
    report_id = make_report()

    response = api.post("/v1/report-processing-runs", json={"report_ids": [report_id]})

    assert response.status_code == 202  # no AI work inline
    body = response.json()
    assert len(body["items"]) == 1
    item = body["items"][0]
    assert item["report_id"] == report_id
    assert item["outcome"] == "created"
    # Published to the queue during the request, so it lands as queued.
    assert item["status"] == RunItemStatus.QUEUED.value


def test_submit_rejects_unknown_report_ids(api, make_report):
    known = make_report()
    response = api.post("/v1/report-processing-runs", json={"report_ids": [known, 999_000_111]})

    assert response.status_code == 404
    body = response.json()
    assert body["error"]["code"] == "report_not_found"
    assert body["error"]["details"]["missing_report_ids"] == [999_000_111]


def test_duplicate_ids_within_one_request_create_one_item(api, make_report):
    report_id = make_report()
    response = api.post(
        "/v1/report-processing-runs", json={"report_ids": [report_id, report_id, report_id]}
    )
    assert response.status_code == 202
    assert len(response.json()["items"]) == 1


def test_resubmitting_an_active_report_reuses_the_item(api, make_report):
    report_id = make_report()
    first = api.post("/v1/report-processing-runs", json={"report_ids": [report_id]}).json()
    second = api.post("/v1/report-processing-runs", json={"report_ids": [report_id]}).json()

    assert second["items"][0]["outcome"] == "reused"
    # Same work, not a second pass over the same report.
    assert second["items"][0]["item_id"] == first["items"][0]["item_id"]
    assert second["run_id"] != first["run_id"]


def test_completed_report_is_not_reprocessed_without_force(api, make_report, db_session):
    report_id = make_report()
    first = api.post("/v1/report-processing-runs", json={"report_ids": [report_id]}).json()
    _complete(db_session, first["items"][0]["item_id"])

    second = api.post("/v1/report-processing-runs", json={"report_ids": [report_id]}).json()

    assert second["items"][0]["outcome"] == "already_completed"
    assert second["items"][0]["item_id"] == first["items"][0]["item_id"]


def test_force_reprocess_creates_a_new_item_for_a_completed_report(api, make_report, db_session):
    report_id = make_report()
    first = api.post("/v1/report-processing-runs", json={"report_ids": [report_id]}).json()
    _complete(db_session, first["items"][0]["item_id"])

    second = api.post(
        "/v1/report-processing-runs",
        json={"report_ids": [report_id], "force_reprocess": True},
    ).json()

    assert second["items"][0]["outcome"] == "created"
    assert second["items"][0]["item_id"] != first["items"][0]["item_id"]


def test_force_reprocess_does_not_duplicate_in_flight_work(api, make_report):
    """force_reprocess concerns finished results, not work already running."""
    report_id = make_report()
    api.post("/v1/report-processing-runs", json={"report_ids": [report_id]})
    second = api.post(
        "/v1/report-processing-runs",
        json={"report_ids": [report_id], "force_reprocess": True},
    ).json()

    assert second["items"][0]["outcome"] == "reused"


# --- family uploads ---------------------------------------------------------


def test_family_upload_is_accepted(api, make_report, make_user):
    """A report whose uploader differs from its subject must process normally.

    Regression guard: an early design compared the requesting user to
    reports.user_id, which would reject exactly this case. Authorization belongs to
    Spring; see app/api/deps.py.
    """
    relative = make_user()  # the family member doing the upload
    report_id = make_report(created_by=relative)

    response = api.post(
        "/v1/report-processing-runs",
        json={"report_ids": [report_id], "requested_by_user_id": str(relative)},
    )

    assert response.status_code == 202


def test_mismatched_requesting_user_still_processes(api, make_report):
    """An unrelated requested_by_user_id must not block processing.

    It is audit metadata, not an access-control input. If this test starts failing,
    someone has added user-level authorization here by mistake.
    """
    response = api.post(
        "/v1/report-processing-runs",
        json={"report_ids": [make_report()], "requested_by_user_id": str(uuid.uuid4())},
    )
    assert response.status_code == 202


# --- progress ---------------------------------------------------------------


def test_get_run_reports_progress(api, make_report):
    report_ids = [make_report(), make_report()]
    run_id = api.post("/v1/report-processing-runs", json={"report_ids": report_ids}).json()[
        "run_id"
    ]

    body = api.get(f"/v1/report-processing-runs/{run_id}").json()

    assert body["progress"]["total"] == 2
    assert body["progress"]["queued"] == 2
    assert body["finished"] is False
    assert len(body["items"]) == 2


def test_get_run_marks_finished_when_all_items_terminal(api, make_report, db_session):
    run = api.post("/v1/report-processing-runs", json={"report_ids": [make_report()]}).json()
    _complete(db_session, run["items"][0]["item_id"])

    body = api.get(f"/v1/report-processing-runs/{run['run_id']}").json()

    assert body["finished"] is True
    assert body["progress"]["completed"] == 1


def test_get_unknown_run_is_404(api):
    response = api.get(f"/v1/report-processing-runs/{uuid.uuid4()}")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "run_not_found"


# --- cancellation -----------------------------------------------------------


def test_cancel_marks_in_flight_items_cancelled(api, make_report):
    run = api.post("/v1/report-processing-runs", json={"report_ids": [make_report()]}).json()

    body = api.delete(f"/v1/report-processing-runs/{run['run_id']}").json()

    assert len(body["cancelled_item_ids"]) == 1
    after = api.get(f"/v1/report-processing-runs/{run['run_id']}").json()
    assert after["progress"]["cancelled"] == 1
    assert after["finished"] is True


def test_cancel_leaves_completed_items_alone(api, make_report, db_session):
    run = api.post("/v1/report-processing-runs", json={"report_ids": [make_report()]}).json()
    item_id = run["items"][0]["item_id"]
    _complete(db_session, item_id)

    body = api.delete(f"/v1/report-processing-runs/{run['run_id']}").json()

    assert body["cancelled_item_ids"] == []
    assert body["unaffected_item_ids"] == [item_id]


def test_cancelled_report_can_be_submitted_again(api, make_report):
    """Cancelled is terminal, so the partial unique index no longer blocks a new item."""
    report_id = make_report()
    run = api.post("/v1/report-processing-runs", json={"report_ids": [report_id]}).json()
    api.delete(f"/v1/report-processing-runs/{run['run_id']}")

    again = api.post("/v1/report-processing-runs", json={"report_ids": [report_id]}).json()
    assert again["items"][0]["outcome"] == "created"


# --- helpers ----------------------------------------------------------------


def _complete(db_session, item_id: str) -> None:
    from sqlalchemy import text

    db_session.execute(
        text("UPDATE ai_processing_run_items SET status = 'completed' WHERE id = :id"),
        {"id": uuid.UUID(item_id)},
    )
    db_session.flush()
