"""Runs against the live local Postgres. Marked `integration` so unit runs stay fast:

pytest -m "not integration"     # no database needed
pytest -m integration           # requires docker compose --profile localdev up -d db
"""

import pytest
from sqlalchemy import text

from app.core.db import engine
from app.models.processing import AiProcessingRun, AiProcessingRunItem

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def connection():
    try:
        conn = engine.connect()
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"local database unavailable: {exc}")
    yield conn
    conn.close()


def test_spring_schema_is_present(connection):
    count = connection.execute(
        text("SELECT count(*) FROM information_schema.tables WHERE table_schema = 'public'")
    ).scalar_one()
    assert count >= 31


def test_reports_table_matches_plan_assumptions(connection):
    rows = connection.execute(
        text(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_schema = 'public' AND table_name = 'reports'"
        )
    ).all()
    columns = dict(rows)

    # These types drive the API contract: report ids are integers, run ids are UUIDs.
    assert columns["id"] == "integer"
    assert columns["user_id"] == "uuid"
    assert columns["filepath"] == "character varying"
    assert columns["content"] == "jsonb"


def test_ready_endpoint_reports_up_against_live_database(api):
    # Uses the `api` fixture so S3/SQS resolve to moto; /ready now probes all three.
    response = api.get("/ready")
    assert response.status_code == 200
    body = response.json()
    checks = body["checks"]
    assert checks["database"]["status"] == "up"
    assert checks["s3"]["status"] == "up"
    assert checks["sqs"]["status"] == "up"
    # Reported, and reported separately from `checks`.
    assert body["dlq"] == {"status": "up", "messages": 0}


def test_a_full_dead_letter_queue_does_not_make_the_service_unready(api, aws):
    """Depth is visibility, never a health signal.

    A message in the DLQ is a document nobody is processing — worth surfacing, and no
    reason whatsoever to take a working service out of rotation. It is also not
    necessarily recent: a message with an unreadable body is skipped without being
    deleted and lands here after the queue's own maxReceiveCount, then stays for ever.
    Gating readiness on that would mean one stale message from months ago permanently
    failing every deploy's health check.
    """
    _, sqs, _, dlq = aws
    for _ in range(3):
        sqs.send_message(QueueUrl=dlq, MessageBody='{"stranded": true}')

    response = api.get("/ready")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["dlq"]["messages"] == 3
    # And it stayed out of the readiness calculation entirely.
    assert "dlq" not in body["checks"]


def test_run_item_carries_the_auto_filing_columns(db_session) -> None:
    """The four columns filing depends on exist and default to NULL.

    section_row_id replaces reports_id because the filed row is no longer always a report;
    filed_section says which table it is in; source_key is how stages find the document
    after the intake row is deleted.
    """
    run = AiProcessingRun(caller="spring")
    db_session.add(run)
    db_session.flush()
    item = AiProcessingRunItem(run_id=run.id, document_id=1, status="pending")
    db_session.add(item)
    db_session.flush()

    assert item.section_row_id is None
    assert item.filed_section is None
    assert item.intended_section is None
    assert item.source_key is None
