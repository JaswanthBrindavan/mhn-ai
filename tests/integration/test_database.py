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
    checks = response.json()["checks"]
    assert checks["database"]["status"] == "up"
    assert checks["s3"]["status"] == "up"
    assert checks["sqs"]["status"] == "up"


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
