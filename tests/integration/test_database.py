"""Runs against the live local Postgres. Marked `integration` so unit runs stay fast:

pytest -m "not integration"     # no database needed
pytest -m integration           # requires docker compose --profile localdev up -d db
"""

import pytest
from sqlalchemy import text

from app.core.db import engine

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


def test_ready_endpoint_reports_up_against_live_database(client):
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json()["checks"]["database"]["status"] == "up"
