"""Fixtures for tests that need the live database.

Every test runs inside an outer transaction that is rolled back afterwards, so nothing
is left behind. The service layer calls ``session.commit()``, which would normally end
that transaction — ``join_transaction_mode="create_savepoint"`` turns those commits into
savepoint releases so the outer rollback still undoes everything.

This matters more than usual here: the database is shared with the Spring backend, so
tests must never leave rows lying around or touch Spring-owned data they did not create.
"""

import uuid
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Connection, text
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.db import engine, get_session
from app.main import create_app

SERVICE_TOKEN = "test-service-token-at-least-32-chars-long"


@pytest.fixture(scope="session")
def _engine_available() -> None:
    try:
        with engine.connect():
            pass
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"local database unavailable: {exc}")


@pytest.fixture
def db_connection(_engine_available: None) -> Iterator[Connection]:
    connection = engine.connect()
    transaction = connection.begin()
    try:
        yield connection
    finally:
        transaction.rollback()  # undoes everything the test did
        connection.close()


@pytest.fixture
def db_session(db_connection: Connection) -> Iterator[Session]:
    session = Session(bind=db_connection, join_transaction_mode="create_savepoint")
    try:
        yield session
    finally:
        session.close()


def _create_user(session: Session) -> uuid.UUID:
    user_id = uuid.uuid4()
    suffix = user_id.hex[:12]
    session.execute(
        text(
            'INSERT INTO "user" (id, name, email, user_name, health_card_number, hashcode) '
            "VALUES (:id, :name, :email, :user_name, :hcn, :hashcode)"
        ),
        {
            "id": user_id,
            "name": "Test User",
            "email": f"test-{suffix}@example.invalid",
            "user_name": f"test_{suffix}",
            "hcn": f"HC{suffix}",
            "hashcode": f"hash-{suffix}",
        },
    )
    session.flush()
    return user_id


@pytest.fixture
def make_user(db_session: Session):
    """Create a user row. `reports.created_by` has a foreign key to this table, so a
    family-upload test needs a real second user rather than an arbitrary UUID."""

    def _make() -> uuid.UUID:
        return _create_user(db_session)

    return _make


@pytest.fixture
def seed_user(db_session: Session) -> uuid.UUID:
    """A throwaway user row, required by reports' foreign key."""
    return _create_user(db_session)


@pytest.fixture
def make_report(db_session: Session, seed_user: uuid.UUID):
    """Create a reports row. `created_by` may differ from `user_id` (family upload)."""

    def _make(*, created_by: uuid.UUID | None = None) -> int:
        report_id = db_session.execute(
            text(
                "INSERT INTO reports (user_id, filepath, created_by) "
                "VALUES (:user_id, :filepath, :created_by) RETURNING id"
            ),
            {
                "user_id": seed_user,
                "filepath": f"reports/test/{uuid.uuid4().hex}.pdf",
                "created_by": created_by if created_by is not None else seed_user,
            },
        ).scalar_one()
        db_session.flush()
        return int(report_id)

    return _make


@pytest.fixture
def api(db_session: Session) -> Iterator[TestClient]:
    """A client whose requests share the test's transaction, and carry the token."""
    app = create_app()
    app.dependency_overrides[get_session] = lambda: db_session

    settings = get_settings()
    client = TestClient(app, raise_server_exceptions=False)
    client.headers.update({"Authorization": f"Bearer {settings.mhn_service_token}"})
    with client:
        yield client
