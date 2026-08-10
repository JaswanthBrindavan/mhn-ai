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

import boto3
import pytest
from fastapi.testclient import TestClient
from moto import mock_aws
from sqlalchemy import Connection, text
from sqlalchemy.orm import Session

from app.api.deps import s3_client, sqs_client
from app.core.config import Settings, get_settings
from app.core.db import engine, get_session
from app.main import create_app

SERVICE_TOKEN = "test-service-token-at-least-32-chars-long"
REGION = "ap-south-1"
BUCKET = "mhn-reports-test"


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
def aws() -> Iterator[tuple]:
    """moto-backed S3 and SQS, with the bucket and queue already created."""
    with mock_aws():
        s3 = boto3.client("s3", region_name=REGION)
        s3.create_bucket(Bucket=BUCKET, CreateBucketConfiguration={"LocationConstraint": REGION})
        sqs = boto3.client("sqs", region_name=REGION)
        dlq = sqs.create_queue(QueueName="report-processing-dlq")["QueueUrl"]
        queue_url = sqs.create_queue(QueueName="report-processing")["QueueUrl"]
        yield s3, sqs, queue_url, dlq


@pytest.fixture
def test_settings(aws) -> Settings:
    """Settings pointed at the moto resources rather than the developer's .env."""
    _, _, queue_url, dlq = aws
    base = get_settings()
    return Settings(
        database_url=base.database_url,
        mhn_service_token=base.mhn_service_token,
        aws_region=REGION,
        s3_bucket=BUCKET,
        sqs_queue_url=queue_url,
        # Read by /ready to report the queue's depth. Nothing else in the application
        # reads it — the redrive policy itself lives on the main queue in AWS.
        sqs_dlq_url=dlq,
        max_file_bytes=1_000_000,
    )


def document_key(session: Session, document_id: int) -> str:
    """The S3 key a document was seeded with, or "" if it has no intake row.

    Stage tests build a `StageContext` directly, so they need the key the real pipeline
    would have copied onto the run item at submit time.
    """
    key = session.execute(
        text("SELECT filepath FROM unclassified_files WHERE id = :id"), {"id": document_id}
    ).scalar_one_or_none()
    return str(key) if key else ""


@pytest.fixture
def make_document(db_session: Session, seed_user: uuid.UUID, aws):
    """Create an `unclassified_files` row AND its S3 object, and return its id.

    This is the submitted unit of work. `created_by` may differ from `user_id`, which is
    the family-upload case.
    """
    s3 = aws[0]

    def _make(
        *,
        created_by: uuid.UUID | None = None,
        body: bytes = b"%PDF-1.4 fake report",
        content_type: str = "application/pdf",
        suffix: str = ".pdf",
        upload: bool = True,
        key: str | None = None,
    ) -> int:
        key = key or f"uploads/test/{uuid.uuid4().hex}{suffix}"
        if upload:
            s3.put_object(Bucket=BUCKET, Key=key, Body=body, ContentType=content_type)
        document_id = db_session.execute(
            text(
                "INSERT INTO unclassified_files (user_id, filepath, created_by) "
                "VALUES (:user_id, :filepath, :created_by) RETURNING id"
            ),
            {
                "user_id": seed_user,
                "filepath": key,
                "created_by": created_by if created_by is not None else seed_user,
            },
        ).scalar_one()
        db_session.flush()
        return int(document_id)

    return _make


@pytest.fixture
def api(db_session: Session, aws, test_settings: Settings) -> Iterator[TestClient]:
    """A client sharing the test's transaction and moto clients, carrying the token."""
    s3, sqs, _, _ = aws
    app = create_app()
    app.dependency_overrides[get_session] = lambda: db_session
    app.dependency_overrides[s3_client] = lambda: s3
    app.dependency_overrides[sqs_client] = lambda: sqs
    app.dependency_overrides[get_settings] = lambda: test_settings

    client = TestClient(app, raise_server_exceptions=False)
    client.headers.update({"Authorization": f"Bearer {test_settings.mhn_service_token}"})
    with client:
        yield client
