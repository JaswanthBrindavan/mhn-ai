"""The plural identity route: which of these documents are waiting on their owner.

Exists for a list screen. A wallet list holds several intake rows, and asking `/status`
per row turns the most-hit screen in the app into an N+1 across a service boundary.

Two properties matter more than the happy path. It must agree with `/status` about which
attempt counts, because two routes disagreeing over whether a document is yours would be
its own bug. And it must carry **no printed name** — that is the most identifying field a
document has, and a list has no use for it.
"""

from datetime import UTC, datetime

import pytest
from sqlalchemy import text

pytestmark = pytest.mark.integration


def _seed(
    db_session,
    make_document,
    *,
    patient_name: str | None = "Priya Menon",
    name_match: str | None = "mismatch",
    confirmed: bool = False,
    document_id: int | None = None,
    created_at: datetime | None = None,
) -> int:
    """One classified document with an identity verdict, as the gate leaves it."""
    if document_id is None:
        document_id = make_document()
    run_id = db_session.execute(
        text("INSERT INTO ai_processing_runs (caller) VALUES ('test') RETURNING id")
    ).scalar_one()
    item_id = db_session.execute(
        text(
            "INSERT INTO ai_processing_run_items "
            "(run_id, document_id, status, created_at) "
            "VALUES (:r, :d, 'rejected', coalesce(:c, now())) RETURNING id"
        ),
        {"r": run_id, "d": document_id, "c": created_at},
    ).scalar_one()
    db_session.execute(
        text(
            "INSERT INTO ai_report_classifications "
            "(run_item_id, document_id, section, title, confidence, prompt_version, "
            " schema_version, patient_name, name_match, identity_confirmed_at) "
            "VALUES (:i, :d, 'reports', 'CBC', 0.95, 'clf-1', 'clf-1', :n, :m, :c)"
        ),
        {
            "i": item_id,
            "d": document_id,
            "n": patient_name,
            "m": name_match,
            "c": datetime.now(UTC) if confirmed else None,
        },
    )
    db_session.commit()
    return document_id


def _post(api, ids):
    return api.post("/v1/documents/name-checks", json={"document_ids": ids})


def test_returns_a_verdict_per_document(api, db_session, make_document) -> None:
    mismatched = _seed(db_session, make_document)
    matched = _seed(db_session, make_document, name_match="match", patient_name="Praveen")

    response = _post(api, [mismatched, matched])

    assert response.status_code == 200
    by_id = {c["document_id"]: c for c in response.json()["checks"]}
    assert by_id[mismatched]["verdict"] == "mismatch"
    assert by_id[matched]["verdict"] == "match"


def test_carries_no_printed_name(api, db_session, make_document) -> None:
    """A list needs to know a decision is waiting, not who the document names. The name
    is the most identifying field on the page and belongs on the screen that asks."""
    document_id = _seed(db_session, make_document, patient_name="Priya Menon")

    body = _post(api, [document_id]).json()

    assert "Priya Menon" not in _post(api, [document_id]).text
    assert set(body["checks"][0]) == {"document_id", "verdict", "confirmed"}


def test_a_confirmed_mismatch_says_so(api, db_session, make_document) -> None:
    """Settled, so the list must not go on badging it — the user already answered."""
    document_id = _seed(db_session, make_document, confirmed=True)

    check = _post(api, [document_id]).json()["checks"][0]

    assert check["verdict"] == "mismatch"
    assert check["confirmed"] is True


def test_documents_with_no_verdict_are_omitted(api, db_session, make_document) -> None:
    """Absent already means "we have not looked"; a null entry would be a second way of
    saying the same thing, and the client would have to handle both."""
    unread = _seed(db_session, make_document, name_match=None, patient_name=None)
    never_processed = make_document()

    checks = _post(api, [unread, never_processed]).json()["checks"]

    assert checks == []


def test_reads_the_latest_attempt_like_status_does(api, db_session, make_document) -> None:
    """A retry writes a new item and a new classification. If this route read the newest
    classification row directly it could answer from a different attempt than `/status`,
    and the list would contradict the dialog."""
    # Timestamps are set explicitly rather than left to now(): every insert in this suite
    # runs inside ONE transaction, and Postgres's now() is the TRANSACTION time -- so two
    # attempts seeded back to back tie, and "latest" becomes a coin flip.
    earlier = datetime(2026, 8, 1, 9, 0, tzinfo=UTC)
    later = datetime(2026, 8, 2, 9, 0, tzinfo=UTC)
    document_id = _seed(db_session, make_document, name_match="mismatch", created_at=earlier)
    # A later attempt on the SAME document, which the user confirmed.
    _seed(
        db_session,
        make_document,
        document_id=document_id,
        name_match="mismatch",
        confirmed=True,
        created_at=later,
    )

    checks = _post(api, [document_id]).json()["checks"]
    status_body = api.get(f"/v1/documents/{document_id}/status").json()

    assert len(checks) == 1
    assert checks[0]["confirmed"] is True
    assert checks[0]["verdict"] == status_body["name_check"]["verdict"]
    assert checks[0]["confirmed"] == status_body["name_check"]["confirmed"]


def test_an_empty_request_is_an_empty_answer(api) -> None:
    response = _post(api, [])

    assert response.status_code == 200
    assert response.json()["checks"] == []


def test_unknown_ids_are_not_an_error(api) -> None:
    """A list can hold a row the AI has never seen; that is not a failure to report."""
    response = _post(api, [99_123_456])

    assert response.status_code == 200
    assert response.json()["checks"] == []


def test_refuses_an_unknown_field(api) -> None:
    response = api.post(
        "/v1/documents/name-checks", json={"document_ids": [1], "include_names": True}
    )

    assert response.status_code == 422


def test_refuses_more_than_it_will_answer(api) -> None:
    """A cap the caller is told about, rather than a slow query nobody predicted."""
    response = _post(api, list(range(501)))

    assert response.status_code == 422


def test_requires_the_service_token(api) -> None:
    response = api.post(
        "/v1/documents/name-checks",
        json={"document_ids": [1]},
        headers={"Authorization": "Bearer wrong"},
    )

    assert response.status_code == 401
