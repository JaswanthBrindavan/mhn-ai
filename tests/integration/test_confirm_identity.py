"""The two answers offered on a name mismatch: "it's mine", and "whose is it?".

A mismatched document is never filed — the gate refuses before filing — so both routes
act on a document still sitting in `unclassified_files` with its object where it was
uploaded. That is why confirming can simply re-submit it through the ordinary path.

`name-candidates` is a string comparison over a list Spring supplies. It reads no family
table and makes no access decision; the tests below assert only that it matches names.
"""

import uuid

import pytest
from sqlalchemy import text

pytestmark = pytest.mark.integration


def _seed_mismatch(
    db_session,
    make_document,
    *,
    patient_name: str | None = "Priya Menon",
    status: str = "rejected",
    last_error_code: str | None = "name_mismatch",
    name_match: str | None = "mismatch",
    intended_section: str | None = None,
    analyze_now: bool = False,
) -> tuple[int, uuid.UUID]:
    """A document the gate stopped: classified, name read, not filed, intake row intact."""
    document_id = make_document()
    run_id = db_session.execute(
        text("INSERT INTO ai_processing_runs (caller) VALUES ('test') RETURNING id")
    ).scalar_one()
    item_id = db_session.execute(
        text(
            "INSERT INTO ai_processing_run_items "
            "(run_id, document_id, status, last_error_code, intended_section, analyze_now) "
            "VALUES (:r, :d, :s, :e, :i, :a) RETURNING id"
        ),
        {
            "r": run_id,
            "d": document_id,
            "s": status,
            "e": last_error_code,
            "i": intended_section,
            "a": analyze_now,
        },
    ).scalar_one()
    db_session.execute(
        text(
            "INSERT INTO ai_report_classifications "
            "(run_item_id, document_id, section, title, confidence, prompt_version, "
            " schema_version, patient_name, name_match) "
            "VALUES (:i, :d, 'reports', 'CBC', 0.95, 'clf-1', 'clf-1', :n, :m)"
        ),
        {"i": item_id, "d": document_id, "n": patient_name, "m": name_match},
    )
    db_session.commit()
    return document_id, item_id


# --- confirm-identity -------------------------------------------------------


def test_confirm_identity_requeues_the_document(api, db_session, make_document) -> None:
    document_id, item_id = _seed_mismatch(db_session, make_document)

    response = api.post(f"/v1/documents/{document_id}/confirm-identity")

    assert response.status_code == 202
    body = response.json()
    assert body["status"] in {"queued", "pending"}
    assert body["document_id"] == document_id
    assert body["item_id"] != str(item_id)


def test_confirm_identity_records_the_decision(api, db_session, make_document) -> None:
    """The stamp is the whole point: `identity.settled_verdict` reads it on the next pass
    and the gate lets the document through instead of asking again."""
    document_id, _ = _seed_mismatch(db_session, make_document)

    api.post(f"/v1/documents/{document_id}/confirm-identity")

    assert (
        db_session.execute(
            text(
                "SELECT identity_confirmed_at FROM ai_report_classifications WHERE document_id = :d"
            ),
            {"d": document_id},
        ).scalar_one()
        is not None
    )


def test_confirm_identity_keeps_the_users_section_choice(api, db_session, make_document) -> None:
    """The user answered a question about WHOSE the document is, not about where it goes."""
    document_id, _ = _seed_mismatch(db_session, make_document, intended_section="vaccinations")

    item_id = api.post(f"/v1/documents/{document_id}/confirm-identity").json()["item_id"]

    assert (
        db_session.execute(
            text("SELECT intended_section FROM ai_processing_run_items WHERE id = :i"),
            {"i": item_id},
        ).scalar_one()
        == "vaccinations"
    )


def test_confirm_identity_keeps_the_users_read_it_now_choice(
    api, db_session, make_document
) -> None:
    """The other half of the same sentence, and the half that was dropped.

    A user who ticked "read it straight away" and was then asked whose the document is
    has answered a question about the OWNER — not about whether they still want it read.
    Without this the new item defaults `analyze_now` to false, the document files and
    stops, and they are asked to press Analyse for something they already asked for.

    This is the only re-submission where it can happen: every other one is of a document
    that was already FILED, and a filed document skips the pause regardless.
    """
    document_id, _ = _seed_mismatch(db_session, make_document, analyze_now=True)

    item_id = api.post(f"/v1/documents/{document_id}/confirm-identity").json()["item_id"]

    assert (
        db_session.execute(
            text("SELECT analyze_now FROM ai_processing_run_items WHERE id = :i"),
            {"i": item_id},
        ).scalar_one()
        is True
    )


def test_confirm_identity_does_not_invent_a_read_it_now_choice(
    api, db_session, make_document
) -> None:
    """Carried, not defaulted on. A document uploaded without the tick still pauses, so
    confirming an identity never becomes a way to spend money nobody asked to spend."""
    document_id, _ = _seed_mismatch(db_session, make_document, analyze_now=False)

    item_id = api.post(f"/v1/documents/{document_id}/confirm-identity").json()["item_id"]

    assert (
        db_session.execute(
            text("SELECT analyze_now FROM ai_processing_run_items WHERE id = :i"),
            {"i": item_id},
        ).scalar_one()
        is False
    )


def test_confirm_identity_is_409_when_nothing_is_awaiting_a_decision(
    api, db_session, make_document
) -> None:
    document_id, _ = _seed_mismatch(
        db_session,
        make_document,
        status="completed",
        last_error_code=None,
        name_match="match",
    )

    response = api.post(f"/v1/documents/{document_id}/confirm-identity")

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "not_awaiting_identity"


def test_confirm_identity_is_404_for_a_document_with_no_ai_result(api, make_document) -> None:
    response = api.post(f"/v1/documents/{make_document()}/confirm-identity")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "no_ai_result"


# --- name-candidates --------------------------------------------------------


def test_name_candidates_returns_only_matching_ids(api, db_session, make_document) -> None:
    """Spring supplies the list, already filtered to who the caller may write to.

    This endpoint makes no access decision — see the spec's D8.
    """
    document_id, _ = _seed_mismatch(db_session, make_document)

    response = api.post(
        f"/v1/documents/{document_id}/name-candidates",
        json={
            "candidates": [
                {"user_id": "11111111-1111-1111-1111-111111111111", "name": "Rajesh Sharma"},
                {"user_id": "33333333-3333-3333-3333-333333333333", "name": "Priya Menon"},
            ]
        },
    )

    assert response.status_code == 200
    assert response.json()["matches"] == ["33333333-3333-3333-3333-333333333333"]


def test_name_candidates_is_empty_for_an_unreadable_name(api, db_session, make_document) -> None:
    """An unknown name must match nobody, rather than fan out across a whole family."""
    document_id, _ = _seed_mismatch(db_session, make_document, patient_name=None, name_match=None)

    response = api.post(
        f"/v1/documents/{document_id}/name-candidates",
        json={"candidates": [{"user_id": "x", "name": "Sunita Devi"}]},
    )

    assert response.status_code == 200
    assert response.json()["matches"] == []


def test_name_candidates_changes_nothing(api, db_session, make_document) -> None:
    """It answers a question; it does not act on the answer. Spring performs the move."""
    document_id, item_id = _seed_mismatch(db_session, make_document)

    api.post(
        f"/v1/documents/{document_id}/name-candidates",
        json={"candidates": [{"user_id": "a", "name": "Priya Menon"}]},
    )

    row = (
        db_session.execute(
            text("SELECT status, last_error_code FROM ai_processing_run_items WHERE id = :i"),
            {"i": item_id},
        )
        .mappings()
        .one()
    )
    assert row["status"] == "rejected"
    assert row["last_error_code"] == "name_mismatch"
    assert (
        db_session.execute(
            text("SELECT count(*) FROM ai_processing_run_items WHERE document_id = :d"),
            {"d": document_id},
        ).scalar_one()
        == 1
    )


def test_name_candidates_is_404_for_a_document_with_no_ai_result(api, make_document) -> None:
    response = api.post(
        f"/v1/documents/{make_document()}/name-candidates",
        json={"candidates": [{"user_id": "a", "name": "Priya Menon"}]},
    )

    assert response.status_code == 404


def test_name_candidates_refuses_an_unknown_field(api, db_session, make_document) -> None:
    """`extra="forbid"`: a caller sending `users` instead of `candidates` must be told,
    not answered with an empty list that reads as "nobody matches"."""
    document_id, _ = _seed_mismatch(db_session, make_document)

    response = api.post(
        f"/v1/documents/{document_id}/name-candidates",
        json={"candidates": [], "users": []},
    )

    assert response.status_code == 422
