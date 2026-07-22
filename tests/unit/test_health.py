from app.core.db import get_session
from app.core.errors import ApiError
from app.main import create_app


def test_health_does_not_touch_dependencies(client):
    # No database configured in this test path; /health must still answer.
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready_reports_down_database_as_503():
    app = create_app()

    def broken_session():
        class Failing:
            def execute(self, *_args, **_kwargs):
                raise RuntimeError("connection refused")

        yield Failing()

    app.dependency_overrides[get_session] = broken_session

    from fastapi.testclient import TestClient

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/ready")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "not_ready"
    assert body["checks"]["database"]["status"] == "down"
    # The underlying error text must not leak to the client.
    assert "connection refused" not in response.text


def test_error_envelope_is_stable_and_leaks_nothing():
    app = create_app()

    @app.get("/boom")
    def boom() -> None:
        raise ApiError(409, "duplicate_active_item", "Report already being processed")

    @app.get("/unexpected")
    def unexpected() -> None:
        raise RuntimeError("secret-bucket-key-abc123")

    from fastapi.testclient import TestClient

    with TestClient(app, raise_server_exceptions=False) as client:
        handled = client.get("/boom")
        unhandled = client.get("/unexpected")

    assert handled.status_code == 409
    assert handled.json() == {
        "error": {
            "code": "duplicate_active_item",
            "message": "Report already being processed",
            "details": {},
        }
    }

    assert unhandled.status_code == 500
    assert unhandled.json()["error"]["code"] == "internal_error"
    # No stack trace, no internal detail.
    assert "secret-bucket-key-abc123" not in unhandled.text
    assert "Traceback" not in unhandled.text
