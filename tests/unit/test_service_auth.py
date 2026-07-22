"""Service-token authentication (Approach A).

The service authenticates *Spring as a service* and performs no authorization of its
own. These tests pin both halves of that: the token is enforced, and no user-ownership
check exists.
"""

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from app.api.deps import require_service_token
from app.api.v1 import router as v1_router
from app.core.config import (
    MIN_SERVICE_TOKEN_LENGTH,
    Settings,
    get_settings,
    verify_required_settings,
)

VALID_TOKEN = "s" * 40
BASE = {"database_url": "postgresql+psycopg://u:p@h:5432/d"}


def _probe_app(token: str) -> FastAPI:
    """A minimal app guarded by the real dependency."""
    app = FastAPI()

    @app.get("/guarded", dependencies=[Depends(require_service_token)])
    def guarded() -> dict[str, bool]:
        return {"ok": True}

    from app.core.errors import register_error_handlers

    register_error_handlers(app)
    app.dependency_overrides[get_settings] = lambda: Settings(**BASE, mhn_service_token=token)
    return app


# --- token enforcement ------------------------------------------------------


def test_missing_token_is_rejected():
    with TestClient(_probe_app(VALID_TOKEN), raise_server_exceptions=False) as client:
        response = client.get("/guarded")
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "unauthorized"


def test_wrong_token_is_rejected():
    with TestClient(_probe_app(VALID_TOKEN), raise_server_exceptions=False) as client:
        response = client.get("/guarded", headers={"Authorization": "Bearer wrong-token"})
    assert response.status_code == 401


def test_correct_token_is_accepted():
    with TestClient(_probe_app(VALID_TOKEN), raise_server_exceptions=False) as client:
        response = client.get("/guarded", headers={"Authorization": f"Bearer {VALID_TOKEN}"})
    assert response.status_code == 200
    assert response.json() == {"ok": True}


def test_token_prefix_is_not_enough():
    # Guards against a startswith/truncating comparison.
    with TestClient(_probe_app(VALID_TOKEN), raise_server_exceptions=False) as client:
        response = client.get("/guarded", headers={"Authorization": f"Bearer {VALID_TOKEN[:20]}"})
    assert response.status_code == 401


def test_empty_configured_token_never_authenticates():
    """An unset token must not silently mean 'allow everyone'."""
    app = _probe_app("")
    with TestClient(app, raise_server_exceptions=False) as client:
        # Even sending an empty bearer token must fail, not match "" == "".
        no_header = client.get("/guarded")
        empty_bearer = client.get("/guarded", headers={"Authorization": "Bearer "})

    assert no_header.status_code == 503
    assert empty_bearer.status_code == 503
    assert no_header.json()["error"]["code"] == "service_misconfigured"


def test_error_response_does_not_echo_the_token():
    with TestClient(_probe_app(VALID_TOKEN), raise_server_exceptions=False) as client:
        response = client.get("/guarded", headers={"Authorization": f"Bearer {VALID_TOKEN}x"})
    assert VALID_TOKEN not in response.text


# --- fail-closed startup ----------------------------------------------------


def test_startup_refuses_empty_token():
    with pytest.raises(RuntimeError, match="MHN_SERVICE_TOKEN is not set"):
        verify_required_settings(Settings(**BASE, mhn_service_token=""))


def test_startup_refuses_short_token():
    short = "a" * (MIN_SERVICE_TOKEN_LENGTH - 1)
    with pytest.raises(RuntimeError, match="too short"):
        verify_required_settings(Settings(**BASE, mhn_service_token=short))


def test_startup_accepts_a_long_token():
    verify_required_settings(Settings(**BASE, mhn_service_token=VALID_TOKEN))


# --- routing --------------------------------------------------------------


def test_v1_router_requires_the_token_by_default():
    """Attached at the router, so a new endpoint cannot forget to authenticate."""
    dependency_calls = [d.dependency for d in v1_router.dependencies]
    assert require_service_token in dependency_calls


def test_probes_are_reachable_without_a_token(client):
    # Orchestrator health checks have no credentials.
    assert client.get("/health").status_code == 200


# --- authorization is deliberately absent -----------------------------------


def test_service_performs_no_user_ownership_check():
    """Regression guard for a real bug that was caught in review.

    `reports` separates `user_id` (the subject) from `created_by` (the uploader), and
    family-connect lets one user upload a relative's report. A check comparing the
    requesting user to `reports.user_id` therefore rejects legitimate family uploads
    while providing no real protection, since the caller supplies the id.

    Authorization belongs to Spring. If someone adds an ownership check here, this
    test should fail and send them to app/api/deps.py for the reasoning.
    """
    import inspect

    import app.api.deps as deps

    source = inspect.getsource(deps)
    offending = ["reports.user_id", "report.user_id", "created_by"]
    for pattern in offending:
        # Mentions in the explanatory docstring are fine; executable comparisons are not.
        code = source.split('"""', 2)[-1]
        assert pattern not in code, (
            f"{pattern!r} appears in executable code in app/api/deps.py. "
            "This service must not perform user-level authorization; see the module "
            "docstring for why."
        )
