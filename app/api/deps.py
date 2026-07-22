"""Shared FastAPI dependencies.

Authentication model — read before changing anything here.

This service is an internal component that only the Spring backend calls. The token
below authenticates **Spring as a service**, not the end user. Spring has already
decided whether the human is allowed to touch the report before it calls us, so this
service performs **no authorization of its own**.

That is deliberate. Report access in MHN depends on ``family_connect`` (with
asymmetric ``req_file_share``/``acc_file_share`` direction flags),
``family_file_access`` per-resource overrides, ``reports.private``, and
``reports.created_by``. Spring already implements those rules. A second
implementation here would drift, and a drift bug leaks one family member's medical
records to another.

In particular: **do not add a check comparing the requesting user to
``reports.user_id``.** ``reports`` separates ``user_id`` (the subject of the report)
from ``created_by`` (whoever uploaded it), and family-connect lets one user upload a
relative's report. Such a check rejects legitimate family uploads while providing no
real protection, since the caller supplies the id. ``tests/unit/test_service_auth.py``
asserts this check stays absent.

Two conditions make this model safe, and both live outside this file:
  1. The service must not be reachable from the public internet.
  2. Spring must re-authorize on every *read* of AI results, since
     ``GET /v1/reports/{id}/ai-result`` returns lab values for any id presented with
     a valid token.
"""

import secrets

from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.config import Settings, get_settings
from app.core.errors import ApiError

# auto_error=False so a missing header produces our own error envelope rather than
# Starlette's differently-shaped default.
_bearer = HTTPBearer(auto_error=False, description="Shared MHN service token")


def require_service_token(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    settings: Settings = Depends(get_settings),
) -> None:
    """Authenticate the calling service. Says nothing about which user is behind it."""
    expected = settings.mhn_service_token

    # Startup already refuses an empty token; this is defence in depth for anything
    # that builds an app without that check. Never authenticate against "".
    if not expected:
        raise ApiError(
            503,
            "service_misconfigured",
            "Service credentials are not configured",
        )

    if credentials is None:
        raise ApiError(401, "unauthorized", "Missing service credentials")

    # compare_digest, not ==, so response timing cannot leak the token byte by byte.
    if not secrets.compare_digest(credentials.credentials, expected):
        raise ApiError(401, "unauthorized", "Invalid service credentials")
