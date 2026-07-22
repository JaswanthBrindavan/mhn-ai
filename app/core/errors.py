"""Stable, machine-readable error bodies.

Responses never carry stack traces, prompts, credentials, S3 keys, or report
contents. Handlers raise ``ApiError``; the handlers registered here shape it into
a consistent envelope.
"""

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


class ApiError(Exception):
    """Raised by application code to produce a controlled error response."""

    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details or {}


def _envelope(code: str, message: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"error": {"code": code, "message": message, "details": details or {}}}


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    def _handle_api_error(_: Request, exc: ApiError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=_envelope(exc.code, exc.message, exc.details),
        )

    @app.exception_handler(RequestValidationError)
    def _handle_validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        # Field locations and messages only. Never echo submitted values back,
        # since a request body may carry report content.
        fields = [
            {"field": ".".join(str(part) for part in err["loc"]), "reason": err["msg"]}
            for err in exc.errors()
        ]
        return JSONResponse(
            status_code=422,
            content=_envelope("validation_error", "Request validation failed", {"fields": fields}),
        )

    @app.exception_handler(Exception)
    def _handle_unexpected(_: Request, exc: Exception) -> JSONResponse:
        # Full detail to the log, nothing to the client.
        logger.exception("unhandled_exception", exc_info=exc)
        return JSONResponse(
            status_code=500,
            content=_envelope("internal_error", "An internal error occurred"),
        )
