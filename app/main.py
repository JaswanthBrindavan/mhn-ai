"""FastAPI application factory.

No schema work happens here: tables are created by Alembic migrations only, never
by ``create_all()`` at startup.
"""

from fastapi import FastAPI

from app.api.health import router as health_router
from app.api.v1 import router as v1_router
from app.core.config import get_settings, verify_required_settings
from app.core.errors import register_error_handlers
from app.core.logging import configure_logging


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)

    # Fail closed: refuse to start rather than run with authentication disabled.
    verify_required_settings(settings)

    app = FastAPI(
        title="MHN AI",
        description="AI-assisted medical report processing.",
        version="0.1.0",
        openapi_url="/openapi.json",
        docs_url="/docs",
    )

    register_error_handlers(app)

    # Probes stay unauthenticated so orchestrators can reach them.
    app.include_router(health_router)
    # Everything under /v1 requires the service token (see app/api/v1/__init__.py).
    app.include_router(v1_router)

    return app


app = create_app()
