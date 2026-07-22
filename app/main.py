"""FastAPI application factory.

No schema work happens here: tables are created by Alembic migrations only, never
by ``create_all()`` at startup.
"""

from fastapi import FastAPI

from app.api.health import router as health_router
from app.core.config import get_settings
from app.core.errors import register_error_handlers
from app.core.logging import configure_logging


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(
        title="MHN AI",
        description="AI-assisted medical report processing.",
        version="0.1.0",
        openapi_url="/openapi.json",
        docs_url="/docs",
    )

    register_error_handlers(app)

    # Probes stay unversioned; v1 resource routers are added in step 3.
    app.include_router(health_router)

    return app


app = create_app()
