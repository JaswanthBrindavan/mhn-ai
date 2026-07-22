"""Database engine and session management.

Sessions are synchronous on purpose: the worker's other I/O (boto3, the Anthropic
SDK) is synchronous, and FastAPI runs ``def`` handlers in a threadpool. A single
concurrency model is easier to reason about than mixing async and sync sessions.
"""

from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import get_settings


class Base(DeclarativeBase):
    """Declarative base for service-owned tables only.

    Every table defined against this base MUST be named with an ``ai_`` prefix.
    Alembic's ``include_object`` filter relies on that prefix to avoid touching
    the Spring-owned schema sharing this database.
    """


_settings = get_settings()

engine = create_engine(
    _settings.database_url,
    pool_pre_ping=True,  # tolerate connections dropped by an idle timeout
    future=True,
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def get_session() -> Generator[Session, None, None]:
    """FastAPI dependency yielding a session that always closes."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
