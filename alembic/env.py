"""Alembic environment, scoped to service-owned tables.

The ownership rules live in ``app.core.migration_scope`` so they can be unit-tested;
this module only wires them into Alembic.
"""

from logging.config import fileConfig

from alembic import context
from sqlalchemy import Connection, engine_from_config, pool

from app.core.config import get_settings
from app.core.db import Base
from app.core.migration_scope import VERSION_TABLE, include_object

# Importing the models package registers every service-owned table on Base.metadata.
import app.models  # noqa: F401  isort:skip

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Single source of connection truth: the same setting the application uses, so the
# app, Alembic, and tests can never disagree about which database they mean.
config.set_main_option("sqlalchemy.url", get_settings().database_url)

target_metadata = Base.metadata


def _configure(connection: Connection | None = None, url: str | None = None) -> None:
    context.configure(
        connection=connection,
        url=url,
        target_metadata=target_metadata,
        include_object=include_object,
        include_schemas=False,
        version_table=VERSION_TABLE,
        compare_type=True,
        compare_server_default=True,
        literal_binds=url is not None,
        dialect_opts={"paramstyle": "named"} if url is not None else {},
    )


def run_migrations_offline() -> None:
    _configure(url=config.get_main_option("sqlalchemy.url"))
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        _configure(connection=connection)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
