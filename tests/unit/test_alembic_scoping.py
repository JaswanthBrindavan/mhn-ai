"""The include_object filter is the only thing standing between
`alembic revision --autogenerate` and a migration that drops all 31 Spring tables.
It is tested as load-bearing safety code, not as configuration.
"""

from types import SimpleNamespace

import pytest

from app.core.migration_scope import include_object

# Real table names taken from the live mhn_ai database.
SPRING_TABLES = [
    "reports",
    "user",
    "prescriptions",
    "scans_imaging",
    "hospital",
    "medicine_master",
    "vital_reading",
    "family_file_access",
]

OWNED_TABLES = [
    "ai_processing_runs",
    "ai_processing_run_items",
    "ai_report_classifications",
    "ai_report_extractions",
    "ai_report_insights",
    "ai_process_logs",
    "ai_alembic_version",
]


@pytest.mark.parametrize("table_name", SPRING_TABLES)
def test_spring_tables_are_excluded(table_name):
    assert include_object(None, table_name, "table", True, None) is False


@pytest.mark.parametrize("table_name", OWNED_TABLES)
def test_owned_tables_are_included(table_name):
    assert include_object(None, table_name, "table", True, None) is True


def test_index_on_spring_table_is_excluded():
    index = SimpleNamespace(table=SimpleNamespace(name="reports"))
    assert include_object(index, "idx_reports_user_id", "index", True, None) is False


def test_index_on_owned_table_is_included():
    index = SimpleNamespace(table=SimpleNamespace(name="ai_processing_runs"))
    assert include_object(index, "idx_ai_runs_status", "index", True, None) is True


def test_foreign_key_to_spring_table_is_excluded():
    fk = SimpleNamespace(table=SimpleNamespace(name="prescriptions"))
    assert include_object(fk, "fk_x", "foreign_key_constraint", True, None) is False


def test_lookalike_prefix_is_not_owned():
    # "airline" starts with "ai" but not "ai_" — guards against a sloppy check.
    assert include_object(None, "airline_bookings", "table", True, None) is False


def test_none_and_empty_names_are_not_owned():
    assert include_object(None, None, "table", True, None) is False
    assert include_object(None, "", "table", True, None) is False
