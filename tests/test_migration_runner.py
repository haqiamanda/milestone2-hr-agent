from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile

import psycopg
import pytest

from scripts.migrate import (
    MigrationError,
    MIGRATION_PATH,
    apply_migration,
    target_metadata,
    validate_target,
)
from tests.postgres_fixture import EMPLOYEE_ID, PROPOSAL_ID, PostgresTestDatabase


pytestmark = pytest.mark.postgres_integration


def _migration_record(database: PostgresTestDatabase):
    with database.connect() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT to_regclass('public.schema_migrations') AS relation"
            )
            if cursor.fetchone()["relation"] is None:
                return []
            cursor.execute("SELECT version, checksum FROM schema_migrations")
            return cursor.fetchall()


def test_runner_applies_and_skips_same_checksum(
    postgres_database: PostgresTestDatabase,
) -> None:
    postgres_database.reset_baseline()
    assert postgres_database.apply_migration() is True
    assert postgres_database.apply_migration() is False
    assert len(_migration_record(postgres_database)) == 1


def test_runner_rejects_checksum_drift_without_schema_change(
    postgres_database: PostgresTestDatabase,
) -> None:
    postgres_database.reset_baseline()
    assert postgres_database.apply_migration() is True
    with NamedTemporaryFile("w", suffix=".sql", encoding="utf-8", delete=False) as file:
        file.write(MIGRATION_PATH.read_text(encoding="utf-8") + "\n-- drift\n")
        drift_path = Path(file.name)
    try:
        with pytest.raises(MigrationError, match="checksum mismatch"):
            with postgres_database.connect() as connection:
                apply_migration(
                    connection,
                    drift_path,
                    target_kind="test",
                    test_target_id=postgres_database.target_id,
                )
    finally:
        drift_path.unlink()
    assert len(_migration_record(postgres_database)) == 1


def test_runner_rolls_back_failed_migration_and_record(
    postgres_database: PostgresTestDatabase,
) -> None:
    postgres_database.reset_baseline()
    with NamedTemporaryFile("w", suffix=".sql", encoding="utf-8", delete=False) as file:
        file.write(
            "ALTER TABLE action_proposals ADD COLUMN transient_marker TEXT;\n"
            "SELECT no_such_function();\n"
        )
        failed_path = Path(file.name)
    try:
        with pytest.raises(psycopg.errors.UndefinedFunction):
            with postgres_database.connect() as connection:
                apply_migration(connection, failed_path, version="0001_failed")
    finally:
        failed_path.unlink()
    with postgres_database.connect() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'action_proposals'
                  AND column_name = 'transient_marker'
                """
            )
            assert cursor.fetchone() is None
            cursor.execute("SELECT to_regclass('public.schema_migrations') AS relation")
            assert cursor.fetchone()["relation"] is None


def test_runner_preserves_historical_request_without_proposal(
    postgres_database: PostgresTestDatabase,
) -> None:
    postgres_database.reset_baseline()
    with postgres_database.connect() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO employees (employee_id, display_name, role)
                VALUES (%s, %s, %s)
                """,
                (EMPLOYEE_ID, "Historical Employee", "employee"),
            )
            cursor.execute(
                """
                INSERT INTO leave_requests
                    (request_id, employee_id, leave_type, start_date, end_date,
                     business_days, status, created_at)
                VALUES (%s, %s, %s, CURRENT_DATE, CURRENT_DATE, 1,
                        'approved', now())
                """,
                ("LR-HISTORICAL", EMPLOYEE_ID, "annual"),
            )
    postgres_database.apply_migration()
    with postgres_database.connect() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT proposal_id FROM leave_requests WHERE request_id = %s",
                ("LR-HISTORICAL",),
            )
            assert cursor.fetchone()["proposal_id"] is None


def test_runner_rejects_invalid_existing_data_before_migration(
    postgres_database: PostgresTestDatabase,
) -> None:
    postgres_database.reset_baseline()
    with postgres_database.connect() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "ALTER TABLE leave_requests ADD COLUMN proposal_id TEXT"
            )
            cursor.execute(
                """
                INSERT INTO employees (employee_id, display_name, role)
                VALUES (%s, %s, %s)
                """,
                (
                    EMPLOYEE_ID,
                    "Invalid Employee",
                    "employee",
                ),
            )
            cursor.execute(
                """
                INSERT INTO leave_requests
                    (request_id, proposal_id, employee_id, leave_type, start_date,
                     end_date, business_days, status, created_at)
                VALUES (%s, %s, %s, %s, CURRENT_DATE, CURRENT_DATE, 1,
                        'approved', now())
                """,
                (
                    "LR-ORPHAN",
                    "AP-ORPHAN",
                    EMPLOYEE_ID,
                    "annual",
                ),
            )
    with pytest.raises(MigrationError, match="orphan"):
        postgres_database.apply_migration()
    assert _migration_record(postgres_database) == []


def test_runner_rejects_wrong_definition_with_expected_constraint_name(
    postgres_database: PostgresTestDatabase,
) -> None:
    postgres_database.reset_baseline()
    with postgres_database.connect() as connection:
        with connection.cursor() as cursor:
            cursor.execute("ALTER TABLE leave_requests ADD COLUMN proposal_id TEXT")
            cursor.execute(
                """
                ALTER TABLE leave_requests
                ADD CONSTRAINT leave_requests_proposal_fk
                FOREIGN KEY (proposal_id) REFERENCES employees(employee_id)
                """
            )
    with pytest.raises(MigrationError, match="foreign key"):
        postgres_database.apply_migration()
    assert _migration_record(postgres_database) == []


def test_runner_rejects_unconfirmed_non_test_target(
    postgres_database: PostgresTestDatabase,
) -> None:
    with postgres_database.connect() as connection:
        metadata = target_metadata(connection)
        with pytest.raises(MigrationError, match="confirmation bound"):
            validate_target(connection, metadata, target_kind="non-test")


def test_live_like_target_cannot_be_labeled_test(
    postgres_database: PostgresTestDatabase,
) -> None:
    postgres_database.reset_baseline()
    with postgres_database.connect() as connection:
        metadata = target_metadata(connection)
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO migration_test_targets
                    (target_id, database_name, schema_name)
                VALUES (%s, current_database(), current_schema())
                """,
                ("homemade-marker",),
            )
        with pytest.raises(MigrationError, match="disposable test harness"):
            validate_target(
                connection,
                metadata,
                target_kind="test",
                test_target_id="homemade-marker",
            )


def test_runner_rejects_connection_with_active_transaction(
    postgres_database: PostgresTestDatabase,
) -> None:
    with postgres_database.connect() as connection:
        with connection.transaction():
            with pytest.raises(MigrationError, match="top-level idle"):
                apply_migration(
                    connection,
                    MIGRATION_PATH,
                    target_kind="test",
                    test_target_id=postgres_database.target_id,
                )


def test_cli_rejects_non_test_and_direct_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import migrate

    monkeypatch.setattr(
        sys,
        "argv",
        ["migrate.py", "--database-url", "unused", "--target-kind", "non-test"],
    )
    with pytest.raises(SystemExit) as error:
        migrate.main()
    assert error.value.code == 2


def test_concurrent_runners_record_migration_once(
    postgres_database: PostgresTestDatabase,
) -> None:
    postgres_database.reset_baseline()

    def apply():
        return postgres_database.apply_migration()

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: apply(), range(2)))

    assert sorted(results) == [False, True]
    assert len(_migration_record(postgres_database)) == 1
