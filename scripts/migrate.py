from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import psycopg
from psycopg.pq import TransactionStatus


MIGRATION_VERSION = "0001_durable_proposal_confirmation"
MIGRATION_FILE = "0001_durable_proposal_confirmation.sql"
MIGRATION_PATH = Path(__file__).resolve().parents[1] / "migrations" / MIGRATION_FILE
LOCK_KEY = 7_363_120_441
REQUIRED_TABLES = ("employees", "action_proposals", "leave_requests")
CONFIRMATION_COLUMNS = (
    "confirmed_at",
    "confirmed_request_id",
    "confirmed_idempotency_key",
    "confirmed_audit_reference",
)


class MigrationError(RuntimeError):
    pass


_REGISTERED_DISPOSABLE_TARGETS: set[str] = set()


def register_disposable_target(target_id: str) -> None:
    """Register a target created by the disposable PostgreSQL test harness."""
    _REGISTERED_DISPOSABLE_TARGETS.add(target_id)


@dataclass(frozen=True)
class TargetMetadata:
    database: str
    user: str
    schema: str

    @property
    def fingerprint(self) -> str:
        return sha256(f"{self.database}\x00{self.user}\x00{self.schema}".encode()).hexdigest()


def migration_checksum(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _row_value(row: Any, key: str, index: int = 0) -> Any:
    return row[key] if isinstance(row, dict) else row[index]


def target_metadata(connection: psycopg.Connection) -> TargetMetadata:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT current_database() AS database,
                   current_user AS user,
                   current_schema() AS schema
            """
        )
        row = cursor.fetchone()
    return TargetMetadata(
        database=_row_value(row, "database", 0),
        user=_row_value(row, "user", 1),
        schema=_row_value(row, "schema", 2),
    )


def validate_target(
    connection: psycopg.Connection,
    metadata: TargetMetadata,
    *,
    target_kind: str,
    test_target_id: str | None = None,
    confirmed_target_fingerprint: str | None = None,
) -> None:
    if target_kind not in {"test", "non-test"}:
        raise MigrationError("target_kind must be 'test' or 'non-test'")
    if target_kind == "test":
        if not test_target_id:
            raise MigrationError("test target requires a disposable target identity")
        if test_target_id not in _REGISTERED_DISPOSABLE_TARGETS:
            raise MigrationError("test target was not registered by the disposable test harness")
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT 1
                FROM migration_test_targets
                WHERE target_id = %s
                  AND database_name = current_database()
                  AND schema_name = current_schema()
                """,
                (test_target_id,),
            )
            if cursor.fetchone() is None:
                raise MigrationError("database is not the declared disposable test target")
    elif confirmed_target_fingerprint != metadata.fingerprint:
        raise MigrationError(
            "non-test target requires confirmation bound to the inspected target fingerprint"
        )


def _table_exists(cursor: Any, table: str) -> bool:
    cursor.execute("SELECT to_regclass(%s) AS relation", (f"public.{table}",))
    row = cursor.fetchone()
    return (row["relation"] if isinstance(row, dict) else row[0]) is not None


def validate_preconditions(connection: psycopg.Connection) -> None:
    with connection.cursor() as cursor:
        missing = [table for table in REQUIRED_TABLES if not _table_exists(cursor, table)]
        if missing:
            raise MigrationError(f"missing baseline table(s): {', '.join(missing)}")

        cursor.execute(
            """
            SELECT 1
            FROM pg_constraint
            WHERE conrelid = 'action_proposals'::regclass
              AND contype IN ('p', 'u')
              AND conkey @> ARRAY[
                  (SELECT attnum FROM pg_attribute
                   WHERE attrelid = 'action_proposals'::regclass
                     AND attname = 'proposal_id')
              ]::smallint[]
            LIMIT 1
            """
        )
        if cursor.fetchone() is None:
            raise MigrationError("action_proposals.proposal_id must have a primary or unique key")

        cursor.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name = 'leave_requests'
              AND column_name = 'proposal_id'
            """
        )
        has_proposal_id = cursor.fetchone() is not None
        if has_proposal_id:
            cursor.execute(
                """
                SELECT proposal_id
                FROM leave_requests
                WHERE proposal_id IS NOT NULL
                GROUP BY proposal_id
                HAVING count(*) > 1
                LIMIT 1
                """
            )
            if cursor.fetchone() is not None:
                raise MigrationError("existing duplicate non-null leave_requests.proposal_id")

            cursor.execute(
                """
                SELECT 1
                FROM leave_requests lr
                LEFT JOIN action_proposals ap ON ap.proposal_id = lr.proposal_id
                WHERE lr.proposal_id IS NOT NULL AND ap.proposal_id IS NULL
                LIMIT 1
                """
            )
            if cursor.fetchone() is not None:
                raise MigrationError("existing orphan leave_requests.proposal_id")

        existing_columns = set()
        cursor.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name = 'action_proposals'
            """
        )
        existing_columns.update(_row_value(row, "column_name") for row in cursor.fetchall())
        marker_columns = existing_columns.intersection(CONFIRMATION_COLUMNS)
        if marker_columns and marker_columns != set(CONFIRMATION_COLUMNS):
            raise MigrationError("action_proposals has a partial confirmation marker schema")
        if marker_columns == set(CONFIRMATION_COLUMNS):
            cursor.execute(
                """
                SELECT 1
                FROM action_proposals
                WHERE (
                    (confirmed_at IS NULL)::int
                    + (confirmed_request_id IS NULL)::int
                    + (confirmed_idempotency_key IS NULL)::int
                    + (confirmed_audit_reference IS NULL)::int
                ) NOT IN (0, 4)
                LIMIT 1
                """
            )
            if cursor.fetchone() is not None:
                raise MigrationError("existing partial action proposal confirmation marker")


def validate_postconditions(connection: psycopg.Connection) -> None:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name = 'action_proposals'
              AND column_name = ANY(%s)
            """,
            (list(CONFIRMATION_COLUMNS),),
        )
        if {
            _row_value(row, "column_name") for row in cursor.fetchall()
        } != set(CONFIRMATION_COLUMNS):
            raise MigrationError("confirmation marker columns are incomplete")

        cursor.execute(
            """
            SELECT is_nullable
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name = 'leave_requests'
              AND column_name = 'proposal_id'
            """
        )
        row = cursor.fetchone()
        if row is None or _row_value(row, "is_nullable") != "YES":
            raise MigrationError("leave_requests.proposal_id must remain nullable")

        cursor.execute(
            """
            SELECT 1
            FROM pg_constraint
            WHERE conname = 'leave_requests_proposal_fk'
              AND conrelid = 'leave_requests'::regclass
              AND contype = 'f'
              AND confrelid = 'action_proposals'::regclass
              AND conkey = ARRAY[
                  (SELECT attnum FROM pg_attribute
                   WHERE attrelid = 'leave_requests'::regclass
                     AND attname = 'proposal_id')
              ]::smallint[]
              AND confkey = ARRAY[
                  (SELECT attnum FROM pg_attribute
                   WHERE attrelid = 'action_proposals'::regclass
                     AND attname = 'proposal_id')
              ]::smallint[]
            """
        )
        if cursor.fetchone() is None:
            raise MigrationError("leave_requests proposal foreign key is missing")

        cursor.execute(
            """
            SELECT 1
            FROM pg_class index_class
            JOIN pg_index index_info ON index_info.indexrelid = index_class.oid
            JOIN pg_class table_class ON table_class.oid = index_info.indrelid
            JOIN pg_attribute index_column
              ON index_column.attrelid = table_class.oid
             AND index_column.attnum = index_info.indkey[0]
            WHERE index_class.relname = 'leave_requests_proposal_uidx'
              AND table_class.relname = 'leave_requests'
              AND index_info.indisunique
              AND index_info.indnkeyatts = 1
              AND index_column.attname = 'proposal_id'
              AND pg_get_expr(index_info.indpred, index_info.indrelid)
                    ILIKE '%proposal_id IS NOT NULL%'
            """
        )
        if cursor.fetchone() is None:
            raise MigrationError("leave_requests proposal unique partial index is missing")

        cursor.execute(
            """
            SELECT pg_get_constraintdef(oid) AS definition
            FROM pg_constraint
            WHERE conname = 'action_proposals_confirmation_state_ck'
              AND conrelid = 'action_proposals'::regclass
              AND contype = 'c'
            """
        )
        row = cursor.fetchone()
        definition = _row_value(row, "definition") if row is not None else ""
        normalized = " ".join(definition.lower().split())
        if (
            row is None
            or "confirmed_at is null" not in normalized
            or "confirmed_request_id is null" not in normalized
            or "confirmed_idempotency_key is null" not in normalized
            or "confirmed_audit_reference is null" not in normalized
            or "confirmed_at is not null" not in normalized
            or "confirmed_request_id is not null" not in normalized
            or "confirmed_idempotency_key is not null" not in normalized
            or "confirmed_audit_reference is not null" not in normalized
        ):
            raise MigrationError("confirmation state CHECK constraint is missing")


def _ensure_tracking_table(connection: psycopg.Connection) -> None:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version TEXT PRIMARY KEY,
                checksum TEXT NOT NULL,
                applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )


def apply_migration(
    connection: psycopg.Connection,
    migration_path: Path,
    *,
    version: str = MIGRATION_VERSION,
    target_kind: str | None = None,
    test_target_id: str | None = None,
    confirmed_target_fingerprint: str | None = None,
) -> bool:
    if connection.info.transaction_status != TransactionStatus.IDLE:
        raise MigrationError("apply_migration requires a top-level idle connection")
    checksum = migration_checksum(migration_path)
    with connection.transaction():
        metadata = target_metadata(connection)
        if target_kind is not None:
            validate_target(
                connection,
                metadata,
                target_kind=target_kind,
                test_target_id=test_target_id,
                confirmed_target_fingerprint=confirmed_target_fingerprint,
            )
        with connection.cursor() as cursor:
            cursor.execute("SELECT pg_advisory_xact_lock(%s)", (LOCK_KEY,))
        _ensure_tracking_table(connection)
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT checksum FROM schema_migrations WHERE version = %s",
                (version,),
            )
            record = cursor.fetchone()
        if record is not None:
            if _row_value(record, "checksum") != checksum:
                raise MigrationError(f"checksum mismatch for migration {version}")
            return False

        validate_preconditions(connection)
        with connection.cursor() as cursor:
            cursor.execute(migration_path.read_text(encoding="utf-8"))
        validate_postconditions(connection)
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO schema_migrations (version, checksum)
                VALUES (%s, %s)
                """,
                (version, checksum),
            )
    return True


def run(
    database_url: str,
    *,
    migration_path: Path,
    target_kind: str,
    test_target_id: str | None = None,
    confirmed_target_fingerprint: str | None = None,
) -> bool:
    with psycopg.connect(database_url) as connection:
        metadata = target_metadata(connection)
        validate_target(
            connection,
            metadata,
            target_kind=target_kind,
            test_target_id=test_target_id,
            confirmed_target_fingerprint=confirmed_target_fingerprint,
        )
        connection.rollback()
        return apply_migration(
            connection,
            migration_path,
            target_kind=target_kind,
            test_target_id=test_target_id,
            confirmed_target_fingerprint=confirmed_target_fingerprint,
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Disposable PostgreSQL test migration runner. "
            "Not for Neon, development, or other non-test databases."
        )
    )
    parser.add_argument("--database-url", required=True)
    parser.add_argument("--migration", type=Path, default=MIGRATION_PATH)
    parser.add_argument(
        "--target-kind",
        choices=("test", "non-test"),
        required=True,
        help="must be test; non-test targets are rejected",
    )
    parser.add_argument("--test-target-id")
    parser.add_argument("--confirm-target-fingerprint")
    args = parser.parse_args()
    parser.error(
        "direct CLI execution is disabled; use the disposable PostgreSQL pytest harness. "
        "This runner cannot migrate Neon or any non-test database."
    )


if __name__ == "__main__":
    raise SystemExit(main())
