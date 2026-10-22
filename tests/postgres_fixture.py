from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import re
import socket
import subprocess
import time
from uuid import uuid4

import psycopg
import pytest
from psycopg.rows import dict_row

from src.repository import PostgresRepository
from scripts.migrate import MIGRATION_PATH, apply_migration, register_disposable_target


ROOT = Path(__file__).resolve().parents[1]
POSTGRES_IMAGE = "pgvector/pgvector:pg16"
POSTGRES_USER = "postgres"
POSTGRES_PASSWORD = "postgres"
POSTGRES_DB = "hr_agent_test"
EMPLOYEE_ID = "EMP-TEST"
PROPOSAL_ID = "AP-TEST"

BASELINE_SCHEMA = """
CREATE TABLE employees (
    employee_id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    role TEXT NOT NULL
);

CREATE TABLE action_proposals (
    proposal_id TEXT PRIMARY KEY,
    employee_id TEXT NOT NULL REFERENCES employees(employee_id),
    leave_type TEXT NOT NULL,
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    business_days INTEGER NOT NULL CHECK (business_days > 0),
    reason TEXT,
    validation_summary JSONB NOT NULL,
    expires_at TIMESTAMPTZ NOT NULL,
    requires_confirmation BOOLEAN NOT NULL DEFAULT TRUE
);

CREATE TABLE leave_requests (
    request_id TEXT PRIMARY KEY,
    employee_id TEXT NOT NULL REFERENCES employees(employee_id),
    leave_type TEXT NOT NULL,
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    business_days INTEGER NOT NULL CHECK (business_days > 0),
    status TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL
);
"""


@dataclass(frozen=True)
class PostgresTestDatabase:
    url: str
    target_id: str

    def connect(self):
        return psycopg.connect(self.url, row_factory=dict_row)

    def reset_case(self) -> None:
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute("TRUNCATE leave_requests, action_proposals, employees CASCADE")
                cursor.execute(
                    """
                    INSERT INTO employees (employee_id, display_name, role)
                    VALUES (%s, %s, %s)
                    """,
                    (EMPLOYEE_ID, "Integration Test Employee", "employee"),
                )
                cursor.execute(
                    """
                    INSERT INTO action_proposals
                        (proposal_id, employee_id, leave_type, start_date, end_date,
                         business_days, reason, validation_summary, expires_at,
                         requires_confirmation)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        PROPOSAL_ID,
                        EMPLOYEE_ID,
                        "annual",
                        date(2026, 11, 2),
                        date(2026, 11, 3),
                        2,
                        "integration test",
                        '["valid"]',
                        datetime.now(timezone.utc) + timedelta(minutes=10),
                        True,
                    ),
                )

    def reset_baseline(self) -> None:
        with self.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute("DROP SCHEMA public CASCADE")
                cursor.execute("CREATE SCHEMA public")
                cursor.execute(BASELINE_SCHEMA)
                cursor.execute(
                    """
                    CREATE TABLE migration_test_targets (
                        target_id TEXT PRIMARY KEY,
                        database_name TEXT NOT NULL,
                        schema_name TEXT NOT NULL
                    )
                    """
                )
                cursor.execute(
                    """
                    INSERT INTO migration_test_targets
                        (target_id, database_name, schema_name)
                    VALUES (%s, current_database(), current_schema())
                    """,
                    (self.target_id,),
                )

    def apply_migration(self) -> bool:
        with self.connect() as connection:
            return apply_migration(
                connection,
                MIGRATION_PATH,
                target_kind="test",
                test_target_id=self.target_id,
            )

    def repository(self) -> PostgresRepository:
        return PostgresRepository(database_url=self.url, embedding_dimensions=1536)


def _docker(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", *args],
        check=check,
        capture_output=True,
        text=True,
    )


def _docker_available() -> bool:
    try:
        result = _docker("info", "--format", "{{.ServerVersion}}", check=False)
    except OSError:
        return False
    return result.returncode == 0 and bool(result.stdout.strip())


def _free_host_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_database(url: str) -> None:
    deadline = time.monotonic() + 30
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with psycopg.connect(url) as connection:
                with connection.cursor() as cursor:
                    cursor.execute("SELECT 1")
                    if cursor.fetchone() == (1,):
                        return
        except psycopg.Error as exc:
            last_error = exc
        time.sleep(0.5)
    raise RuntimeError("disposable PostgreSQL did not become ready") from last_error


def _container_port(container_name: str) -> int:
    result = _docker("port", container_name, "5432/tcp")
    match = re.search(r":(\d+)\s*$", result.stdout.strip())
    if match is None:
        raise RuntimeError("could not determine disposable PostgreSQL host port")
    return int(match.group(1))


@pytest.fixture(scope="session")
def postgres_database() -> PostgresTestDatabase:
    if not _docker_available():
        pytest.skip("Docker daemon is unavailable; PostgreSQL integration tests skipped")

    container_name = f"hr-agent-pgtest-{uuid4().hex[:12]}"
    host_port = _free_host_port()
    container = _docker(
        "run",
        "--detach",
        "--name",
        container_name,
        "--publish",
        f"127.0.0.1:{host_port}:5432",
        "--env",
        f"POSTGRES_USER={POSTGRES_USER}",
        "--env",
        f"POSTGRES_PASSWORD={POSTGRES_PASSWORD}",
        "--env",
        f"POSTGRES_DB={POSTGRES_DB}",
        "--rm",
        POSTGRES_IMAGE,
    )
    if not container.stdout.strip():
        raise RuntimeError("Docker did not return a disposable PostgreSQL container ID")

    url = (
        f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
        f"@127.0.0.1:{host_port}/{POSTGRES_DB}"
    )
    try:
        _wait_for_database(url)
        with psycopg.connect(url) as connection:
            with connection.cursor() as cursor:
                cursor.execute(BASELINE_SCHEMA)
                target_id = uuid4().hex
                cursor.execute(
                    """
                    CREATE TABLE migration_test_targets (
                        target_id TEXT PRIMARY KEY,
                        database_name TEXT NOT NULL,
                        schema_name TEXT NOT NULL
                    )
                    """
                )
                cursor.execute(
                    """
                    INSERT INTO migration_test_targets
                        (target_id, database_name, schema_name)
                    VALUES (%s, current_database(), current_schema())
                    """,
                    (target_id,),
                )
                connection.commit()
            register_disposable_target(target_id)
            apply_migration(
                connection,
                MIGRATION_PATH,
                target_kind="test",
                test_target_id=target_id,
            )
        database = PostgresTestDatabase(url, target_id)
        yield database
    finally:
        _docker("rm", "--force", container_name, check=False)
