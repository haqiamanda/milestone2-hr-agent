"""Original policy/API/readiness regressions with memory and fake dependencies."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import date
from unittest.mock import Mock

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import src.api
from src.api import create_app
from src.config import Settings
from src.contracts import AuthContext, LeaveBalance, PrepareLeaveArgs
from src.domain import DomainError, validate_leave_request
from src.repository import ConflictError, MemoryRepository, PostgresRepository
from src.security import MedicalPrivacyError
from src.tools import ToolExecutor


@pytest.mark.parametrize("today,start,allowed", [
    ("2026-10-19", "2026-10-19", False),  # Same day.
    ("2026-10-19", "2026-10-20", False),  # One weekday.
    ("2026-10-19", "2026-10-21", False),  # Two weekdays.
    ("2026-10-19", "2026-10-22", True),   # Exactly three weekdays.
    ("2026-10-21", "2026-10-26", True),   # Wed -> Mon, excluding weekend.
    ("2026-10-23", "2026-10-27", False),  # Fri -> Tue = two weekdays.
    ("2026-10-23", "2026-10-28", True),   # Fri -> Wed = three weekdays.
    ("2026-10-24", "2026-10-28", False),  # Sat -> Wed = two weekdays.
    ("2026-10-24", "2026-10-29", True),   # Sat -> Thu = three weekdays.
])
def test_annual_notice_business_day_boundaries(today, start, allowed):
    start_date = date.fromisoformat(start)
    args = PrepareLeaveArgs(start_date=start_date, end_date=start_date, leave_type="annual")
    balances = [LeaveBalance(leave_type="annual", available_days=8, as_of_date=date(2026, 1, 1))]
    if allowed:
        assert validate_leave_request(args, balances, False, date.fromisoformat(today))[0] == 1
    else:
        with pytest.raises(DomainError, match="tiga hari kerja"):
            validate_leave_request(args, balances, False, date.fromisoformat(today))


@pytest.mark.parametrize("leave_type", ["sick", "parental"])
def test_annual_notice_does_not_apply_to_other_leave_types(leave_type):
    today = date(2026, 10, 19)
    args = PrepareLeaveArgs(start_date=today, end_date=today, leave_type=leave_type)
    balances = [LeaveBalance(leave_type=leave_type, available_days=10, as_of_date=today)]
    assert validate_leave_request(args, balances, False, today)[0] == 1


def test_notice_is_enforced_by_tool_before_proposal_persistence():
    repo = MemoryRepository()
    executor = ToolExecutor(repo, Mock(), Settings(_env_file=None, model_mode="fake"),
                            today_factory=lambda: date(2026, 10, 19))
    with pytest.raises(DomainError, match="tiga hari kerja"):
        executor.execute("prepare_leave_request", {
            "start_date": "2026-10-21", "end_date": "2026-10-21", "leave_type": "annual",
        }, AuthContext(employee_id="EMP-001", display_name="Alya"))
    assert not repo._proposals and not repo._requests
    assert repo.get_balances("EMP-001")[0].available_days == 8


def fake_client(*, repo=None, cache_ready=True, cache_error=False, config_ready=True):
    repo = repo if repo is not None else MemoryRepository()
    cache = Mock()
    cache.health.return_value = cache_ready
    if cache_error:
        cache.health.side_effect = RuntimeError("private-cache-url-canary")
    agent = Mock()
    settings = Settings(_env_file=None, model_mode="fake")
    if not config_ready:
        settings = settings.model_copy(update={"model_mode": "live", "openai_api_key": None})
    return TestClient(create_app(settings=settings, repository=repo, cache=cache, agent=agent),
                      raise_server_exceptions=False), agent


def assert_problem(response, expected_status):
    assert response.status_code == expected_status
    assert response.headers["content-type"].startswith("application/problem+json")
    body = response.json()
    assert set(body) == {"type", "title", "status", "detail"}
    assert body["type"] == "about:blank" and body["status"] == expected_status
    assert body["title"] and body["detail"]
    assert "canary" not in response.text and "Traceback" not in response.text


@pytest.mark.parametrize("exception,status", [
    (DomainError("private-diagnosis-canary"), 422),
    (ConflictError("private-db-url-canary"), 409),
    (MedicalPrivacyError(), 403),
    (HTTPException(403, "private-token-canary"), 403),
    (RuntimeError("private-password-canary"), 500),
])
def test_safe_consistent_domain_http_and_internal_problems(exception, status):
    client, agent = fake_client()
    agent.run.side_effect = exception
    result = client.post("/api/v1/chat", headers={"Authorization": "Bearer token-alya"},
                         json={"message": "Bantuan HR."})
    assert_problem(result, status)


def test_auth_and_authorization_problem_statuses_are_preserved():
    client, _ = fake_client()
    for headers in ({}, {"Authorization": "Bearer private-token-canary"}):
        result = client.post("/api/v1/chat", headers=headers, json={"message": "Bantuan HR."})
        assert_problem(result, 401)
        assert result.headers["www-authenticate"] == "Bearer"
    assert_problem(client.get("/api/v1/monitoring/summary",
                              headers={"Authorization": "Bearer token-alya"}), 403)
    assert_problem(client.get("/private-resource-canary"), 404)
    assert_problem(client.post("/healthz"), 405)


@pytest.mark.parametrize("body", [
    {}, {"message": {"private-token-canary": "private-diagnosis-canary"}},
    {"message": "private-diagnosis-canary" * 250},
])
def test_request_validation_never_echoes_rejected_input(body):
    client, _ = fake_client()
    result = client.post("/api/v1/chat", headers={"Authorization": "Bearer token-alya"}, json=body)
    assert_problem(result, 422)
    assert "input" not in result.json() and "errors" not in result.json()


def test_malformed_json_does_not_echo_body():
    client, _ = fake_client()
    result = client.post("/api/v1/chat", headers={
        "Authorization": "Bearer token-alya", "Content-Type": "application/json",
    }, content='{"message": "private-diagnosis-canary",')
    assert_problem(result, 422)


@pytest.mark.parametrize("operation,status", [("missing", 404), ("expired", 409)])
def test_confirmation_expected_errors_keep_safe_problem_contract(operation, status, monkeypatch):
    repo = MemoryRepository()
    client, _ = fake_client(repo=repo)
    if operation == "expired":
        monkeypatch.setattr(repo, "get_proposal", lambda proposal_id: ("EMP-001", object()))
        def fail(*args):
            raise ConflictError("private-proposal-canary")
        monkeypatch.setattr(repo, "confirm_proposal", fail)
    result = client.post("/api/v1/action-proposals/AP-private-canary/confirm",
                         headers={"Authorization": "Bearer token-alya"},
                         json={"idempotency_key": "test-idempotency-key"})
    assert_problem(result, status)


class ReadyRepository(MemoryRepository):
    def __init__(self, checks=None, unavailable=False):
        super().__init__()
        self.checks = checks if checks is not None else {"database": True, "schema": True, "index": True}
        self.unavailable = unavailable

    def readiness(self, index_version):
        assert index_version == "policies-2026-01"
        if self.unavailable:
            raise RuntimeError("private-db-url-canary")
        return self.checks


@pytest.mark.parametrize("cache_ready,cache_error", [(True, False), (False, False), (False, True)])
def test_readiness_reports_cache_degradation_without_blocking_core(cache_ready, cache_error):
    client, _ = fake_client(repo=ReadyRepository(), cache_ready=cache_ready, cache_error=cache_error)
    result = client.get("/readyz")
    assert result.status_code == 200 and result.json() == {"status": "ready"}
    assert result.headers["X-Cache-Status"] == ("ready" if cache_ready else "degraded")
    assert "canary" not in result.text


@pytest.mark.parametrize("failed", ["database", "schema", "index"])
def test_readiness_requires_every_essential_repository_check(failed):
    checks = {"database": True, "schema": True, "index": True}
    checks[failed] = False
    client, _ = fake_client(repo=ReadyRepository(checks))
    result = client.get("/readyz")
    assert_problem(result, 503)
    assert failed in result.json()["detail"]
    assert client.get("/healthz").json() == {"status": "ok"}


def test_readiness_checks_configuration_and_handles_repository_failure():
    client, _ = fake_client(repo=ReadyRepository(), config_ready=False)
    result = client.get("/readyz")
    assert_problem(result, 503)
    assert "model" in result.json()["detail"]
    client, _ = fake_client(repo=ReadyRepository(unavailable=True))
    assert_problem(client.get("/readyz"), 503)


def test_unseeded_memory_index_is_not_ready():
    client, _ = fake_client()
    result = client.get("/readyz")
    assert_problem(result, 503)
    assert "index" in result.json()["detail"]


def test_factory_keeps_health_and_readiness_available_without_model_configuration(monkeypatch):
    monkeypatch.setattr(src.api, "PostgresRepository", lambda **kwargs: ReadyRepository())
    monkeypatch.setattr(src.api, "ResilientCache", lambda url: Mock(health=lambda: True))
    settings = Settings(_env_file=None).model_copy(update={"model_mode": "live", "openai_api_key": None})
    client = TestClient(create_app(settings=settings))
    assert client.get("/healthz").json() == {"status": "ok"}
    assert_problem(client.get("/readyz"), 503)
    assert_problem(client.post("/api/v1/chat", headers={"Authorization": "Bearer token-alya"},
                              json={"message": "Bantuan HR."}), 503)


@contextmanager
def readiness_connection(queries, *, missing_table=None, vector=True, index=True):
    class Cursor:
        last = ""

        def execute(self, query, parameters=None):
            self.last = query
            queries.append((query, parameters))
            if missing_table and f"FROM {missing_table} LIMIT 0" in query:
                raise RuntimeError("private-schema-canary")

        def fetchone(self):
            if self.last == "SELECT 1":
                return {"?column?": 1}
            return {"available": vector if "pg_extension" in self.last else index}

    class Connection:
        @contextmanager
        def cursor(self):
            yield Cursor()

    yield Connection()


@pytest.mark.parametrize("missing_table", [None, "employees", "leave_balances", "policy_chunks",
                                          "action_proposals", "leave_requests", "tickets", "traces"])
def test_postgres_readiness_requires_schema_columns_without_mutations(missing_table, monkeypatch):
    repo = PostgresRepository("unused", 1536)
    queries = []
    monkeypatch.setattr(repo, "_connect", lambda: readiness_connection(queries, missing_table=missing_table))
    result = repo.readiness("test-version")
    assert result == {"database": True, "schema": missing_table is None, "index": missing_table is None}
    assert queries[0][0] == "SET TRANSACTION READ ONLY"
    assert all(query.lstrip().startswith("SELECT") or query == "SET TRANSACTION READ ONLY" for query, _ in queries)
    if missing_table is None:
        assert queries[-1][1] == ("test-version", 1536)
        assert "vector_dims(embedding)" in queries[-1][0]
        assert "i.indisvalid AND i.indisready" in queries[-1][0]


@pytest.mark.parametrize("vector,index", [(False, True), (True, False)])
def test_postgres_readiness_requires_extension_and_populated_current_index(vector, index, monkeypatch):
    repo = PostgresRepository("unused", 1536)
    monkeypatch.setattr(repo, "_connect", lambda: readiness_connection([], vector=vector, index=index))
    assert repo.readiness("test-version") == {"database": True, "schema": vector, "index": False}


def test_postgres_readiness_connection_failure_returns_only_safe_flags(monkeypatch):
    repo = PostgresRepository("unused", 1536)
    def fail():
        raise RuntimeError("private-db-url-canary")
    monkeypatch.setattr(repo, "_connect", fail)
    assert repo.readiness("test-version") == {"database": False, "schema": False, "index": False}
