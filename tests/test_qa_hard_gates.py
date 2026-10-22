"""Real API/service regressions, fake provider and memory storage only."""
from copy import deepcopy
from datetime import date
import json

from fastapi.testclient import TestClient
import pytest

from src.agent import AgentService
from src.api import create_app
from src.cache import NullCache
from src.config import Settings
from src.contracts import ModelToolCall, ModelTurn
from src.qa_hard_gates import (
    Verifier, all_checks, instrument, verdict, reconcile_tools, tool_metadata, interrupted_result,
)
from src.repository import MemoryRepository
from src.run_evaluation import load_cases
from src.tools import ToolExecutor


class Provider:
    def __init__(self, call=None, answer="Permintaan tidak dapat diproses."):
        self.call, self.answer = call, answer

    def respond(self, **kwargs):
        call, self.call = self.call, None
        return ModelTurn(response_id="unit", text=self.answer,
                         tool_calls=[call] if call else [])


class NoRetrieval:
    def search(self, *args):
        raise AssertionError("No retrieval authorized by these security probes")


class Reader:
    def __init__(self, repository):
        self.repository = repository

    def business_snapshot(self):
        repo = self.repository
        proposals = []
        for identifier, (owner, proposal) in sorted(repo._proposals.items()):
            row = dict(proposal.model_dump(mode="json"), employee_id=owner)
            if identifier in repo._confirmed_proposals:
                key, result = repo._confirmed_proposals[identifier]
                row.update(confirmed_at="synthetic-time", confirmed_request_id=result.request_id,
                           confirmed_idempotency_key=key, confirmed_audit_reference=result.audit_reference)
            proposals.append(row)
        return deepcopy({
            "leave_balances": [{"employee_id": owner, **b.model_dump(mode="json")}
                               for owner, balances in sorted(repo._balances.items()) for b in balances],
            "action_proposals": proposals,
            "leave_requests": [{"employee_id": owner, **r.model_dump(mode="json")}
                               for owner, rows in sorted(repo._requests.items()) for r in rows],
            "tickets": list(repo._tickets.values()),
        })

    def trace(self, identifier):
        return self.repository.get_trace(identifier)


@pytest.fixture
def runtime():
    repo = MemoryRepository()
    settings = Settings(_env_file=None, database_url="unused", redis_url="unused", model_mode="fake")
    tools = ToolExecutor(repo, NoRetrieval(), settings, today_factory=lambda: date(2026, 10, 22))
    agent = AgentService(repo, Provider(), tools, settings)
    observed, tool_observer = instrument(repo, agent)
    app = create_app(settings=settings, repository=observed, cache=NullCache(), agent=agent)
    with TestClient(app, headers={"Authorization": "Bearer token-alya"}) as client:
        verifier = Verifier(client, Reader(repo), observed, tool_observer,
                            "EMP-001", "EMP-002", ["token-alya", "token-bima", "unit-secret-key"])
        yield verifier, repo, agent


def case(index):
    return load_cases("data/evaluation_cases.json")[index]


def test_foreign_balance_and_request_binding_has_observed_api_and_storage_evidence(runtime):
    verifier, repo, _ = runtime
    response, auth, checks = verifier.security(case(6), case(6)["input"])
    result = verdict(case(6), response, checks, authorization=auth)
    assert result["hard_gate_status"] == "PASS"
    assert all_checks(checks) is True
    assert not any(e["operation"] == "get_balances" for e in verifier.repository_calls.events)
    assert "EMP-002" not in json.dumps(result) and "token-alya" not in json.dumps(result)


def test_refusal_without_persisted_trace_remains_unknown(runtime, monkeypatch):
    verifier, _, _ = runtime
    monkeypatch.setattr(verifier.reader, "trace", lambda identifier: None)
    response, auth, checks = verifier.security(case(6), case(6)["input"])
    assert verdict(case(6), response, checks, authorization=auth)["hard_gate_status"] == "UNKNOWN"


@pytest.mark.parametrize("answer", ["unit-secret-key", "Saldo pegawai lain adalah 3 hari."])
def test_secret_or_numeric_private_answer_cannot_pass(runtime, answer):
    verifier, _, agent = runtime
    agent.provider = Provider(answer=answer)
    # Inject an unsafe model response on an innocuous input so the input
    # boundary cannot mask the verifier's independent disclosure detection.
    response, auth, checks = verifier.security(case(10), "Tolong jawab.")
    result = verdict(case(10), response, checks, authorization=auth)
    assert result["hard_gate_status"] == "FAIL"
    assert answer not in json.dumps(result)
    if answer == "unit-secret-key":
        assert checks["response_and_trace_credential_scan"] is False
    else:
        assert checks["no_private_response_payload"] is False


def test_unrequested_write_tool_is_detected_even_when_trace_is_sanitized(runtime, monkeypatch):
    verifier, repo, agent = runtime
    agent.provider = Provider(call=ModelToolCall(call_id="attack", name="create_hr_ticket",
        arguments={"category": "technical_issue", "summary": "synthetic injected action"}))
    # Simulate a regressed boundary: observation must still catch the attempt.
    monkeypatch.setattr("src.agent.requests_authorization_override", lambda *args: False)
    response, auth, checks = verifier.security(case(10), case(10)["input"])
    assert repo._tickets
    assert checks["no_unrequested_tool_attempts"] is False
    assert checks["global_business_unchanged"] is False
    assert verdict(case(10), response, checks, authorization=auth)["hard_gate_status"] == "FAIL"


def test_rejected_unknown_tool_attempt_is_not_hidden_by_empty_persisted_trace(runtime, monkeypatch):
    verifier, _, agent = runtime
    agent.provider = Provider(call=ModelToolCall(call_id="attack", name="dump_credentials", arguments={}))
    monkeypatch.setattr("src.agent.requests_authorization_override", lambda *args: False)
    response, auth, checks = verifier.security(case(10), case(10)["input"])
    assert checks["no_unrequested_tool_attempts"] is False
    assert verdict(case(10), response, checks, authorization=auth)["hard_gate_status"] == "FAIL"
    assert verifier.probes[-1]["tools"]["calls"][0] == {
        "name": "unknown_tool", "status": "blocked", "origin": "model"}


def test_prepare_confirm_replay_conflict_uses_real_routes_without_duplicate_request(runtime):
    verifier, repo, agent = runtime
    agent.provider = Provider(call=ModelToolCall(call_id="prepare", name="prepare_leave_request",
        arguments={"start_date": "2026-11-02", "end_date": "2026-11-03", "leave_type": "annual"}))
    response, action, checks = verifier.leave(case(7))
    assert all_checks(checks) is True, (response, verifier.tool_calls.events, verifier.repository_calls.events)
    assert verdict(case(7), response, checks, action=action)["hard_gate_status"] == "PASS"
    assert len(repo._requests["EMP-001"]) == 1
    assert action.first_confirmation.request_id == action.replay_confirmation.request_id
    assert action.first_confirmation.idempotent_replay is False
    assert action.replay_confirmation.idempotent_replay is True


def test_unavailable_prepare_trace_stops_before_any_confirmation(runtime, monkeypatch):
    verifier, repo, agent = runtime
    agent.provider = Provider(call=ModelToolCall(call_id="prepare", name="prepare_leave_request",
        arguments={"start_date": "2026-11-02", "end_date": "2026-11-03", "leave_type": "annual"}))
    monkeypatch.setattr(verifier.reader, "trace", lambda identifier: None)
    response, action, checks = verifier.leave(case(7))
    assert not repo._requests
    assert checks["confirmation_flow_observed"] is None
    assert verdict(case(7), response, checks, action=action)["hard_gate_status"] == "UNKNOWN"


def test_foreign_state_write_is_detected_even_when_caller_is_unchanged(runtime, monkeypatch):
    verifier, repo, agent = runtime
    original = agent.run

    def attacked(*args, **kwargs):
        response = original(*args, **kwargs)
        repo._balances["EMP-002"][0].available_days -= 1
        return response

    monkeypatch.setattr(agent, "run", attacked)
    response, auth, checks = verifier.security(case(6), case(6)["input"])
    assert checks["global_business_unchanged"] is False
    assert verdict(case(6), response, checks, authorization=auth)["hard_gate_status"] == "FAIL"


def test_unknown_extra_check_cannot_be_overridden_by_other_passing_checks(runtime):
    verifier, _, _ = runtime
    response, auth, checks = verifier.security(case(6), case(6)["input"])
    checks["missing_observation"] = None
    assert verdict(case(6), response, checks, authorization=auth)["hard_gate_status"] == "UNKNOWN"


def test_trace_owner_switch_cannot_pass(runtime, monkeypatch):
    verifier, _, _ = runtime
    monkeypatch.setattr(verifier.reader, "trace", lambda identifier: {
        "trace_id": identifier, "employee_id": "EMP-002", "tool_calls": []})
    response, auth, checks = verifier.security(case(10), case(10)["input"])
    assert verdict(case(10), response, checks, authorization=auth)["hard_gate_status"] == "FAIL"


def test_conflict_response_with_persistent_write_fails(runtime, monkeypatch):
    verifier, repo, agent = runtime
    agent.provider = Provider(call=ModelToolCall(call_id="prepare", name="prepare_leave_request",
        arguments={"start_date": "2026-11-02", "end_date": "2026-11-03", "leave_type": "annual"}))
    original = repo.confirm_proposal

    def bad_conflict(*args):
        try:
            return original(*args)
        except Exception:
            repo._balances["EMP-002"][0].available_days -= 1
            raise

    monkeypatch.setattr(repo, "confirm_proposal", bad_conflict)
    response, action, checks = verifier.leave(case(7))
    assert action.different_key_rejected is True
    assert checks["different_key_conflict_no_write"] is False
    assert verdict(case(7), response, checks, action=action)["hard_gate_status"] == "FAIL"


def test_global_snapshot_is_fixed_readonly_business_sql():
    from src.qa_evidence import QaStateReader
    from tests.test_qa_evidence import Connection, Cursor, ENVIRONMENT, PROJECT
    cursor = Cursor()
    connections = []

    def connect(dsn, **kwargs):
        connections.append(kwargs)
        return Connection(cursor)

    reader = QaStateReader("unit-only", PROJECT, connect=connect, environ=ENVIRONMENT)
    snapshot = reader.business_snapshot()
    assert set(snapshot) == {"leave_balances", "action_proposals", "leave_requests", "tickets"}
    assert "default_transaction_read_only=on" in connections[0]["options"]
    assert len(cursor.statements) == 5
    assert all(sql.startswith(("SELECT ", "SHOW ")) for sql, _ in cursor.statements)
    assert "employees" not in repr(cursor.statements) and "token" not in repr(cursor.statements)


def test_cli_requires_explicit_live_and_confirmation_authorization(monkeypatch):
    from src.qa_hard_gates import main
    monkeypatch.setattr("sys.argv", ["qa_hard_gates", "--qa-project", "hr-qa-b21b139719e8",
                                     "--output", "unused.json"])
    with pytest.raises(SystemExit) as missing_flags:
        main()
    assert missing_flags.value.code == 2


def record(name="search_hr_policy", status="ok"):
    return {"name": name, "status": status, "duration_ms": 0}


@pytest.mark.parametrize("reported", [[], [record(status="error")],
    [record("get_my_leave_balance")], [record(), record()]])
def test_reconciliation_detects_real_count_name_and_status_mismatches(reported):
    events = [{"operation": "search_hr_policy", "status": "executed"}]
    assert reconcile_tools(events, reported) is False


@pytest.mark.parametrize("status,reported", [
    ("executed", "ok"), ("failed", "error"), ("blocked", "error"), ("blocked", "blocked"),
])
def test_reconciliation_accepts_nonempty_consistent_observations(status, reported):
    assert reconcile_tools([{"operation": "search_hr_policy", "status": status}],
                           [record(status=reported)]) is True


@pytest.mark.parametrize("events,reported", [
    (None, []), ([], None), ([], [{}]),
    ([{"operation": "search_hr_policy", "status": "attempted"}], []),
])
def test_missing_or_incomplete_instrumentation_remains_unknown(events, reported):
    assert reconcile_tools(events, reported) is None


def test_authorized_policy_fallback_matches_response_and_trace_without_empty_list_assumption():
    from tests.test_prompt_injection_boundary import runtime as policy_runtime
    from src.contracts import AuthContext, ChatRequest
    service, repo, _, _ = policy_runtime(
        "Kebijakan keamanan password dan token: jangan bagikan kredensial. "
        "Laporkan insiden privasi melalui HR. [HR-PRIV-001]")
    _, calls = instrument(repo, service)
    response = service.run(ChatRequest(message="Jelaskan kebijakan keamanan password dan token."),
                           AuthContext(employee_id="EMP-001", display_name="Alya"))
    assert response.outcome == "answered"
    assert reconcile_tools(calls.events, response.model_dump()["tool_calls"]) is True
    assert reconcile_tools(calls.events, repo.get_trace(response.trace_id)["tool_calls"]) is True
    assert tool_metadata(calls.events)["calls"] == [{
        "name": "search_hr_policy", "status": "executed", "origin": "policy_fallback"}]


def test_security_probe_records_no_tools_after_boundary_and_preserves_strict_requirement(runtime):
    verifier, _, _ = runtime
    response, auth, checks = verifier.security(case(10), case(10)["input"])
    assert verdict(case(10), response, checks, authorization=auth)["hard_gate_status"] == "PASS"
    assert checks["trace_calls_match_observer"] is True
    assert checks["response_calls_match_observer"] is True
    assert checks["no_unrequested_tool_attempts"] is True
    assert verifier.probes == [{"probe": "original", "tools": {
        "attempted_count": 0, "status_counts": {"attempted": 0, "blocked": 0, "executed": 0, "failed": 0},
        "calls": []}}]


def test_correctly_logged_unauthorized_tool_still_fails(runtime, monkeypatch):
    verifier, _, agent = runtime
    monkeypatch.setattr("src.agent.requests_authorization_override", lambda *args: False)
    agent.provider = Provider(call=ModelToolCall(call_id="unsafe", name="create_hr_ticket",
        arguments={"category": "technical_issue", "summary": "synthetic unrequested write"}))
    response, auth, checks = verifier.security(case(10), case(10)["input"])
    assert checks["response_calls_match_observer"] is True
    assert checks["trace_calls_match_observer"] is True
    assert checks["no_unrequested_tool_attempts"] is False
    assert verdict(case(10), response, checks, authorization=auth)["hard_gate_status"] == "FAIL"


def test_failed_tool_records_origin_without_exception_or_arguments(runtime, monkeypatch):
    verifier, _, agent = runtime
    monkeypatch.setattr("src.agent.requests_authorization_override", lambda *args: False)
    def unavailable(*args, **kwargs):
        raise RuntimeError("unit-secret-key private diagnostic")
    monkeypatch.setattr(agent.tools.delegate.retrieval, "search", unavailable)
    response, _, checks = verifier.security(case(10), case(10)["input"])
    assert checks["response_calls_match_observer"] is True
    assert checks["trace_calls_match_observer"] is True
    metadata = verifier.probes[-1]["tools"]
    assert metadata["status_counts"]["failed"] == 1
    assert metadata["calls"] == [{"name": "search_hr_policy", "status": "failed", "origin": "policy_fallback"}]
    assert "unit-secret-key" not in json.dumps(metadata)
    assert "arguments" not in json.dumps(metadata)


def test_unknown_tool_name_and_unknown_origin_cannot_export_untrusted_strings():
    metadata = tool_metadata([{"operation": "unit-secret-key", "status": "blocked",
                               "origin": "private text", "owner": "EMP-002"}])
    assert metadata["calls"] == [{"name": "unknown_tool", "status": "blocked", "origin": "unknown"}]
    assert "unit-secret-key" not in json.dumps(metadata) and "EMP-002" not in json.dumps(metadata)


def test_interrupted_observation_cannot_erase_known_unauthorized_attempt():
    events = [{"operation": "create_hr_ticket", "status": "failed", "origin": "model"}]
    assert interrupted_result(case(10), events)["hard_gate_status"] == "FAIL"
    assert interrupted_result(case(10), [])["hard_gate_status"] == "UNKNOWN"
    assert interrupted_result(case(7), events)["hard_gate_status"] == "UNKNOWN"


def test_origin_context_resets_after_failed_execution(runtime, monkeypatch):
    from src.tools import TOOL_EXECUTION_ORIGIN
    verifier, _, _ = runtime
    monkeypatch.setattr("src.agent.requests_authorization_override", lambda *args: False)
    verifier.security(case(10), case(10)["input"])
    assert verifier.tool_calls.events[-1]["origin"] == "policy_fallback"
    assert TOOL_EXECUTION_ORIGIN.get() == "unknown"
