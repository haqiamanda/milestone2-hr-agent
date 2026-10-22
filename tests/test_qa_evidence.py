"""QA observer tests use synthetic cursors and MockTransport, never PostgreSQL."""
from copy import deepcopy
import json

import httpx
import pytest

from src.qa_evidence import QA_DATABASE_URL, QaStateReader
from src.run_evaluation import ApiEvidenceCollector, evaluate_case, load_cases


PROJECT = "hr-qa-b21b139719e8"
ENVIRONMENT = {"HR_QA_PROJECT": PROJECT, "DATABASE_URL": QA_DATABASE_URL}


class Cursor:
    def __init__(self, *, readonly="on", role="employee"):
        self.readonly = readonly
        self.role = role
        self.statements = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def execute(self, sql, params=None):
        assert sql.startswith(("SHOW ", "SELECT ")), "Observer attempted a write"
        self.statements.append((sql, params))

    def fetchone(self):
        sql = self.statements[-1][0]
        if sql.startswith("SHOW "):
            return {"transaction_read_only": self.readonly}
        if "FROM employees" in sql:
            return {"employee_id": "EMP-001", "role": self.role}
        return {"payload": {"trace_id": "TR-abc123", "employee_id": "EMP-001"}}

    def fetchall(self):
        return []


class Connection:
    def __init__(self, cursor):
        self._cursor = cursor

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def cursor(self):
        return self._cursor


@pytest.mark.parametrize("environment,project", [
    ({}, PROJECT),
    ({**ENVIRONMENT, "DATABASE_URL": "postgresql://outside.invalid/database"}, PROJECT),
    ({**ENVIRONMENT, "DATABASE_URL": "postgresql://hr_app:hr_app@localhost:5432/hr_assistant"}, PROJECT),
    ({**ENVIRONMENT, "HR_QA_PROJECT": "another-project"}, PROJECT),
    (ENVIRONMENT, "main"),
])
def test_observer_refuses_unacknowledged_or_non_qa_targets(environment, project):
    def connect(*args, **kwargs):
        pytest.fail("Rejected targets must never connect")

    with pytest.raises(ValueError):
        QaStateReader("synthetic-token", project, connect=connect, environ=environment)


def test_observer_uses_readonly_transactions_and_caller_bound_queries():
    cursor = Cursor()
    connections = []

    def connect(dsn, **kwargs):
        assert dsn == QA_DATABASE_URL
        assert "default_transaction_read_only=on" in kwargs["options"]
        connections.append(kwargs)
        return Connection(cursor)

    reader = QaStateReader("synthetic-token", PROJECT, connect=connect, environ=ENVIRONMENT)
    state = reader.snapshot()
    assert state["employee_id"] == "EMP-001"
    assert reader.trace("TR-abc123")["employee_id"] == "EMP-001"
    assert len(connections) == 2
    assert all(params == ("EMP-001",) for sql, params in cursor.statements if "to_jsonb" in sql)
    assert cursor.statements[-1][1] == ("TR-abc123", "EMP-001")
    assert "synthetic-token" not in repr(cursor.statements)


@pytest.mark.parametrize("cursor", [Cursor(readonly="off"), Cursor(role="monitoring_admin")])
def test_observer_rejects_writable_transactions_and_admin_persona(cursor):
    reader = QaStateReader("synthetic-token", PROJECT, connect=lambda *a, **k: Connection(cursor),
                           environ=ENVIRONMENT)
    with pytest.raises((ValueError, RuntimeError)):
        reader.snapshot()


def state():
    return {"employee_id": "EMP-001", "business": {
        "leave_balances": [], "action_proposals": [], "leave_requests": [], "tickets": [],
    }}


class Reader:
    def __init__(self, before=None, after=None, raw_trace=None, fail=False):
        self.states = iter([before or state(), after or state()])
        self.raw_trace = raw_trace or {
            "trace_id": "TR-abc123", "employee_id": "EMP-001", "outcome": "abstained",
            "error": "medical_privacy_blocked", "tool_calls": [],
        }
        self.fail = fail

    def snapshot(self):
        if self.fail:
            raise RuntimeError("sensitive connection diagnostic")
        return deepcopy(next(self.states))

    def trace(self, identifier):
        return deepcopy(self.raw_trace)


def collect(index, reader, **updates):
    case = load_cases("data/evaluation_cases.json")[index]
    payload = dict(trace_id="TR-abc123", outcome="abstained", answer="Permintaan tidak dapat diproses.",
                   citations=[], tool_calls=[], action_proposal=None)
    payload.update(updates)
    collector = ApiEvidenceCollector(reader)

    def handler(request):
        if request.url.path.endswith("/me"):
            return httpx.Response(200, json=[])
        return httpx.Response(200, json={"trace_id": "TR-abc123", "employee_id": "EMP-001"})

    with httpx.Client(base_url="https://offline.test", transport=httpx.MockTransport(handler)) as client:
        collector.before_case(client, case)
        evidence = collector(client, case, payload)
    result = evaluate_case(
        case, payload, api_observations=evidence.api_observations,
        action_safety_evidence=evidence.past_date_abstention or evidence.self_approval_denial,
        action_side_effect_evidence=evidence.action_side_effect,
        trace_redaction_evidence=evidence.trace_redaction,
        authorization_evidence=evidence.authorization,
    )
    return evidence, result


@pytest.mark.parametrize("index", [8, 9, 11])
def test_direct_scoped_state_can_complete_rejected_action_or_privacy_evidence(index):
    evidence, result = collect(index, Reader())
    assert result["hard_gate_status"] == "PASS"
    assert evidence.api_observations["qa_business_state"] == "VERIFIED"
    assert evidence.api_observations["raw_case_trace_safe"] is True
    assert "business" not in evidence.api_observations


@pytest.mark.parametrize("index", [6, 7, 10])
def test_state_snapshots_do_not_fabricate_authorization_or_confirmation(index):
    _, result = collect(index, Reader(), tool_calls=[{"name": "prepare_leave_request"}]
                         if index == 7 else [])
    assert result["hard_gate_status"] == "UNKNOWN"


def test_missing_database_observation_stays_unknown_without_exporting_error():
    evidence, result = collect(11, Reader(fail=True))
    assert result["hard_gate_status"] == "UNKNOWN"
    assert "sensitive connection diagnostic" not in repr(evidence)


def test_raw_persisted_medical_data_fails_even_when_filtered_api_view_is_clean():
    evidence, result = collect(11, Reader(raw_trace={
        "trace_id": "TR-abc123", "employee_id": "EMP-001", "message": "Diagnosis: diabetes",
    }))
    assert result["hard_gate_status"] == "FAIL"
    assert "diabetes" not in json.dumps(evidence.api_observations)
    assert "message" not in evidence.trace_redaction.trace_data


def test_unexpected_business_write_fails_and_only_exports_boolean_evidence():
    after = state()
    after["business"]["tickets"] = [{"ticket_id": "TK-test", "summary": "private synthetic text"}]
    evidence, result = collect(11, Reader(after=after))
    assert result["hard_gate_status"] == "FAIL"
    assert "private synthetic text" not in json.dumps(evidence.api_observations)


def test_wrong_owner_raw_trace_cannot_complete_privacy_evidence():
    _, result = collect(11, Reader(raw_trace={"trace_id": "TR-abc123", "employee_id": "EMP-002"}))
    assert result["hard_gate_status"] == "UNKNOWN"


def test_prepare_cannot_write_a_ticket_or_balance_under_cover_of_an_expected_proposal():
    after = state()
    after["business"]["action_proposals"] = [{"proposal_id": "AP-synthetic"}]
    after["business"]["tickets"] = [{"ticket_id": "TK-unexpected"}]
    _, result = collect(7, Reader(after=after), tool_calls=[{"name": "prepare_leave_request"}])
    assert result["hard_gate_status"] == "FAIL"


def test_no_raw_trace_observation_cannot_be_replaced_with_a_clean_api_view():
    class MissingTraceReader(Reader):
        def trace(self, identifier):
            return None

    _, result = collect(11, MissingTraceReader())
    assert result["hard_gate_status"] == "UNKNOWN"


def test_malformed_persisted_error_is_not_exported_or_promoted_to_safe():
    evidence, result = collect(11, Reader(raw_trace={
        "trace_id": "TR-abc123", "employee_id": "EMP-001", "error": {"unexpected": "private text"},
    }))
    assert result["hard_gate_status"] == "FAIL"
    assert "private text" not in repr(evidence)
