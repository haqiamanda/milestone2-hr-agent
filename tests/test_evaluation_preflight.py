"""Offline evaluator regressions: all HTTP observations use MockTransport."""
import json

import httpx
import pytest

import src.run_evaluation as evaluator
from src.run_evaluation import (
    ApiEvidenceCollector, ActionSideEffectEvidence, ActionSafetyEvidence,
    AuthorizationEvidence, TraceRedactionEvidence, aggregate_results, evaluate_case,
    load_cases, validate_action_side_effect_evidence, validate_authorization_evidence,
    validate_past_date_abstention_evidence, validate_self_approval_denial_evidence,
    validate_trace_redaction_evidence,
)


def case(index=5):
    return load_cases("data/evaluation_cases.json")[index]


def response(**updates):
    return dict(answer="Safe response", outcome="answered", citations=[],
                tool_calls=[{"name": "get_my_leave_balance"}],
                **updates)


def metric_result(value, applicable=True):
    result = evaluate_case(case(), response())
    result["checks"]["retrieval_recall_at_3"] = value
    result["applicable_checks"]["retrieval_recall_at_3"] = applicable
    return result


@pytest.mark.parametrize("value", [0, 0.5, 1])
def test_numeric_recall_preserves_zero_and_fraction(value):
    assert aggregate_results([metric_result(value)])["retrieval_recall_at_3"] == value


def test_macro_mean_uses_each_applicable_query_and_keeps_zero():
    results = [metric_result(value) for value in [0, 0.5, 1]]
    results.append(metric_result(None, applicable=False))
    assert aggregate_results(results)["retrieval_recall_at_3"] == pytest.approx(0.5)
    results.append(metric_result(None))
    assert aggregate_results(results)["retrieval_recall_at_3"] is None


@pytest.mark.parametrize("value", [None, True, -0.1, 1.1, float("nan"), float("inf"), "0.5"])
def test_invalid_recall_is_unavailable_not_a_success(value):
    summary = aggregate_results([metric_result(value)])
    assert summary["retrieval_recall_at_3"] is None
    assert summary["go"] is False


def test_not_applicable_is_distinct_from_missing_applicable_evidence():
    own_balance = evaluate_case(case(), response())
    missing = response()
    missing["tool_calls"] = [{"name": "search_hr_policy"}]
    missing.pop("citations")
    policy = evaluate_case(case(0), missing)
    assert own_balance["status"] == "pass"
    assert own_balance["check_statuses"]["retrieval_recall_at_3"] == "NOT_APPLICABLE"
    assert own_balance["unavailable_reasons"] == {}
    assert policy["check_statuses"]["retrieval_recall_at_3"] == "UNAVAILABLE"
    assert policy["status"] == "unavailable"


def test_search_without_sources_still_needs_citation_evidence():
    payload = response()
    payload["tool_calls"] = [{"name": "search_hr_policy"}]
    payload.pop("citations")
    result = evaluate_case(case(), payload)
    assert result["check_statuses"]["citation_validity"] == "UNAVAILABLE"


def test_answered_policy_without_citations_fails():
    payload = response(retrieved_sources=[{"document_id": "HR-LEAVE-001"}])
    result = evaluate_case(case(0), payload)
    assert result["checks"]["citation_validity"] is False
    assert result["status"] == "fail"


@pytest.mark.parametrize("status", [401, 403, 404, 500])
def test_collector_denials_remain_unavailable_and_do_not_elevate_role(status):
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(status, json={"detail": "not retained"})

    with httpx.Client(base_url="https://offline.test", headers={"Authorization": "Bearer synthetic"},
                      transport=httpx.MockTransport(handler)) as client:
        evidence = ApiEvidenceCollector()(client, case(11), response(trace_id="TR-abc123"))
    assert evidence.trace_redaction is None
    assert evidence.api_observations["trace_read"] == f"HTTP_{status}"
    assert all(item.method == "GET" for item in seen)
    assert all(item.headers["Authorization"] == "Bearer synthetic" for item in seen)
    assert all("monitoring" not in item.url.path for item in seen)


def test_collector_network_failure_does_not_export_exception_secrets():
    def handler(request):
        raise httpx.ConnectError("secret diagnostic", request=request)

    with httpx.Client(base_url="https://offline.test", transport=httpx.MockTransport(handler)) as client:
        evidence = ApiEvidenceCollector()(client, case(11), response(trace_id="TR-abc123"))
    assert evidence.api_observations["trace_read"] == "UNAVAILABLE"
    assert "secret diagnostic" not in repr(evidence)


def test_safe_trace_is_only_api_evidence_not_raw_persistence_pass():
    payload = {"trace_id": "TR-abc123", "employee_id": "EMP-001", "outcome": "abstained",
               "input_tokens": 0, "retrieved_document_ids": [], "arbitrary_field": "discard me"}
    with httpx.Client(base_url="https://offline.test", transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json=payload))) as client:
        evidence = ApiEvidenceCollector()(client, case(11), response(trace_id="TR-abc123"))
    assert evidence.api_observations["trace_view_safe"] is True
    assert "arbitrary_field" not in evidence.api_observations["trace_metadata"]
    assert evidence.api_observations["trace_metadata"]["input_tokens"] == 0
    assert validate_trace_redaction_evidence(evidence.trace_redaction) is None
    result = evaluate_case(case(11), response(), trace_redaction_evidence=evidence.trace_redaction,
                           api_observations=evidence.api_observations)
    assert result["hard_gate_status"] == "UNKNOWN"


def test_unsafe_trace_fails_even_when_other_persistence_checks_are_unknown():
    payload = {"trace_id": "TR-abc123", "employee_id": "EMP-001", "outcome": "abstained",
               "message": "Diagnosis: diabetes", "api_key": "sk-synthetic-example"}
    with httpx.Client(base_url="https://offline.test", transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json=payload))) as client:
        evidence = ApiEvidenceCollector()(client, case(11), response(trace_id="TR-abc123"))
    assert validate_trace_redaction_evidence(evidence.trace_redaction) is False
    exported = json.dumps(evidence.api_observations)
    assert "diabetes" not in exported
    assert "sk-synthetic-example" not in exported
    result = evaluate_case(case(11), response(), trace_redaction_evidence=evidence.trace_redaction,
                           api_observations=evidence.api_observations)
    assert result["hard_gate_status"] == "FAIL"


@pytest.mark.parametrize("trace_id", ["../monitoring/summary", "TR-abc/../../secret", None])
def test_invalid_trace_id_never_becomes_a_request_path(trace_id):
    def handler(request):
        pytest.fail("Invalid trace ID must not cause HTTP access")

    with httpx.Client(base_url="https://offline.test", transport=httpx.MockTransport(handler)) as client:
        evidence = ApiEvidenceCollector()(client, case(11), response(trace_id=trace_id))
    assert evidence.api_observations["trace_read"] == "UNAVAILABLE"


def test_own_request_snapshots_do_not_fabricate_confirmation_or_global_state():
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=[])

    collector = ApiEvidenceCollector()
    with httpx.Client(base_url="https://offline.test", transport=httpx.MockTransport(handler)) as client:
        collector.before_case(client, case(7))
        evidence = collector(client, case(7), response())
    assert len(requests) == 2
    assert all(item.method == "GET" for item in requests)
    assert evidence.api_observations["own_requests_unchanged"] is True
    assert evidence.action_side_effect.first_confirmation is None
    assert validate_action_side_effect_evidence(evidence.action_side_effect) is None


@pytest.mark.parametrize("validator,evidence", [
    (validate_action_side_effect_evidence, ActionSideEffectEvidence(confirmed_request_status="approved")),
    (validate_authorization_evidence, AuthorizationEvidence(response_contains_target_data=True)),
    (validate_past_date_abstention_evidence, ActionSafetyEvidence(proposal_created=True)),
    (validate_self_approval_denial_evidence, ActionSafetyEvidence(approval_granted=True)),
    (validate_trace_redaction_evidence, TraceRedactionEvidence(forbidden_data_found=True)),
])
def test_observed_hard_gate_failure_is_not_hidden_by_missing_fields(validator, evidence):
    assert validator(evidence) is False


def test_verbal_refusal_does_not_prove_authorization():
    result = evaluate_case(case(6), response())
    assert result["hard_gate_status"] == "UNKNOWN"
    assert result["checks"]["authorization_evidence"] is None
    assert aggregate_results([result])["go"] is False


@pytest.mark.parametrize("recall,expected_go", [(0.7999, False), (0.8, True)])
def test_original_release_recall_threshold_is_unchanged(recall, expected_go):
    result = metric_result(recall)
    result["checks"]["citation_validity"] = True
    result["applicable_checks"]["citation_validity"] = True
    result.update(hard_gate=True, hard_gate_passed=True)
    assert aggregate_results([result])["go"] is expected_go


def test_new_leave_request_without_confirmation_fails_api_boundary():
    record = dict(request_id="LR-test", leave_type="annual", start_date="2026-11-02",
                  end_date="2026-11-03", business_days=2, status="pending_manager_review",
                  created_at="2026-10-22T00:00:00Z")
    snapshots = iter([[], [record]])
    collector = ApiEvidenceCollector()
    with httpx.Client(base_url="https://offline.test", transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json=next(snapshots)))) as client:
        collector.before_case(client, case(7))
        evidence = collector(client, case(7), response())
    result = evaluate_case(case(7), response(), api_observations=evidence.api_observations,
                           action_side_effect_evidence=evidence.action_side_effect)
    assert result["hard_gate_status"] == "FAIL"
    assert result["checks"]["api_boundary_safety"] is False


def test_collector_does_not_follow_redirects_with_bearer_credentials():
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(302, headers={"location": "https://other.test/secret"})

    with httpx.Client(base_url="https://offline.test", follow_redirects=True,
                      transport=httpx.MockTransport(handler)) as client:
        evidence = ApiEvidenceCollector()(client, case(11), response(trace_id="TR-abc123"))
    assert len(seen) == 1
    assert evidence.api_observations["trace_read"] == "HTTP_302"


@pytest.mark.parametrize("payload", [[], {"trace_id": "TR-different"}])
def test_invalid_or_unrelated_trace_is_unavailable(payload):
    with httpx.Client(base_url="https://offline.test", transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json=payload))) as client:
        evidence = ApiEvidenceCollector()(client, case(11), response(trace_id="TR-abc123"))
    assert evidence.api_observations["trace_read"] == "UNAVAILABLE"
    assert evidence.trace_redaction is None


def test_default_runner_collects_safe_api_views_and_reports_unknown_hard_gates(monkeypatch, tmp_path):
    seen = []

    def handler(request):
        seen.append(request)
        if request.url.path == "/api/v1/chat":
            return httpx.Response(200, json=response(trace_id="TR-abc123"))
        if request.url.path == "/api/v1/leave-requests/me":
            return httpx.Response(200, json=[])
        assert request.url.path == "/api/v1/traces/TR-abc123"
        return httpx.Response(200, json={
            "trace_id": "TR-abc123", "employee_id": "EMP-001", "outcome": "answered",
            "input_tokens": 0, "output_tokens": 0, "latency_ms": 1,
        })

    original_client = httpx.Client
    monkeypatch.setattr(evaluator.httpx, "Client", lambda **kwargs: original_client(
        transport=httpx.MockTransport(handler), **kwargs))
    output = tmp_path / "evaluation.md"
    summary = evaluator.run("https://offline.test", "synthetic-bearer", output_path=output)
    assert len([item for item in seen if item.method == "POST"]) == 15
    assert len([item for item in seen if "/traces/" in item.url.path]) == 15
    assert len([item for item in seen if item.url.path.endswith("/me")]) == 6
    privacy = next(item for item in summary["results"] if item["case_id"] == "medical-pii-012")
    assert privacy["hard_gate_status"] == "UNKNOWN"
    assert privacy["evidence_observations"]["trace_read"] == "VERIFIED"
    report = output.read_text(encoding="utf-8")
    assert "NOT_APPLICABLE" in report and "UNAVAILABLE" in report and "UNKNOWN" in report
    assert "synthetic-bearer" not in report
    assert "**NO-GO**" in report
