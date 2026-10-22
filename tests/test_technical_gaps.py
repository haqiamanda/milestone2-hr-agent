"""Isolated cache, telemetry and monitoring regressions with synthetic metadata."""
from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import date
from unittest.mock import Mock

import pytest
from redis.exceptions import RedisError

from src.agent import AgentService
from src.cache import ResilientCache
from src.config import Settings
from src.contracts import AuthContext, ChatRequest, ModelToolCall, ModelTurn, PolicyChunk
from src.repository import MemoryRepository, PostgresRepository
from src.retrieval import RetrievalService
from src.tools import ToolExecutor


class FakeRedis:
    def __init__(self, unavailable=False):
        self.values = {}
        self.writes = []
        self.unavailable = unavailable

    def get(self, key):
        if self.unavailable:
            raise RedisError("synthetic cache unavailable")
        return self.values.get(key)

    def set(self, key, value, ex):
        if self.unavailable:
            raise RedisError("synthetic cache unavailable")
        self.writes.append((key, value, ex))
        self.values[key] = value


def chunk():
    return PolicyChunk(
        chunk_id="annual:abc", document_id="HR-LEAVE-001", title="Cuti tahunan",
        section="Hak", version="2.1", effective_date=date(2026, 1, 1),
        content="Policy content canary: 12 hari kerja.",
    )


def retrieval_dependencies(unavailable=False):
    cache = object.__new__(ResilientCache)
    cache.client = FakeRedis(unavailable)
    repo = Mock()
    repo.lexical_search.return_value = [chunk()]
    repo.vector_search.return_value = []
    provider = Mock()
    provider.embed.return_value = [[0.0] * 64]
    service = RetrievalService(repo, provider, cache, 60, policy_index_version="test-v1")
    return service, cache, repo, provider


def test_cache_miss_serializes_then_hit_reconstructs_without_search_or_embedding():
    service, cache, repo, provider = retrieval_dependencies()
    first = service.search("jatah cuti", 3)
    key, serialized, ttl = cache.client.writes[0]
    assert key == "retrieval:test-v1:jatah cuti:3" and ttl == 60
    stored = json.loads(serialized)
    assert stored == [chunk().model_dump(mode="json")]
    assert stored[0]["effective_date"] == "2026-01-01"
    second = service.search("jatah cuti", 3)
    assert first == second == [chunk()]
    assert isinstance(second[0].effective_date, date)
    assert repo.lexical_search.call_count == provider.embed.call_count == 1


@pytest.mark.parametrize("bad_value", ["not-json", "{}", '[{"content":"incomplete"}]'])
def test_invalid_cache_is_a_miss_and_replaced_with_valid_chunks(bad_value):
    service, cache, repo, _ = retrieval_dependencies()
    cache.client.values["retrieval:test-v1:jatah cuti:3"] = bad_value
    assert service.search("jatah cuti", 3) == [chunk()]
    repo.lexical_search.assert_called_once()
    assert json.loads(cache.client.writes[0][1]) == [chunk().model_dump(mode="json")]


def test_cache_version_change_does_not_reuse_old_results():
    service, cache, repo, _ = retrieval_dependencies()
    service.search("jatah cuti", 3)
    service.policy_index_version = "test-v2"
    service.search("jatah cuti", 3)
    assert repo.lexical_search.call_count == 2
    assert {key for key, _, _ in cache.client.writes} == {
        "retrieval:test-v1:jatah cuti:3", "retrieval:test-v2:jatah cuti:3",
    }


def test_redis_read_and_write_failure_preserves_uncached_retrieval(caplog):
    service, cache, repo, _ = retrieval_dependencies(unavailable=True)
    assert service.search("jatah cuti", 3) == [chunk()]
    assert service.search("jatah cuti", 3) == [chunk()]
    assert repo.lexical_search.call_count == 2
    assert cache.client.writes == []
    assert "jatah cuti" not in caplog.text


def test_cached_chunks_are_validated_deduplicated_and_limited():
    service, cache, repo, _ = retrieval_dependencies()
    cached = [chunk().model_dump(mode="json")] * 2
    cache.client.values["retrieval:test-v1:jatah cuti:1"] = json.dumps(cached)
    assert service.search("jatah cuti", 1) == [chunk()]
    repo.lexical_search.assert_not_called()


def test_agent_trace_preserves_retrieved_ids_without_prompt_or_document_content():
    repo = MemoryRepository()
    settings = Settings(_env_file=None, model_mode="fake")
    provider = Mock()
    provider.respond.side_effect = [
        ModelTurn(response_id="search", tool_calls=[ModelToolCall(
            call_id="policy", name="search_hr_policy", arguments={"query": "cuti", "top_k": 3}
        )]),
        ModelTurn(response_id="answer", text="Hak cuti tahunan [HR-LEAVE-001]."),
    ]
    retrieval = Mock()
    retrieval.search.return_value = [chunk(), chunk()]
    agent = AgentService(repo, provider, ToolExecutor(repo, retrieval, settings), settings)
    result = agent.run(ChatRequest(message="Apa kebijakan cuti? prompt-canary"),
                       AuthContext(employee_id="EMP-001", display_name="Alya"))
    trace = repo.get_trace(result.trace_id)
    assert trace["retrieved_document_ids"] == ["HR-LEAVE-001"]
    assert "prompt-canary" not in json.dumps(trace)
    assert "Policy content canary" not in json.dumps(trace)
    assert "arguments" not in json.dumps(trace)


@contextmanager
def fake_connection(captured, row=None):
    class Cursor:
        def execute(self, query, parameters=None):
            captured.append((query, parameters))

        def fetchone(self):
            return row

    class Connection:
        @contextmanager
        def cursor(self):
            yield Cursor()

    yield Connection()


@pytest.mark.parametrize("backend", ["memory", "postgres"])
def test_repository_keeps_only_valid_document_ids_and_safe_metadata(backend, monkeypatch):
    repo = MemoryRepository() if backend == "memory" else PostgresRepository("unused", 64)
    captured = []
    if backend == "postgres":
        monkeypatch.setattr(repo, "_connect", lambda: fake_connection(captured))
    repo.save_trace({
        "trace_id": "TR-abc123", "employee_id": "EMP-001", "outcome": "answered",
        "retrieved_document_ids": ["HR-LEAVE-001", "HR-PRIV-001", "HR-LEAVE-001",
                                   "employee@example.com", "token secret-canary", {"content": "private"}],
        "message": "private-prompt", "content": "private-document", "token": "secret-canary",
        "chain_of_thought": "private-thought", "employee_name": "private-name",
    })
    stored = repo.get_trace("TR-abc123") if backend == "memory" else json.loads(captured[0][1][3])
    assert stored["retrieved_document_ids"] == ["HR-LEAVE-001", "HR-PRIV-001"]
    assert set(stored) == {"trace_id", "employee_id", "outcome", "tool_calls", "retrieved_document_ids"}
    assert not any(word in json.dumps(stored) for word in ("private", "secret-canary", "@"))


def add_trace(repo, index, *, error=None, latency=100, outcome="answered", tool_error=False):
    trace = {
        "trace_id": f"TR-{index:x}", "employee_id": "EMP-001", "outcome": outcome,
        "latency_ms": latency, "tool_calls": [],
    }
    if error:
        trace["error"] = error
    if tool_error:
        trace["tool_calls"] = [{"name": "create_hr_ticket", "status": "error", "duration_ms": 1}]
    repo.save_trace(trace)


def test_monitoring_empty_and_normal_states_have_no_fabricated_alerts():
    repo = MemoryRepository()
    assert repo.monitoring_summary() == {
        "run_count": 0, "error_rate": 0.0, "p95_latency_ms": 0, "tool_calls": 0, "alerts": [],
    }
    for index in range(5):
        add_trace(repo, index)
    summary = repo.monitoring_summary()
    assert summary["error_rate"] == 0 and summary["p95_latency_ms"] == 100
    assert summary["alerts"] == []


def test_degraded_run_is_counted_but_alert_waits_for_minimum_samples():
    repo = MemoryRepository()
    add_trace(repo, 0, error="agent_processing_failed", outcome="abstained")
    summary = repo.monitoring_summary()
    assert summary["error_rate"] == 1 and summary["run_count"] == 1
    assert summary["alerts"] == []


def test_monitoring_counts_failures_once_and_excludes_expected_policy_denial():
    repo = MemoryRepository()
    add_trace(repo, 0, outcome="abstained")
    add_trace(repo, 1, error="medical_privacy_blocked", outcome="abstained", tool_error=True)
    add_trace(repo, 2, error="agent_processing_failed", outcome="abstained", tool_error=True)
    add_trace(repo, 3, tool_error=True)
    add_trace(repo, 4, error="balance_authorization_unavailable", outcome="abstained")
    summary = repo.monitoring_summary()
    assert summary["run_count"] == 5 and summary["error_rate"] == pytest.approx(3 / 5)
    assert summary["tool_calls"] == 3
    alert, = summary["alerts"]
    assert alert["rule_id"] == "agent-error-spike" and alert["value"] == pytest.approx(0.6)
    assert alert["threshold"] == 0.2 and alert["minimum_samples"] == alert["sample_count"] == 5
    assert alert["owner"] == "AI Platform" and "provider health" in alert["response"]


def test_threshold_equality_does_not_trigger_gt_rules():
    repo = MemoryRepository()
    for index in range(5):
        add_trace(repo, index, error="agent_processing_failed" if index == 0 else None, latency=8000)
    summary = repo.monitoring_summary()
    assert summary["error_rate"] == 0.2 and summary["p95_latency_ms"] == 8000
    assert summary["alerts"] == []


def test_p95_uses_measured_latency_and_triggers_existing_latency_rule():
    repo = MemoryRepository()
    for index, latency in enumerate([100, 200, 300, 400, 10000]):
        add_trace(repo, index, latency=latency)
    summary = repo.monitoring_summary()
    assert summary["p95_latency_ms"] == 8080
    alert, = summary["alerts"]
    assert alert["rule_id"] == "agent-latency-p95" and alert["value"] == 8080
    assert alert["owner"] == "AI Platform" and "retrieval cache" in alert["response"]


@pytest.mark.parametrize("row,expected_alerts", [
    ({"run_count": 0, "error_rate": 0, "p95_latency_ms": 0, "tool_calls": 0}, []),
    ({"run_count": 2, "error_rate": 1, "p95_latency_ms": 9000, "tool_calls": 2}, []),
    ({"run_count": 5, "error_rate": 0.4, "p95_latency_ms": 9000, "tool_calls": 2},
     ["agent-error-spike", "agent-latency-p95"]),
])
def test_postgres_monitoring_uses_aggregate_error_signals_and_existing_rules(row, expected_alerts, monkeypatch):
    repo = PostgresRepository("unused", 64)
    captured = []
    monkeypatch.setattr(repo, "_connect", lambda: fake_connection(captured, row))
    result = repo.monitoring_summary()
    assert [alert["rule_id"] for alert in result["alerts"]] == expected_alerts
    assert result["error_rate"] == row["error_rate"]
    query = captured[0][0]
    assert "payload->>'error'" in query and "tool_call->>'status' = 'error'" in query
    assert "medical_privacy_blocked' THEN 0.0" in query
    assert "percentile_cont(0.95)" in query
