from __future__ import annotations

from datetime import date

from src.agent import AgentService
from src.cache import NullCache
from src.config import Settings
from src.contracts import AuthContext, ChatRequest, ModelToolCall, ModelTurn
from src.repository import MemoryRepository
from src.retrieval import RetrievalService
from src.setup_lab import ROOT, chunk_policy
from src.tools import ToolExecutor


class FailingProvider:
    def embed(self, texts):
        return [[0.0] * 8 for _ in texts]

    def respond(self, **kwargs):
        raise TimeoutError("provider timeout for secret-token_ABCDEF123")


class BadCitationProvider:
    def __init__(self) -> None:
        self.turn = 0

    def embed(self, texts):
        return [[0.0] * 8 for _ in texts]

    def respond(self, **kwargs):
        self.turn += 1
        if self.turn == 1:
            return ModelTurn(
                response_id="one",
                tool_calls=[
                    ModelToolCall(
                        call_id="search", name="search_hr_policy", arguments={"query": "cuti tahunan", "top_k": 3}
                    )
                ],
            )
        return ModelTurn(response_id="two", text="Jawaban tanpa bukti [HR-FAKE-999].")


def service(repository: MemoryRepository, provider) -> AgentService:
    settings = Settings(model_mode="fake")
    retrieval = RetrievalService(repository, provider, NullCache(), 0)
    tools = ToolExecutor(repository, retrieval, settings, today_factory=lambda: date(2026, 9, 23))
    return AgentService(repository, provider, tools, settings)


def test_provider_failure_is_side_effect_free_and_redacted(repository: MemoryRepository) -> None:
    response = service(repository, FailingProvider()).run(
        ChatRequest(message="Tolong jawab"),
        AuthContext(employee_id="EMP-001", display_name="Alya"),
    )
    assert response.outcome == "abstained"
    assert not repository.list_leave_requests("EMP-001")
    trace = repository.get_trace(response.trace_id)
    assert "ABCDEF123" not in trace["error"]


def test_invalid_citation_forces_abstention(repository: MemoryRepository) -> None:
    response = service(repository, BadCitationProvider()).run(
        ChatRequest(message="Berapa cuti tahunan?"),
        AuthContext(employee_id="EMP-001", display_name="Alya"),
    )
    assert response.outcome == "abstained"
    assert response.citations == []


def test_policy_chunk_ids_and_metadata_are_stable() -> None:
    path = ROOT / "data" / "policies" / "annual-leave.md"
    first = chunk_policy(path)
    second = chunk_policy(path)
    assert first == second
    assert first[0]["document_id"] == "HR-LEAVE-001"
    assert first[0]["version"] == "2.1"
    assert first[0]["effective_date"] == "2026-01-01"

