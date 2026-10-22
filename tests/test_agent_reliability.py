from __future__ import annotations

from datetime import date
import json

from src.agent import AgentService
from src.cache import NullCache
from src.config import Settings
from src.contracts import (
    AuthContext,
    ChatRequest,
    ModelToolCall,
    ModelTurn,
    PolicyChunk,
)
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


class MissingCitationProvider:
    def embed(self, texts):
        return [[0.0] * 8 for _ in texts]

    def respond(self, **kwargs):
        return ModelTurn(response_id="missing-citation", text="Cuti tahunan tersedia.")


class FallbackPolicyProvider:
    def __init__(self) -> None:
        self.turn = 0

    def embed(self, texts):
        return [[0.0] * 8 for _ in texts]

    def respond(self, **kwargs):
        self.turn += 1
        return ModelTurn(
            response_id=f"fallback-{self.turn}",
            text=("Kebijakan ini merujuk pada HR-LEAVE-001." if self.turn == 1 else
                  "Pegawai tetap memperoleh 12 hari kerja cuti tahunan. [HR-LEAVE-001]"),
        )


class CombinedLeaveProvider:
    def __init__(self) -> None:
        self.turn = 0
        self.calls = []

    def embed(self, texts):
        return [[0.0] * 8 for _ in texts]

    def respond(self, **kwargs):
        self.calls.append(kwargs)
        self.turn += 1
        if self.turn == 1:
            return ModelTurn(
                response_id="combined-tools",
                tool_calls=[
                    ModelToolCall(
                        call_id="policy",
                        name="search_hr_policy",
                        arguments={"query": "cuti tahunan pengajuan", "top_k": 2},
                    ),
                    ModelToolCall(
                        call_id="balance",
                        name="get_my_leave_balance",
                        arguments={},
                    ),
                ],
            )

        return ModelTurn(
            response_id="combined-answer",
            text=(
                "Hak cuti tahunan pegawai tetap adalah 12 hari kerja per tahun kalender. "
                "Saldo operasional pribadi Anda saat ini 8 hari, dan saldo ini tidak sama "
                "dengan hak tahunan. Pengajuan dilakukan paling lambat 3 hari kerja "
                "sebelum tanggal mulai. [HR-LEAVE-001]"
            ),
        )


class PersonalLeaveBalanceProvider:
    def __init__(self) -> None:
        self.turn = 0

    def embed(self, texts):
        return [[0.0] * 8 for _ in texts]

    def respond(self, **kwargs):
        self.turn += 1
        if self.turn == 1:
            return ModelTurn(
                response_id="balance-tool",
                tool_calls=[
                    ModelToolCall(
                        call_id="balance",
                        name="get_my_leave_balance",
                        arguments={},
                    )
                ],
            )
        return ModelTurn(
            response_id="balance-answer",
            text="Saldo cuti tahunan Alya saat ini adalah 8 hari.",
        )


class LeaveProposalProvider:
    def __init__(self) -> None:
        self.turn = 0

    def embed(self, texts):
        return [[0.0] * 8 for _ in texts]

    def respond(self, **kwargs):
        self.turn += 1
        if self.turn == 1:
            return ModelTurn(
                response_id="proposal-tool",
                tool_calls=[
                    ModelToolCall(
                        call_id="proposal",
                        name="prepare_leave_request",
                        arguments={
                            "start_date": "2026-10-05",
                            "end_date": "2026-10-06",
                            "leave_type": "annual",
                            "reason": "Keperluan pribadi",
                        },
                    )
                ],
            )
        return ModelTurn(
            response_id="proposal-answer",
            text="Proposal cuti tahunan untuk 2 hari kerja sudah disiapkan dan menunggu konfirmasi.",
        )


class PlainTextProvider:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls = []

    def embed(self, texts):
        return [[0.0] * 8 for _ in texts]

    def respond(self, **kwargs):
        self.calls.append(kwargs)
        return ModelTurn(response_id="plain-text", text=self.text)


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


def test_policy_tool_retrieval_is_exposed_as_ordered_provenance(
    repository: MemoryRepository,
) -> None:
    response = service(repository, BadCitationProvider()).run(
        ChatRequest(message="Berapa cuti tahunan?"),
        AuthContext(employee_id="EMP-001", display_name="Alya"),
    )

    assert response.retrieved_sources is not None
    assert [
        (source.rank, source.chunk_id, source.document_id, source.version)
        for source in response.retrieved_sources
    ] == [(1, "chunk-annual", "HR-LEAVE-001", "2.1")]
    assert response.retrieved_sources[0].effective_date == date(2026, 1, 1)


def test_fallback_policy_retrieval_is_exposed_as_provenance(
    repository: MemoryRepository,
) -> None:
    response = service(repository, FallbackPolicyProvider()).run(
        ChatRequest(message="Jelaskan aturan cuti tahunan."),
        AuthContext(employee_id="EMP-001", display_name="Alya"),
    )

    assert response.outcome == "answered"
    assert response.retrieved_sources is not None
    assert response.retrieved_sources[0].document_id == "HR-LEAVE-001"
    assert response.retrieved_sources[0].rank == 1


def test_missing_policy_citation_forces_abstention(repository: MemoryRepository) -> None:
    response = service(repository, MissingCitationProvider()).run(
        ChatRequest(message="Berapa cuti tahunan?"),
        AuthContext(employee_id="EMP-001", display_name="Alya"),
    )
    assert response.outcome == "abstained"
    assert response.citations == []


def test_combined_leave_answer_uses_policy_and_personal_balance_evidence() -> None:
    repository = MemoryRepository()
    repository.chunks = [
        PolicyChunk(
            chunk_id="chunk-annual-right",
            document_id="HR-LEAVE-001",
            title="Kebijakan Cuti Tahunan",
            section="Hak dan saldo",
            version="2.1",
            effective_date=date(2026, 1, 1),
            content=(
                "Pegawai tetap memperoleh 12 hari kerja cuti tahunan per tahun kalender. "
                "Saldo tersedia pada sistem HR merupakan sumber operasional."
            ),
        ),
        PolicyChunk(
            chunk_id="chunk-annual-application",
            document_id="HR-LEAVE-001",
            title="Kebijakan Cuti Tahunan",
            section="Pengajuan",
            version="2.1",
            effective_date=date(2026, 1, 1),
            content="Pengajuan dilakukan paling lambat tiga hari kerja sebelum tanggal mulai.",
        ),
    ]
    provider = CombinedLeaveProvider()

    response = service(repository, provider).run(
        ChatRequest(
            message="Berapa hak cuti tahunan, saldo pribadi saya, dan apa syarat pengajuannya?"
        ),
        AuthContext(employee_id="EMP-001", display_name="Alya"),
    )

    assert response.outcome == "answered"
    assert "12 hari kerja" in response.answer
    assert "8 hari" in response.answer
    assert "3 hari kerja" in response.answer
    assert response.citations
    assert {citation.document_id for citation in response.citations} == {"HR-LEAVE-001"}
    assert [item.name for item in response.tool_calls] == [
        "search_hr_policy",
        "get_my_leave_balance",
    ]

    assert len(provider.calls) == 2
    instructions = provider.calls[0]["instructions"]
    assert "jawab setiap subpertanyaan" in instructions
    assert "Bedakan hak atau ketentuan" in instructions
    assert "saldo operasional" in instructions
    assert "Jangan menyamakan atau menghitung ulang" in instructions

    second_input = provider.calls[1]["input_data"]
    assert second_input[0]["content"] == "Berapa hak cuti tahunan, saldo pribadi saya, dan apa syarat pengajuannya?"
    tool_messages = [item for item in second_input if item["role"] == "tool"]
    assert {item["name"] for item in tool_messages} == {
        "search_hr_policy",
        "get_my_leave_balance",
    }
    policy_output = next(
        json.loads(item["content"])["result"] for item in tool_messages if item["name"] == "search_hr_policy"
    )
    balance_output = next(
        json.loads(item["content"])["result"] for item in tool_messages if item["name"] == "get_my_leave_balance"
    )
    assert {item["section"] for item in policy_output} == {"Hak dan saldo", "Pengajuan"}
    assert any(item["available_days"] == 8 for item in balance_output)


def test_personal_annual_leave_balance_answers_with_balance_tool() -> None:
    repository = MemoryRepository()
    provider = PersonalLeaveBalanceProvider()

    response = service(repository, provider).run(
        ChatRequest(message="Berapa saldo cuti tahunan saya?"),
        AuthContext(employee_id="EMP-001", display_name="Alya"),
    )

    assert response.outcome == "answered"
    assert "8 hari" in response.answer
    assert [item.name for item in response.tool_calls] == ["get_my_leave_balance"]


def test_agent_prepares_future_leave_proposal_without_creating_request() -> None:
    repository = MemoryRepository()
    provider = LeaveProposalProvider()

    response = service(repository, provider).run(
        ChatRequest(
            message="Siapkan proposal cuti tahunan 2026-10-05 sampai 2026-10-06."
        ),
        AuthContext(employee_id="EMP-001", display_name="Alya"),
    )

    assert response.outcome == "action_proposed"
    assert response.action_proposal is not None
    assert response.action_proposal.requires_confirmation is True
    assert [
        (item.name, item.status) for item in response.tool_calls
    ] == [("prepare_leave_request", "ok")]
    assert repository.list_leave_requests("EMP-001") == []


def test_explicit_complete_proposal_request_does_not_end_as_plain_answer(
    repository: MemoryRepository,
) -> None:
    provider = PlainTextProvider(
        "Kebijakan cuti tahunan tercantum pada HR-LEAVE-001."
    )

    response = service(repository, provider).run(
        ChatRequest(
            message=(
                "Siapkan proposal cuti tahunan dari 2026-10-05 sampai 2026-10-06."
            )
        ),
        AuthContext(employee_id="EMP-001", display_name="Alya"),
    )

    assert response.outcome == "action_proposed"
    assert response.action_proposal is not None
    assert response.action_proposal.start_date == date(2026, 10, 5)
    assert response.action_proposal.end_date == date(2026, 10, 6)
    assert response.action_proposal.leave_type == "annual"
    assert response.action_proposal.requires_confirmation is True
    assert [item.name for item in response.tool_calls].count("prepare_leave_request") == 1
    assert repository.list_leave_requests("EMP-001") == []


def test_ordinary_leave_policy_question_does_not_prepare_proposal(
    repository: MemoryRepository,
) -> None:
    provider = PlainTextProvider(
        "Kebijakan cuti tahunan pegawai tetap tercantum pada HR-LEAVE-001."
    )

    response = service(repository, provider).run(
        ChatRequest(message="Apa kebijakan cuti tahunan pegawai tetap?"),
        AuthContext(employee_id="EMP-001", display_name="Alya"),
    )

    assert response.outcome == "answered"
    assert response.action_proposal is None
    assert all(item.name != "prepare_leave_request" for item in response.tool_calls)
    assert repository._proposals == {}
    assert repository.list_leave_requests("EMP-001") == []


def test_incomplete_explicit_proposal_request_asks_for_missing_leave_type(
    repository: MemoryRepository,
) -> None:
    provider = PlainTextProvider(
        "Kebijakan cuti tahunan tercantum pada HR-LEAVE-001."
    )

    response = service(repository, provider).run(
        ChatRequest(
            message=(
                "Siapkan proposal cuti dari 2026-10-05 sampai 2026-10-06."
            )
        ),
        AuthContext(employee_id="EMP-001", display_name="Alya"),
    )

    assert "jenis cuti" in response.answer.casefold()
    assert "2026-10-05" not in response.answer
    assert "2026-10-06" not in response.answer
    assert response.action_proposal is None
    assert all(item.name != "prepare_leave_request" for item in response.tool_calls)
    assert repository._proposals == {}
    assert repository.list_leave_requests("EMP-001") == []


def test_incomplete_explicit_proposal_request_asks_for_missing_dates(
    repository: MemoryRepository,
) -> None:
    provider = PlainTextProvider(
        "Kebijakan cuti tahunan tercantum pada HR-LEAVE-001."
    )

    response = service(repository, provider).run(
        ChatRequest(message="Siapkan proposal cuti tahunan."),
        AuthContext(employee_id="EMP-001", display_name="Alya"),
    )

    answer = response.answer.casefold()
    assert "tanggal mulai" in answer
    assert "tanggal akhir" in answer
    assert response.action_proposal is None
    assert all(item.name != "prepare_leave_request" for item in response.tool_calls)
    assert repository._proposals == {}
    assert repository.list_leave_requests("EMP-001") == []


def test_policy_chunk_ids_and_metadata_are_stable() -> None:
    path = ROOT / "data" / "policies" / "annual-leave.md"
    first = chunk_policy(path)
    second = chunk_policy(path)
    assert first == second
    assert first[0]["document_id"] == "HR-LEAVE-001"
    assert first[0]["version"] == "2.1"
    assert first[0]["effective_date"] == "2026-01-01"
