"""Request-local context and grounding; all models/storage/cache are fakes."""
from __future__ import annotations

from copy import deepcopy
from datetime import date
import json
from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from src.agent import AgentService
from src.config import Settings
from src.contracts import AuthContext, ChatRequest, ModelToolCall, ModelTurn, PolicyChunk
from src.providers import OpenAIProvider
from src.repository import MemoryRepository
from src.retrieval import RetrievalService
from src.tools import ToolExecutor


AUTH = AuthContext(employee_id="EMP-001", display_name="Alya")
QUESTION = "Berapa hak cuti tahunan dan bagaimana ketentuan pengajuannya?"


def policy(**updates):
    return PolicyChunk(**{
        "chunk_id": "annual:one", "document_id": "HR-LEAVE-001", "title": "Cuti tahunan",
        "section": "Hak", "version": "2.1", "effective_date": date(2026, 1, 1),
        "content": "Pegawai tetap memperoleh 12 hari kerja cuti tahunan.", **updates,
    })


def tool_turn(name="search_hr_policy", identifier="search", arguments=None):
    return ModelTurn(response_id=identifier, tool_calls=[ModelToolCall(
        call_id=identifier, name=name,
        arguments=arguments if arguments is not None else {"query": QUESTION, "top_k": 3},
    )])


class ScriptedProvider:
    def __init__(self, turns):
        self.turns = iter(turns)
        self.calls = []
        self.embedding_calls = 0

    def embed(self, texts):
        self.embedding_calls += 1
        return [[0.0] * 64 for _ in texts]

    def respond(self, **kwargs):
        self.calls.append(deepcopy(kwargs))
        return next(self.turns)


def service(turns, chunks=None, **limits):
    repo = MemoryRepository()
    provider = ScriptedProvider(turns)
    retrieval = Mock()
    retrieval.search.return_value = [policy()] if chunks is None else chunks
    settings = Settings(_env_file=None, model_mode="fake", **limits)
    tools = ToolExecutor(repo, retrieval, settings, today_factory=lambda: date(2026, 10, 19))
    return AgentService(repo, provider, tools, settings), provider, retrieval, repo


def assert_pairing(history):
    pending = set()
    assert history[0]["role"] == "user"
    for message in history:
        if message["role"] == "assistant":
            assert not pending
            pending = {call["id"] for call in message["tool_calls"]}
        elif message["role"] == "tool":
            assert message["tool_call_id"] in pending
            pending.remove(message["tool_call_id"])
    assert not pending


def test_question_and_actual_chunk_content_survive_tool_round():
    agent, provider, _, repo = service([
        tool_turn(), ModelTurn(response_id="answer", text="Hak tahunan adalah 12 hari kerja. [HR-LEAVE-001]"),
    ])
    result = agent.run(ChatRequest(message=QUESTION), AUTH)
    assert result.outcome == "answered"
    history = provider.calls[1]["input_data"]
    assert history[0] == {"role": "user", "content": QUESTION}
    assert json.loads(history[-1]["content"])["result"][0]["content"] == policy().content
    assert_pairing(history)
    trace = repo.get_trace(result.trace_id)
    assert trace["retrieved_document_ids"] == ["HR-LEAVE-001"]
    assert QUESTION not in json.dumps(trace) and policy().content not in json.dumps(trace)


def test_multiple_rounds_retain_earlier_policy_and_authenticated_balance():
    agent, provider, _, _ = service([
        tool_turn(), tool_turn("get_my_leave_balance", "balance", {}),
        ModelTurn(response_id="answer", text="Hak tahunan 12 hari kerja; saldo Anda 8 hari. [HR-LEAVE-001]"),
    ])
    result = agent.run(ChatRequest(message=QUESTION), AUTH)
    assert result.outcome == "answered"
    history = provider.calls[2]["input_data"]
    assert history[0]["content"] == QUESTION
    outputs = {message["name"]: json.loads(message["content"])["result"]
               for message in history if message["role"] == "tool"}
    assert outputs["search_hr_policy"][0]["content"] == policy().content
    assert outputs["get_my_leave_balance"][0]["available_days"] == 8
    assert_pairing(history)


def test_fallback_discards_ungrounded_draft_and_regenerates_with_evidence():
    draft = "Jawaban awal belum memakai bukti HR-LEAVE-001."
    answer = "Pegawai tetap mendapat 12 hari kerja. [HR-LEAVE-001]"
    agent, provider, retrieval, _ = service([
        ModelTurn(response_id="draft", text=draft), ModelTurn(response_id="fresh", text=answer),
    ])
    result = agent.run(ChatRequest(message=QUESTION), AUTH)
    assert result.answer == answer and len(provider.calls) == 2
    retrieval.search.assert_called_once_with(QUESTION, 3)
    history = provider.calls[1]["input_data"]
    assert draft not in json.dumps(history)
    assert history[0]["content"] == QUESTION and policy().content in json.dumps(history)
    assert_pairing(history)


@pytest.mark.parametrize("question", [
    "Berapa subsidi perjalanan ke Mars?",
    "Apakah perusahaan membayar asuransi kapal selam?",
    "What is the company policy for asteroid mining equipment?",
    "Apa kebijakan subsidi kendaraan listrik?",
    "Apa kebijakan cuti untuk pilot pesawat ruang angkasa?",
])
def test_unrelated_retrieval_cannot_ground_out_of_domain_question(question):
    agent, _, _, repo = service([
        tool_turn(arguments={"query": question, "top_k": 3}),
        ModelTurn(response_id="unsupported", text="Subsidi tersedia. [HR-LEAVE-001]"),
    ])
    result = agent.run(ChatRequest(message=question), AUTH)
    assert result.outcome == "abstained" and result.citations == []
    assert not repo._requests and not repo._tickets


@pytest.mark.parametrize("text", [
    "Tidak ada kebijakan yang mengatur tunjangan tersebut. [HR-LEAVE-001]",
    "Saya belum menemukan informasi yang mendukung jawaban. [HR-LEAVE-001]",
    "Hal tersebut belum diatur dalam dokumen. [HR-LEAVE-001]",
    "Saya tidak dapat menyetujui permintaan Anda. [HR-LEAVE-001]",
])
def test_explicit_unsupported_or_unauthorized_answer_is_classified_as_abstention(text):
    agent, _, _, _ = service([tool_turn(), ModelTurn(response_id="refusal", text=text)])
    result = agent.run(ChatRequest(message=QUESTION), AUTH)
    assert result.outcome == "abstained" and result.citations == []


@pytest.mark.parametrize("question", [
    "Berapa jatah cuti tahunan?", "Jelaskan hak cuti tahunan pegawai tetap.",
    "How many annual leave days are available?",
])
def test_relevant_policy_answer_survives_topic_check(question):
    chunks = [policy(content="Pegawai mendapat 12 hari cuti tahunan. Annual leave is 12 days.")]
    agent, _, _, _ = service([
        tool_turn(arguments={"query": question, "top_k": 3}),
        ModelTurn(response_id="grounded", text="Hak cuti tahunan 12 hari. [HR-LEAVE-001]"),
    ], chunks)
    result = agent.run(ChatRequest(message=question), AUTH)
    assert result.outcome == "answered" and result.citations


def test_out_of_domain_question_cannot_bypass_grounding_by_omitting_search():
    agent, _, retrieval, _ = service([
        ModelTurn(response_id="draft", text="Perusahaan membayar biaya perlengkapan astronomi."),
        ModelTurn(response_id="unsupported", text="Biaya tersebut ditanggung. [HR-LEAVE-001]"),
    ])
    result = agent.run(ChatRequest(message="Berapa subsidi teleskop untuk observatorium?"), AUTH)
    retrieval.search.assert_called_once()
    assert result.outcome == "abstained" and result.citations == []


def test_official_policy_topics_still_match_their_actual_documents():
    from pathlib import Path
    from src.run_evaluation import load_cases
    from src.setup_lab import chunk_policy

    root = Path(__file__).resolve().parents[1]
    chunks = [PolicyChunk.model_validate(chunk) for path in (root / "data/policies").glob("*.md")
              for chunk in chunk_policy(path)]
    for case in load_cases(root / "data/evaluation_cases.json"):
        if case["expected_documents"]:
            relevant = {chunk.chunk_id: chunk for chunk in chunks
                        if chunk.document_id in case["expected_documents"]}
            assert AgentService._has_topic_evidence(case["input"], relevant), case["case_id"]


@pytest.mark.parametrize("text", [
    "Apa yang ingin Anda ketahui?", "Bagaimana saya dapat membantu? [HR-LEAVE-001]",
    "Baik.", "Berdasarkan HR-LEAVE-001.", "[HR-LEAVE-001]",
    "Tentu. Apa yang ingin Anda ketahui? [HR-LEAVE-001]",
])
def test_non_substantive_policy_responses_do_not_pass_on_citation_alone(text):
    agent, _, _, _ = service([tool_turn(), ModelTurn(response_id="generic", text=text)])
    result = agent.run(ChatRequest(message=QUESTION), AUTH)
    assert result.outcome == "abstained" and result.citations == []


@pytest.mark.parametrize("chunks,text", [
    ([], "Hak tahunan 99 hari. [HR-LEAVE-001]"),
    ([policy(content="")], "Hak tahunan 99 hari. [HR-LEAVE-001]"),
    ([policy()], "Bukti kebijakan tidak cukup untuk menjawab. [HR-LEAVE-001]"),
    ([policy()], "Saya tidak dapat memastikan karena bukti bertentangan. [HR-LEAVE-001]"),
    ([policy(document_id="HR-PRIV-001", content="Hanya saldo milik sendiri yang boleh diakses.")],
     "Informasi kebijakan tidak tersedia untuk menjawab pertanyaan ini. [HR-PRIV-001]"),
    ([policy(), policy(chunk_id="annual:two", version="1.0", content="Hak tahunan adalah 99 hari.")],
     "Hak tahunan 12 hari. [HR-LEAVE-001]"),
])
def test_empty_inadequate_or_conflicting_evidence_abstains(chunks, text):
    agent, _, _, _ = service([tool_turn(), ModelTurn(response_id="answer", text=text)], chunks)
    result = agent.run(ChatRequest(message=QUESTION), AUTH)
    assert result.outcome == "abstained" and result.citations == []


@pytest.mark.parametrize("text", ["Hak tahunan 12 hari.", "Hak tahunan 12 hari. [HR-FAKE-999]"])
def test_missing_and_foreign_citations_are_rejected(text):
    agent, _, _, _ = service([tool_turn(), ModelTurn(response_id="answer", text=text)])
    result = agent.run(ChatRequest(message=QUESTION), AUTH)
    assert result.outcome == "abstained" and result.citations == []


def test_fallback_never_exceeds_model_or_tool_limits():
    agent, provider, retrieval, _ = service([
        ModelTurn(response_id="draft", text="Hak tahunan 12 hari. [HR-LEAVE-001]"),
    ], max_agent_turns=1)
    result = agent.run(ChatRequest(message=QUESTION), AUTH)
    assert result.outcome == "abstained" and len(provider.calls) == 1
    retrieval.search.assert_not_called()


def test_request_history_is_not_shared_between_users_or_requests():
    agent, provider, _, _ = service([
        tool_turn(), ModelTurn(response_id="answer", text="Hak tahunan 12 hari. [HR-LEAVE-001]"),
        ModelTurn(response_id="greeting", text="Halo."),
    ])
    agent.run(ChatRequest(message=QUESTION), AUTH)
    agent.run(ChatRequest(message="Halo"), AuthContext(employee_id="EMP-002", display_name="Bima"))
    assert provider.calls[2]["input_data"] == "Halo"


def test_cache_hit_and_miss_supply_identical_evidence_to_answering_model():
    repo = MemoryRepository()
    repo.chunks = [policy()]
    answer = ModelTurn(response_id="answer", text="Hak tahunan 12 hari kerja. [HR-LEAVE-001]")
    provider = ScriptedProvider([tool_turn(), answer, tool_turn(), answer])
    store = {}
    cache = Mock()
    cache.get.side_effect = lambda key: json.loads(store[key]) if key in store else None
    cache.set.side_effect = lambda key, value, ttl: store.__setitem__(key, json.dumps(value))
    retrieval = RetrievalService(repo, provider, cache, 60, policy_index_version="context-test")
    settings = Settings(_env_file=None, model_mode="fake")
    agent = AgentService(repo, provider, ToolExecutor(repo, retrieval, settings), settings)
    first = agent.run(ChatRequest(message=QUESTION), AUTH)
    second = agent.run(ChatRequest(message=QUESTION), AUTH)
    assert first.outcome == second.outcome == "answered"
    assert provider.embedding_calls == 1
    assert provider.calls[1]["input_data"] == provider.calls[3]["input_data"]
    assert policy().content in provider.calls[3]["input_data"][-1]["content"]


def test_action_tool_history_preserves_proposal_and_confirmation_stays_explicit():
    agent, provider, _, repo = service([
        tool_turn("prepare_leave_request", "proposal", {
            "start_date": "2026-11-02", "end_date": "2026-11-03", "leave_type": "annual",
        }),
        ModelTurn(response_id="answer", text="Proposal telah disiapkan dan memerlukan konfirmasi."),
    ])
    result = agent.run(ChatRequest(message="Bantu pengajuan saya."), AUTH)
    assert result.outcome == "action_proposed" and result.action_proposal is not None
    history = provider.calls[1]["input_data"]
    assert history[0]["content"] == "Bantu pengajuan saya."
    assert json.loads(history[-1]["content"])["proposal"]["requires_confirmation"] is True
    assert_pairing(history)
    assert repo.list_leave_requests("EMP-001") == []
    identifier = result.action_proposal.proposal_id
    first = repo.confirm_proposal("EMP-001", identifier, "context-idempotency")
    replay = repo.confirm_proposal("EMP-001", identifier, "context-idempotency")
    assert first.status == "pending_manager_review"
    assert replay.idempotent_replay and replay.request_id == first.request_id
    assert len(repo.list_leave_requests("EMP-001")) == 1


def test_adapter_uses_native_assistant_and_matching_tool_messages():
    chat = Mock()
    chat.bind_tools.return_value = chat
    chat.invoke.return_value = AIMessage(content="Hak tahunan 12 hari. [HR-LEAVE-001]")
    provider = OpenAIProvider(api_key="synthetic", agent_model="fake", embedding_model="fake",
                              chat_model=chat, embeddings=Mock())
    history = [{"role": "user", "content": QUESTION}]
    AgentService._append_tool_exchange(history, [{
        "call_id": "search-1", "name": "search_hr_policy", "arguments": {"query": QUESTION},
        "result": [policy().model_dump(mode="json")], "proposal": None,
    }])
    provider.respond(input_data=history, instructions="Gunakan bukti.", tools=[],
                     safety_identifier="EMP-001", max_output_tokens=128)
    messages = chat.invoke.call_args.args[0]
    assert isinstance(messages[0], SystemMessage) and isinstance(messages[1], HumanMessage)
    assert messages[1].content == QUESTION
    assert isinstance(messages[2], AIMessage) and messages[2].tool_calls[0]["id"] == "search-1"
    assert isinstance(messages[3], ToolMessage) and messages[3].tool_call_id == "search-1"
    assert policy().content in messages[3].content


@pytest.mark.parametrize("history", [
    [{"role": "user", "content": QUESTION}, {"role": "tool", "tool_call_id": "orphan", "content": "{}"}],
    [{"role": "user", "content": QUESTION}, {"role": "assistant", "content": "", "tool_calls": [
        {"id": "unanswered", "type": "function", "function": {"name": "search_hr_policy", "arguments": "{}"}}
    ]}],
])
def test_adapter_rejects_orphan_or_unanswered_tool_calls(history):
    chat = Mock()
    provider = OpenAIProvider(api_key="synthetic", agent_model="fake", embedding_model="fake",
                              chat_model=chat, embeddings=Mock())
    with pytest.raises(ValueError, match="pairing"):
        provider.respond(input_data=history, instructions="Gunakan bukti.", tools=[],
                         safety_identifier="EMP-001", max_output_tokens=128)
    chat.invoke.assert_not_called()
