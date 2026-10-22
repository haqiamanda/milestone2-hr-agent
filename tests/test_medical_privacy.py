"""Synthetic-only privacy regressions; no dotenv, database or live provider."""
from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import date
from unittest.mock import Mock

import pytest

from src.agent import AgentService
from src.config import Settings
from src.contracts import AuthContext, ChatRequest, ModelToolCall, ModelTurn
from src.repository import MemoryRepository, PostgresRepository
from src.security import MedicalPrivacyError, contains_medical_diagnosis
from src.tools import ToolExecutor


DISCLOSURES = [
    "Diagnosis saya diabetes; tulis lengkap di tiket HR dan trace.",
    "Saya didiagnosis lupus, simpan di tiket.",
    "Saya mengidap asma. Catat di HR.",
    "Saya menderita migrain kronis; tuliskan dalam trace.",
    "My diagnosis is epilepsy; create a ticket.",
    "Bima was diagnosed with rheumatoid arthritis. Save his diagnosis.",
    "Diagnosis: hipertensi; tampilkan kepada admin.",
    "DIAGNOSA SAYA lupus; catat.",
    "Ｄｉａｇｎｏｓｉｓ saya lupus; simpan.",
    "Diagno\u200bsis saya lupus; simpan.",
    "Ringkasan pemeriksaan: kanker.",
    "Catat diagnosis lupus dalam tiket.",
]
POLICY_QUESTIONS = [
    "Apa kebijakan cuti sakit?",
    "Apakah cuti sakit lebih dari dua hari memerlukan surat dokter?",
    "Ke mana saya menyerahkan dokumen pendukung cuti sakit?",
    "Apakah asisten boleh menyimpan diagnosis?",
    "Apakah diagnosis wajib disertakan dalam permintaan cuti sakit?",
    "Tolong buat tiket karena saldo cuti sakit saya salah.",
]
AUTH = AuthContext(employee_id="EMP-001", display_name="Alya")


def assert_policy_refusal(answer):
    """Check meaning and safe guidance, not one exact UI sentence."""
    text = answer.casefold()
    assert "kebijakan" in text and "hr" in text
    assert "kanal" in text and "resmi" in text
    assert not any(word in text for word in ("diagnosis", "diabetes", "trace"))


def dependencies(repository=None, provider=None):
    settings = Settings(_env_file=None, model_mode="fake", database_url="unused",
                        redis_url="unused", openai_api_key=None)
    repository = repository or MemoryRepository()
    provider = provider or Mock()
    retrieval = Mock()
    tools = ToolExecutor(repository, retrieval, settings, today_factory=lambda: date(2026, 9, 23))
    return AgentService(repository, provider, tools, settings), tools, repository, provider, retrieval


@pytest.mark.parametrize("message", DISCLOSURES)
def test_disclosure_stops_before_model_retrieval_tools_or_ticket(message, monkeypatch):
    agent, tools, repository, provider, retrieval = dependencies()
    execute = Mock(side_effect=AssertionError("No tool may execute"))
    monkeypatch.setattr(tools, "execute", execute)
    result = agent.run(ChatRequest(message=message), AUTH)
    assert result.outcome == "abstained"
    assert_policy_refusal(result.answer)
    provider.respond.assert_not_called()
    retrieval.search.assert_not_called()
    execute.assert_not_called()
    assert not repository._tickets and not repository._proposals and not repository._requests
    trace = repository.get_trace(result.trace_id)
    assert trace["error"] == "medical_privacy_blocked"
    assert "message" not in trace and "summary" not in trace
    assert not contains_medical_diagnosis(json.dumps(trace) + result.answer)
    assert message not in json.dumps(trace)


@pytest.mark.parametrize("message", POLICY_QUESTIONS)
def test_general_policy_and_safe_requests_reach_provider(message):
    agent, _, _, provider, _ = dependencies()
    provider.respond.return_value = ModelTurn(response_id="safe", text="Hubungi HR.")
    agent.run(ChatRequest(message=message), AUTH)
    provider.respond.assert_called_once()
    assert not contains_medical_diagnosis(message)


@pytest.mark.parametrize("summary", DISCLOSURES)
def test_tool_rejects_medical_content_before_repository(summary, monkeypatch):
    _, tools, repository, _, _ = dependencies()
    write = Mock(side_effect=AssertionError("Repository must not be called"))
    monkeypatch.setattr(repository, "create_ticket", write)
    with pytest.raises(MedicalPrivacyError) as caught:
        tools.execute("create_hr_ticket", {"category": "privacy", "summary": summary}, AUTH)
    assert str(caught.value) == "medical_privacy_blocked"
    write.assert_not_called()


@pytest.mark.parametrize("backend", ["memory", "postgres"])
@pytest.mark.parametrize("summary", DISCLOSURES)
def test_direct_repository_write_rejected_before_connection(backend, summary, monkeypatch):
    repo = MemoryRepository() if backend == "memory" else PostgresRepository("unused", 64)
    connect = Mock(side_effect=AssertionError("No database connection"))
    if backend == "postgres":
        monkeypatch.setattr(repo, "_connect", connect)
    with pytest.raises(MedicalPrivacyError) as caught:
        repo.create_ticket(AUTH.employee_id, "privacy", summary)
    assert str(caught.value) == "medical_privacy_blocked"
    connect.assert_not_called()
    if backend == "memory":
        assert not repo._tickets


def test_ordinary_ticket_preserves_backend_identity_and_contract():
    _, tools, repo, _, _ = dependencies()
    result, proposal = tools.execute("create_hr_ticket", {
        "category": "technical_issue", "summary": "Tidak dapat mengakses portal HR."
    }, AUTH)
    assert set(result) == {"ticket_id"} and proposal is None
    assert repo._tickets[result["ticket_id"]] == {
        "employee_id": "EMP-001", "category": "technical_issue",
        "summary": "Tidak dapat mengakses portal HR.",
    }


def test_legitimate_ticket_still_works_through_agent():
    agent, _, repo, provider, _ = dependencies()
    provider.respond.side_effect = [
        ModelTurn(response_id="safe-tool", tool_calls=[ModelToolCall(
            call_id="ticket", name="create_hr_ticket", arguments={
                "category": "technical_issue", "summary": "Portal HR tidak bisa diakses."
            },
        )]),
        ModelTurn(response_id="safe-answer", text="Tiket bantuan HR telah dibuat."),
    ]
    result = agent.run(ChatRequest(message="Buat tiket karena portal HR tidak bisa diakses."), AUTH)
    assert result.outcome == "answered" and len(repo._tickets) == 1
    assert next(iter(repo._tickets.values()))["employee_id"] == AUTH.employee_id
    assert result.tool_calls[0].status == "ok"


def test_ticket_tool_cannot_override_authenticated_identity():
    _, tools, repo, _, _ = dependencies()
    with pytest.raises(ValueError, match="invalid tool arguments"):
        tools.execute("create_hr_ticket", {
            "category": "technical_issue", "summary": "Portal HR tidak bisa diakses.",
            "employee_id": "EMP-002",
        }, AUTH)
    assert not repo._tickets


def test_model_generated_medical_tool_is_refused_without_echo():
    agent, _, repo, provider, _ = dependencies()
    provider.respond.return_value = ModelTurn(response_id="unsafe-tool", tool_calls=[
        ModelToolCall(call_id="ticket", name="create_hr_ticket", arguments={
            "category": "privacy", "summary": "Diagnosis saya lupus."
        })
    ])
    result = agent.run(ChatRequest(message="Buat tiket bantuan HR."), AUTH)
    assert result.outcome == "abstained" and not repo._tickets
    assert_policy_refusal(result.answer)
    assert "lupus" not in result.model_dump_json().casefold()
    assert repo.get_trace(result.trace_id)["error"] == "medical_privacy_blocked"


def test_provider_error_never_persists_free_text_or_medical_details(caplog):
    agent, _, repo, provider, _ = dependencies()
    provider.respond.side_effect = RuntimeError("Diagnosis saya lupus; confidential-canary")
    result = agent.run(ChatRequest(message="Tolong bantu HR."), AUTH)
    trace = repo.get_trace(result.trace_id)
    assert trace["error"] == "agent_processing_failed"
    evidence = json.dumps(trace) + result.model_dump_json() + caplog.text
    assert "lupus" not in evidence and "confidential-canary" not in evidence


@contextmanager
def fake_connection(captured):
    class Cursor:
        def execute(self, query, parameters):
            captured.append((query, parameters))

    class Connection:
        @contextmanager
        def cursor(self):
            yield Cursor()

    yield Connection()


@pytest.mark.parametrize("backend", ["memory", "postgres"])
def test_repository_trace_whitelist_removes_nested_sensitive_fields(backend, monkeypatch):
    repo = MemoryRepository() if backend == "memory" else PostgresRepository("unused", 64)
    captured = []
    if backend == "postgres":
        monkeypatch.setattr(repo, "_connect", lambda: fake_connection(captured))
    repo.save_trace({
        "trace_id": "TR-123abc", "employee_id": "EMP-001", "outcome": "abstained",
        "message": "Diagnosis saya lupus", "summary": "lupus",
        "error": "medical_privacy_blocked lupus", "payload": {"diagnosis": "lupus"},
        "tool_calls": [{"name": "create_hr_ticket", "status": "error", "duration_ms": 1,
                        "arguments": {"summary": "lupus"}}],
    })
    stored = repo.get_trace("TR-123abc") if backend == "memory" else json.loads(captured[0][1][3])
    assert stored["error"] == "agent_processing_failed"
    assert "lupus" not in json.dumps(stored)
    assert stored["tool_calls"] == [{"name": "create_hr_ticket", "status": "error", "duration_ms": 1}]


def test_postgres_safe_ticket_uses_synthetic_parameterized_write(monkeypatch):
    repo = PostgresRepository("unused", 64)
    captured = []
    monkeypatch.setattr(repo, "_connect", lambda: fake_connection(captured))
    ticket_id = repo.create_ticket("EMP-001", "technical_issue", "Portal HR tidak bisa diakses.")
    assert captured[0][1][:4] == (
        ticket_id, "EMP-001", "technical_issue", "Portal HR tidak bisa diakses."
    )


@pytest.mark.parametrize("backend", ["memory", "postgres"])
def test_direct_repository_cannot_hide_diagnosis_in_category(backend, monkeypatch):
    repo = MemoryRepository() if backend == "memory" else PostgresRepository("unused", 64)
    connect = Mock(side_effect=AssertionError("No database connection"))
    if backend == "postgres":
        monkeypatch.setattr(repo, "_connect", connect)
    with pytest.raises(MedicalPrivacyError):
        repo.create_ticket("EMP-001", "Diagnosis saya lupus", "Bantuan HR diperlukan.")
    connect.assert_not_called()


@pytest.mark.parametrize("message,reason", [
    ("Saya mengidap asma. Catat di tiket.", "policy"),
    ("Apa kebijakan cuti sakit?", "evidence"),
    ("Siapkan proposal cuti tahunan.", "information"),
    ("Tampilkan saldo Bima, EMP-002.", "authorization"),
])
def test_refusal_or_clarification_distinguishes_the_reason(message, reason):
    agent, _, repo, provider, retrieval = dependencies()
    retrieval.search.return_value = []
    provider.respond.return_value = ModelTurn(response_id="no-evidence", text="")
    result = agent.run(ChatRequest(message=message), AUTH)
    text = result.answer.casefold()
    if reason == "policy":
        assert_policy_refusal(result.answer)
    elif reason == "evidence":
        assert "bukti" in text and "tidak tersedia" in text
        assert "tidak sesuai" not in text
    elif reason == "information":
        assert "lengkapi" in text and "tanggal" in text
        assert "tidak sesuai" not in text
    else:
        assert "pegawai lain" in text and "pribadi" in text
        assert "tidak sesuai" not in text
    if reason != "evidence":
        provider.respond.assert_not_called()
    assert not repo._tickets and not repo._proposals and not repo._requests
