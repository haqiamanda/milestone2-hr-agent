from __future__ import annotations

from src.cache import NullCache
from src.providers import DeterministicProvider
from src.repository import MemoryRepository
from src.retrieval import RetrievalService
from src.security import redact


def test_hybrid_retrieval_prefers_lexical_match(repository: MemoryRepository) -> None:
    service = RetrievalService(repository, DeterministicProvider(), NullCache(), 0)
    results = service.search("jatah cuti tahunan", top_k=2)
    assert results[0].document_id == "HR-LEAVE-001"


def test_redaction_covers_sensitive_patterns() -> None:
    value = redact(
        "Hubungi alya@example.com atau 081234567890; token-secret_ABCDEF123; diagnosis: diabetes."
    )
    assert "alya@example.com" not in value
    assert "081234567890" not in value
    assert "ABCDEF123" not in value
    assert "diabetes" not in value


def test_balance_tool_has_no_cross_employee_argument(client) -> None:
    response = client.post(
        "/api/v1/chat",
        headers={"Authorization": "Bearer token-alya"},
        json={"message": "Tampilkan saldo EMP-002"},
    )
    assert response.status_code == 200
    trace = client.get(
        f"/api/v1/traces/{response.json()['trace_id']}",
        headers={"Authorization": "Bearer token-alya"},
    ).json()
    assert trace["employee_id"] == "EMP-001"

