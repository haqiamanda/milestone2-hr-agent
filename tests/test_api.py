from __future__ import annotations

from fastapi.testclient import TestClient

import src.api
from src.config import Settings


AUTH = {"Authorization": "Bearer token-alya"}


def test_factory_builds_runtime_when_dependencies_are_missing(monkeypatch) -> None:
    repository = type("Repository", (), {
        "readiness": lambda self, version: {"database": True, "schema": True, "index": True},
    })()
    cache = type("Cache", (), {"health": lambda self: True})()
    agent = object()
    calls = []

    def fake_build_runtime(settings):
        calls.append(settings)
        return repository, cache, agent

    monkeypatch.setattr(src.api, "build_runtime", fake_build_runtime)

    client = TestClient(src.api.create_app(settings=Settings(model_mode="fake")))

    assert client.get("/readyz").json() == {"status": "ready"}
    assert len(calls) == 1


def test_global_app_is_asgi_entrypoint() -> None:
    assert callable(src.api.app)


def test_health_and_auth_contract(client: TestClient) -> None:
    assert client.get("/healthz").json() == {"status": "ok"}
    assert client.post("/api/v1/chat", json={"message": "saldo cuti"}).status_code == 401


def test_chat_returns_contract_and_trace(client: TestClient) -> None:
    response = client.post("/api/v1/chat", headers=AUTH, json={"message": "Berapa jatah cuti tahunan?"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["outcome"] == "answered"
    assert payload["citations"][0]["document_id"] == "HR-LEAVE-001"
    assert payload["retrieved_sources"][0]["rank"] == 1
    assert payload["retrieved_sources"][0]["document_id"] == "HR-LEAVE-001"
    trace = client.get(f"/api/v1/traces/{payload['trace_id']}", headers=AUTH)
    assert trace.status_code == 200
    assert "message" not in trace.json()


def test_chat_prepares_leave_proposal_without_creating_request(
    client: TestClient, repository
) -> None:
    response = client.post(
        "/api/v1/chat",
        headers=AUTH,
        json={
            "message": (
                "Siapkan proposal cuti tahunan dari 2026-11-02 sampai "
                "2026-11-03 karena keperluan keluarga."
            )
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["outcome"] == "action_proposed"
    assert payload["action_proposal"] is not None
    assert payload["action_proposal"]["requires_confirmation"] is True
    proposal_id = payload["action_proposal"]["proposal_id"]
    assert repository.get_proposal(proposal_id) is not None
    assert repository.list_leave_requests("EMP-001") == []


def test_monitoring_is_admin_only(client: TestClient) -> None:
    assert client.get("/api/v1/monitoring/summary", headers=AUTH).status_code == 403
    response = client.get(
        "/api/v1/monitoring/summary", headers={"Authorization": "Bearer token-admin"}
    )
    assert response.status_code == 200


def test_trace_cannot_be_read_by_another_employee(client: TestClient) -> None:
    chat = client.post("/api/v1/chat", headers=AUTH, json={"message": "saldo saya"}).json()
    response = client.get(
        f"/api/v1/traces/{chat['trace_id']}",
        headers={"Authorization": "Bearer token-bima"},
    )
    assert response.status_code == 403
