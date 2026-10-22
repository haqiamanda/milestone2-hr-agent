from __future__ import annotations

from datetime import date

import pytest
from langchain_core.tools import StructuredTool

from src.cache import NullCache
from src.contracts import AuthContext
from src.providers import DeterministicProvider
from src.retrieval import RetrievalService
from src.tools import TOOL_SCHEMAS, ToolExecutor, build_langchain_tools


@pytest.fixture
def retrieval(repository):
    return RetrievalService(repository, DeterministicProvider(), NullCache(), 0)


def test_tool_schemas_are_strict_and_do_not_accept_employee_id() -> None:
    assert [schema["name"] for schema in TOOL_SCHEMAS] == [
        "search_hr_policy",
        "get_my_leave_balance",
        "prepare_leave_request",
        "create_hr_ticket",
    ]

    for schema in TOOL_SCHEMAS:
        assert schema["strict"] is True
        assert schema["parameters"]["additionalProperties"] is False
        properties = schema["parameters"].get("properties", {})
        assert set(schema["parameters"]["required"]) == set(properties)
        assert "employee_id" not in schema["parameters"].get("properties", {})


def test_langchain_tools_are_request_scoped_and_reject_extra_identity(
    repository, retrieval, settings
) -> None:
    executor = ToolExecutor(
        repository, retrieval, settings, today_factory=lambda: date(2026, 9, 23)
    )
    auth = AuthContext(employee_id="EMP-001", display_name="Alya")

    tools = build_langchain_tools(executor, auth)

    assert all(isinstance(tool, StructuredTool) for tool in tools)
    assert {tool.name for tool in tools} == {
        "search_hr_policy",
        "get_my_leave_balance",
        "prepare_leave_request",
        "create_hr_ticket",
    }

    balance_tool = next(tool for tool in tools if tool.name == "get_my_leave_balance")
    assert "employee_id" not in balance_tool.args_schema.model_fields
    with pytest.raises(ValueError, match="invalid tool arguments"):
        executor.execute(
            "get_my_leave_balance",
            {"employee_id": "EMP-002"},
            auth,
        )


def test_langchain_tool_delegates_business_execution_to_executor(
    repository, retrieval, settings, monkeypatch
) -> None:
    executor = ToolExecutor(
        repository, retrieval, settings, today_factory=lambda: date(2026, 9, 23)
    )
    auth = AuthContext(employee_id="EMP-001", display_name="Alya")
    calls = []

    def fake_execute(name, arguments, received_auth):
        calls.append((name, arguments, received_auth))
        return {"ok": True}, None

    monkeypatch.setattr(executor, "execute", fake_execute)
    policy_tool = next(
        tool for tool in build_langchain_tools(executor, auth)
        if tool.name == "search_hr_policy"
    )

    assert policy_tool.invoke({"query": "cuti", "top_k": 1}) == {"ok": True}
    assert calls == [
        (
            "search_hr_policy",
            {"query": "cuti", "top_k": 1},
            auth,
        )
    ]


def test_prepare_leave_request_remains_proposal_only(
    repository, retrieval, settings
) -> None:
    executor = ToolExecutor(
        repository, retrieval, settings, today_factory=lambda: date(2026, 9, 23)
    )
    auth = AuthContext(employee_id="EMP-001", display_name="Alya")
    prepare_tool = next(
        tool for tool in build_langchain_tools(executor, auth)
        if tool.name == "prepare_leave_request"
    )

    result = prepare_tool.invoke(
        {
            "start_date": "2026-10-05",
            "end_date": "2026-10-06",
            "leave_type": "annual",
            "reason": "Keperluan pribadi",
        }
    )

    assert result["requires_confirmation"] is True
    assert repository.list_leave_requests("EMP-001") == []
    assert repository.get_proposal(result["proposal_id"])[0] == "EMP-001"
