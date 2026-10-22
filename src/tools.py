"""Strict tool schemas and identity-bound execution."""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, ValidationError

from .config import Settings
from .contracts import (
    ActionProposal,
    AuthContext,
    PrepareLeaveArgs,
    SearchPolicyArgs,
    TicketArgs,
)
from .domain import validate_leave_request
from .repository import Repository
from .security import validate_privacy
from .retrieval import RetrievalService


TOOL_EXECUTION_ORIGIN = ContextVar("tool_execution_origin", default="unknown")


@contextmanager
def tool_execution_origin(origin: str):
    """Request-local provenance for passive QA observers; never model-supplied."""
    token = TOOL_EXECUTION_ORIGIN.set(origin)
    try:
        yield
    finally:
        TOOL_EXECUTION_ORIGIN.reset(token)


class ToolBlockedError(ValueError):
    """Rejected before dispatch by the tool allowlist or argument schema."""


class _NoToolArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


_ToolDefinition = tuple[str, str, type[BaseModel] | None]
_TOOL_DEFINITIONS: tuple[_ToolDefinition, ...] = (
    (
        "search_hr_policy",
        "Cari kebijakan HR yang relevan dan kembalikan metadata dokumen.",
        SearchPolicyArgs,
    ),
    (
        "get_my_leave_balance",
        "Ambil saldo cuti milik pegawai yang sedang terautentikasi.",
        None,
    ),
    (
        "prepare_leave_request",
        "Siapkan proposal pengajuan cuti tanpa membuat permintaan cuti.",
        PrepareLeaveArgs,
    ),
    (
        "create_hr_ticket",
        "Buat tiket HR untuk permintaan yang perlu ditindaklanjuti.",
        TicketArgs,
    ),
)


def _tool_parameters(model: type[BaseModel] | None) -> dict[str, Any]:
    parameters = (model or _NoToolArgs).model_json_schema()
    parameters.pop("title", None)
    parameters["additionalProperties"] = False
    parameters["required"] = list(parameters.get("properties", {}))
    return parameters


TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "name": name,
        "description": description,
        "parameters": _tool_parameters(model),
        "strict": True,
    }
    for name, description, model in _TOOL_DEFINITIONS
]


def build_langchain_tools(executor: "ToolExecutor", auth: AuthContext) -> list[Any]:
    """Build request-scoped LangChain tools without exposing backend identity."""
    from langchain_core.tools import StructuredTool

    tools: list[Any] = []
    for name, description, model in _TOOL_DEFINITIONS:
        args_schema = model or _NoToolArgs

        def invoke_tool(
            _name: str = name, _executor: ToolExecutor = executor, **arguments: Any
        ) -> Any:
            result, _ = _executor.execute(_name, arguments, auth)
            return result

        tools.append(
            StructuredTool.from_function(
                func=invoke_tool,
                name=name,
                description=description,
                args_schema=args_schema,
            )
        )
    return tools


class ToolExecutor:
    def __init__(self, repository: Repository, retrieval: RetrievalService, settings: Settings, today_factory=None) -> None:
        self.repository = repository
        self.retrieval = retrieval
        self.settings = settings
        self.today_factory = today_factory

    def execute(self, name: str, arguments: dict[str, Any], auth: AuthContext) -> tuple[Any, ActionProposal | None]:
        if name == "search_hr_policy":
            args = self._validate_arguments(SearchPolicyArgs, arguments)
            chunks = self.retrieval.search(args.query, args.top_k)
            return [chunk.model_dump(mode="json") for chunk in chunks], None

        if name == "get_my_leave_balance":
            self._validate_arguments(None, arguments)
            balances = self.repository.get_balances(auth.employee_id)
            return [balance.model_dump(mode="json") for balance in balances], None

        if name == "prepare_leave_request":
            args = self._validate_arguments(PrepareLeaveArgs, arguments)
            balances = self.repository.get_balances(auth.employee_id)
            today = self.today_factory() if self.today_factory else date.today()
            business_day_count, validation_summary = validate_leave_request(
                args,
                balances,
                self.repository.has_overlap(
                    auth.employee_id, args.start_date, args.end_date
                ),
                today,
            )
            proposal = ActionProposal(
                proposal_id=f"AP-{uuid4().hex[:12]}",
                leave_type=args.leave_type,
                start_date=args.start_date,
                end_date=args.end_date,
                business_days=business_day_count,
                reason=args.reason,
                validation_summary=validation_summary,
                expires_at=datetime.now(timezone.utc)
                + timedelta(minutes=self.settings.proposal_ttl_minutes),
            )
            self.repository.save_proposal(auth.employee_id, proposal)
            return proposal.model_dump(mode="json"), proposal

        if name == "create_hr_ticket":
            # Before Pydantic: validation errors must not echo medical arguments.
            if isinstance(arguments, dict):
                for field in ("category", "summary"):
                    if isinstance(arguments.get(field), str):
                        validate_privacy(arguments[field])
            args = self._validate_arguments(TicketArgs, arguments)
            ticket_id = self.repository.create_ticket(
                auth.employee_id, args.category, args.summary
            )
            return {"ticket_id": ticket_id}, None

        raise ToolBlockedError("unknown tool")

    @staticmethod
    def _validate_arguments(model, arguments: dict[str, Any] | None):
        if not isinstance(arguments, dict):
            raise ToolBlockedError("tool arguments must be an object")

        expected = set(model.model_fields) if model is not None else set()
        if not set(arguments).issubset(expected):
            raise ToolBlockedError("invalid tool arguments")
        try:
            return model.model_validate(arguments) if model is not None else None
        except ValidationError:
            raise ToolBlockedError("invalid tool arguments") from None
