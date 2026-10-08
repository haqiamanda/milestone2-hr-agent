"""Strict tool schemas and identity-bound execution."""
from __future__ import annotations

from typing import Any

from .config import Settings
from .contracts import ActionProposal, AuthContext
from .repository import Repository
from .retrieval import RetrievalService


# TODO(M2): Define all four Responses API function schemas with strict=True,
# additionalProperties=False, and every property listed in required.
TOOL_SCHEMAS: list[dict[str, Any]] = []


class ToolExecutor:
    def __init__(self, repository: Repository, retrieval: RetrievalService, settings: Settings, today_factory=None) -> None:
        self.repository = repository
        self.retrieval = retrieval
        self.settings = settings
        self.today_factory = today_factory

    def execute(self, name: str, arguments: dict[str, Any], auth: AuthContext) -> tuple[Any, ActionProposal | None]:
        # TODO(M2): Validate arguments with Pydantic and bind all private-data tools to auth.employee_id.
        raise NotImplementedError

