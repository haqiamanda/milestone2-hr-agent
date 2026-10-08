from __future__ import annotations

from .config import Settings
from .contracts import AuthContext, ChatRequest, ChatResponse
from .providers import ModelProvider
from .repository import Repository
from .tools import ToolExecutor


SYSTEM_INSTRUCTIONS = """TODO(M2): Write a concise Bahasa Indonesia system instruction covering
grounding, citations, untrusted retrieved text, identity binding, pending-only actions,
abstention, and the prohibition on exposing chain-of-thought."""


class AgentService:
    def __init__(self, repository: Repository, provider: ModelProvider, tools: ToolExecutor, settings: Settings) -> None:
        self.repository = repository
        self.provider = provider
        self.tools = tools
        self.settings = settings

    def run(self, request: ChatRequest, auth: AuthContext) -> ChatResponse:
        # TODO(M2): Implement bounded tool loop, citation validation, fallback, usage, and redacted trace.
        raise NotImplementedError

