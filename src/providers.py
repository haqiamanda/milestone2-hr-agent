from __future__ import annotations

from typing import Any, Protocol

from .contracts import ModelTurn


class ModelProvider(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...
    def respond(
        self, *, input_data: str | list[Any], instructions: str,
        tools: list[dict[str, Any]],
        safety_identifier: str, max_output_tokens: int,
    ) -> ModelTurn: ...


class OpenAIProvider:
    """TODO(M2): Implement Responses API function calling and batched embeddings.

    Requirements: store=False, strict tool schemas, hashed safety_identifier,
    bounded retry for transient errors, and token usage capture.
    """

    def __init__(self, **kwargs: Any) -> None:
        raise NotImplementedError


class DeterministicProvider:
    """TODO(M2): A network-free fake for tests; never production default."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    def respond(self, **kwargs: Any) -> ModelTurn:
        raise NotImplementedError
