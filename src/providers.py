from __future__ import annotations

import hashlib
import json
from typing import Any, Protocol

from .contracts import ModelToolCall, ModelTurn


class ModelProvider(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...
    def respond(
        self, *, input_data: str | list[Any], instructions: str,
        tools: list[dict[str, Any]],
        safety_identifier: str, max_output_tokens: int,
    ) -> ModelTurn: ...


class OpenAIProvider:
    """LangChain adapter for OpenAI chat completions and embeddings."""

    def __init__(self, **kwargs: Any) -> None:
        api_key = kwargs.pop("api_key", None)
        base_url = kwargs.pop("base_url", None)
        self.agent_model = kwargs.pop("agent_model", kwargs.pop("model", None))
        self.embedding_model = kwargs.pop("embedding_model", None)
        self.embedding_dimensions = kwargs.pop("embedding_dimensions", None)
        timeout = kwargs.pop("timeout", kwargs.pop("timeout_seconds", 30.0))
        chat_model = kwargs.pop("chat_model", None)
        embeddings = kwargs.pop("embeddings", None)

        if not api_key:
            raise ValueError("api_key is required")
        if not self.agent_model:
            raise ValueError("agent_model is required")
        if not self.embedding_model:
            raise ValueError("embedding_model is required")
        if self.embedding_dimensions is not None and self.embedding_dimensions <= 0:
            raise ValueError("embedding_dimensions must be positive")

        if chat_model is None or embeddings is None:
            try:
                from langchain_openai import ChatOpenAI, OpenAIEmbeddings
            except ImportError as exc:
                raise RuntimeError(
                    "langchain-openai package is required for OpenAIProvider"
                ) from exc

            common_kwargs: dict[str, Any] = {"api_key": api_key}
            if base_url:
                common_kwargs["base_url"] = base_url
            if chat_model is None:
                chat_model = ChatOpenAI(
                    model=self.agent_model,
                    timeout=timeout,
                    reasoning_effort="none",
                    **common_kwargs,
                )
            if embeddings is None:
                embedding_kwargs = {
                    **common_kwargs,
                    "model": self.embedding_model,
                    "request_timeout": timeout,
                }
                if self.embedding_dimensions is not None:
                    embedding_kwargs["dimensions"] = self.embedding_dimensions
                embeddings = OpenAIEmbeddings(**embedding_kwargs)

        self.chat_model = chat_model
        self.embeddings = embeddings

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not isinstance(texts, list) or not texts or any(
            not isinstance(text, str) or not text.strip() for text in texts
        ):
            raise ValueError("texts must be a non-empty list of non-blank strings")

        vectors = self.embeddings.embed_documents(texts)
        if not isinstance(vectors, list) or len(vectors) != len(texts):
            raise RuntimeError("embedding provider returned an unexpected batch size")
        if any(
            not isinstance(vector, list)
            or not vector
            or any(not isinstance(value, (int, float)) for value in vector)
            for vector in vectors
        ):
            raise RuntimeError("embedding provider returned an invalid vector")
        if self.embedding_dimensions is not None and any(
            len(vector) != self.embedding_dimensions for vector in vectors
        ):
            raise RuntimeError("embedding provider returned an unexpected vector dimension")
        return [[float(value) for value in vector] for vector in vectors]

    def respond(
        self,
        *,
        input_data: str | list[Any],
        instructions: str,
        tools: list[dict[str, Any]],
        safety_identifier: str,
        max_output_tokens: int,
    ) -> ModelTurn:
        if not isinstance(instructions, str) or not instructions.strip():
            raise ValueError("instructions must be a non-blank string")
        if not isinstance(safety_identifier, str) or not safety_identifier.strip():
            raise ValueError("safety_identifier must be a non-blank string")
        if not isinstance(max_output_tokens, int) or max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")

        is_history = isinstance(input_data, list) and bool(input_data) and all(
            isinstance(item, dict) and item.get("role") in {"user", "assistant", "tool"}
            for item in input_data
        )
        if is_history:
            self._validate_history(input_data)
            history = input_data
        else:
            # Preserve the original string/legacy-list adapter contract.
            content = input_data if isinstance(input_data, str) else json.dumps(
                input_data, ensure_ascii=False, default=str
            )
            history = [{"role": "user", "content": content}]
        try:
            from langchain_core.messages import SystemMessage, convert_to_messages
        except ImportError:
            messages = [{"role": "system", "content": instructions}, *history]
        else:
            messages = [SystemMessage(content=instructions), *convert_to_messages(history)]
        bound_model = self.chat_model
        if isinstance(tools, list) and tools:
            bound_model = bound_model.bind_tools(
                [self._chat_tool_schema(tool) for tool in tools]
            )
        response = bound_model.invoke(
            messages,
            safety_identifier=hashlib.sha256(
                safety_identifier.encode("utf-8")
            ).hexdigest(),
            max_tokens=max_output_tokens,
        )

        tool_calls: list[ModelToolCall] = []
        for item in getattr(response, "tool_calls", []) or []:
            arguments = item.get("args", {})
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError as exc:
                    raise RuntimeError("provider returned invalid tool arguments") from exc
            if not isinstance(arguments, dict):
                raise RuntimeError("provider returned invalid tool arguments")
            tool_calls.append(
                ModelToolCall(
                    call_id=str(item.get("id", "")),
                    name=str(item.get("name", "")),
                    arguments=arguments,
                )
            )

        metadata = getattr(response, "usage_metadata", {}) or {}
        response_metadata = getattr(response, "response_metadata", {}) or {}
        token_usage = response_metadata.get("token_usage", {})
        return ModelTurn(
            response_id=str(response_metadata.get("id", getattr(response, "id", ""))),
            text=self._message_text(getattr(response, "content", "")),
            tool_calls=tool_calls,
            output_items=[response],
            input_tokens=int(
                metadata.get("input_tokens", token_usage.get("prompt_tokens", 0)) or 0
            ),
            output_tokens=int(
                metadata.get("output_tokens", token_usage.get("completion_tokens", 0))
                or 0
            ),
        )

    @staticmethod
    def _validate_history(history: list[dict]) -> None:
        """Reject orphan, duplicated or incomplete tool exchanges before invoke."""
        pending: set[str] = set()
        used: set[str] = set()
        if history[0]["role"] != "user":
            raise ValueError("conversation must start with a user message")
        for message in history:
            role = message["role"]
            if role == "tool":
                identifier = message.get("tool_call_id")
                if identifier not in pending:
                    raise ValueError("invalid tool message pairing")
                pending.remove(identifier)
                continue
            if pending:
                raise ValueError("incomplete tool message pairing")
            if role == "assistant":
                for call in message.get("tool_calls", []):
                    identifier = call.get("id")
                    if not isinstance(identifier, str) or not identifier or identifier in used:
                        raise ValueError("invalid tool call identifier")
                    pending.add(identifier)
                    used.add(identifier)
        if pending:
            raise ValueError("incomplete tool message pairing")

    @staticmethod
    def _chat_tool_schema(tool: dict[str, Any]) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("parameters", {}),
                "strict": tool.get("strict", False),
            },
        }

    @staticmethod
    def _message_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(
                item.get("text", "") if isinstance(item, dict) else str(item)
                for item in content
            )
        return str(content or "")


class DeterministicProvider:
    """TODO(M2): A network-free fake for tests; never production default."""

    def __init__(self, dimensions: int = 8) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self.dimensions = dimensions

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not isinstance(texts, list) or any(
            not isinstance(text, str) for text in texts
        ):
            raise ValueError("texts must be a list of strings")
        vectors: list[list[float]] = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            vectors.append(
                [
                    int.from_bytes(digest[index : index + 2], "big") / 65535.0
                    for index in range(0, self.dimensions * 2, 2)
                ]
            )
        return vectors

    def respond(
        self,
        *,
        input_data: str | list[Any],
        instructions: str,
        tools: list[dict[str, Any]],
        safety_identifier: str,
        max_output_tokens: int,
    ) -> ModelTurn:
        if isinstance(input_data, str):
            prompt = input_data.casefold()
        else:
            prompt = json.dumps(input_data, ensure_ascii=False).casefold()

        if "cuti" in prompt or "kebijakan" in prompt:
            text = (
                "Berdasarkan Kebijakan Cuti Tahunan (HR-LEAVE-001), "
                "pegawai tetap memperoleh 12 hari kerja cuti tahunan per tahun kalender."
            )
        elif "saldo" in prompt:
            text = "Saldo cuti dapat diperiksa melalui data HR milik pegawai yang terautentikasi."
        else:
            text = "Maaf, saya tidak memiliki bukti kebijakan yang cukup untuk menjawab."
        return ModelTurn(response_id="deterministic", text=text)
