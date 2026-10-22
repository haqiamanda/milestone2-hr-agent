from src.providers import OpenAIProvider
import sys
import types
import pytest

import src.config
import src.providers
import src.setup_lab


class FakeEmbeddings:
    def embed_documents(self, texts):
        return [[float(index)] for index, _ in enumerate(texts)]


class FakeResponse:
    content = "Jawaban"
    tool_calls = [
        {"id": "call-1", "name": "search_hr_policy", "args": {"query": "cuti"}}
    ]
    usage_metadata = {"input_tokens": 11, "output_tokens": 7}
    response_metadata = {"id": "response-1"}


class FakeChatModel:
    def __init__(self):
        self.bound_tools = None
        self.calls = []

    def bind_tools(self, tools):
        self.bound_tools = tools
        return self

    def invoke(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return FakeResponse()


def test_langchain_adapter_preserves_contract_and_usage() -> None:
    chat_model = FakeChatModel()
    provider = OpenAIProvider(
        api_key="test-key",
        agent_model="agent",
        embedding_model="embedding",
        embeddings=FakeEmbeddings(),
        chat_model=chat_model,
    )

    turn = provider.respond(
        input_data="Berapa cuti?",
        instructions="Jawab singkat.",
        tools=[
            {
                "type": "function",
                "name": "search_hr_policy",
                "description": "Cari kebijakan.",
                "parameters": {"type": "object"},
                "strict": True,
            }
        ],
        safety_identifier="EMP-001",
        max_output_tokens=128,
    )

    assert turn.response_id == "response-1"
    assert turn.text == "Jawaban"
    assert turn.input_tokens == 11
    assert turn.output_tokens == 7
    assert turn.tool_calls[0].arguments == {"query": "cuti"}
    assert chat_model.bound_tools[0]["function"]["name"] == "search_hr_policy"
    assert chat_model.bound_tools[0]["function"]["strict"] is True
    assert chat_model.calls[0][1]["safety_identifier"] != "EMP-001"
    assert len(provider.embed(["a", "b"])) == 2


def test_openai_provider_disables_reasoning_for_function_tools(monkeypatch) -> None:
    captured = {}

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            captured["chat"] = kwargs

    class FakeOpenAIEmbeddings:
        def __init__(self, **kwargs):
            captured["embeddings"] = kwargs

    fake_module = types.SimpleNamespace(
        ChatOpenAI=FakeChatOpenAI,
        OpenAIEmbeddings=FakeOpenAIEmbeddings,
    )
    monkeypatch.setitem(sys.modules, "langchain_openai", fake_module)

    OpenAIProvider(
        api_key="test-key",
        agent_model="gpt-6-luna",
        embedding_model="embedding",
    )

    assert captured["chat"]["model"] == "gpt-6-luna"
    assert captured["chat"]["reasoning_effort"] == "none"


def test_reset_and_seed_passes_agent_model_to_provider(monkeypatch) -> None:
    captured = {}

    class FakeSettings:
        openai_api_key = "test-key"
        openai_base_url = None
        agent_model = "configured-agent"
        embedding_model = "configured-embedding"
        embedding_dimensions = 1536
        live_model_ready = True

    class StopBeforeDatabase:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            raise RuntimeError("stop before database setup")

    monkeypatch.setattr(src.config, "get_settings", lambda: FakeSettings())
    monkeypatch.setattr(src.providers, "OpenAIProvider", StopBeforeDatabase)

    with pytest.raises(RuntimeError, match="stop before database setup"):
        src.setup_lab.reset_and_seed()

    assert captured == {
        "api_key": "test-key",
        "base_url": None,
        "agent_model": "configured-agent",
        "embedding_model": "configured-embedding",
        "embedding_dimensions": 1536,
    }
