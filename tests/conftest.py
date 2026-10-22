from __future__ import annotations

from datetime import date

import pytest
from fastapi.testclient import TestClient

from src.agent import AgentService
from src.api import create_app
from src.cache import NullCache
from src.config import Settings
from src.contracts import PolicyChunk
from src.providers import DeterministicProvider
from src.repository import MemoryRepository
from src.retrieval import RetrievalService
from src.tools import ToolExecutor
from tests.postgres_fixture import postgres_database


@pytest.fixture
def settings() -> Settings:
    return Settings(
        model_mode="fake",
        proposal_ttl_minutes=15,
        max_agent_turns=4,
        max_tool_calls=6,
    )


@pytest.fixture
def repository() -> MemoryRepository:
    repo = MemoryRepository()
    repo.chunks = [
        PolicyChunk(
            chunk_id="chunk-annual",
            document_id="HR-LEAVE-001",
            title="Kebijakan Cuti Tahunan",
            section="Hak dan saldo",
            version="2.1",
            effective_date=date(2026, 1, 1),
            content="Pegawai tetap memperoleh 12 hari kerja cuti tahunan per tahun kalender.",
        ),
        PolicyChunk(
            chunk_id="chunk-privacy",
            document_id="HR-PRIV-001",
            title="Privasi Data Pegawai",
            section="Akses minimum",
            version="2.0",
            effective_date=date(2026, 1, 1),
            content="Pegawai hanya boleh melihat saldo dan permintaan miliknya sendiri.",
        ),
    ]
    return repo


@pytest.fixture
def client(repository: MemoryRepository, settings: Settings) -> TestClient:
    provider = DeterministicProvider()
    cache = NullCache()
    retrieval = RetrievalService(repository, provider, cache, 0)
    tools = ToolExecutor(repository, retrieval, settings, today_factory=lambda: date(2026, 9, 23))
    agent = AgentService(repository, provider, tools, settings)
    app = create_app(settings=settings, repository=repository, cache=cache, agent=agent)
    return TestClient(app)
