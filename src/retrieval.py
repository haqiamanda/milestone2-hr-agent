from __future__ import annotations

from .cache import ResilientCache
from .contracts import PolicyChunk
from .providers import ModelProvider
from .repository import Repository


class RetrievalService:
    def __init__(self, repository: Repository, provider: ModelProvider, cache: ResilientCache, cache_ttl_seconds: int) -> None:
        self.repository = repository
        self.provider = provider
        self.cache = cache
        self.cache_ttl_seconds = cache_ttl_seconds

    def search(self, query: str, top_k: int = 3) -> list[PolicyChunk]:
        # TODO(M2): Version-aware cache -> embed -> lexical/vector candidates -> RRF -> keyword rerank.
        raise NotImplementedError

