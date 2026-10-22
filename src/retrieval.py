from __future__ import annotations

import re

from .cache import ResilientCache
from .config import get_settings
from .contracts import PolicyChunk
from .providers import ModelProvider
from .repository import Repository


class RetrievalService:
    def __init__(
        self,
        repository: Repository,
        provider: ModelProvider,
        cache: ResilientCache,
        cache_ttl_seconds: int,
        policy_index_version: str | None = None,
    ) -> None:
        self.repository = repository
        self.provider = provider
        self.cache = cache
        self.cache_ttl_seconds = cache_ttl_seconds
        self.policy_index_version = (
            policy_index_version
            if policy_index_version is not None
            else get_settings().policy_index_version
        )

    def search(self, query: str, top_k: int = 3) -> list[PolicyChunk]:
        if top_k <= 0:
            return []

        cache_key = f"retrieval:{self.policy_index_version}:{query}:{top_k}"
        get = getattr(self.cache, "get", None)
        set_value = getattr(self.cache, "set", None)
        if callable(get):
            try:
                cached = get(cache_key)
            except Exception:
                cached = None
            if isinstance(cached, list):
                try:
                    cached_chunks = [
                        item
                        if isinstance(item, PolicyChunk)
                        else PolicyChunk.model_validate(item)
                        for item in cached
                    ]
                except (TypeError, ValueError):
                    cached_chunks = []
                if len(cached_chunks) == len(cached):
                    deduplicated: list[PolicyChunk] = []
                    seen: set[str] = set()
                    for chunk in cached_chunks:
                        if chunk.chunk_id not in seen:
                            deduplicated.append(chunk)
                            seen.add(chunk.chunk_id)
                    return deduplicated[:top_k]

        candidate_limit = max(top_k * 3, top_k)
        lexical_search = getattr(self.repository, "lexical_search", None)
        lexical = (
            lexical_search(query, candidate_limit)
            if callable(lexical_search)
            else []
        )
        if not lexical:
            chunks = getattr(self.repository, "chunks", [])
            terms = self._terms(query)
            lexical = sorted(
                chunks,
                key=lambda chunk: self._keyword_score(chunk, terms),
                reverse=True,
            )[:candidate_limit]

        vector: list[PolicyChunk] = []
        try:
            embeddings = self.provider.embed([query])
            if embeddings:
                vector_search = getattr(self.repository, "vector_search", None)
                if callable(vector_search):
                    vector = vector_search(embeddings[0], candidate_limit)
        except (NotImplementedError, AttributeError):
            vector = []

        ranked: dict[str, tuple[PolicyChunk, float, int]] = {}
        for rank, chunk in enumerate(lexical, start=1):
            previous = ranked.get(chunk.chunk_id)
            ranked[chunk.chunk_id] = (
                previous[0] if previous else chunk,
                (previous[1] if previous else 0.0) + 1 / (60 + rank),
                (previous[2] if previous else 0) + 1,
            )
        for rank, chunk in enumerate(vector, start=1):
            previous = ranked.get(chunk.chunk_id)
            ranked[chunk.chunk_id] = (
                previous[0] if previous else chunk,
                (previous[1] if previous else 0.0) + 1 / (60 + rank),
                (previous[2] if previous else 0) + 1,
            )

        terms = self._terms(query)
        results = sorted(
            ranked.values(),
            key=lambda item: (
                self._keyword_score(item[0], terms),
                item[1],
                item[2],
            ),
            reverse=True,
        )[:top_k]
        output = [item[0] for item in results]
        if callable(set_value) and self.cache_ttl_seconds > 0:
            try:
                set_value(
                    cache_key,
                    [chunk.model_dump(mode="json") for chunk in output],
                    self.cache_ttl_seconds,
                )
            except Exception:
                pass
        return output

    @staticmethod
    def _terms(query: str) -> set[str]:
        return set(re.findall(r"\w+", query.casefold()))

    @classmethod
    def _keyword_score(cls, chunk: PolicyChunk, terms: set[str]) -> int:
        text = " ".join((chunk.title, chunk.section, chunk.content)).casefold()
        return sum(term in set(re.findall(r"\w+", text)) for term in terms)
