from __future__ import annotations

from fastapi import FastAPI

from .config import Settings, get_settings


def build_runtime(settings: Settings):
    # TODO(M2): Wire PostgresRepository, ResilientCache, provider, retrieval, tools, and agent.
    raise NotImplementedError


def create_app(*, settings: Settings | None = None, repository=None, cache=None, agent=None) -> FastAPI:
    """Build the documented API without changing endpoint or response contracts."""
    # TODO(M2): Add auth dependencies, problem details, eight endpoints, and authorization checks.
    raise NotImplementedError

