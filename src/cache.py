from __future__ import annotations

from typing import Any


class ResilientCache:
    def __init__(self, redis_url: str) -> None:
        # TODO(M2): Initialize Redis with a short socket timeout.
        raise NotImplementedError

    def health(self) -> bool:
        raise NotImplementedError

    def get(self, key: str) -> Any | None:
        # TODO(M2): Any Redis failure must behave like a cache miss.
        raise NotImplementedError

    def set(self, key: str, value: Any, ttl_seconds: int) -> None:
        # TODO(M2): Cache JSON only and swallow infrastructure failure after logging safely.
        raise NotImplementedError


class NullCache(ResilientCache):
    def __init__(self) -> None:
        self.client = None

