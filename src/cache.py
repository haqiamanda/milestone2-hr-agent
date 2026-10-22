from __future__ import annotations

import json
import logging
from typing import Any

import redis
from redis.exceptions import RedisError


logger = logging.getLogger(__name__)


class ResilientCache:
    def __init__(self, redis_url: str) -> None:
        self.client = redis.Redis.from_url(
            redis_url,
            decode_responses=True,
            socket_connect_timeout=1.0,
            socket_timeout=1.0,
        )

    def health(self) -> bool:
        try:
            return bool(self.client.ping())
        except RedisError:
            return False

    def get(self, key: str) -> Any | None:
        try:
            value = self.client.get(key)
        except RedisError:
            return None

        if value is None:
            return None
        try:
            return json.loads(value)
        except (TypeError, ValueError):
            return None

    def set(self, key: str, value: Any, ttl_seconds: int) -> None:
        try:
            serialized = json.dumps(value)
        except (TypeError, ValueError):
            logger.debug("Cache value is not JSON serializable")
            return

        try:
            self.client.set(key, serialized, ex=ttl_seconds)
        except RedisError:
            logger.debug("Redis cache write failed")


class NullCache(ResilientCache):
    def __init__(self) -> None:
        self.client = None

    def health(self) -> bool:
        return False

    def get(self, key: str) -> Any | None:
        return None

    def set(self, key: str, value: Any, ttl_seconds: int) -> None:
        return None
