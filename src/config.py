from __future__ import annotations

from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "HR Policy and Leave Assistant"
    environment: str = "development"
    database_url: str = "postgresql://hr_app:hr_app@postgres:5432/hr_assistant"
    redis_url: str = "redis://redis:6379/0"
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    agent_model: str = ""
    embedding_model: str = ""
    embedding_dimensions: int = Field(default=1536, ge=64, le=4096)
    policy_index_version: str = "policies-2026-01"
    model_mode: str = "live"
    max_agent_turns: int = Field(default=4, ge=1, le=8)
    max_tool_calls: int = Field(default=6, ge=1, le=12)
    max_output_tokens: int = Field(default=700, ge=128, le=4096)
    model_timeout_seconds: float = Field(default=30.0, ge=1, le=120)
    proposal_ttl_minutes: int = Field(default=15, ge=1, le=60)
    retrieval_cache_ttl_seconds: int = Field(default=900, ge=0, le=86400)
    input_cost_per_1m: float | None = Field(default=None, ge=0)
    output_cost_per_1m: float | None = Field(default=None, ge=0)

    @field_validator(
        "openai_api_key",
        "openai_base_url",
        "input_cost_per_1m",
        "output_cost_per_1m",
        mode="before",
    )
    @classmethod
    def empty_string_is_none(cls, value):
        return None if value == "" else value

    @property
    def live_model_ready(self) -> bool:
        return bool(self.openai_api_key and self.agent_model and self.embedding_model)


@lru_cache
def get_settings() -> Settings:
    return Settings()
