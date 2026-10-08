"""Security helpers. Complete every TODO before persisting telemetry."""
from __future__ import annotations


def token_hash(token: str) -> str:
    # TODO(M2): Return a deterministic SHA-256 hex digest; never store the raw token.
    raise NotImplementedError


def stable_safety_id(employee_id: str) -> str:
    # TODO(M2): Return a stable, non-identifying hash suitable for provider safety_identifier.
    raise NotImplementedError


def redact(value: str) -> str:
    # TODO(M2): Redact email, Indonesian phone numbers, token-like values, and medical detail.
    raise NotImplementedError

