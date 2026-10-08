"""Deterministic leave-domain rules; this module must never call the model."""
from __future__ import annotations

from datetime import date

from .contracts import LeaveBalance, PrepareLeaveArgs


class DomainError(ValueError):
    pass


def business_days(start: date, end: date) -> int:
    # TODO(M2): Validate ordering, exclude weekends, and reject zero business-day ranges.
    raise NotImplementedError


def validate_leave_request(
    args: PrepareLeaveArgs,
    balances: list[LeaveBalance],
    overlaps: bool,
    today: date,
) -> tuple[int, list[str]]:
    # TODO(M2): Apply past-date, range, overlap, leave-type, and balance rules.
    raise NotImplementedError


def estimate_cost(
    input_tokens: int,
    output_tokens: int,
    input_rate: float | None,
    output_rate: float | None,
) -> float | None:
    # TODO(M2): Return None unless both instructor-supplied rates are configured.
    raise NotImplementedError

