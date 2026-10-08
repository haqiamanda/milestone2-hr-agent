from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from src.contracts import LeaveBalance, PrepareLeaveArgs
from src.domain import DomainError, business_days, validate_leave_request


def test_business_days_excludes_weekend() -> None:
    assert business_days(date(2026, 9, 25), date(2026, 9, 28)) == 2


@pytest.mark.parametrize(
    "start,end,message",
    [
        (date(2026, 9, 22), date(2026, 9, 24), "masa lalu"),
        (date(2026, 9, 25), date(2026, 9, 24), "sebelum"),
    ],
)
def test_invalid_dates_are_blocked(start: date, end: date, message: str) -> None:
    args = PrepareLeaveArgs(start_date=start, end_date=end, leave_type="annual", reason=None)
    with pytest.raises(DomainError, match=message):
        validate_leave_request(
            args,
            [LeaveBalance(leave_type="annual", available_days=8, as_of_date=date(2026, 1, 1))],
            False,
            date(2026, 9, 23),
        )


def test_insufficient_balance_and_overlap_are_blocked() -> None:
    args = PrepareLeaveArgs(
        start_date=date(2026, 10, 1), end_date=date(2026, 10, 9), leave_type="annual", reason=None
    )
    balance = [LeaveBalance(leave_type="annual", available_days=2, as_of_date=date(2026, 1, 1))]
    with pytest.raises(DomainError, match="tidak cukup"):
        validate_leave_request(args, balance, False, date(2026, 9, 23))
    with pytest.raises(DomainError, match="bertumpang tindih"):
        validate_leave_request(args, balance, True, date(2026, 9, 23))


def test_unsupported_leave_type_is_rejected_by_contract() -> None:
    with pytest.raises(ValidationError):
        PrepareLeaveArgs(
            start_date=date(2026, 10, 1),
            end_date=date(2026, 10, 2),
            leave_type="study",
            reason=None,
        )
