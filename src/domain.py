"""Deterministic leave-domain rules; this module must never call the model."""
from __future__ import annotations

from datetime import date, timedelta

from .contracts import LeaveBalance, PrepareLeaveArgs


class DomainError(ValueError):
    pass


def business_days(start: date, end: date) -> int:
    if start > end:
        raise DomainError("tanggal mulai tidak boleh setelah tanggal akhir")

    days = (end - start).days + 1
    weekdays = sum(
        (start + timedelta(days=offset)).weekday() < 5 for offset in range(days)
    )
    if weekdays == 0:
        raise DomainError("rentang tanggal tidak memiliki hari kerja")
    return weekdays


def validate_leave_request(
    args: PrepareLeaveArgs,
    balances: list[LeaveBalance],
    overlaps: bool,
    today: date,
) -> tuple[int, list[str]]:
    if args.start_date < today:
        raise DomainError("tanggal mulai tidak boleh berada di masa lalu")
    if args.start_date > args.end_date:
        raise DomainError("tanggal mulai tidak boleh sebelum tanggal akhir")
    if args.leave_type not in {"annual", "sick", "parental"}:
        raise DomainError("jenis cuti tidak didukung")
    if overlaps:
        raise DomainError("tanggal cuti bertumpang tindih dengan permintaan aktif")

    requested_days = business_days(args.start_date, args.end_date)
    # HR-LEAVE-001: submission date through the day before leave starts.
    # Weekends do not count; public-holiday calendars are outside this policy.
    notice_days = sum(
        (today + timedelta(days=offset)).weekday() < 5
        for offset in range((args.start_date - today).days)
    )
    if args.leave_type == "annual" and notice_days < 3:
        raise DomainError("cuti tahunan harus diajukan minimal tiga hari kerja sebelum tanggal mulai")
    balance = next(
        (item for item in balances if item.leave_type == args.leave_type),
        None,
    )
    if balance is None or balance.available_days < requested_days:
        raise DomainError("saldo cuti tidak cukup")

    return requested_days, [
        f"jenis cuti {args.leave_type}",
        f"{requested_days} hari kerja",
        "tidak ada tanggal yang bertumpang tindih",
        "saldo cuti mencukupi",
    ]


def estimate_cost(
    input_tokens: int,
    output_tokens: int,
    input_rate: float | None,
    output_rate: float | None,
) -> float | None:
    if input_rate is None or output_rate is None:
        return None

    cost = (
        input_tokens * input_rate + output_tokens * output_rate
    ) / 1_000_000
    return round(cost, 6)
