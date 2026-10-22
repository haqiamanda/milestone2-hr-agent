"""Aggregate safe run metadata and evaluate the assignment's monitoring rules."""
from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def _rules() -> list[dict]:
    path = Path(__file__).resolve().parents[1] / "data" / "monitoring_rules.json"
    return json.loads(path.read_text(encoding="utf-8"))


def actionable_alerts(metrics: dict) -> list[dict]:
    """Emit only measured breaches; insufficient samples do not trigger alerts."""
    alerts = []
    for rule in _rules():
        if rule["operator"] != "gt":
            raise ValueError("unsupported monitoring rule operator")
        value = metrics.get(rule["metric"])
        if (value is not None and metrics["run_count"] >= rule["minimum_samples"]
                and value > rule["threshold"]):
            alerts.append({
                **rule, "value": value, "sample_count": metrics["run_count"],
            })
    return alerts


def failed_run(trace: dict) -> bool:
    # Expected policy denials are not operational outages. Plain abstention
    # (e.g. insufficient evidence) is not a failure without an error signal.
    if trace.get("error") == "medical_privacy_blocked":
        return False
    return bool(
        trace.get("outcome") == "error" or trace.get("error")
        or any(call.get("status") == "error" for call in trace.get("tool_calls", []))
    )


def memory_metrics(traces: list[dict]) -> dict:
    latencies = sorted(trace.get("latency_ms", 0) for trace in traces)
    p95 = 0
    if latencies:
        # PostgreSQL percentile_cont(0.95) uses the same linear interpolation.
        position = (len(latencies) - 1) * 0.95
        lower, upper = math.floor(position), math.ceil(position)
        p95 = round(latencies[lower] + (latencies[upper] - latencies[lower]) * (position - lower))
    return {
        "run_count": len(traces),
        "error_rate": sum(failed_run(trace) for trace in traces) / len(traces) if traces else 0.0,
        "p95_latency_ms": p95,
        "tool_calls": sum(len(trace.get("tool_calls", [])) for trace in traces),
    }
