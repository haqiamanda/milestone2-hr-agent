"""Testable persona selection and API contract for the supplied Streamlit client."""

from __future__ import annotations

import json
from pathlib import Path


API_PATHS = {
    "chat": "/api/v1/chat",
    "confirm": "/api/v1/action-proposals/{proposal_id}/confirm",
    "my_requests": "/api/v1/leave-requests/me",
    "monitoring_summary": "/api/v1/monitoring/summary",
    "monitoring_alerts": "/api/v1/monitoring/alerts",
}


def load_personas(path: Path) -> list[dict[str, str]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("A nonempty demo persona catalog is required.")
    for row in rows:
        if not isinstance(row, dict) or not all(
            isinstance(row.get(key), str) and row[key]
            for key in ("employee_id", "label", "scenario", "demo_token")
        ):
            raise ValueError("Invalid demo persona.")
    return rows


def select_persona(state: dict, persona: dict[str, str]) -> None:
    """Clear responses and confirmation keys when the active identity changes."""
    if state.get("active_employee_id") == persona["employee_id"]:
        return
    state["active_employee_id"] = persona["employee_id"]
    state.pop("last_chat", None)
    for key in list(state):
        if key.startswith("idem_"):
            state.pop(key, None)

