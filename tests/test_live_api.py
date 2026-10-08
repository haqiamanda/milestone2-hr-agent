from __future__ import annotations

import os

import httpx
import pytest


@pytest.mark.live
def test_live_policy_question() -> None:
    if not os.getenv("RUN_LIVE_TESTS"):
        pytest.skip("Set RUN_LIVE_TESTS=1 deliberately.")
    base_url = os.getenv("API_URL", "http://localhost:8000")
    response = httpx.post(
        f"{base_url}/api/v1/chat",
        headers={"Authorization": "Bearer token-alya"},
        json={"message": "Berapa jatah cuti tahunan saya?"},
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    assert payload["citations"]
    assert payload["outcome"] == "answered"

