from pathlib import Path

from ui_contract import API_PATHS, load_personas, select_persona


def test_supplied_personas_match_public_identity_contract() -> None:
    personas = load_personas(Path(__file__).resolve().parents[1] / "demo_personas.json")
    assert {item["employee_id"] for item in personas} == {"EMP-001", "EMP-002", "ADM-001"}
    assert len({item["demo_token"] for item in personas}) == len(personas)


def test_persona_change_clears_prior_response_and_confirmation_key() -> None:
    state = {
        "active_employee_id": "EMP-001",
        "last_chat": {"answer": "Alya's answer"},
        "idem_AP-OLD": "same-key",
    }
    select_persona(state, {"employee_id": "EMP-002"})
    assert state == {"active_employee_id": "EMP-002"}


def test_supplied_ui_uses_documented_api_paths() -> None:
    assert API_PATHS["chat"] == "/api/v1/chat"
    assert API_PATHS["confirm"].format(proposal_id="AP-1") == "/api/v1/action-proposals/AP-1/confirm"

