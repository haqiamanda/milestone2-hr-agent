from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from src.contracts import ActionProposal
from src.repository import ConflictError, MemoryRepository


def proposal() -> ActionProposal:
    return ActionProposal(
        proposal_id="AP-TEST",
        leave_type="annual",
        start_date=date(2026, 11, 2),
        end_date=date(2026, 11, 3),
        business_days=2,
        validation_summary=["valid"],
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=10),
    )


def test_confirmation_is_pending_only_and_idempotent(repository: MemoryRepository) -> None:
    repository.save_proposal("EMP-001", proposal())
    first = repository.confirm_proposal("EMP-001", "AP-TEST", "idem-12345678")
    replay = repository.confirm_proposal("EMP-001", "AP-TEST", "idem-12345678")
    assert first.status == "pending_manager_review"
    assert replay.request_id == first.request_id
    assert replay.idempotent_replay is True
    assert repository.list_leave_requests("EMP-001")[0].status == "pending_manager_review"


def test_confirmation_rejects_wrong_identity_and_key(repository: MemoryRepository) -> None:
    repository.save_proposal("EMP-001", proposal())
    with pytest.raises(KeyError):
        repository.confirm_proposal("EMP-002", "AP-TEST", "idem-12345678")
    repository.confirm_proposal("EMP-001", "AP-TEST", "idem-12345678")
    with pytest.raises(ConflictError):
        repository.confirm_proposal("EMP-001", "AP-TEST", "another-key")


def test_expired_proposal_is_rejected(repository: MemoryRepository) -> None:
    item = proposal().model_copy(update={"expires_at": datetime.now(timezone.utc) - timedelta(seconds=1)})
    repository.save_proposal("EMP-001", item)
    with pytest.raises(ConflictError, match="expired"):
        repository.confirm_proposal("EMP-001", "AP-TEST", "idem-12345678")

