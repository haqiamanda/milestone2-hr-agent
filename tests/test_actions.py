from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from src.contracts import ActionProposal
from src.repository import ConflictError, MemoryRepository, PostgresRepository


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


def test_confirmation_is_single_use_after_replay(repository: MemoryRepository) -> None:
    repository.save_proposal("EMP-001", proposal())
    first = repository.confirm_proposal("EMP-001", "AP-TEST", "idem-12345678")

    replay = repository.confirm_proposal("EMP-001", "AP-TEST", "idem-12345678")

    assert replay.idempotent_replay is True
    assert replay.request_id == first.request_id
    assert len(repository.list_leave_requests("EMP-001")) == 1


def test_postgres_confirmation_rolls_back_before_marker_update(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = PostgresRepository(
        database_url="unused",
        embedding_dimensions=3,
    )

    row = {
        "proposal_id": "AP-TEST",
        "employee_id": "EMP-001",
        "leave_type": "annual",
        "start_date": date(2026, 11, 2),
        "end_date": date(2026, 11, 3),
        "business_days": 2,
        "reason": None,
        "validation_summary": ["valid"],
        "expires_at": datetime.now(timezone.utc) + timedelta(minutes=10),
        "requires_confirmation": True,
        "confirmed_request_id": None,
        "confirmed_idempotency_key": None,
        "confirmed_audit_reference": None,
    }

    class FailingCursor:
        def __init__(self, connection):
            self.connection = connection

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def execute(self, query, parameters):
            if "FOR UPDATE" in query:
                self.connection.row = row
            elif "INSERT INTO leave_requests" in query:
                self.connection.working_inserted = True
            elif "UPDATE action_proposals" in query:
                raise RuntimeError("marker update failed")

        def fetchone(self):
            return self.connection.row

        @property
        def rowcount(self):
            return 1

    class FailingConnection:
        def __init__(self):
            self.working_inserted = False
            self.persisted_inserted = False
            self.committed = False
            self.rolled_back = False
            self.row = None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            if exc_type is None:
                self.persisted_inserted = self.working_inserted
                self.committed = True
            else:
                self.working_inserted = False
                self.rolled_back = True
            return False

        def cursor(self):
            return FailingCursor(self)

    connection = FailingConnection()
    monkeypatch.setattr(repository, "_connect", lambda: connection)

    with pytest.raises(RuntimeError, match="marker update failed"):
        repository.confirm_proposal("EMP-001", "AP-TEST", "idem-12345678")

    assert connection.working_inserted is False
    assert connection.persisted_inserted is False
    assert connection.committed is False
    assert connection.rolled_back is True
    assert not hasattr(repository, "_idempotency")


def test_expired_proposal_is_rejected(repository: MemoryRepository) -> None:
    item = proposal().model_copy(update={"expires_at": datetime.now(timezone.utc) - timedelta(seconds=1)})
    repository.save_proposal("EMP-001", item)
    with pytest.raises(ConflictError, match="expired"):
        repository.confirm_proposal("EMP-001", "AP-TEST", "idem-12345678")
