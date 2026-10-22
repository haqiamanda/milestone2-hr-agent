from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from src.repository import ConflictError
from tests.postgres_fixture import EMPLOYEE_ID, PROPOSAL_ID, PostgresTestDatabase


pytestmark = pytest.mark.postgres_integration


def _count_requests(database: PostgresTestDatabase) -> int:
    with database.connect() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT count(*) AS count FROM leave_requests WHERE proposal_id = %s",
                (PROPOSAL_ID,),
            )
            return int(cursor.fetchone()["count"])


def _proposal_markers(database: PostgresTestDatabase) -> dict:
    with database.connect() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT confirmed_at, confirmed_request_id,
                       confirmed_idempotency_key, confirmed_audit_reference
                FROM action_proposals
                WHERE proposal_id = %s
                """,
                (PROPOSAL_ID,),
            )
            return cursor.fetchone()


def test_first_confirmation_creates_one_request_and_all_markers(
    postgres_database: PostgresTestDatabase,
) -> None:
    postgres_database.reset_case()
    result = postgres_database.repository().confirm_proposal(
        EMPLOYEE_ID, PROPOSAL_ID, "idem-first-123456"
    )

    assert result.idempotent_replay is False
    assert result.status == "pending_manager_review"
    assert _count_requests(postgres_database) == 1
    markers = _proposal_markers(postgres_database)
    assert markers["confirmed_at"] is not None
    assert markers["confirmed_request_id"] == result.request_id
    assert markers["confirmed_idempotency_key"] == "idem-first-123456"
    assert markers["confirmed_audit_reference"] == result.audit_reference


def test_new_repository_replays_durable_result(
    postgres_database: PostgresTestDatabase,
) -> None:
    postgres_database.reset_case()
    first = postgres_database.repository().confirm_proposal(
        EMPLOYEE_ID, PROPOSAL_ID, "idem-replay-1234"
    )

    replay = postgres_database.repository().confirm_proposal(
        EMPLOYEE_ID, PROPOSAL_ID, "idem-replay-1234"
    )

    assert replay.idempotent_replay is True
    assert replay.request_id == first.request_id
    assert replay.audit_reference == first.audit_reference
    assert _count_requests(postgres_database) == 1


def test_different_key_conflicts_without_duplicate_request(
    postgres_database: PostgresTestDatabase,
) -> None:
    postgres_database.reset_case()
    postgres_database.repository().confirm_proposal(
        EMPLOYEE_ID, PROPOSAL_ID, "idem-original-1234"
    )

    with pytest.raises(ConflictError):
        postgres_database.repository().confirm_proposal(
            EMPLOYEE_ID, PROPOSAL_ID, "idem-other-12345"
        )

    assert _count_requests(postgres_database) == 1


def test_concurrent_same_key_has_one_original_and_one_replay(
    postgres_database: PostgresTestDatabase,
) -> None:
    postgres_database.reset_case()
    barrier = Barrier(2)

    def confirm():
        barrier.wait()
        return postgres_database.repository().confirm_proposal(
            EMPLOYEE_ID, PROPOSAL_ID, "idem-concurrent-123"
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: confirm(), range(2)))

    assert sorted(result.idempotent_replay for result in results) == [False, True]
    assert results[0].request_id == results[1].request_id
    assert results[0].audit_reference == results[1].audit_reference
    assert _count_requests(postgres_database) == 1


def test_concurrent_different_keys_has_one_success_and_one_conflict(
    postgres_database: PostgresTestDatabase,
) -> None:
    postgres_database.reset_case()
    barrier = Barrier(2)
    keys = ("idem-concurrent-a", "idem-concurrent-b")

    def confirm(key: str):
        barrier.wait()
        try:
            return ("success", postgres_database.repository().confirm_proposal(
                EMPLOYEE_ID, PROPOSAL_ID, key
            ))
        except ConflictError:
            return ("conflict", None)

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(confirm, keys))

    assert sorted(outcome[0] for outcome in outcomes) == ["conflict", "success"]
    assert _count_requests(postgres_database) == 1


def test_marker_failure_rolls_back_request_and_retry_succeeds(
    postgres_database: PostgresTestDatabase,
) -> None:
    postgres_database.reset_case()
    trigger_function = "test_fail_confirmation_marker"
    trigger_name = "test_fail_confirmation_marker_trigger"
    with postgres_database.connect() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f"""
                CREATE OR REPLACE FUNCTION {trigger_function}()
                RETURNS trigger
                LANGUAGE plpgsql
                AS $$
                BEGIN
                    RAISE EXCEPTION 'intentional marker failure';
                END;
                $$;
                CREATE TRIGGER {trigger_name}
                BEFORE UPDATE OF confirmed_at ON action_proposals
                FOR EACH ROW EXECUTE FUNCTION {trigger_function}();
                """
            )

    try:
        with pytest.raises(Exception, match="intentional marker failure"):
            postgres_database.repository().confirm_proposal(
                EMPLOYEE_ID, PROPOSAL_ID, "idem-rollback-123"
            )

        assert _count_requests(postgres_database) == 0
        markers = _proposal_markers(postgres_database)
        assert all(value is None for value in markers.values())
    finally:
        with postgres_database.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(f"DROP TRIGGER IF EXISTS {trigger_name} ON action_proposals")
                cursor.execute(f"DROP FUNCTION IF EXISTS {trigger_function}()")

    retry = postgres_database.repository().confirm_proposal(
        EMPLOYEE_ID, PROPOSAL_ID, "idem-rollback-123"
    )
    assert retry.idempotent_replay is False
    assert _count_requests(postgres_database) == 1
