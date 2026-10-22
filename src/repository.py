"""Repository contracts. Implement MemoryRepository first, then PostgreSQL/pgvector."""
from __future__ import annotations

from datetime import date, datetime, timezone
import hashlib
import json
from threading import RLock
from typing import Any, Protocol
from uuid import uuid4

import psycopg
from psycopg.rows import dict_row
from .security import sanitize_trace, validate_privacy
from .monitoring import actionable_alerts, memory_metrics

from .contracts import (
    ActionProposal,
    ActionResult,
    AuthContext,
    LeaveBalance,
    LeaveRequestView,
    PolicyChunk,
)


class Repository(Protocol):
    def health(self) -> bool: ...
    def readiness(self, index_version: str) -> dict[str, bool]: ...
    def authenticate(self, token: str) -> AuthContext | None: ...
    def list_employee_identities(self) -> list[tuple[str, str]]: ...
    def lexical_search(self, query: str, limit: int) -> list[PolicyChunk]: ...
    def vector_search(self, embedding: list[float], limit: int) -> list[PolicyChunk]: ...
    def get_balances(self, employee_id: str) -> list[LeaveBalance]: ...
    def has_overlap(self, employee_id: str, start: date, end: date) -> bool: ...
    def save_proposal(self, employee_id: str, proposal: ActionProposal) -> None: ...
    def get_proposal(self, proposal_id: str) -> tuple[str, ActionProposal] | None: ...
    def confirm_proposal(self, employee_id: str, proposal_id: str, idempotency_key: str) -> ActionResult: ...
    def list_leave_requests(self, employee_id: str) -> list[LeaveRequestView]: ...
    def create_ticket(self, employee_id: str, category: str, summary: str) -> str: ...
    def save_trace(self, trace: dict[str, Any]) -> None: ...
    def get_trace(self, trace_id: str) -> dict[str, Any] | None: ...
    def monitoring_summary(self) -> dict[str, Any]: ...


class ConflictError(RuntimeError):
    pass


class MemoryRepository:
    def __init__(self) -> None:
        self.chunks: list[PolicyChunk] = []
        self._identities = {
            "token-alya": AuthContext(
                employee_id="EMP-001", display_name="Alya Pratama", role="employee"
            ),
            "token-bima": AuthContext(
                employee_id="EMP-002", display_name="Bima Santoso", role="employee"
            ),
            "token-admin": AuthContext(
                employee_id="ADM-001",
                display_name="Monitoring Admin",
                role="monitoring_admin",
            ),
        }
        as_of = date(2026, 1, 1)
        self._balances = {
            "EMP-001": [
                LeaveBalance(leave_type="annual", available_days=8, as_of_date=as_of),
                LeaveBalance(leave_type="sick", available_days=10, as_of_date=as_of),
                LeaveBalance(leave_type="parental", available_days=60, as_of_date=as_of),
            ],
            "EMP-002": [
                LeaveBalance(leave_type="annual", available_days=3, as_of_date=as_of),
                LeaveBalance(leave_type="sick", available_days=10, as_of_date=as_of),
                LeaveBalance(leave_type="parental", available_days=60, as_of_date=as_of),
            ],
        }
        self._proposals: dict[str, tuple[str, ActionProposal]] = {}
        self._requests: dict[str, list[LeaveRequestView]] = {}
        self._confirmed_proposals: dict[str, tuple[str, ActionResult]] = {}
        self._confirmation_lock = RLock()
        self._tickets: dict[str, dict[str, str]] = {}
        self._traces: dict[str, dict[str, Any]] = {}

    def health(self) -> bool:
        return True

    def readiness(self, index_version: str) -> dict[str, bool]:
        # In-memory dependencies are used by deterministic tests only.
        return {"database": True, "schema": True, "index": bool(self.chunks)}

    def authenticate(self, token: str) -> AuthContext | None:
        auth = self._identities.get(token)
        return auth.model_copy(deep=True) if auth else None

    def list_employee_identities(self) -> list[tuple[str, str]]:
        return [(auth.employee_id, auth.display_name) for auth in self._identities.values()]

    def lexical_search(self, query: str, limit: int) -> list[PolicyChunk]:
        terms = set(query.casefold().split())
        scored = []
        for chunk in self.chunks:
            text = " ".join((chunk.title, chunk.section, chunk.content)).casefold()
            score = sum(term in text for term in terms)
            if score:
                scored.append((score, chunk))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [chunk for _, chunk in scored[: max(0, limit)]]

    def vector_search(self, embedding: list[float], limit: int) -> list[PolicyChunk]:
        return []

    def get_balances(self, employee_id: str) -> list[LeaveBalance]:
        return [item.model_copy(deep=True) for item in self._balances.get(employee_id, [])]

    def has_overlap(self, employee_id: str, start: date, end: date) -> bool:
        for request in self._requests.get(employee_id, []):
            if request.status not in {"cancelled", "rejected"} and not (
                end < request.start_date or start > request.end_date
            ):
                return True
        return False

    def save_proposal(self, employee_id: str, proposal: ActionProposal) -> None:
        if proposal.proposal_id in self._proposals:
            raise ConflictError("proposal already exists")
        self._proposals[proposal.proposal_id] = (employee_id, proposal.model_copy(deep=True))

    def get_proposal(self, proposal_id: str) -> tuple[str, ActionProposal] | None:
        item = self._proposals.get(proposal_id)
        if item is None:
            return None
        return item[0], item[1].model_copy(deep=True)

    def confirm_proposal(
        self, employee_id: str, proposal_id: str, idempotency_key: str
    ) -> ActionResult:
        with self._confirmation_lock:
            item = self._proposals.get(proposal_id)
            if item is None or item[0] != employee_id:
                raise KeyError(proposal_id)

            confirmed = self._confirmed_proposals.get(proposal_id)
            if confirmed is not None:
                confirmed_key, result = confirmed
                if confirmed_key != idempotency_key:
                    raise ConflictError("idempotency key conflict")
                return result.model_copy(update={"idempotent_replay": True})

            proposal = item[1]
            expires_at = proposal.expires_at
            if expires_at.tzinfo is None:
                expires_at = expires_at.replace(tzinfo=timezone.utc)
            if expires_at <= datetime.now(timezone.utc):
                raise ConflictError("proposal expired")

            request_id = f"LR-{uuid4().hex[:12]}"
            request = LeaveRequestView(
                request_id=request_id,
                leave_type=proposal.leave_type,
                start_date=proposal.start_date,
                end_date=proposal.end_date,
                business_days=proposal.business_days,
                status="pending_manager_review",
                created_at=datetime.now(timezone.utc),
            )
            result = ActionResult(
                request_id=request_id,
                idempotent_replay=False,
                audit_reference=f"AUD-{uuid4().hex[:12]}",
            )
            self._requests.setdefault(employee_id, []).append(request)
            self._confirmed_proposals[proposal_id] = (idempotency_key, result)
            return result

    def list_leave_requests(self, employee_id: str) -> list[LeaveRequestView]:
        return [
            request.model_copy(deep=True)
            for request in self._requests.get(employee_id, [])
        ]

    def create_ticket(self, employee_id: str, category: str, summary: str) -> str:
        validate_privacy(category)
        validate_privacy(summary)
        ticket_id = f"T-{uuid4().hex[:12]}"
        self._tickets[ticket_id] = {
            "employee_id": employee_id,
            "category": category,
            "summary": summary,
        }
        return ticket_id

    def save_trace(self, trace: dict[str, Any]) -> None:
        trace = sanitize_trace(trace)
        trace_id = trace.get("trace_id")
        if not trace_id:
            raise ValueError("trace_id is required")
        self._traces[trace_id] = dict(trace)

    def get_trace(self, trace_id: str) -> dict[str, Any] | None:
        trace = self._traces.get(trace_id)
        return dict(trace) if trace is not None else None

    def monitoring_summary(self) -> dict[str, Any]:
        metrics = memory_metrics(list(self._traces.values()))
        return {**metrics, "alerts": actionable_alerts(metrics)}


class PostgresRepository:
    def __init__(self, database_url: str, embedding_dimensions: int) -> None:
        self.database_url = database_url
        self.embedding_dimensions = embedding_dimensions

    def _connect(self):
        return psycopg.connect(self.database_url, row_factory=dict_row)

    @staticmethod
    def _proposal_from_row(row: dict[str, Any]) -> ActionProposal:
        return ActionProposal(
            proposal_id=row["proposal_id"],
            leave_type=row["leave_type"],
            start_date=row["start_date"],
            end_date=row["end_date"],
            business_days=row["business_days"],
            reason=row["reason"],
            validation_summary=row["validation_summary"],
            expires_at=row["expires_at"],
            requires_confirmation=row["requires_confirmation"],
        )

    def health(self) -> bool:
        try:
            with self._connect() as connection:
                with connection.cursor() as cursor:
                    cursor.execute("SELECT 1")
                    return cursor.fetchone() is not None
        except psycopg.Error:
            return False

    def readiness(self, index_version: str) -> dict[str, bool]:
        """Check the runtime schema and populated policy index without writes."""
        checks = {"database": False, "schema": False, "index": False}
        required_columns = {
            "employees": "employee_id, display_name, role, token_hash",
            "leave_balances": "employee_id, leave_type, available_days, as_of_date",
            "policy_chunks": "chunk_id, document_id, title, section, version, effective_date, content, embedding, index_version",
            "action_proposals": "proposal_id, employee_id, leave_type, start_date, end_date, business_days, reason, validation_summary, expires_at, requires_confirmation, confirmed_at, confirmed_request_id, confirmed_idempotency_key, confirmed_audit_reference",
            "leave_requests": "request_id, proposal_id, employee_id, leave_type, start_date, end_date, business_days, status, created_at",
            "tickets": "ticket_id, employee_id, category, summary, created_at",
            "traces": "trace_id, employee_id, outcome, payload, created_at",
        }
        try:
            with self._connect() as connection:
                with connection.cursor() as cursor:
                    cursor.execute("SET TRANSACTION READ ONLY")
                    cursor.execute("SELECT 1")
                    checks["database"] = cursor.fetchone() is not None
                    # Identifiers below are fixed source constants, never input.
                    for table, columns in required_columns.items():
                        cursor.execute(f"SELECT {columns} FROM {table} LIMIT 0")
                    cursor.execute("SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector') AS available")
                    if not cursor.fetchone()["available"]:
                        return checks
                    checks["schema"] = True
                    cursor.execute(
                        """
                        SELECT EXISTS (
                            SELECT 1 FROM policy_chunks
                            WHERE index_version = %s AND vector_dims(embedding) = %s
                        ) AND EXISTS (
                            SELECT 1 FROM pg_index i
                            JOIN pg_class c ON c.oid = i.indexrelid
                            WHERE i.indrelid = 'policy_chunks'::regclass
                              AND c.relname = 'policy_chunks_search_idx'
                              AND i.indisvalid AND i.indisready
                        ) AS available
                        """,
                        (index_version, self.embedding_dimensions),
                    )
                    checks["index"] = bool(cursor.fetchone()["available"])
        except Exception:
            # Never return driver messages, URLs, schema rows or credentials.
            pass
        return checks

    def authenticate(self, token: str) -> AuthContext | None:
        token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
        try:
            with self._connect() as connection:
                with connection.cursor() as cursor:
                    cursor.execute(
                        """
                        SELECT employee_id, display_name, role
                        FROM employees
                        WHERE token_hash = %s
                        """,
                        (token_hash,),
                    )
                    row = cursor.fetchone()
        except psycopg.Error:
            return None
        return AuthContext(**row) if row else None

    def list_employee_identities(self) -> list[tuple[str, str]]:
        """Backend-only directory for request authorization, never a model tool."""
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT employee_id, display_name FROM employees")
                rows = cursor.fetchall()
        return [(row["employee_id"], row["display_name"]) for row in rows]

    def lexical_search(self, query: str, limit: int) -> list[PolicyChunk]:
        if limit <= 0:
            return []
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT chunk_id, document_id, title, section, version,
                           effective_date, content,
                           ts_rank(
                               to_tsvector('simple', title || ' ' || section || ' ' || content),
                               websearch_to_tsquery('simple', %s)
                           ) AS score
                    FROM policy_chunks
                    WHERE to_tsvector(
                        'simple', title || ' ' || section || ' ' || content
                    ) @@ websearch_to_tsquery('simple', %s)
                    ORDER BY score DESC, chunk_id
                    LIMIT %s
                    """,
                    (query, query, limit),
                )
                rows = cursor.fetchall()
        return [PolicyChunk(**row) for row in rows]

    def vector_search(self, embedding: list[float], limit: int) -> list[PolicyChunk]:
        if limit <= 0 or len(embedding) != self.embedding_dimensions:
            return []
        vector = "[" + ",".join(str(value) for value in embedding) + "]"
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT chunk_id, document_id, title, section, version,
                           effective_date, content,
                           1 - (embedding <=> %s::vector) AS score
                    FROM policy_chunks
                    ORDER BY embedding <=> %s::vector, chunk_id
                    LIMIT %s
                    """,
                    (vector, vector, limit),
                )
                rows = cursor.fetchall()
        return [PolicyChunk(**row) for row in rows]

    def get_balances(self, employee_id: str) -> list[LeaveBalance]:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT leave_type, available_days, as_of_date
                    FROM leave_balances
                    WHERE employee_id = %s
                    ORDER BY leave_type
                    """,
                    (employee_id,),
                )
                rows = cursor.fetchall()
        return [LeaveBalance(**row) for row in rows]

    def has_overlap(self, employee_id: str, start: date, end: date) -> bool:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT EXISTS (
                        SELECT 1
                        FROM leave_requests
                        WHERE employee_id = %s
                          AND status NOT IN ('cancelled', 'rejected')
                          AND start_date <= %s
                          AND end_date >= %s
                    ) AS has_overlap
                    """,
                    (employee_id, end, start),
                )
                row = cursor.fetchone()
        return bool(row["has_overlap"])

    def save_proposal(self, employee_id: str, proposal: ActionProposal) -> None:
        try:
            with self._connect() as connection:
                with connection.cursor() as cursor:
                    cursor.execute(
                        """
                        INSERT INTO action_proposals
                            (proposal_id, employee_id, leave_type, start_date, end_date,
                             business_days, reason, validation_summary, expires_at,
                             requires_confirmation)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            proposal.proposal_id,
                            employee_id,
                            proposal.leave_type,
                            proposal.start_date,
                            proposal.end_date,
                            proposal.business_days,
                            proposal.reason,
                            json.dumps(proposal.validation_summary),
                            proposal.expires_at,
                            proposal.requires_confirmation,
                        ),
                    )
        except psycopg.errors.UniqueViolation as exc:
            raise ConflictError("proposal already exists") from exc

    def get_proposal(self, proposal_id: str) -> tuple[str, ActionProposal] | None:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT proposal_id, employee_id, leave_type, start_date, end_date,
                           business_days, reason, validation_summary, expires_at,
                           requires_confirmation
                    FROM action_proposals
                    WHERE proposal_id = %s
                    """,
                    (proposal_id,),
                )
                row = cursor.fetchone()
        if row is None:
            return None
        return row["employee_id"], self._proposal_from_row(row)

    def confirm_proposal(
        self, employee_id: str, proposal_id: str, idempotency_key: str
    ) -> ActionResult:
        now = datetime.now(timezone.utc)
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT proposal_id, employee_id, leave_type, start_date, end_date,
                           business_days, reason, validation_summary, expires_at,
                           requires_confirmation, confirmed_at, confirmed_request_id,
                           confirmed_idempotency_key, confirmed_audit_reference
                    FROM action_proposals
                    WHERE proposal_id = %s
                    FOR UPDATE
                    """,
                    (proposal_id,),
                )
                row = cursor.fetchone()
                if row is None or row["employee_id"] != employee_id:
                    raise KeyError(proposal_id)

                confirmed_request_id = row["confirmed_request_id"]
                if confirmed_request_id is not None:
                    confirmed_key = row["confirmed_idempotency_key"]
                    if confirmed_key != idempotency_key:
                        raise ConflictError("idempotency key conflict")
                    return ActionResult(
                        request_id=confirmed_request_id,
                        idempotent_replay=True,
                        audit_reference=row["confirmed_audit_reference"],
                    )

                expires_at = row["expires_at"]
                if expires_at.tzinfo is None:
                    expires_at = expires_at.replace(tzinfo=timezone.utc)
                if expires_at <= now:
                    raise ConflictError("proposal expired")

                request_id = f"LR-{uuid4().hex[:12]}"
                audit_reference = f"AUD-{uuid4().hex[:12]}"
                cursor.execute(
                    """
                    INSERT INTO leave_requests
                        (request_id, proposal_id, employee_id, leave_type, start_date,
                         end_date, business_days, status, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s,
                            'pending_manager_review', %s)
                    """,
                    (
                        request_id,
                        proposal_id,
                        employee_id,
                        row["leave_type"],
                        row["start_date"],
                        row["end_date"],
                        row["business_days"],
                        now,
                    ),
                )
                cursor.execute(
                    """
                    UPDATE action_proposals
                    SET confirmed_at = %s,
                        confirmed_request_id = %s,
                        confirmed_idempotency_key = %s,
                        confirmed_audit_reference = %s
                    WHERE proposal_id = %s
                    """,
                    (
                        now,
                        request_id,
                        idempotency_key,
                        audit_reference,
                        proposal_id,
                    ),
                )
                if cursor.rowcount != 1:
                    raise RuntimeError("proposal confirmation marker update failed")

        return ActionResult(
            request_id=request_id,
            idempotent_replay=False,
            audit_reference=audit_reference,
        )

    def list_leave_requests(self, employee_id: str) -> list[LeaveRequestView]:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT request_id, leave_type, start_date, end_date,
                           business_days, status, created_at
                    FROM leave_requests
                    WHERE employee_id = %s
                    ORDER BY created_at, request_id
                    """,
                    (employee_id,),
                )
                rows = cursor.fetchall()
        return [LeaveRequestView(**row) for row in rows]

    def create_ticket(self, employee_id: str, category: str, summary: str) -> str:
        validate_privacy(category)
        validate_privacy(summary)
        ticket_id = f"T-{uuid4().hex[:12]}"
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO tickets
                        (ticket_id, employee_id, category, summary, created_at)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (ticket_id, employee_id, category, summary, datetime.now(timezone.utc)),
                )
        return ticket_id

    def save_trace(self, trace: dict[str, Any]) -> None:
        trace = sanitize_trace(trace)
        trace_id = trace.get("trace_id")
        employee_id = trace.get("employee_id")
        outcome = trace.get("outcome")
        if not trace_id or not employee_id or not outcome:
            raise ValueError("trace_id, employee_id, and outcome are required")
        payload = dict(trace)
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO traces (trace_id, employee_id, outcome, payload, created_at)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (trace_id) DO UPDATE SET
                        employee_id = EXCLUDED.employee_id,
                        outcome = EXCLUDED.outcome,
                        payload = EXCLUDED.payload
                    """,
                    (trace_id, employee_id, outcome, json.dumps(payload), datetime.now(timezone.utc)),
                )

    def get_trace(self, trace_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT payload FROM traces WHERE trace_id = %s",
                    (trace_id,),
                )
                row = cursor.fetchone()
        return dict(row["payload"]) if row else None

    def monitoring_summary(self) -> dict[str, Any]:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT
                        COUNT(*) AS run_count,
                        COALESCE(
                            AVG(CASE
                                WHEN payload->>'error' = 'medical_privacy_blocked' THEN 0.0
                                WHEN outcome = 'error'
                                    OR COALESCE(payload->>'error', '') <> ''
                                    OR EXISTS (
                                        SELECT 1 FROM jsonb_array_elements(
                                            COALESCE(payload->'tool_calls', '[]'::jsonb)
                                        ) AS tool_call
                                        WHERE tool_call->>'status' = 'error'
                                    ) THEN 1.0
                                ELSE 0.0
                            END),
                            0.0
                        ) AS error_rate,
                        COALESCE(
                            percentile_cont(0.95) WITHIN GROUP (
                                ORDER BY COALESCE((payload->>'latency_ms')::double precision, 0)
                            ),
                            0
                        ) AS p95_latency_ms,
                        COALESCE(
                            SUM(jsonb_array_length(COALESCE(payload->'tool_calls', '[]'::jsonb))),
                            0
                        ) AS tool_calls
                    FROM traces
                    """
                )
                row = cursor.fetchone()
        metrics = {
            "run_count": int(row["run_count"]),
            "error_rate": float(row["error_rate"]),
            "p95_latency_ms": round(row["p95_latency_ms"]),
            "tool_calls": int(row["tool_calls"]),
        }
        return {**metrics, "alerts": actionable_alerts(metrics)}
