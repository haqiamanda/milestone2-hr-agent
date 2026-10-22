"""Opt-in, read-only observations on the explicitly verified Compose QA network.

No dotenv loading, seed, migration, confirmation, arbitrary SQL, or report export.
Raw synthetic business state is compared in memory by the evaluator, never printed.
"""
from contextlib import contextmanager
import hashlib
import os
import re

import psycopg
from psycopg.rows import dict_row


QA_DATABASE_URL = "postgresql://hr_app:hr_app@postgres:5432/hr_assistant"


class QaStateReader:
    def __init__(self, token: str, project: str, *, connect=None, environ=None):
        # Caller must independently verify Docker project/network/volume labels.
        # The extra environment acknowledgement prevents accidental default use.
        environment = os.environ if environ is None else environ
        if (not re.fullmatch(r"hr-qa-[0-9a-f]{12}", project)
                or environment.get("HR_QA_PROJECT") != project):
            raise ValueError("Explicit verified QA project acknowledgement required")
        if environment.get("DATABASE_URL") != QA_DATABASE_URL:
            raise ValueError("QA observer requires the exact internal Compose QA database")
        self._connect = connect or psycopg.connect
        self._token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
        self.employee_id = None

    @contextmanager
    def _cursor(self):
        with self._connect(
            QA_DATABASE_URL, connect_timeout=3, row_factory=dict_row,
            options="-c default_transaction_read_only=on -c statement_timeout=3000",
        ) as connection:
            with connection.cursor() as cursor:
                cursor.execute("SHOW transaction_read_only")
                if cursor.fetchone()["transaction_read_only"] != "on":
                    raise RuntimeError("QA evidence transaction must be read-only")
                yield cursor

    def snapshot(self):
        with self._cursor() as cursor:
            cursor.execute(
                "SELECT employee_id, role FROM employees WHERE token_hash = %s",
                (self._token_hash,),
            )
            identity = cursor.fetchone()
            if not identity or identity["role"] != "employee":
                raise ValueError("QA evaluation requires an authenticated employee")
            if self.employee_id and identity["employee_id"] != self.employee_id:
                raise ValueError("QA observer identity changed")
            self.employee_id = identity["employee_id"]
            business = {}
            # Fixed table identifiers, caller-bound parameters, no credential columns.
            for table, key in (
                ("leave_balances", "leave_type"), ("action_proposals", "proposal_id"),
                ("leave_requests", "request_id"), ("tickets", "ticket_id"),
            ):
                cursor.execute(
                    f"SELECT to_jsonb(t) AS record FROM {table} t "
                    f"WHERE employee_id = %s ORDER BY {key}", (self.employee_id,),
                )
                business[table] = [row["record"] for row in cursor.fetchall()]
        return {"employee_id": self.employee_id, "business": business}

    def trace(self, trace_id):
        if (not self.employee_id or not isinstance(trace_id, str)
                or not re.fullmatch(r"TR-[a-fA-F0-9]{1,100}", trace_id)):
            return None
        with self._cursor() as cursor:
            cursor.execute(
                "SELECT payload FROM traces WHERE trace_id = %s AND employee_id = %s",
                (trace_id, self.employee_id),
            )
            row = cursor.fetchone()
        return row["payload"] if row else None

    def business_snapshot(self):
        """QA-only global comparison; raw rows must never be exported.

        Includes foreign owners so a caller-scoped snapshot cannot conceal an
        injected write. Traces and retrieval caches are technical side effects.
        """
        business = {}
        with self._cursor() as cursor:
            for table, key in (
                ("leave_balances", "employee_id, leave_type"),
                ("action_proposals", "proposal_id"),
                ("leave_requests", "request_id"), ("tickets", "ticket_id"),
            ):
                cursor.execute(f"SELECT to_jsonb(t) AS record FROM {table} t ORDER BY {key}")
                business[table] = [row["record"] for row in cursor.fetchall()]
        return business
