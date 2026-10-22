from __future__ import annotations

import hashlib
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]


def chunk_policy(path: Path) -> list[dict[str, str]]:
    # TODO(M2): Parse front matter and deterministic section chunks with stable IDs.
    text = path.read_text(encoding="utf-8")
    parts = text.split("---", 2)
    if len(parts) != 3 or parts[0].strip():
        raise ValueError("invalid policy front matter")

    metadata: dict[str, str] = {}
    for line in parts[1].splitlines():
        if not line.strip():
            continue
        key, separator, value = line.partition(":")
        if not separator or not key.strip() or not value.strip():
            raise ValueError("invalid policy metadata")
        metadata[key.strip()] = value.strip()

    required = {"document_id", "title", "version", "effective_date"}
    if required - metadata.keys():
        raise ValueError("incomplete policy metadata")

    body = parts[2].strip()
    sections = list(re.finditer(r"(?m)^##\s+(.+?)\s*$", body))
    if not sections:
        raise ValueError("policy has no sections")

    chunks: list[dict[str, str]] = []
    for index, match in enumerate(sections):
        content_start = match.end()
        content_end = sections[index + 1].start() if index + 1 < len(sections) else len(body)
        content = body[content_start:content_end].strip()
        if not content:
            continue
        section = match.group(1).strip()
        digest_input = "\n".join(
            (metadata["document_id"], metadata["version"], section, content)
        )
        chunk_id = f"{metadata['document_id']}:{hashlib.sha256(digest_input.encode('utf-8')).hexdigest()[:16]}"
        chunks.append(
            {
                "chunk_id": chunk_id,
                "document_id": metadata["document_id"],
                "title": metadata["title"],
                "section": section,
                "version": metadata["version"],
                "effective_date": metadata["effective_date"],
                "content": content,
            }
        )
    if not chunks:
        raise ValueError("policy has no content")
    return chunks


def reset_and_seed() -> None:
    # TODO(M2): Create pgvector schema, seed synthetic identities/balances, batch embed, and upsert chunks.
    import json

    from .config import get_settings
    from .providers import OpenAIProvider

    settings = get_settings()
    if not settings.live_model_ready:
        raise RuntimeError(
            "live embedding is not configured; set OPENAI_API_KEY, "
            "EMBEDDING_MODEL, and AGENT_MODEL before seeding"
        )

    try:
        provider = OpenAIProvider(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            agent_model=settings.agent_model,
            embedding_model=settings.embedding_model,
            embedding_dimensions=settings.embedding_dimensions,
        )
    except NotImplementedError as exc:
        raise RuntimeError(
            "embedding provider is not implemented; policy chunks were not seeded"
        ) from exc

    try:
        import psycopg
    except ImportError as exc:
        raise RuntimeError("psycopg is required to seed PostgreSQL") from exc

    employees_path = ROOT / "data" / "employees.json"
    employees = json.loads(employees_path.read_text(encoding="utf-8"))
    policy_chunks = [
        chunk
        for path in sorted((ROOT / "data" / "policies").glob("*.md"))
        for chunk in chunk_policy(path)
    ]
    dimensions = settings.embedding_dimensions

    schema = f"""
        CREATE EXTENSION IF NOT EXISTS vector;

        CREATE TABLE IF NOT EXISTS employees (
            employee_id TEXT PRIMARY KEY,
            display_name TEXT NOT NULL,
            role TEXT NOT NULL CHECK (role IN ('employee', 'monitoring_admin')),
            token_hash TEXT NOT NULL UNIQUE
        );
        CREATE TABLE IF NOT EXISTS leave_balances (
            employee_id TEXT NOT NULL REFERENCES employees(employee_id),
            leave_type TEXT NOT NULL,
            available_days INTEGER NOT NULL CHECK (available_days >= 0),
            as_of_date DATE NOT NULL,
            PRIMARY KEY (employee_id, leave_type)
        );
        CREATE TABLE IF NOT EXISTS policy_chunks (
            chunk_id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL,
            title TEXT NOT NULL,
            section TEXT NOT NULL,
            version TEXT NOT NULL,
            effective_date DATE NOT NULL,
            content TEXT NOT NULL,
            embedding vector({dimensions}) NOT NULL,
            index_version TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS policy_chunks_document_idx
            ON policy_chunks (document_id, version);
        CREATE INDEX IF NOT EXISTS policy_chunks_search_idx
            ON policy_chunks USING GIN (to_tsvector('simple', content));
        CREATE TABLE IF NOT EXISTS action_proposals (
            proposal_id TEXT PRIMARY KEY,
            employee_id TEXT NOT NULL REFERENCES employees(employee_id),
            leave_type TEXT NOT NULL,
            start_date DATE NOT NULL,
            end_date DATE NOT NULL,
            business_days INTEGER NOT NULL CHECK (business_days > 0),
            reason TEXT,
            validation_summary JSONB NOT NULL,
            expires_at TIMESTAMPTZ NOT NULL,
            requires_confirmation BOOLEAN NOT NULL DEFAULT TRUE,
            confirmed_at TIMESTAMPTZ,
            confirmed_request_id TEXT,
            confirmed_idempotency_key TEXT,
            confirmed_audit_reference TEXT,
            CONSTRAINT action_proposals_confirmation_state_ck CHECK (
                (
                    confirmed_at IS NULL
                    AND confirmed_request_id IS NULL
                    AND confirmed_idempotency_key IS NULL
                    AND confirmed_audit_reference IS NULL
                )
                OR
                (
                    confirmed_at IS NOT NULL
                    AND confirmed_request_id IS NOT NULL
                    AND confirmed_idempotency_key IS NOT NULL
                    AND confirmed_audit_reference IS NOT NULL
                )
            )
        );
        CREATE TABLE IF NOT EXISTS leave_requests (
            request_id TEXT PRIMARY KEY,
            proposal_id TEXT REFERENCES action_proposals(proposal_id),
            employee_id TEXT NOT NULL REFERENCES employees(employee_id),
            leave_type TEXT NOT NULL,
            start_date DATE NOT NULL,
            end_date DATE NOT NULL,
            business_days INTEGER NOT NULL CHECK (business_days > 0),
            status TEXT NOT NULL CHECK (
                status IN ('pending_manager_review', 'approved', 'rejected', 'cancelled')
            ),
            created_at TIMESTAMPTZ NOT NULL
        );
        CREATE UNIQUE INDEX IF NOT EXISTS leave_requests_proposal_uidx
            ON leave_requests (proposal_id)
            WHERE proposal_id IS NOT NULL;
        CREATE TABLE IF NOT EXISTS tickets (
            ticket_id TEXT PRIMARY KEY,
            employee_id TEXT NOT NULL REFERENCES employees(employee_id),
            category TEXT NOT NULL,
            summary TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL
        );
        CREATE TABLE IF NOT EXISTS traces (
            trace_id TEXT PRIMARY KEY,
            employee_id TEXT NOT NULL,
            outcome TEXT NOT NULL,
            payload JSONB NOT NULL,
            created_at TIMESTAMPTZ NOT NULL
        );
    """

    with psycopg.connect(settings.database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(schema)
            for employee in employees:
                token_digest = hashlib.sha256(
                    employee["token"].encode("utf-8")
                ).hexdigest()
                cursor.execute(
                    """
                    INSERT INTO employees (employee_id, display_name, role, token_hash)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (employee_id) DO UPDATE SET
                        display_name = EXCLUDED.display_name,
                        role = EXCLUDED.role,
                        token_hash = EXCLUDED.token_hash
                    """,
                    (
                        employee["employee_id"],
                        employee["display_name"],
                        employee["role"],
                        token_digest,
                    ),
                )
                for leave_type, available_days in employee.get("balances", {}).items():
                    cursor.execute(
                        """
                        INSERT INTO leave_balances
                            (employee_id, leave_type, available_days, as_of_date)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (employee_id, leave_type) DO UPDATE SET
                            available_days = EXCLUDED.available_days,
                            as_of_date = EXCLUDED.as_of_date
                        """,
                        (
                            employee["employee_id"],
                            leave_type,
                            available_days,
                            "2026-01-01",
                        ),
                    )

            embeddings = provider.embed([chunk["content"] for chunk in policy_chunks])
            if len(embeddings) != len(policy_chunks):
                raise RuntimeError("embedding provider returned an unexpected batch size")
            for chunk, embedding in zip(policy_chunks, embeddings):
                if len(embedding) != dimensions:
                    raise RuntimeError(
                        f"embedding dimension mismatch for {chunk['chunk_id']}: "
                        f"expected {dimensions}, got {len(embedding)}"
                    )
                cursor.execute(
                    """
                    INSERT INTO policy_chunks
                        (chunk_id, document_id, title, section, version,
                         effective_date, content, embedding, index_version)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s::vector, %s)
                    ON CONFLICT (chunk_id) DO UPDATE SET
                        document_id = EXCLUDED.document_id,
                        title = EXCLUDED.title,
                        section = EXCLUDED.section,
                        version = EXCLUDED.version,
                        effective_date = EXCLUDED.effective_date,
                        content = EXCLUDED.content,
                        embedding = EXCLUDED.embedding,
                        index_version = EXCLUDED.index_version
                    """,
                    (
                        chunk["chunk_id"],
                        chunk["document_id"],
                        chunk["title"],
                        chunk["section"],
                        chunk["version"],
                        chunk["effective_date"],
                        chunk["content"],
                        "[" + ",".join(str(value) for value in embedding) + "]",
                        settings.policy_index_version,
                    ),
                )


if __name__ == "__main__":
    reset_and_seed()
