from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def chunk_policy(path: Path) -> list[dict[str, str]]:
    # TODO(M2): Parse front matter and deterministic section chunks with stable IDs.
    raise NotImplementedError


def reset_and_seed() -> None:
    # TODO(M2): Create pgvector schema, seed synthetic identities/balances, batch embed, and upsert chunks.
    raise NotImplementedError


if __name__ == "__main__":
    reset_and_seed()

