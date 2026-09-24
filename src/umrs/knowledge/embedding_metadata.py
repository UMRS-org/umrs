"""Embedding metadata helpers for knowledge artifact rows (stability S10).

The durable re-index pattern (``JobType.KNOWLEDGE_REINDEX``):

1. **sweep** — :func:`select_stale_artifacts` pages active artifacts whose
   embedding metadata is missing or targets another model;
2. **compute** — submissions ride the canonical ``INDEX_BUILD`` durable
   path (EMBEDDING_BATCH dispatch is explicitly unimplemented, #216 —
   embeddings are computed into the RAG index);
3. **stamp** — :func:`stamp_embeddings_current` records model/dim/time per
   completed artifact set; a model bump first calls
   :func:`mark_embeddings_stale` so the next sweep re-selects them.

Registered payload contract for KNOWLEDGE_REINDEX jobs:
``{"target_model": str, "index_name": str}`` — both required; the spec
never re-derives defaults client-side (deliberate, operator-visible).

Dispatcher wire-up (KNOWLEDGE_REINDEX → sweep/submit/stamp) is a
recorded follow-up; the registered type exists so no caller scripts
re-indexing ad hoc (S9 premise).
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any

TARGET_MODEL_KEY = "target_model"
INDEX_NAME_KEY = "index_name"
REINDEX_PAYLOAD_KEYS = frozenset({TARGET_MODEL_KEY, INDEX_NAME_KEY})

VALID_EMBEDDING_STATUSES = frozenset({"pending", "current", "stale"})


class EmbeddingMetadataError(ValueError):
    """Invalid payload, identifier set, or status."""


def validate_reindex_payload(spec: dict) -> dict:
    """Enforce the registered KNOWLEDGE_REINDEX payload contract."""
    if not isinstance(spec, dict):
        raise EmbeddingMetadataError("reindex spec must be a dict")
    missing = REINDEX_PAYLOAD_KEYS - spec.keys()
    if missing:
        raise EmbeddingMetadataError(f"reindex spec missing keys: {sorted(missing)}")
    model = str(spec.get(TARGET_MODEL_KEY) or "").strip()
    index = str(spec.get(INDEX_NAME_KEY) or "").strip()
    if not model:
        raise EmbeddingMetadataError("target_model must be non-empty")
    if not index:
        raise EmbeddingMetadataError("index_name must be non-empty")
    return {TARGET_MODEL_KEY: model, INDEX_NAME_KEY: index}


def select_stale_artifacts(conn, target_model: str, *, limit: int = 500) -> list[str]:
    """Active artifact ids needing (re)embedding under ``target_model``."""
    model = str(target_model or "").strip()
    if not model:
        raise EmbeddingMetadataError("target_model must be non-empty")
    cur = conn.cursor()
    cur.execute(
        "SELECT artifact_id FROM knowledge_artifacts "
        "WHERE status = 'active' "
        "AND (embedding_status <> 'current' OR embedding_model <> %s) "
        "ORDER BY created_at LIMIT %s",
        (model, int(limit)),
    )
    return [row[0] for row in cur.fetchall()]


def stamp_embeddings_current(
    conn,
    artifact_ids: Sequence[str],
    *,
    model: str,
    dim: int | None = None,
    embedded_at: str | None = None,
) -> int:
    """Mark a completed batch's artifacts current. Returns updated count.

    ``dim`` may be None (unknown/not recorded); when given it must be
    positive (the column stores NULL otherwise — never a silent 0).
    """
    ids = [str(a).strip() for a in artifact_ids if str(a).strip()]
    if not ids:
        return 0
    model = str(model or "").strip()
    if not model:
        raise EmbeddingMetadataError("model must be non-empty")
    if dim is not None and int(dim) <= 0:
        raise EmbeddingMetadataError("dim must be positive when given")
    dim_value = None if dim is None else int(dim)
    ts = embedded_at or datetime.now(timezone.utc).isoformat()
    cur = conn.cursor()
    cur.execute(
        "UPDATE knowledge_artifacts "
        "SET embedding_model = %s, embedding_dim = %s, embedded_at = %s, "
        "embedding_status = 'current', updated_at = %s "
        "WHERE artifact_id = ANY(%s)",
        (model, dim_value, ts, ts, ids),
    )
    return int(getattr(cur, "rowcount", 0) or 0)


def mark_embeddings_stale(conn, *, previous_model: str | None = None) -> int:
    """Flag embeddings as stale after a model change.

    With ``previous_model`` set, only rows embedding under OTHER models are
    flagged (the just-bumped-away-from model's rows stay current); without
    it, every active current row is flagged (full re-index).
    """
    ts = datetime.now(timezone.utc).isoformat()
    cur = conn.cursor()
    if previous_model is not None:
        cur.execute(
            "UPDATE knowledge_artifacts SET embedding_status = 'stale', updated_at = %s "
            "WHERE status = 'active' AND embedding_model <> %s",
            (ts, previous_model),
        )
    else:
        cur.execute(
            "UPDATE knowledge_artifacts SET embedding_status = 'stale', updated_at = %s "
            "WHERE status = 'active' AND embedding_status = 'current'",
            (ts,),
        )
    return int(getattr(cur, "rowcount", 0) or 0)


async def stamp_artifact_embedded(
    pool: Any, artifact_id: str, *, model: str, dim: int
) -> None:
    """Mark ONE artifact current after its vectors were actually written
    (KNOWLEDGE_EMBED — the only honest stamper; asyncpg pool)."""
    if not model or dim <= 0:
        raise EmbeddingMetadataError(
            "stamp needs the producing model and a positive dim"
        )
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE knowledge_artifacts "
            "SET embedding_model = $1, embedding_dim = $2, embedded_at = now(), "
            "embedding_status = 'current', updated_at = now() "
            "WHERE artifact_id = $3",
            model,
            dim,
            artifact_id,
        )
