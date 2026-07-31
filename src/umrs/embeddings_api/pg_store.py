"""Postgres-backed embedding store (durable K1 implementation).

Implements the same interface as :class:`InMemoryEmbeddingStore` against
the ``sat_embeddings`` table (migration 0060) using pgvector for cosine
similarity search. The in-memory store remains the reference
implementation and test fake.
"""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any

from . import (
    EmbedderProtocol,
    Embedding,
    EmbeddingMatch,
    EmbeddingStatus,
    hash_based_vector,
)

logger = logging.getLogger(__name__)

_COLUMNS = (
    "embedding_id, project_id, object_id, model, vector::text AS vector, "
    "text, metadata, status, created_at, updated_at"
)


def _vector_literal(vector: list[float]) -> str:
    return "[" + ",".join(repr(float(v)) for v in vector) + "]"


def _row_to_embedding(row: dict[str, Any]) -> Embedding:
    data = dict(row)
    if isinstance(data.get("metadata"), str):
        data["metadata"] = json.loads(data["metadata"])
    vector = data.get("vector")
    if isinstance(vector, str):
        vector = [float(x) for x in vector.strip("[]").split(",") if x]
    data["vector"] = vector or []
    return Embedding.from_dict(data)


class PgEmbeddingStore:
    """Durable embedding store backed by Postgres + pgvector."""

    def __init__(
        self,
        db_pool: Any,
        embedder: EmbedderProtocol | None = None,
    ) -> None:
        self._pool = db_pool
        self._embedder = embedder

    @contextmanager
    def _conn(self):
        conn = self._pool.getconn()
        try:
            with conn:
                yield conn
        finally:
            self._pool.putconn(conn)

    @contextmanager
    def _cursor(self):
        from psycopg2.extras import RealDictCursor

        with self._conn() as conn:
            yield conn.cursor(cursor_factory=RealDictCursor)

    def _embed(self, text: str) -> list[float]:
        if self._embedder:
            return self._embedder.embed(text)
        return hash_based_vector(text)

    def create_embedding(
        self,
        *,
        project_id: str,
        object_id: str,
        text: str,
        model: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Create a new embedding and return its embedding_id.

        With no embedder configured a deterministic hash vector is used
        (testing fallback only), matching the in-memory store.
        """
        vector = self._embed(text)
        resolved_model = model or (
            self._embedder.model_name if self._embedder else "hash-based"
        )
        embedding = Embedding(
            project_id=project_id,
            object_id=object_id,
            model=resolved_model,
            vector=vector,
            text=text,
            metadata=metadata or {},
        )
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO sat_embeddings ("
                "embedding_id, project_id, object_id, model, vector, text, "
                "metadata, status, created_at, updated_at) "
                "VALUES (%s, %s, %s, %s, %s::vector, %s, %s::jsonb, %s, %s, %s)",
                (
                    embedding.embedding_id,
                    embedding.project_id,
                    embedding.object_id,
                    embedding.model,
                    _vector_literal(embedding.vector),
                    embedding.text,
                    json.dumps(embedding.metadata),
                    embedding.status.value,
                    embedding.created_at,
                    embedding.updated_at,
                ),
            )
        return embedding.embedding_id

    def get_embedding(self, embedding_id: str) -> Embedding | None:
        """Get an embedding by ID, or None."""
        with self._cursor() as cur:
            cur.execute(
                f"SELECT {_COLUMNS} FROM sat_embeddings WHERE embedding_id = %s",
                (embedding_id,),
            )
            row = cur.fetchone()
        return _row_to_embedding(row) if row else None

    def search_similar(
        self,
        *,
        query_embedding: list[float],
        top_k: int = 10,
        project_id: str | None = None,
    ) -> list[EmbeddingMatch]:
        """Cosine-similarity search (ACTIVE rows), score clamped to [0, 1].

        pgvector ``<=>`` is cosine distance; ``1 - distance`` reproduces the
        in-memory store's cosine similarity, clamped identically.
        """
        clauses = ["status = %s"]
        # Placeholder order: query vector (SELECT), then WHERE, then LIMIT.
        params: list[Any] = [
            _vector_literal(query_embedding),
            EmbeddingStatus.ACTIVE.value,
        ]
        if project_id:
            clauses.append("project_id = %s")
            params.append(project_id)
        params.append(top_k)
        with self._cursor() as cur:
            cur.execute(
                "SELECT embedding_id, object_id, text, metadata, "
                "greatest(0.0, least(1.0, 1 - (vector <=> %s::vector))) AS score "
                f"FROM sat_embeddings WHERE {' AND '.join(clauses)} "
                "ORDER BY score DESC LIMIT %s",
                params,
            )
            rows = cur.fetchall()
        return [
            EmbeddingMatch(
                embedding_id=str(row["embedding_id"]),
                object_id=str(row["object_id"]),
                score=float(row["score"]),
                text=str(row["text"]),
                metadata=(
                    json.loads(row["metadata"])
                    if isinstance(row["metadata"], str)
                    else dict(row["metadata"] or {})
                ),
            )
            for row in rows
        ]

    def update_embedding(
        self,
        embedding_id: str,
        *,
        text: str | None = None,
        model: str | None = None,
    ) -> bool:
        """Update text/model and recompute the vector; True when found."""
        with self._cursor() as cur:
            cur.execute(
                "SELECT text, model FROM sat_embeddings WHERE embedding_id = %s "
                "FOR UPDATE",
                (embedding_id,),
            )
            row = cur.fetchone()
            if not row:
                return False
            new_text = text if text is not None else str(row["text"])
            new_model = model if model is not None else str(row["model"])
            new_vector = self._embed(new_text)
            now = datetime.now(timezone.utc).isoformat()
            cur.execute(
                "UPDATE sat_embeddings SET text = %s, model = %s, "
                "vector = %s::vector, updated_at = %s WHERE embedding_id = %s",
                (new_text, new_model, _vector_literal(new_vector), now, embedding_id),
            )
            return True

    def delete_embedding(self, embedding_id: str) -> bool:
        """Delete an embedding; True when it existed."""
        with self._cursor() as cur:
            cur.execute(
                "DELETE FROM sat_embeddings WHERE embedding_id = %s",
                (embedding_id,),
            )
            return cur.rowcount > 0

    def list_embeddings(
        self,
        *,
        project_id: str | None = None,
        status: EmbeddingStatus | None = None,
        limit: int = 100,
    ) -> list[Embedding]:
        """List embeddings (ACTIVE by default), newest first."""
        effective_status = status if status is not None else EmbeddingStatus.ACTIVE
        clauses = ["status = %s"]
        params: list[Any] = [effective_status.value]
        if project_id:
            clauses.append("project_id = %s")
            params.append(project_id)
        params.append(limit)
        with self._cursor() as cur:
            cur.execute(
                f"SELECT {_COLUMNS} FROM sat_embeddings "
                f"WHERE {' AND '.join(clauses)} "
                "ORDER BY created_at DESC LIMIT %s",
                params,
            )
            return [_row_to_embedding(row) for row in cur.fetchall()]

    def count(
        self,
        *,
        project_id: str | None = None,
        status: EmbeddingStatus | None = None,
    ) -> int:
        """Count embeddings with optional filters."""
        clauses: list[str] = []
        params: list[Any] = []
        if project_id:
            clauses.append("project_id = %s")
            params.append(project_id)
        if status:
            clauses.append("status = %s")
            params.append(status.value)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._cursor() as cur:
            cur.execute(f"SELECT count(*) AS n FROM sat_embeddings {where}", params)
            return int(cur.fetchone()["n"])
