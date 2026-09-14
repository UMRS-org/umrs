"""Embeddings API — embedding creation, storage, retrieval, and updates.

Phase 1, Step 2.7 of the RC1 migration plan.

UMRS owns embedding management. R4 consumes this API.

This module provides an embeddings API for:

  - Creating embeddings from text (using a pluggable embedder)
  - Storing embeddings with project scoping
  - Retrieving embeddings by ID
  - Searching for similar embeddings (cosine similarity)
  - Updating embeddings when content changes
  - Deleting embeddings

The store uses in-memory vectors for the reference implementation.
The Postgres-backed implementation will use pgvector.

Embeddings are project-scoped: searches only return results from the
same project (unless project_id is None, in which case all projects
are searched).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Protocol
from uuid import uuid4

logger = logging.getLogger(__name__)

__all__ = [
    "DuplicateEmbeddingError",
    "EmbedderProtocol",
    "Embedding",
    "EmbeddingMatch",
    "EmbeddingNotFoundError",
    "EmbeddingStatus",
    "InMemoryEmbeddingStore",
    "build_default_store",
    "hash_based_vector",
]


# ── Embedding Status ─────────────────────────────────────────────────────


class EmbeddingStatus(str, Enum):
    """Status of an embedding."""

    ACTIVE = "active"
    ARCHIVED = "archived"
    DEPRECATED = "deprecated"


# ── Embedding ────────────────────────────────────────────────────────────


@dataclass
class Embedding:
    """A single embedding vector with metadata.

    Attributes:
        embedding_id: Unique identifier for the embedding
        project_id: The project this embedding belongs to
        object_id: The object this embedding was created from
        model: The embedding model used (e.g., "sentence-transformers/all-MiniLM-L6-v2")
        vector: The embedding vector (list of floats)
        text: The text that was embedded (may be truncated)
        metadata: Additional metadata
        status: Current status of the embedding
        created_at: When the embedding was created
        updated_at: When the embedding was last updated
    """

    embedding_id: str = ""
    project_id: str = ""
    object_id: str = ""
    model: str = ""
    vector: list[float] = field(default_factory=list)
    text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    status: EmbeddingStatus = EmbeddingStatus.ACTIVE
    created_at: str = ""
    updated_at: str = ""

    def __post_init__(self) -> None:
        if not self.embedding_id:
            self.embedding_id = f"emb-{uuid4().hex[:12]}"
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()
        if not self.updated_at:
            self.updated_at = self.created_at

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "embedding_id": self.embedding_id,
            "project_id": self.project_id,
            "object_id": self.object_id,
            "model": self.model,
            "vector": self.vector,
            "text": self.text,
            "metadata": self.metadata,
            "status": self.status.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Embedding:
        """Create from dictionary."""
        status_raw = data.get("status", "active")
        status = (
            EmbeddingStatus(status_raw) if isinstance(status_raw, str) else status_raw
        )
        return cls(
            embedding_id=data.get("embedding_id", ""),
            project_id=data.get("project_id", ""),
            object_id=data.get("object_id", ""),
            model=data.get("model", ""),
            vector=data.get("vector", []),
            text=data.get("text", ""),
            metadata=data.get("metadata", {}),
            status=status,
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
        )


# ── Embedding Match ──────────────────────────────────────────────────────


@dataclass
class EmbeddingMatch:
    """A search result from embedding similarity search.

    Attributes:
        embedding_id: The matching embedding ID
        object_id: The object ID of the matching embedding
        score: Similarity score (0.0–1.0, higher is more similar)
        text: The text of the matching embedding (may be truncated)
        metadata: Additional metadata from the matching embedding
    """

    embedding_id: str = ""
    object_id: str = ""
    score: float = 0.0
    text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "embedding_id": self.embedding_id,
            "object_id": self.object_id,
            "score": self.score,
            "text": self.text,
            "metadata": self.metadata,
        }


# ── Embedder Protocol ───────────────────────────────────────────────────


class EmbedderProtocol(Protocol):
    """Protocol for an embedder that creates embedding vectors from text."""

    def embed(self, text: str) -> list[float]:
        """Create an embedding vector from text.

        Args:
            text: The text to embed

        Returns:
            A list of floats representing the embedding vector
        """
        ...

    @property
    def model_name(self) -> str:
        """The name of the embedding model."""
        ...


# ── Exceptions ────────────────────────────────────────────────────────────


class EmbeddingNotFoundError(Exception):
    """Raised when an embedding is not found."""


class DuplicateEmbeddingError(Exception):
    """Raised when a duplicate embedding is added."""


def hash_based_vector(text: str, dim: int = 384) -> list[float]:
    """Deterministic hash-based vector (testing fallback, NOT a real embedding).

    Shared by the in-memory and Postgres stores when no embedder is
    configured.
    """
    import hashlib

    hash_bytes = hashlib.sha256(text.encode("utf-8")).digest()
    return [(hash_bytes[i % len(hash_bytes)] / 255.0) * 2 - 1 for i in range(dim)]


# ── In-Memory Embedding Store ────────────────────────────────────────────


class InMemoryEmbeddingStore:
    """In-memory store for embeddings.

    This is the reference implementation. The Postgres-backed implementation
    will use pgvector for vector storage and similarity search.

    The store supports:
      - Creating embeddings from text using a pluggable embedder
      - Storing embeddings with project scoping
      - Retrieving embeddings by ID
      - Searching for similar embeddings (cosine similarity)
      - Updating embeddings when content changes
      - Deleting embeddings
    """

    def __init__(self, embedder: EmbedderProtocol | None = None) -> None:
        self._embeddings: dict[str, Embedding] = {}
        self._embedder = embedder

    def create_embedding(
        self,
        *,
        project_id: str,
        object_id: str,
        text: str,
        model: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Create a new embedding from text.

        If an embedder is configured, it will be used to create the embedding
        vector. Otherwise, a simple hash-based vector will be created (for
        testing purposes only).

        Args:
            project_id: The project this embedding belongs to
            object_id: The object this embedding was created from
            text: The text to embed
            model: The embedding model used (if empty, uses embedder's model name)
            metadata: Additional metadata

        Returns:
            The embedding_id of the created embedding
        """
        # Create embedding vector
        if self._embedder:
            vector = self._embedder.embed(text)
            model = model or self._embedder.model_name
        else:
            # Fallback: simple hash-based vector (for testing only)
            vector = self._create_hash_vector(text)
            model = model or "hash-based"

        embedding = Embedding(
            project_id=project_id,
            object_id=object_id,
            model=model,
            vector=vector,
            text=text,
            metadata=metadata or {},
        )
        self._embeddings[embedding.embedding_id] = embedding
        return embedding.embedding_id

    def get_embedding(self, embedding_id: str) -> Embedding | None:
        """Get an embedding by ID.

        Args:
            embedding_id: The embedding ID

        Returns:
            The embedding, or None if not found
        """
        return self._embeddings.get(embedding_id)

    def search_similar(
        self,
        *,
        query_embedding: list[float],
        top_k: int = 10,
        project_id: str | None = None,
    ) -> list[EmbeddingMatch]:
        """Search for embeddings similar to the query embedding.

        Uses cosine similarity to compare the query embedding to all
        stored embeddings.

        Args:
            query_embedding: The query embedding vector
            top_k: Maximum number of results to return
            project_id: Filter by project ID (if None, search all projects)

        Returns:
            List of EmbeddingMatch objects, sorted by similarity score descending
        """
        results: list[EmbeddingMatch] = []
        for emb in self._embeddings.values():
            if emb.status != EmbeddingStatus.ACTIVE:
                continue
            if project_id and emb.project_id != project_id:
                continue
            # Calculate cosine similarity
            score = self._cosine_similarity(query_embedding, emb.vector)
            results.append(
                EmbeddingMatch(
                    embedding_id=emb.embedding_id,
                    object_id=emb.object_id,
                    score=score,
                    text=emb.text,
                    metadata=emb.metadata,
                )
            )
        # Sort by score descending
        results.sort(key=lambda m: m.score, reverse=True)
        return results[:top_k]

    def update_embedding(
        self,
        embedding_id: str,
        *,
        text: str | None = None,
        model: str | None = None,
    ) -> bool:
        """Update an embedding's text and/or model.

        This re-creates the embedding vector with the new text.

        Args:
            embedding_id: The embedding ID
            text: New text to embed (optional)
            model: New model to use (optional)

        Returns:
            True if the embedding was found and updated, False otherwise
        """
        emb = self._embeddings.get(embedding_id)
        if not emb:
            return False

        # Update text and model
        if text is not None:
            emb.text = text
        if model is not None:
            emb.model = model

        # Re-create embedding vector
        if self._embedder:
            emb.vector = self._embedder.embed(emb.text)
        else:
            emb.vector = self._create_hash_vector(emb.text)

        emb.updated_at = datetime.now(timezone.utc).isoformat()
        return True

    def delete_embedding(self, embedding_id: str) -> bool:
        """Delete an embedding.

        Args:
            embedding_id: The embedding ID

        Returns:
            True if the embedding was found and deleted, False otherwise
        """
        if embedding_id not in self._embeddings:
            return False
        del self._embeddings[embedding_id]
        return True

    def list_embeddings(
        self,
        *,
        project_id: str | None = None,
        status: EmbeddingStatus | None = None,
        limit: int = 100,
    ) -> list[Embedding]:
        """List embeddings with optional filters.

        Args:
            project_id: Filter by project ID
            status: Filter by status
            limit: Maximum number of embeddings to return

        Returns:
            List of matching embeddings, sorted by created_at descending
        """
        if status is None:
            status = EmbeddingStatus.ACTIVE
        results: list[Embedding] = []
        for emb in self._embeddings.values():
            if project_id and emb.project_id != project_id:
                continue
            if status is not None and emb.status != status:
                continue
            results.append(emb)
        results.sort(key=lambda e: e.created_at, reverse=True)
        return results[:limit]

    def count(
        self,
        *,
        project_id: str | None = None,
        status: EmbeddingStatus | None = None,
    ) -> int:
        """Count embeddings with optional filters.

        Args:
            project_id: Filter by project ID
            status: Filter by status

        Returns:
            Number of matching embeddings
        """
        count = 0
        for emb in self._embeddings.values():
            if project_id and emb.project_id != project_id:
                continue
            if status and emb.status != status:
                continue
            count += 1
        return count

    # ── Internal Helper Methods ───────────────────────────────────────────

    def _create_hash_vector(self, text: str, dim: int = 384) -> list[float]:
        """Create a simple hash-based vector for testing purposes.

        This is NOT a real embedding — it's just a deterministic vector
        based on the text content. Used only when no real embedder is
        configured.
        """
        return hash_based_vector(text, dim)

    def _cosine_similarity(
        self,
        vec_a: list[float],
        vec_b: list[float],
    ) -> float:
        """Calculate cosine similarity between two vectors.

        Args:
            vec_a: First vector
            vec_b: Second vector

        Returns:
            Cosine similarity score (0.0–1.0)
        """
        if not vec_a or not vec_b:
            return 0.0
        if len(vec_a) != len(vec_b):
            return 0.0

        # Calculate dot product
        dot_product = sum(a * b for a, b in zip(vec_a, vec_b))

        # Calculate magnitudes
        mag_a = sum(a * a for a in vec_a) ** 0.5
        mag_b = sum(b * b for b in vec_b) ** 0.5

        if mag_a == 0 or mag_b == 0:
            return 0.0

        # Cosine similarity
        similarity = dot_product / (mag_a * mag_b)

        # Clamp to [0, 1] (cosine similarity can be [-1, 1])
        return max(0.0, min(1.0, similarity))


# ── Default Store Factory ───────────────────────────────────────────────


def build_default_store(
    db_pool: Any = None, embedder: EmbedderProtocol | None = None
) -> Any:
    """Build the default embedding store backend.

    Selection via ``UMRS_EMBEDDINGS_STORE_BACKEND`` (default ``auto``):

      - ``pg``     — force Postgres + pgvector (raises if unavailable)
      - ``memory`` — force the in-memory reference store
      - ``auto``   — Postgres when a pool is provided or the environment
                     points at a database, otherwise in-memory

    Postgres uses the ``sat_embeddings`` table (migration 0060).
    """
    import os as _os

    from umrs.shared.pg import resolve_satellite_pool

    backend = _os.environ.get("UMRS_EMBEDDINGS_STORE_BACKEND", "auto").lower()
    if backend == "memory":
        return InMemoryEmbeddingStore(embedder)
    if backend not in ("auto", "pg"):
        logger.warning("Unknown store backend %r; using auto", backend)
        backend = "auto"

    pool = resolve_satellite_pool(backend, db_pool, "Embeddings store")
    if pool is None:
        return InMemoryEmbeddingStore(embedder)

    from .pg_store import PgEmbeddingStore

    return PgEmbeddingStore(pool, embedder)
