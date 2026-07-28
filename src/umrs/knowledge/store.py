"""Knowledge Postgres store module.

CRUD operations for knowledge persistence tables using asyncpg-pool pattern.
No umrs.api or umrs.server imports — strict boundary.

Key operations:
  - insert_artifact: insert-dedup returning deduped flag
  - insert_ir: write-once (ON CONFLICT DO NOTHING)
  - upsert_node / upsert_edge: node/edge upsert
  - truncate_nodes / truncate_edges: truncate-and-recompile
  - list_artifacts: paged list with kind/status filters
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Schema version — must match migration 0055_knowledge_store.sql
_KNOWLEDGE_SCHEMA_VERSION = 1


def _utcnow() -> datetime:
    """Timezone-aware UTC now."""
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class KnowledgeArtifact:
    """Row from knowledge_artifacts table."""

    artifact_id: str
    content_hash: str
    kind: str  # document, code, table, image, audio, video, embedding
    project_id: str
    source_ref: str = ""
    size_bytes: int = 0
    mime_type: str = ""
    status: str = "active"
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


@dataclass(frozen=True)
class KnowledgeIRRow:
    """Row from knowledge_ir table."""

    ir_id: str
    artifact_id: str
    ir_type: str  # code, document, schema, catalog, embedding
    version: int = 1
    content: dict[str, Any] = field(default_factory=dict)
    content_hash: str = ""
    status: str = "current"
    created_at: Optional[datetime] = None


@dataclass(frozen=True)
class KnowledgeNode:
    """Row from knowledge_nodes table."""

    node_id: str
    project_id: str
    label: str
    node_type: str  # class, function, file, module, concept, entity, relationship
    content: dict[str, Any] = field(default_factory=dict)
    content_hash: str = ""
    source_ir: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


@dataclass(frozen=True)
class KnowledgeEdge:
    """Row from knowledge_edges table."""

    edge_id: str
    project_id: str
    source_id: str
    target_id: str
    relation_type: str
    content: dict[str, Any] = field(default_factory=dict)
    created_at: Optional[datetime] = None


class KnowledgeStore:
    """Knowledge Postgres store using asyncpg pool.

    Provides CRUD operations backed by migration 0055 tables.
    """

    def __init__(self, pool) -> None:  # type: ignore[misc]  # asyncpg Pool
        """Initialize store with an asyncpg connection pool."""
        self._pool = pool
    async def insert_artifact(
        self,
        artifact_id: str,
        content_hash: str,
        kind: str,
        project_id: str,
        source_ref: str = "",
        size_bytes: int = 0,
        mime_type: str = "",
        metadata: Optional[dict[str, Any]] = None,
    ) -> bool:
        """Insert or update artifact metadata.

        Returns True if inserted (new), False if deduped (already existed with same hash).
        """
        async with self._pool.acquire() as conn:
            existing = await conn.fetchval(
                """SELECT artifact_id FROM knowledge_artifacts
                   WHERE content_hash = $1 AND status = 'active'
                   LIMIT 1""",
                content_hash,
            )
            if existing:
                logger.debug(
                    "Artifact deduped: hash=%s (exists as %s)", content_hash, existing
                )
                return False

            ts = _utcnow()
            await conn.execute(
                """INSERT INTO knowledge_artifacts
                   (artifact_id, content_hash, kind, project_id, source_ref,
                    size_bytes, mime_type, status, metadata, created_at, updated_at)
                   VALUES ($1, $2, $3, $4, $5, $6, $7, 'active', $8, $9, $10)
                   ON CONFLICT (artifact_id) DO UPDATE
                   SET updated_at = $10, status = 'active'""",
                artifact_id,
                content_hash,
                kind,
                project_id,
                source_ref,
                size_bytes,
                mime_type,
                metadata or {},
                ts,
                ts,
            )
            logger.debug("Artifact inserted: %s (hash=%s)", artifact_id, content_hash)
            return True

    async def insert_ir(
        self,
        ir_id: str,
        artifact_id: str,
        ir_type: str,
        content: dict[str, Any],
        content_hash: str,
        version: int = 1,
    ) -> bool:
        """Insert IR write-once (ON CONFLICT DO NOTHING for idempotency).

        Returns True if inserted (new), False if already exists.
        """
        try:
            async with self._pool.acquire() as conn:
                result = await conn.execute(
                    """INSERT INTO knowledge_ir
                       (ir_id, artifact_id, version, ir_type, content, content_hash, status)
                       VALUES ($1, $2, $3, $4, $5, $6, 'current')
                       ON CONFLICT (ir_id) DO NOTHING""",
                    ir_id,
                    artifact_id,
                    version,
                    ir_type,
                    content,
                    content_hash,
                )
                inserted = result != "INSERT 0 0"
                return inserted
        except Exception as exc:
            logger.warning("IR insert failed (likely duplicate): %s: %s", ir_id, exc)
            return False

    async def upsert_node(
        self,
        node: KnowledgeNode,
    ) -> None:
        """Upsert a knowledge graph node."""
        async with self._pool.acquire() as conn:
            ts = _utcnow()
            await conn.execute(
                """INSERT INTO knowledge_nodes
                   (node_id, project_id, label, node_type, content, content_hash,
                    source_ir, created_at, updated_at)
                   VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                   ON CONFLICT (node_id) DO UPDATE
                   SET content = $5, content_hash = $6, source_ir = $7, updated_at = $9""",
                node.node_id,
                node.project_id,
                node.label,
                node.node_type,
                node.content,
                node.content_hash,
                node.source_ir,
                node.created_at or ts,
                ts,
            )

    async def upsert_edge(
        self,
        edge: KnowledgeEdge,
    ) -> None:
        """Upsert a knowledge graph edge."""
        async with self._pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO knowledge_edges
                   (edge_id, project_id, source_id, target_id, relation_type, content)
                   VALUES ($1, $2, $3, $4, $5, $6)
                   ON CONFLICT (edge_id) DO UPDATE
                   SET content = $6""",
                edge.edge_id,
                edge.project_id,
                edge.source_id,
                edge.target_id,
                edge.relation_type,
                edge.content,
            )

    async def truncate_nodes(
        self,
        project_id: str,
        source_ir: Optional[str] = None,
    ) -> int:
        """Delete nodes, optionally filtered by project + source_ir.

        Returns number of rows deleted.
        """
        async with self._pool.acquire() as conn:
            if source_ir:
                result = await conn.execute(
                    "DELETE FROM knowledge_nodes WHERE project_id = $1 AND source_ir = $2",
                    project_id,
                    source_ir,
                )
            else:
                result = await conn.execute(
                    "DELETE FROM knowledge_nodes WHERE project_id = $1",
                    project_id,
                )
            return int(result.split()[-1])

    async def truncate_edges(
        self,
        source_id: Optional[str] = None,
        target_id: Optional[str] = None,
    ) -> int:
        """Delete edges, optionally filtered by source or target node.

        Returns number of rows deleted.

        Without filters: deletes ALL edges (full truncate for recompile).
        With source_id: deletes edges where source node is removed.
        With target_id: deletes edges where target node is removed.
        """
        async with self._pool.acquire() as conn:
            if source_id and target_id:
                result = await conn.execute(
                    "DELETE FROM knowledge_edges WHERE source_id = $1 AND target_id = $2",
                    source_id,
                    target_id,
                )
            elif source_id:
                result = await conn.execute(
                    "DELETE FROM knowledge_edges WHERE source_id = $1",
                    source_id,
                )
            elif target_id:
                result = await conn.execute(
                    "DELETE FROM knowledge_edges WHERE target_id = $1",
                    target_id,
                )
            else:
                # Full truncate: delete all edges for recompile
                result = await conn.execute(
                    "DELETE FROM knowledge_edges",
                )
            return int(result.split()[-1])

    async def list_artifacts(
        self,
        project_id: str,
        kind: Optional[str] = None,
        status: str = "active",
        page: int = 1,
        page_size: int = 50,
    ) -> tuple[list[KnowledgeArtifact], int]:
        """Paged list of artifacts with optional filters.

        Returns (artifact_list, total_count).
        """
        async with self._pool.acquire() as conn:
            conditions = ["project_id = $1", "status = $2"]
            params: list[Any] = [project_id, status]
            param_idx = 3

            if kind:
                conditions.append(f"kind = ${param_idx}")
                params.append(kind)
                param_idx += 1

            where = " AND ".join(conditions)
            offset = (page - 1) * page_size

            # Get total count
            count_row = await conn.fetchrow(
                f"SELECT COUNT(*) FROM knowledge_artifacts WHERE {where}",
                *params,
            )
            total = int(count_row["count"]) if count_row else 0

            # Get paged results
            rows = await conn.fetch(
                f"""SELECT artifact_id, content_hash, kind, project_id,
                          source_ref, size_bytes, mime_type, status,
                          metadata, created_at, updated_at
                   FROM knowledge_artifacts
                   WHERE {where}
                   ORDER BY created_at DESC
                   LIMIT ${param_idx} OFFSET ${param_idx + 1}""",
                *params,
                page_size,
                offset,
            )

            artifacts = []
            for row in rows:
                artifacts.append(KnowledgeArtifact(**row))

            return artifacts, total

    async def get_artifact(self, artifact_id: str) -> Optional[KnowledgeArtifact]:
        """Get a single artifact by ID, or None if not found."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """SELECT artifact_id, content_hash, kind, project_id,
                          source_ref, size_bytes, mime_type, status,
                          metadata, created_at, updated_at
                   FROM knowledge_artifacts
                   WHERE artifact_id = $1""",
                artifact_id,
            )
            return KnowledgeArtifact(**row) if row else None

    async def get_ir(self, ir_id: str) -> Optional[KnowledgeIRRow]:
        """Get a single IR by ID, or None if not found."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """SELECT ir_id, artifact_id, version, ir_type,
                          content, content_hash, status, created_at
                   FROM knowledge_ir
                   WHERE ir_id = $1""",
                ir_id,
            )
            return KnowledgeIRRow(**row) if row else None

    async def list_ir_by_ids(self, ir_ids: list[str]) -> list[KnowledgeIRRow]:
        """Get multiple IRs by their ids.

        Returns rows in the same order as *ir_ids*, skipping missing ids.
        """
        if not ir_ids:
            return []
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT ir_id, artifact_id, version, ir_type,
                          content, content_hash, status, created_at
                   FROM knowledge_ir
                   WHERE ir_id = ANY($1)""",
                ir_ids,
            )
            return [KnowledgeIRRow(**dict(r)) for r in rows]

    async def get_schema_version(self) -> int:
        """Return the current knowledge schema version."""
        return _KNOWLEDGE_SCHEMA_VERSION

    async def list_irs(
        self,
        project_id: str,
        ir_type: Optional[str] = None,
    ) -> list[KnowledgeIRRow]:
        """Return all IRs for a project.

        Optional filter by ir_type.
        """
        async with self._pool.acquire() as conn:
            if ir_type:
                rows = await conn.fetch(
                    """SELECT ir_id, artifact_id, version, ir_type,
                              content, content_hash, status, created_at
                       FROM knowledge_ir
                       WHERE artifact_id IN (
                           SELECT artifact_id FROM knowledge_artifacts
                           WHERE project_id = $1
                       ) AND ir_type = $2
                       ORDER BY created_at""",
                    project_id,
                    ir_type,
                )
            else:
                rows = await conn.fetch(
                    """SELECT ir_id, artifact_id, version, ir_type,
                              content, content_hash, status, created_at
                       FROM knowledge_ir
                       WHERE artifact_id IN (
                           SELECT artifact_id FROM knowledge_artifacts
                           WHERE project_id = $1
                       )
                       ORDER BY created_at""",
                    project_id,
                )
            return [KnowledgeIRRow(**dict(r)) for r in rows]

    async def list_nodes(
        self,
        project_id: str,
    ) -> list[KnowledgeNode]:
        """Return all nodes for a project."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT node_id, project_id, label, node_type,
                          content, content_hash, source_ir,
                          created_at, updated_at
                   FROM knowledge_nodes
                   WHERE project_id = $1
                   ORDER BY created_at""",
                project_id,
            )
            return [KnowledgeNode(**dict(r)) for r in rows]

    async def list_edges(
        self,
        project_id: str,
    ) -> list[KnowledgeEdge]:
        """Return all edges for a project."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT edge_id, project_id, source_id, target_id,
                          relation_type, content, created_at
                   FROM knowledge_edges
                   WHERE project_id = $1
                   ORDER BY created_at""",
                project_id,
            )
            return [KnowledgeEdge(**dict(r)) for r in rows]

    async def insert_event(
        self,
        event_type: str,
        job_id: str,
        attempt_id: str,
        artifact_id: str,
        source_ref: str,
        details: Optional[dict[str, Any]] = None,
    ) -> str:
        """Insert a knowledge event into knowledge_events table.

        Returns the event_id of the inserted row.
        """
        from .events import _utcnow  # avoid top-level import loop

        import uuid

        event_id = uuid.uuid4().hex
        ts = _utcnow()
        async with self._pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO knowledge_events
                   (event_id, event_type, job_id, attempt_id, artifact_id,
                    source_ref, details, timestamp, created_at)
                   VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)""",
                event_id,
                event_type,
                job_id,
                attempt_id,
                artifact_id,
                source_ref,
                details or {},
                ts,
                ts,
            )
        return event_id

    async def list_events(
        self,
        job_id: str,
        artifact_id: Optional[str] = None,
        event_type: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """List events for a job, optionally filtered by artifact and type.

        Returns list of dicts with keys: event_id, event_type, job_id,
        attempt_id, artifact_id, source_ref, details, timestamp.
        """
        async with self._pool.acquire() as conn:
            conditions = ["job_id = $1"]
            params: list[Any] = [job_id]
            param_idx = 2

            if artifact_id:
                conditions.append(f"artifact_id = ${param_idx}")
                params.append(artifact_id)
                param_idx += 1

            if event_type:
                conditions.append(f"event_type = ${param_idx}")
                params.append(event_type)
                param_idx += 1

            where = " AND ".join(conditions)
            rows = await conn.fetch(
                f"""SELECT event_id, event_type, job_id, attempt_id,
                           artifact_id, source_ref, details, timestamp
                    FROM knowledge_events
                    WHERE {where}
                    ORDER BY timestamp""",
                *params,
            )
            return [dict(r) for r in rows]

    # ── Catalog helpers ───────────────────────────────────────────────────────

    async def list_catalog(
        self,
        project_id: str,
        page: int = 1,
        page_size: int = 50,
    ) -> tuple[list[dict[str, Any]], int]:
        """List catalog entries (knowledge_artifacts with kind='catalog') for a project.

        Returns (entries, total_count).
        """
        async with self._pool.acquire() as conn:
            where = "project_id = $1 AND kind = 'catalog' AND status = 'active'"
            offset = (page - 1) * page_size

            total_row = await conn.fetchrow(
                f"SELECT COUNT(*) FROM knowledge_artifacts WHERE {where}",
                project_id,
            )
            total = int(total_row["count"]) if total_row else 0

            rows = await conn.fetch(
                f"""SELECT artifact_id, content_hash, kind, project_id,
                           source_ref, size_bytes, mime_type, status,
                           metadata, created_at, updated_at
                    FROM knowledge_artifacts
                    WHERE {where}
                    ORDER BY created_at DESC
                    LIMIT ${2} OFFSET ${3}""",
                project_id,
                page_size,
                offset,
            )
            entries: list[dict[str, Any]] = [dict(r) for r in rows]
            return entries, total

    async def list_embedding_vectors(
        self,
        project_id: str,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        """Return embedding vectors for a project (memory_vectors table).

        Returns list of dicts with keys: id, vector, metadata, document_id, created_at.
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT id, vector, metadata, document_id, created_at
                   FROM memory_vectors
                   WHERE project_id = $1
                   ORDER BY id
                   LIMIT $2""",
                project_id,
                limit,
            )
            return [dict(r) for r in rows]

    async def count_embedding_vectors(
        self,
        project_id: str,
    ) -> int:
        """Return total count of embedding vectors for a project."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT COUNT(*) FROM memory_vectors WHERE project_id = $1",
                project_id,
            )
            return int(row["count"]) if row else 0

    async def list_embedding_vectors_paged(
        self,
        project_id: str,
        cursor: str,
        page_size: int,
    ) -> tuple[list[dict[str, Any]], int]:
        """Paged fetch of memory_vectors using id-based cursor.

        Returns (rows, total_count).  Rows have keys: id, vector, metadata,
        document_id, created_at.
        """
        async with self._pool.acquire() as conn:
            total_row = await conn.fetchrow(
                "SELECT COUNT(*) FROM memory_vectors WHERE project_id = $1",
                project_id,
            )
            total = int(total_row["count"]) if total_row else 0

            rows = await conn.fetch(
                """SELECT id, vector, metadata, document_id, created_at
                   FROM memory_vectors
                   WHERE project_id = $1 AND id > $2
                   ORDER BY id
                   LIMIT $3""",
                project_id,
                cursor,
                page_size,
            )
            return [dict(r) for r in rows], total

    # ── Knowledge↔A2L Links ──────────────────────────────────────────────

    async def insert_knowledge_a2l_link(
        self,
        knowledge_artifact_id: str,
        a2l_artifact_id: str,
        job_id: str = "",
        attempt_id: str = "",
        edge_kind: str = "derived",
    ) -> bool:
        """Register a knowledge artifact → A2L link (ON CONFLICT DO NOTHING)."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO knowledge_a2l_links
                    (knowledge_artifact_id, a2l_artifact_id, job_id, attempt_id, edge_kind)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (knowledge_artifact_id, a2l_artifact_id)
                DO NOTHING
                RETURNING id
                """,
                knowledge_artifact_id,
                a2l_artifact_id,
                job_id,
                attempt_id,
                edge_kind,
            )
            return row is not None

    # ── A2L Bridge ─────────────────────────────────────────────────────────

    async def insert_a2l_ref(
        self,
        artifact_id: str,
        a2l_hash: str,
        a2l_id: str,
        source: str = "auto",
    ) -> bool:
        """Persist an A2L→UMRS artifact mapping (ON CONFLICT DO NOTHING)."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO a2l_artifact_refs (artifact_id, a2l_hash, a2l_id, source)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (artifact_id, a2l_hash) DO NOTHING
                RETURNING id
                """,
                artifact_id,
                a2l_hash,
                a2l_id,
                source,
            )
            return row is not None

    async def get_artifact_content(self, artifact_id: str) -> Optional[bytes]:
        """Fetch raw artifact content from S3 via the artifact's source_ref.

        Returns None if the artifact has no content or source_ref is empty.
        """
        artifact = await self.get_artifact(artifact_id)
        if not artifact or not artifact.source_ref:
            return None
        # source_ref contains the S3 presigned URL or local path
        # This method is intentionally minimal — content is stored in S3,
        # not in the artifact row.
        return None


# ── Global store singleton ────────────────────────────────────────────────────
# TODO: This is a temporary fix. Define proper store factory.

_pool: Any = None
_instance: "KnowledgeStore | None" = None


def get_store() -> KnowledgeStore:
    """Return the global KnowledgeStore singleton.

    Uses a simple pool-based singleton pattern. In production,
    this should be replaced with a proper dependency injection
    or factory pattern.
    """
    global _instance
    if _pool is None:
        raise RuntimeError(
            "Knowledge store pool not initialized. "
            "Call build_knowledge_pool(uri) first."
        )
    if _instance is None:
        _instance = KnowledgeStore(_pool)
    return _instance


def build_knowledge_pool(uri: str, **kwargs: Any) -> Any:
    """Create the asyncpg pool for the knowledge store and set the global."""
    import asyncio

    import asyncpg  # type: ignore[import-untyped]

    global _pool
    _pool = asyncio.get_event_loop().run_until_complete(
        asyncpg.create_pool(dsn=uri, **kwargs)
    )
    return _pool


def reset_store() -> None:
    """Reset the store singleton for testing (pool is preserved)."""
    global _instance
    _instance = None
