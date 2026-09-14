"""
ArtifactRegistry data access layer — ArtifactStore.

CRUD operations for the artifact registry.  Follows the same pattern as
jobs/store.py: conn_factory / db_pool style, parameterized SQL only.

No imports from umrs.api or umrs.server — foundation module.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from typing import Any

from psycopg2.extras import Json

from umrs.artifact_registry.models import (
    Artifact,
    ArtifactLineageEdge,
    ArtifactQuery,
)

logger = logging.getLogger(__name__)


class ArtifactStoreError(Exception):
    """Base exception for ArtifactStore errors."""


class ArtifactNotFoundError(ArtifactStoreError):
    """Raised when an artifact is not found."""


class ArtifactStore:
    """Artifact Registry data access layer."""

    def __init__(
        self,
        conn_factory: Callable[[], Any],
        *,
        conn_release: Callable[[Any], None] | None = None,
    ) -> None:
        """Initialize the store.

        Args:
            conn_factory: callable that yields a database connection via
                ``with conn_factory() as conn:`` (context manager) or returns
                a raw connection for ``cur.execute()`` calls.
            conn_release: optional callable returning a connection to its
                pool (e.g. ``psycopg2.pool.ThreadedConnectionPool.putconn``).
                The store leases every acquired connection and returns them
                all in :meth:`close`. Previously the API-layer caller tried
                to return the connection via ``pool.putconn(conn_factory())``
                — acquiring a SECOND connection to return, leaking one per
                request until ``pool exhausted`` 500s in production.
        """
        self._conn_factory = conn_factory
        self._conn_release = conn_release
        self._leased: list[Any] = []

    def close(self) -> None:
        """Release every leased connection (idempotent)."""
        while self._leased:
            conn = self._leased.pop()
            try:
                if self._conn_release is not None:
                    self._conn_release(conn)
                else:
                    conn.close()
            except Exception:
                logger.warning(
                    "Failed to release artifact-store connection", exc_info=True
                )

    def __enter__(self) -> ArtifactStore:
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    # -- public CRUD -----------------------------------------------------------

    def create(self, artifact: Artifact) -> Artifact:
        """Register a new artifact.  Returns the artifact with id populated."""
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO artifacts (
                        id, artifact_type, name, owner_id, project_id,
                        uri, bucket, s3_key, size_bytes, checksum,
                        metadata, execution_id, workflow_id, job_id,
                        version
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s
                    )
                    RETURNING *
                    """,
                    (
                        str(artifact.id or uuid.uuid4()),
                        artifact.artifact_type,
                        artifact.name,
                        artifact.owner_id,
                        artifact.project_id,
                        artifact.uri,
                        artifact.bucket,
                        artifact.s3_key,
                        artifact.size_bytes,
                        artifact.checksum,
                        Json(artifact.metadata),
                        artifact.execution_id,
                        artifact.workflow_id,
                        artifact.job_id,
                        artifact.version,
                    ),
                )
                row = cur.fetchone()
                result = Artifact.from_row(row)
                conn.commit()
                return result
        except Exception as exc:
            conn.rollback()
            logger.error("Failed to create artifact: %s", exc)
            raise

    def get(self, artifact_id: uuid.UUID) -> Artifact | None:
        """Get an artifact by id.  Returns None if not found."""
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM artifacts WHERE id = %s", (str(artifact_id),))
            row = cur.fetchone()
            if row is None:
                return None
            return Artifact.from_row(row)

    def list(self, query: ArtifactQuery) -> list[Artifact]:
        """List artifacts matching the query filters."""
        conn = self._get_conn()
        conditions: list[str] = []
        params: list[Any] = []
        param_idx = 1

        if query.artifact_type:
            conditions.append("artifact_type = %s")
            params.append(query.artifact_type)
            param_idx += 1

        if query.owner_id:
            conditions.append("owner_id = %s")
            params.append(query.owner_id)
            param_idx += 1

        if query.project_id:
            conditions.append("project_id = %s")
            params.append(query.project_id)
            param_idx += 1

        if query.execution_id:
            conditions.append("execution_id = %s")
            params.append(query.execution_id)
            param_idx += 1

        if query.name_prefix:
            conditions.append("name LIKE %s")
            params.append(f"{query.name_prefix}%")
            param_idx += 1

        sql = "SELECT * FROM artifacts"
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        sql += " ORDER BY created_at DESC"
        sql += " LIMIT %s OFFSET %s"
        params.extend([query.limit, query.offset])

        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
            return [Artifact.from_row(r) for r in rows]

    def update_metadata(self, artifact_id: uuid.UUID, patch: dict[str, Any]) -> None:
        """Merge patch dict into the artifact's metadata JSONB."""
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE artifacts
                    SET metadata = jsonb_set(metadata, %s, %s::jsonb),
                        updated_at = NOW()
                    WHERE id = %s
                    """,
                    (
                        "$response",
                        str(patch),
                        str(artifact_id),
                    ),
                )
                if cur.rowcount == 0:
                    raise ArtifactNotFoundError(f"Artifact {artifact_id} not found")
                conn.commit()
        except Exception as exc:
            conn.rollback()
            if isinstance(exc, ArtifactNotFoundError):
                raise
            logger.error("Failed to update metadata: %s", exc)
            raise

    def add_lineage(
        self,
        artifact_id: uuid.UUID,
        parent_artifact_id: uuid.UUID,
        relation: str = "derived_from",
    ) -> None:
        """Record a lineage edge (artifact → parent)."""
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO artifact_lineage (
                        artifact_id, parent_artifact_id, relation
                    )
                    VALUES (%s, %s, %s)
                    """,
                    (
                        str(artifact_id),
                        str(parent_artifact_id),
                        relation,
                    ),
                )
                conn.commit()
        except Exception as exc:
            conn.rollback()
            logger.error("Failed to add lineage: %s", exc)
            raise

    def lineage(
        self, artifact_id: uuid.UUID, depth: int = 3
    ) -> list[ArtifactLineageEdge]:
        """Get lineage edges (up to depth levels)."""
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(
                """
                WITH RECURSIVE lineage_path(
                        artifact_id, parent_artifact_id, relation, depth
                    ) AS (
                    SELECT artifact_id, parent_artifact_id, relation, 1
                    FROM artifact_lineage
                    WHERE artifact_id = %s
                    UNION ALL
                    SELECT al.artifact_id, al.parent_artifact_id,
                            al.relation, lp.depth + 1
                    FROM artifact_lineage al
                    INNER JOIN lineage_path lp ON al.artifact_id = lp.parent_artifact_id
                    WHERE lp.depth < %s
                )
                SELECT artifact_id, parent_artifact_id, relation
                FROM lineage_path
                """,
                (str(artifact_id), depth),
            )
            rows = cur.fetchall()
            return [
                ArtifactLineageEdge(
                    artifact_id=uuid.UUID(r[0]),
                    parent_artifact_id=uuid.UUID(r[1]),
                    relation=r[2],
                )
                for r in rows
            ]

    def delete(self, artifact_id: uuid.UUID) -> bool:
        """Delete an artifact and its lineage edges.  Returns True if deleted."""
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM artifacts WHERE id = %s", (str(artifact_id),))
                deleted = cur.rowcount > 0
                conn.commit()
                return deleted
        except Exception as exc:
            conn.rollback()
            logger.error("Failed to delete artifact: %s", exc)
            raise

    # -- helpers ---------------------------------------------------------------

    def _get_conn(self):
        """Acquire a connection from the factory, tracked for release."""
        conn = self._conn_factory()
        self._leased.append(conn)
        return conn
