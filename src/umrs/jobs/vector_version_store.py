"""Vector versioning for embedding model upgrades (MEM-C2).

Tracks which embedding model produced each vector in ``memory_vectors``
and supports re-embedding with a new model while preserving old vectors
until migration completes.

Key concepts:
- ``VectorVersion``: metadata about a specific model version in use
- ``ReembedJob``: tracks a re-embedding operation (old model → new model)
- ``VectorVersionStore``: manages version tracking and re-embedding state

This module lives in ``umrs.jobs`` because it reads/writes durable state
tied to embedding vectors and does not import from ``umrs.api`` or
``umrs.server``.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


def _generate_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt else None


VALID_REEMBED_STATUSES = {"pending", "running", "succeeded", "failed", "cancelled"}


@dataclass
class VectorVersion:
    """Metadata about an embedding model version in use."""

    version_id: str
    model_name: str
    model_version: str
    model_path: str | None = None
    embedding_dimension: int = 0
    vector_count: int = 0
    is_active: bool = True  # current model for new embeddings
    created_at: datetime = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version_id": self.version_id,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "model_path": self.model_path,
            "embedding_dimension": self.embedding_dimension,
            "vector_count": self.vector_count,
            "is_active": self.is_active,
            "created_at": _iso(self.created_at),
        }


@dataclass
class ReembedJob:
    """Tracks a re-embedding migration from one model version to another."""

    reembed_id: str
    source_model: str
    source_version: str
    target_model: str
    target_version: str
    target_dimension: int = 0
    status: str = "pending"
    total_vectors: int = 0
    processed_vectors: int = 0
    failed_vectors: int = 0
    job_id: str | None = None  # linked durable job
    error: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    created_at: datetime = field(default_factory=_now)

    @property
    def progress(self) -> float:
        if self.total_vectors == 0:
            return 0.0
        return self.processed_vectors / self.total_vectors

    def to_dict(self) -> dict[str, Any]:
        return {
            "reembed_id": self.reembed_id,
            "source_model": self.source_model,
            "source_version": self.source_version,
            "target_model": self.target_model,
            "target_version": self.target_version,
            "target_dimension": self.target_dimension,
            "status": self.status,
            "total_vectors": self.total_vectors,
            "processed_vectors": self.processed_vectors,
            "failed_vectors": self.failed_vectors,
            "progress": self.progress,
            "job_id": self.job_id,
            "error": self.error,
            "started_at": _iso(self.started_at),
            "completed_at": _iso(self.completed_at),
            "created_at": _iso(self.created_at),
        }


@dataclass
class VersionSummary:
    """Summary of embedding model versions in use."""

    total_vectors: int
    model_versions: list[dict[str, Any]]
    active_version: str | None
    pending_reembeds: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_vectors": self.total_vectors,
            "model_versions": self.model_versions,
            "active_version": self.active_version,
            "pending_reembeds": self.pending_reembeds,
        }


class _PooledConnection:
    """Context-manager shim: borrow a conn from the shared pool (putconn on exit)."""

    def __init__(self, pool):
        self._pool = pool
        self._conn = None

    def __enter__(self):
        self._conn = self._pool.getconn()
        return self._conn

    def __exit__(self, exc_type, _exc, _tb):
        try:
            if exc_type is None:
                self._conn.commit()
            else:
                self._conn.rollback()
        finally:
            self._pool.putconn(self._conn)
        return False


def _apply_reembed_status(job: ReembedJob, status: str) -> None:
    """Status transition with started/completed timestamps (in-memory path)."""
    job.status = status
    if status == "running" and job.started_at is None:
        job.started_at = _now()
    elif status in ("succeeded", "failed", "cancelled"):
        job.completed_at = _now()


class VectorVersionStore:
    """Manages embedding model version tracking and re-embedding.

    Falls back to in-memory storage when ``db_url`` is not set (dev mode).
    """

    def __init__(self, db_url: str | None = None, *, pool: Any | None = None) -> None:
        self._db_url = db_url
        # Pool-injected mode (WP-06 r2 / venue wiring): reuse the server's
        # shared ThreadedConnectionPool instead of burning a fresh connection
        # per call — same connection lifecycle as every other umrs store.
        self._pool = pool
        self._psycopg2 = None
        if pool is not None:
            import psycopg2.pool  # type: ignore

            self._psycopg2 = psycopg2.pool
            return
        if db_url:
            try:
                import psycopg2  # type: ignore

                self._psycopg2 = psycopg2
            except ImportError:
                logger.warning(
                    "db_url set but psycopg2 unavailable; "
                    "vector versioning will use in-memory store"
                )
        # In-memory fallback
        self._versions: dict[str, VectorVersion] = {}
        self._reembed_jobs: dict[str, ReembedJob] = {}

    @property
    def db_enabled(self) -> bool:
        return self._psycopg2 is not None and (
            self._db_url is not None or self._pool is not None
        )

    def _get_conn(self):
        if self._pool is not None:
            return _PooledConnection(self._pool)
        from umrs.shared.pg_settings import pg_connect_kwargs

        return self._psycopg2.connect(
            self._db_url, **pg_connect_kwargs("vector-versions")
        )

    # ------------------------------------------------------------------
    # Version tracking
    # ------------------------------------------------------------------

    def register_version(
        self,
        model_name: str,
        model_version: str,
        model_path: str | None = None,
        embedding_dimension: int = 0,
        set_active: bool = True,
    ) -> VectorVersion:
        """Register a new embedding model version.

        If set_active=True, deactivates all other versions for this model.
        """
        version_id = _generate_id("emv")

        version = VectorVersion(
            version_id=version_id,
            model_name=model_name,
            model_version=model_version,
            model_path=model_path,
            embedding_dimension=embedding_dimension,
            is_active=set_active,
        )

        if self.db_enabled:
            try:
                self._db_register_version(version, set_active)
                return version
            except Exception as exc:
                logger.warning("DB register_version failed, using memory: %s", exc)

        if set_active:
            for v in self._versions.values():
                if v.model_name == model_name:
                    v.is_active = False
        self._versions[version_id] = version
        return version

    def get_active_version(self, model_name: str) -> VectorVersion | None:
        """Get the currently active version for a model."""
        if self.db_enabled:
            try:
                return self._db_get_active_version(model_name)
            except Exception as exc:
                logger.warning("DB get_active_version failed: %s", exc)

        for v in self._versions.values():
            if v.model_name == model_name and v.is_active:
                return v
        return None

    def list_versions(self, model_name: str | None = None) -> list[VectorVersion]:
        """List all registered model versions, optionally filtered."""
        if self.db_enabled:
            try:
                return self._db_list_versions(model_name)
            except Exception as exc:
                logger.warning("DB list_versions failed: %s", exc)

        versions = list(self._versions.values())
        if model_name:
            versions = [v for v in versions if v.model_name == model_name]
        return sorted(versions, key=lambda v: v.created_at)

    def get_version_summary(self, model_name: str | None = None) -> VersionSummary:
        """Get a summary of all model versions and their vector counts."""
        versions = self.list_versions(model_name)

        total_vectors = sum(v.vector_count for v in versions)
        active_version = None
        for v in versions:
            if v.is_active:
                active_version = f"{v.model_name}:{v.model_version}"
                break

        pending_reembeds = len(
            [r for r in self._list_reembed_jobs() if r.status in ("pending", "running")]
        )

        return VersionSummary(
            total_vectors=total_vectors,
            model_versions=[v.to_dict() for v in versions],
            active_version=active_version,
            pending_reembeds=pending_reembeds,
        )

    def count_vectors_by_version(self, model_name: str, model_version: str) -> int:
        """Count vectors produced by a specific model version."""
        if self.db_enabled:
            try:
                return self._db_count_vectors(model_name, model_version)
            except Exception as exc:
                logger.warning("DB count_vectors failed: %s", exc)

        # In-memory: check registered version
        for v in self._versions.values():
            if v.model_name == model_name and v.model_version == model_version:
                return v.vector_count
        return 0

    # ------------------------------------------------------------------
    # Re-embedding
    # ------------------------------------------------------------------

    def create_reembed_job(
        self,
        source_model: str,
        source_version: str,
        target_model: str,
        target_version: str,
        target_dimension: int = 0,
        job_id: str | None = None,
    ) -> ReembedJob:
        """Create a re-embedding job to migrate vectors from one model to another."""
        reembed_id = _generate_id("reb")

        # Count source vectors
        total = self.count_vectors_by_version(source_model, source_version)

        job = ReembedJob(
            reembed_id=reembed_id,
            source_model=source_model,
            source_version=source_version,
            target_model=target_model,
            target_version=target_version,
            target_dimension=target_dimension,
            total_vectors=total,
            job_id=job_id,
        )

        if self.db_enabled:
            try:
                self._db_create_reembed(job)
                return job
            except Exception as exc:
                logger.warning("DB create_reembed failed, using memory: %s", exc)

        self._reembed_jobs[reembed_id] = job
        return job

    def update_reembed_progress(
        self,
        reembed_id: str,
        processed: int,
        failed: int = 0,
        status: str | None = None,
        error: str | None = None,
    ) -> ReembedJob | None:
        """Update progress of a re-embedding job."""
        if status and status not in VALID_REEMBED_STATUSES:
            raise ValueError(f"Invalid reembed status: {status}")

        if self.db_enabled:
            try:
                return self._db_update_reembed(
                    reembed_id, processed, failed, status, error
                )
            except Exception as exc:
                logger.warning("DB update_reembed failed: %s", exc)

        job = self._reembed_jobs.get(reembed_id)
        if job is None:
            return None
        job.processed_vectors = processed
        job.failed_vectors = failed
        if status:
            _apply_reembed_status(job, status)
        if error:
            job.error = error
        return job

    def get_reembed_job(self, reembed_id: str) -> ReembedJob | None:
        """Get a single re-embedding job."""
        if self.db_enabled:
            try:
                return self._db_get_reembed(reembed_id)
            except Exception as exc:
                logger.warning("DB get_reembed failed: %s", exc)
        return self._reembed_jobs.get(reembed_id)

    def list_reembed_jobs(self, status: str | None = None) -> list[ReembedJob]:
        """List re-embedding jobs, optionally filtered by status."""
        jobs = self._list_reembed_jobs()
        if status:
            jobs = [j for j in jobs if j.status == status]
        return jobs

    def _list_reembed_jobs(self) -> list[ReembedJob]:
        if self.db_enabled:
            try:
                return self._db_list_reembeds()
            except Exception as exc:
                logger.warning("DB list_reembeds failed: %s", exc)
        return list(self._reembed_jobs.values())

    def cleanup_old_vectors(self, model_name: str, model_version: str) -> int:
        """Remove vectors from a specific model version after migration.

        Only call this after confirming the re-embedding is complete
        and the new vectors are validated.
        """
        if self.db_enabled:
            try:
                return self._db_cleanup_vectors(model_name, model_version)
            except Exception as exc:
                logger.warning("DB cleanup_vectors failed: %s", exc)

        # In-memory: just reset count
        for v in self._versions.values():
            if v.model_name == model_name and v.model_version == model_version:
                count = v.vector_count
                v.vector_count = 0
                return count
        return 0

    # ------------------------------------------------------------------
    # Postgres implementation
    # ------------------------------------------------------------------

    def _db_register_version(self, version: VectorVersion, set_active: bool) -> None:
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                if set_active:
                    cur.execute(
                        """
                        UPDATE embedding_model_versions
                        SET is_active = FALSE
                        WHERE model_name = %s AND is_active = TRUE
                        """,
                        (version.model_name,),
                    )
                cur.execute(
                    """
                    INSERT INTO embedding_model_versions
                        (version_id, model_name, model_version, model_path,
                         embedding_dimension, is_active, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        version.version_id,
                        version.model_name,
                        version.model_version,
                        version.model_path,
                        version.embedding_dimension,
                        version.is_active,
                        version.created_at,
                    ),
                )
            conn.commit()

    def _db_get_active_version(self, model_name: str) -> VectorVersion | None:
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT version_id, model_name, model_version, model_path,
                           embedding_dimension, is_active, created_at
                    FROM embedding_model_versions
                    WHERE model_name = %s AND is_active = TRUE
                    LIMIT 1
                    """,
                    (model_name,),
                )
                row = cur.fetchone()
                if row is None:
                    return None
                return self._row_to_version(row)

    def _db_list_versions(self, model_name: str | None) -> list[VectorVersion]:
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                if model_name:
                    cur.execute(
                        """
                        SELECT version_id, model_name, model_version, model_path,
                               embedding_dimension, is_active, created_at
                        FROM embedding_model_versions
                        WHERE model_name = %s
                        ORDER BY created_at
                        """,
                        (model_name,),
                    )
                else:
                    cur.execute(
                        """
                        SELECT version_id, model_name, model_version, model_path,
                               embedding_dimension, is_active, created_at
                        FROM embedding_model_versions
                        ORDER BY created_at
                        """,
                    )
                return [self._row_to_version(row) for row in cur.fetchall()]

    def _db_count_vectors(self, model_name: str, model_version: str) -> int:
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT COUNT(*)
                    FROM memory_vectors
                    WHERE model = %s AND model_version = %s
                    """,
                    (model_name, model_version),
                )
                return cur.fetchone()[0]

    def _db_create_reembed(self, job: ReembedJob) -> None:
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO reembed_jobs
                        (reembed_id, source_model, source_version,
                         target_model, target_version, target_dimension,
                         status, total_vectors, job_id, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        job.reembed_id,
                        job.source_model,
                        job.source_version,
                        job.target_model,
                        job.target_version,
                        job.target_dimension,
                        job.status,
                        job.total_vectors,
                        job.job_id,
                        job.created_at,
                    ),
                )
            conn.commit()

    def _db_update_reembed(
        self,
        reembed_id: str,
        processed: int,
        failed: int,
        status: str | None,
        error: str | None,
    ) -> ReembedJob | None:
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                sets = ["processed_vectors = %s", "failed_vectors = %s"]
                params: list[Any] = [processed, failed]
                if status:
                    sets.append("status = %s")
                    params.append(status)
                    if status == "running":
                        sets.append("started_at = COALESCE(started_at, NOW())")
                    elif status in ("succeeded", "failed", "cancelled"):
                        sets.append("completed_at = NOW()")
                if error:
                    sets.append("error = %s")
                    params.append(error)
                params.append(reembed_id)
                cur.execute(
                    f"UPDATE reembed_jobs SET {', '.join(sets)} WHERE reembed_id = %s",
                    params,
                )
            conn.commit()
        return self.get_reembed_job(reembed_id)

    def _db_get_reembed(self, reembed_id: str) -> ReembedJob | None:
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT reembed_id, source_model, source_version,
                           target_model, target_version, target_dimension,
                           status, total_vectors, processed_vectors,
                           failed_vectors, job_id, error,
                           started_at, completed_at, created_at
                    FROM reembed_jobs
                    WHERE reembed_id = %s
                    """,
                    (reembed_id,),
                )
                row = cur.fetchone()
                if row is None:
                    return None
                return self._row_to_reembed(row)

    def _db_list_reembeds(self) -> list[ReembedJob]:
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT reembed_id, source_model, source_version,
                           target_model, target_version, target_dimension,
                           status, total_vectors, processed_vectors,
                           failed_vectors, job_id, error,
                           started_at, completed_at, created_at
                    FROM reembed_jobs
                    ORDER BY created_at DESC
                    """,
                )
                return [self._row_to_reembed(row) for row in cur.fetchall()]

    def _db_cleanup_vectors(self, model_name: str, model_version: str) -> int:
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    DELETE FROM memory_vectors
                    WHERE model = %s AND model_version = %s
                    """,
                    (model_name, model_version),
                )
                count = cur.rowcount
            conn.commit()
            return count

    @staticmethod
    def _row_to_version(row) -> VectorVersion:
        return VectorVersion(
            version_id=row[0],
            model_name=row[1],
            model_version=row[2],
            model_path=row[3],
            embedding_dimension=row[4] or 0,
            is_active=row[5],
            created_at=row[6],
        )

    @staticmethod
    def _row_to_reembed(row) -> ReembedJob:
        return ReembedJob(
            reembed_id=row[0],
            source_model=row[1],
            source_version=row[2],
            target_model=row[3],
            target_version=row[4],
            target_dimension=row[5] or 0,
            status=row[6],
            total_vectors=row[7],
            processed_vectors=row[8],
            failed_vectors=row[9],
            job_id=row[10],
            error=row[11],
            started_at=row[12],
            completed_at=row[13],
            created_at=row[14],
        )
