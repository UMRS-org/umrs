"""
Persistent Endpoint Registry Backends (U-109).

This module provides pluggable storage backends for the endpoint registry:
- InMemoryBackend: Default, for single-instance deployment
- PostgresBackend: For HA deployment with persistence
- RedisBackend: For fast distributed registry (future)

The registry backend is responsible for:
- Persisting endpoint state
- Surviving UMRS restarts
- Supporting multiple UMRS instances (HA)
"""

from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# Import from the main registry module to avoid circular imports
# These will be imported at runtime


def _resolve_backend_type(backend_type: Optional[str] = None) -> str:
    """Resolve the registry backend type from explicit input or environment."""
    if backend_type:
        return backend_type.lower()

    explicit_backend = os.environ.get("UMRS_REGISTRY_BACKEND")
    if explicit_backend:
        return explicit_backend.lower()

    if os.environ.get("UMRS_REGISTRY_POSTGRES_DSN"):
        return "postgres"

    return "memory"


def _parse_datetime(value: Optional[str]) -> Optional[datetime]:
    """Parse ISO format datetime string."""
    if not value:
        return None
    # Handle both with and without 'Z' suffix
    if value.endswith("Z"):
        value = value[:-1]
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


class RegistryBackend(ABC):
    """
    Abstract base class for endpoint registry storage backends.

    All operations are synchronous. For async support, use run_in_executor.
    """

    @abstractmethod
    def save(self, endpoint: "RegisteredEndpoint") -> None:
        """Save or update an endpoint."""

    @abstractmethod
    def load(self, project_id: str, model: str) -> Optional["RegisteredEndpoint"]:
        """Load an endpoint by (project_id, model)."""

    @abstractmethod
    def load_by_job_id(self, job_id: str) -> Optional["RegisteredEndpoint"]:
        """Load an endpoint by job_id."""

    @abstractmethod
    def delete(self, job_id: str) -> bool:
        """Delete an endpoint by job_id. Returns True if deleted."""

    @abstractmethod
    def list_all(
        self,
        project_id: Optional[str] = None,
        status_filter: Optional[List[str]] = None,
    ) -> List["RegisteredEndpoint"]:
        """List all endpoints with optional filters."""

    @abstractmethod
    def clear_all(self) -> int:
        """Clear all endpoints. Returns count deleted."""

    @abstractmethod
    def health_check(self) -> bool:
        """Check if backend is healthy."""


class InMemoryBackend(RegistryBackend):
    """
    In-memory registry backend (default).

    Fast but not persistent. Suitable for:
    - Development
    - Single-instance deployments where loss on restart is acceptable
    """

    def __init__(self):
        self._by_key: Dict[Tuple[str, str], "RegisteredEndpoint"] = {}
        self._by_job_id: Dict[str, "RegisteredEndpoint"] = {}

    def save(self, endpoint: "RegisteredEndpoint") -> None:
        """Save or update an endpoint."""
        key = (endpoint.project_id, endpoint.model)
        self._by_key[key] = endpoint
        self._by_job_id[endpoint.job_id] = endpoint

    def load(self, project_id: str, model: str) -> Optional["RegisteredEndpoint"]:
        """Load an endpoint by (project_id, model)."""
        return self._by_key.get((project_id, model))

    def load_by_job_id(self, job_id: str) -> Optional["RegisteredEndpoint"]:
        """Load an endpoint by job_id."""
        return self._by_job_id.get(job_id)

    def delete(self, job_id: str) -> bool:
        """Delete an endpoint by job_id."""
        endpoint = self._by_job_id.pop(job_id, None)
        if endpoint:
            key = (endpoint.project_id, endpoint.model)
            if self._by_key.get(key) is endpoint:
                del self._by_key[key]
            return True
        return False

    def list_all(
        self,
        project_id: Optional[str] = None,
        status_filter: Optional[List[str]] = None,
    ) -> List["RegisteredEndpoint"]:
        """List all endpoints with optional filters."""
        endpoints = list(self._by_key.values())

        if project_id:
            endpoints = [e for e in endpoints if e.project_id == project_id]

        if status_filter:
            endpoints = [e for e in endpoints if e.status.value in status_filter]

        return endpoints

    def clear_all(self) -> int:
        """Clear all endpoints."""
        count = len(self._by_key)
        self._by_key.clear()
        self._by_job_id.clear()
        return count

    def health_check(self) -> bool:
        """In-memory backend is always healthy."""
        return True


class PostgresBackend(RegistryBackend):
    """
    PostgreSQL registry backend for persistent storage.

    Provides:
    - Persistence across UMRS restarts
    - Support for multiple UMRS instances (HA)
    - ACID guarantees

    Requires environment variable:
        UMRS_REGISTRY_POSTGRES_DSN=postgres://user:pass@host:5432/dbname

    Table schema (auto-created):
        CREATE TABLE IF NOT EXISTS endpoint_registry (
            job_id VARCHAR(255) PRIMARY KEY,
            project_id VARCHAR(255) NOT NULL,
            model VARCHAR(255) NOT NULL,
            host VARCHAR(255) NOT NULL,
            port INTEGER NOT NULL,
            status VARCHAR(50) NOT NULL,
            backend VARCHAR(50) NOT NULL,
            created_at TIMESTAMP NOT NULL,
            ready_at TIMESTAMP,
            last_used_at TIMESTAMP,
            expires_at TIMESTAMP,
            error_message TEXT,
            consecutive_failures INTEGER DEFAULT 0,
            metadata JSONB DEFAULT '{}',
            UNIQUE(project_id, model)
        );
        CREATE INDEX idx_registry_project ON endpoint_registry(project_id);
        CREATE INDEX idx_registry_status ON endpoint_registry(status);
    """

    def __init__(self, dsn: Optional[str] = None):
        """
        Initialize Postgres backend.

        Args:
            dsn: PostgreSQL connection string. If not provided, reads from
                 UMRS_REGISTRY_POSTGRES_DSN environment variable.
        """
        self._dsn = dsn or os.environ.get("UMRS_REGISTRY_POSTGRES_DSN")
        if not self._dsn:
            raise ValueError(
                "PostgreSQL DSN required. Set UMRS_REGISTRY_POSTGRES_DSN environment variable."
            )

        self._conn = None
        self._init_schema()

    def _get_connection(self):
        """Get or create database connection."""
        try:
            import psycopg2
        except ImportError:
            raise ImportError("psycopg2 required for PostgreSQL backend: pip install psycopg2-binary")

        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(self._dsn)
            self._conn.autocommit = True
        return self._conn

    def _init_schema(self) -> None:
        """Initialize database schema."""
        conn = self._get_connection()
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS endpoint_registry (
                    job_id VARCHAR(255) PRIMARY KEY,
                    project_id VARCHAR(255) NOT NULL,
                    model VARCHAR(255) NOT NULL,
                    host VARCHAR(255) NOT NULL,
                    port INTEGER NOT NULL,
                    status VARCHAR(50) NOT NULL,
                    backend VARCHAR(50) NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    ready_at TIMESTAMP,
                    last_used_at TIMESTAMP,
                    expires_at TIMESTAMP,
                    error_message TEXT,
                    consecutive_failures INTEGER DEFAULT 0,
                    metadata JSONB DEFAULT '{}',
                    UNIQUE(project_id, model)
                )
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_registry_project
                ON endpoint_registry(project_id)
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_registry_status
                ON endpoint_registry(status)
            """)
        logger.info("PostgreSQL registry schema initialized")

    def _row_to_endpoint(self, row) -> "RegisteredEndpoint":
        """Convert database row to RegisteredEndpoint."""
        from .endpoint_registry import EndpointStatus, RegisteredEndpoint

        return RegisteredEndpoint(
            job_id=row[0],
            project_id=row[1],
            model=row[2],
            host=row[3],
            port=row[4],
            status=EndpointStatus(row[5]),
            backend=row[6],
            created_at=row[7],
            ready_at=row[8],
            last_used_at=row[9],
            expires_at=row[10],
            error_message=row[11],
            consecutive_failures=row[12] or 0,
            metadata=row[13] or {},
        )

    def save(self, endpoint: "RegisteredEndpoint") -> None:
        """Save or update an endpoint."""
        conn = self._get_connection()
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO endpoint_registry (
                    job_id, project_id, model, host, port, status, backend,
                    created_at, ready_at, last_used_at, expires_at,
                    error_message, consecutive_failures, metadata
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (job_id) DO UPDATE SET
                    project_id = EXCLUDED.project_id,
                    model = EXCLUDED.model,
                    host = EXCLUDED.host,
                    port = EXCLUDED.port,
                    status = EXCLUDED.status,
                    backend = EXCLUDED.backend,
                    ready_at = EXCLUDED.ready_at,
                    last_used_at = EXCLUDED.last_used_at,
                    expires_at = EXCLUDED.expires_at,
                    error_message = EXCLUDED.error_message,
                    consecutive_failures = EXCLUDED.consecutive_failures,
                    metadata = EXCLUDED.metadata
                """,
                (
                    endpoint.job_id,
                    endpoint.project_id,
                    endpoint.model,
                    endpoint.host,
                    endpoint.port,
                    endpoint.status.value,
                    endpoint.backend,
                    endpoint.created_at,
                    endpoint.ready_at,
                    endpoint.last_used_at,
                    endpoint.expires_at,
                    endpoint.error_message,
                    endpoint.consecutive_failures,
                    json.dumps(endpoint.metadata),
                ),
            )
        logger.debug(f"Saved endpoint {endpoint.job_id} to Postgres")

    def load(self, project_id: str, model: str) -> Optional["RegisteredEndpoint"]:
        """Load an endpoint by (project_id, model)."""
        conn = self._get_connection()
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT job_id, project_id, model, host, port, status, backend,
                       created_at, ready_at, last_used_at, expires_at,
                       error_message, consecutive_failures, metadata
                FROM endpoint_registry
                WHERE project_id = %s AND model = %s
                """,
                (project_id, model),
            )
            row = cur.fetchone()
            return self._row_to_endpoint(row) if row else None

    def load_by_job_id(self, job_id: str) -> Optional["RegisteredEndpoint"]:
        """Load an endpoint by job_id."""
        conn = self._get_connection()
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT job_id, project_id, model, host, port, status, backend,
                       created_at, ready_at, last_used_at, expires_at,
                       error_message, consecutive_failures, metadata
                FROM endpoint_registry
                WHERE job_id = %s
                """,
                (job_id,),
            )
            row = cur.fetchone()
            return self._row_to_endpoint(row) if row else None

    def delete(self, job_id: str) -> bool:
        """Delete an endpoint by job_id."""
        conn = self._get_connection()
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM endpoint_registry WHERE job_id = %s",
                (job_id,),
            )
            deleted = cur.rowcount > 0
        if deleted:
            logger.debug(f"Deleted endpoint {job_id} from Postgres")
        return deleted

    def list_all(
        self,
        project_id: Optional[str] = None,
        status_filter: Optional[List[str]] = None,
    ) -> List["RegisteredEndpoint"]:
        """List all endpoints with optional filters."""
        conn = self._get_connection()
        query = """
            SELECT job_id, project_id, model, host, port, status, backend,
                   created_at, ready_at, last_used_at, expires_at,
                   error_message, consecutive_failures, metadata
            FROM endpoint_registry WHERE 1=1
        """
        params: List[Any] = []

        if project_id:
            query += " AND project_id = %s"
            params.append(project_id)

        if status_filter:
            placeholders = ", ".join(["%s"] * len(status_filter))
            query += f" AND status IN ({placeholders})"
            params.extend(status_filter)

        with conn.cursor() as cur:
            cur.execute(query, params)
            return [self._row_to_endpoint(row) for row in cur.fetchall()]

    def clear_all(self) -> int:
        """Clear all endpoints."""
        conn = self._get_connection()
        with conn.cursor() as cur:
            cur.execute("DELETE FROM endpoint_registry")
            count = cur.rowcount
        logger.warning(f"Cleared {count} endpoints from Postgres registry")
        return count

    def health_check(self) -> bool:
        """Check database connectivity."""
        try:
            conn = self._get_connection()
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
            return True
        except Exception as e:
            logger.error(f"Postgres health check failed: {e}")
            return False


def get_backend(backend_type: Optional[str] = None) -> RegistryBackend:
    """
    Get the appropriate registry backend based on configuration.

    Args:
        backend_type: Override backend type. If None, auto-detect from environment.

    Environment variables:
        UMRS_REGISTRY_BACKEND: "memory" | "postgres" | "redis"
        UMRS_REGISTRY_POSTGRES_DSN: PostgreSQL connection string (for postgres backend)
        UMRS_REGISTRY_REDIS_URL: Redis URL (for redis backend, future)

    Returns:
        Configured registry backend.
    """
    backend_type = _resolve_backend_type(backend_type)

    if backend_type == "memory":
        logger.info("Using in-memory registry backend")
        return InMemoryBackend()

    elif backend_type == "postgres":
        dsn = os.environ.get("UMRS_REGISTRY_POSTGRES_DSN")
        if not dsn:
            raise ValueError(
                "UMRS_REGISTRY_POSTGRES_DSN required for postgres backend"
            )
        logger.info("Using PostgreSQL registry backend")
        return PostgresBackend(dsn)

    elif backend_type == "redis":
        raise NotImplementedError("Redis backend not yet implemented")

    else:
        raise ValueError(f"Unknown registry backend: {backend_type}")
