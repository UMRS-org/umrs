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
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .endpoint_registry import RegisteredEndpoint

logger = logging.getLogger(__name__)


# Import from the main registry module to avoid circular imports
# These will be imported at runtime


def _resolve_backend_type(backend_type: str | None = None) -> str:
    """Resolve the registry backend type from explicit input or environment."""
    if backend_type:
        return backend_type.lower()

    explicit_backend = os.environ.get("UMRS_REGISTRY_BACKEND")
    if explicit_backend:
        return explicit_backend.lower()

    if os.environ.get("UMRS_REGISTRY_POSTGRES_DSN"):
        return "postgres"

    if _shared_registry_enabled():
        return "postgres"

    return "memory"


def _shared_registry_enabled() -> bool:
    """ADR-060 U-1: the endpoint registry lives in the shared ``umrs_db``
    (one hub registry), venue-scoped by site name — set
    ``UMRS_ENDPOINT_REGISTRY_SHARED`` on every umrs-front server."""
    return os.environ.get("UMRS_ENDPOINT_REGISTRY_SHARED", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def _env_first(*names: str, default: str) -> str:
    """First non-empty env value among *names* (mirrors shared/pg's chain)."""
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return default


def _shared_registry_dsn() -> str:
    """DSN of the shared jobs DB built from the standard DB_* components
    (the same resolution chain ``umrs.shared.pg`` pools use)."""
    from urllib.parse import quote

    db = _env_first("UMRS_JOBS_DB", "POSTGRES_DB", default="umrs_db")
    host = _env_first("DB_HOST", default="127.0.0.1")
    port = _env_first("DB_PORT", "POSTGRES_PORT", default="5432")
    user = _env_first("DB_USER", "DB_SUPERUSER", "POSTGRES_USER", default="postgres")
    password = _env_first(
        "DB_PASSWORD", "DB_SUPERUSER_PASSWORD", "POSTGRES_PASSWORD", default="postgres"
    )
    return f"postgres://{quote(user, safe='')}:{quote(password, safe='')}@{host}:{port}/{db}"


def _shared_registry_venue() -> str:
    """This server's venue: the site.yaml ``name`` (jobs' claim scope)."""
    try:
        from umrs.jobs.store import configured_site_name

        return configured_site_name()
    except Exception:  # noqa: BLE001 — no site config: unscoped dev lane
        return ""


def _registry_read_only() -> bool:
    """``UMRS_REGISTRY_READ_ONLY``: this process reads a registry that other
    processes own (umrs-rag over the hub table the servers write) — it never
    runs DDL, never upserts, never deletes (U-1 follow-up 2026-10-03)."""
    return os.environ.get("UMRS_REGISTRY_READ_ONLY", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def drop_read_only_flag_in_owner(process: str = "umrs-server") -> bool:
    """An owning server never runs a read-only registry (2026-10-03).

    On umrs-front every server loads ``env_file: .env``: a
    ``UMRS_REGISTRY_READ_ONLY`` put there for umrs-rag would make every
    server a reader that persists no endpoint, so its endpoints vanish at
    the next restart and the janitor reaps their jobs as orphans. The owner
    drops the flag from its environment, loudly. Returns True when dropped.
    """
    if not _registry_read_only():
        return False
    logger.error(
        "UMRS_REGISTRY_READ_ONLY is set in the %s process — IGNORED: a server "
        "owns its endpoint registry and must persist it. The flag is for "
        "umrs-rag only (start.sh passes it there); remove it from .env.",
        process,
    )
    os.environ.pop("UMRS_REGISTRY_READ_ONLY", None)
    return True


def _reader_venue() -> str:
    """Venue a read-only reader sees: ``UMRS_REGISTRY_VENUE`` when set (a
    reader such as umrs-rag carries no site.yaml), else the site name; ""
    leaves the read unscoped — every venue — and is logged as such."""
    venue = (
        os.environ.get("UMRS_REGISTRY_VENUE", "").strip() or _shared_registry_venue()
    )
    if not venue:
        logger.warning(
            "read-only registry has no venue (UMRS_REGISTRY_VENUE unset, no "
            "site.yaml) — it reads EVERY venue's endpoints"
        )
    return venue


class RegistryReadOnlyError(RuntimeError):
    """A write reached a read-only registry backend."""


def _parse_datetime(value: str | None) -> datetime | None:
    """Parse ISO format datetime string."""
    if not value:
        return None
    # Handle both with and without 'Z' suffix
    value = value.removesuffix("Z")
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


class RegistryBackend(ABC):
    """
    Abstract base class for endpoint registry storage backends.

    All operations are synchronous. For async support, use run_in_executor.
    """

    #: True for a reader of a registry other processes own: the
    #: EndpointRegistry then never writes it and re-reads it on list_all.
    read_only: bool = False

    @abstractmethod
    def save(self, endpoint: RegisteredEndpoint) -> None:
        """Save or update an endpoint."""

    @abstractmethod
    def load(self, project_id: str, model: str) -> RegisteredEndpoint | None:
        """Load an endpoint by (project_id, model)."""

    @abstractmethod
    def load_by_job_id(self, job_id: str) -> RegisteredEndpoint | None:
        """Load an endpoint by job_id."""

    @abstractmethod
    def delete(self, job_id: str) -> bool:
        """Delete an endpoint by job_id. Returns True if deleted."""

    @abstractmethod
    def list_all(
        self,
        project_id: str | None = None,
        status_filter: list[str] | None = None,
    ) -> list[RegisteredEndpoint]:
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
        self._by_key: dict[tuple[str, str], RegisteredEndpoint] = {}
        self._by_job_id: dict[str, RegisteredEndpoint] = {}

    def save(self, endpoint: RegisteredEndpoint) -> None:
        """Save or update an endpoint."""
        key = (endpoint.project_id, endpoint.model)
        self._by_key[key] = endpoint
        self._by_job_id[endpoint.job_id] = endpoint

    def load(self, project_id: str, model: str) -> RegisteredEndpoint | None:
        """Load an endpoint by (project_id, model)."""
        return self._by_key.get((project_id, model))

    def load_by_job_id(self, job_id: str) -> RegisteredEndpoint | None:
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
        project_id: str | None = None,
        status_filter: list[str] | None = None,
    ) -> list[RegisteredEndpoint]:
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
            variant VARCHAR(255),
            created_at TIMESTAMP NOT NULL,
            ready_at TIMESTAMP,
            last_used_at TIMESTAMP,
            expires_at TIMESTAMP,
            error_message TEXT,
            consecutive_failures INTEGER DEFAULT 0,
            metadata JSONB DEFAULT '{}'
        );
        CREATE INDEX idx_registry_project ON endpoint_registry(project_id);
        CREATE INDEX idx_registry_status ON endpoint_registry(status);
    """

    def __init__(
        self, dsn: str | None = None, *, venue: str = "", read_only: bool = False
    ):
        """
        Initialize Postgres backend.

        Args:
            dsn: PostgreSQL connection string. If not provided, reads from
                 UMRS_REGISTRY_POSTGRES_DSN environment variable.
            venue: ADR-060 U-1 venue scope (site.yaml ``name``). When set,
                 every row is stamped and every read filtered to this
                 venue — one shared registry table, per-site visibility.
            read_only: reader of a table other processes own (umrs-rag):
                 no DDL at start, a read-only session, and save/delete/
                 clear_all raise RegistryReadOnlyError — a reader's boot
                 snapshot must never overwrite the owner's rows (its
                 upsert would rewrite every column, venue included).
        """
        self._dsn = dsn or os.environ.get("UMRS_REGISTRY_POSTGRES_DSN")
        if not self._dsn:
            raise ValueError(
                "PostgreSQL DSN required. Set UMRS_REGISTRY_POSTGRES_DSN environment variable."
            )
        self._venue = venue
        self.read_only = read_only

        self._conn = None
        if read_only:
            # The owning servers and migration 0117 own the schema.
            return
        self._ensure_database()
        self._init_schema()
        self._migrate_schema()

    def _refuse_write(self, operation: str) -> None:
        """Raise when this backend is a read-only reader."""
        if self.read_only:
            raise RegistryReadOnlyError(
                f"endpoint registry backend is read-only "
                f"(venue={getattr(self, '_venue', '')!r}); refused {operation}"
            )

    def _venue_clause(self, params: list[Any]) -> str:
        """``AND venue = %s`` (+ param) when venue-scoped, else nothing."""
        venue = getattr(self, "_venue", "")
        if not venue:
            return ""
        params.append(venue)
        return " AND venue = %s"

    def _ensure_database(self) -> None:
        """Create the target database if it does not exist.

        Parses the DSN to extract the database name, then connects to the
        default ``postgres`` database to issue ``CREATE DATABASE`` if needed.
        This fixes the deployment issue where the DSN points to a database
        that was never created (e.g. ``umrs_rccs``).
        """
        try:
            import psycopg2
        except ImportError:
            raise ImportError(
                "psycopg2 required for PostgreSQL backend: pip install psycopg2-binary"
            ) from None

        # Parse DSN: postgres://user:pass@host:port/dbname
        from urllib.parse import urlparse

        parsed = urlparse(self._dsn)
        db_name = parsed.path.lstrip("/")
        if not db_name:
            return  # No database in DSN, nothing to create

        # Build a DSN to the default 'postgres' database
        admin_dsn = self._dsn.rsplit("/", 1)[0] + "/postgres"

        from umrs.shared.pg_settings import pg_connect_kwargs

        try:
            conn = psycopg2.connect(
                admin_dsn, **pg_connect_kwargs("orchestrator-admin")
            )
            conn.autocommit = True
            with conn.cursor() as cur:
                cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (db_name,))
                if cur.fetchone() is None:
                    # Database doesn't exist, create it
                    cur.execute(f'CREATE DATABASE "{db_name}"')
                    logger.info(f"Auto-created database: {db_name}")
            conn.close()
        except Exception as e:
            logger.error(f"Failed to ensure database exists: {e}")
            raise

    def _get_connection(self):
        """Get or create database connection."""
        try:
            import psycopg2
        except ImportError:
            raise ImportError(
                "psycopg2 required for PostgreSQL backend: pip install psycopg2-binary"
            ) from None

        from umrs.shared.pg_settings import pg_connect_kwargs

        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(
                self._dsn, **pg_connect_kwargs("orchestrator-registry")
            )
            if self.read_only:
                # Server-enforced too: default_transaction_read_only.
                self._conn.set_session(readonly=True, autocommit=True)
            else:
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
                    variant VARCHAR(255),
                    created_at TIMESTAMP NOT NULL,
                    ready_at TIMESTAMP,
                    last_used_at TIMESTAMP,
                    expires_at TIMESTAMP,
                    error_message TEXT,
                    consecutive_failures INTEGER DEFAULT 0,
                    metadata JSONB DEFAULT '{}'
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

    @staticmethod
    def _select_columns() -> tuple[str, ...]:
        """Authoritative positional column contract (ADR-058).

        ``_row_to_endpoint`` indexes rows positionally; appending new
        columns must stay *at the end* with these names so old runs of the
        reader map identical fields. Pinned by test to catch reorderings.
        """
        return (
            "job_id",
            "project_id",
            "model",
            "host",
            "port",
            "status",
            "backend",
            "variant",
            "created_at",
            "ready_at",
            "last_used_at",
            "expires_at",
            "error_message",
            "consecutive_failures",
            "metadata",
            "backend_job_id",
        )

    def _migrate_schema(self) -> None:
        """Apply incremental schema migrations for existing tables.

        Handles upgrades from older schema versions:
        - Adds ``variant`` column if missing
        - Removes legacy ``UNIQUE(project_id, model)`` constraint if present
        """
        conn = self._get_connection()
        with conn.cursor() as cur:
            # Check if variant column exists
            cur.execute("""
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'endpoint_registry'
                AND column_name = 'variant'
            """)
            if cur.fetchone() is None:
                cur.execute(
                    "ALTER TABLE endpoint_registry ADD COLUMN variant VARCHAR(255)"
                )
                logger.info("Added 'variant' column to endpoint_registry")

            # ADR-058 D-1 (WP-22): backend's own job identity, set-once.
            # Nullable honestly (rows pre-dating the contract or never
            # claim-resolved keep NULL — never fabricated).
            cur.execute("""
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'endpoint_registry'
                AND column_name = 'backend_job_id'
            """)
            if cur.fetchone() is None:
                cur.execute(
                    "ALTER TABLE endpoint_registry ADD COLUMN backend_job_id VARCHAR(255)"
                )
                logger.info("Added 'backend_job_id' column to endpoint_registry")

            # Check for and remove legacy unique constraint
            cur.execute("""
                SELECT conname FROM pg_constraint
                WHERE conrelid = 'endpoint_registry'::regclass
                AND contype = 'u'
            """)
            for row in cur.fetchall():
                constraint_name = row[0]
                cur.execute(
                    f"ALTER TABLE endpoint_registry DROP CONSTRAINT {constraint_name}"
                )
                logger.info(f"Removed legacy constraint: {constraint_name}")

            # ADR-060 U-1: one hub registry — venue column scopes each row to
            # its site (stamped on write, filtered on read when venue-mode).
            cur.execute("""
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'endpoint_registry'
                AND column_name = 'venue'
            """)
            if cur.fetchone() is None:
                cur.execute(
                    "ALTER TABLE endpoint_registry "
                    "ADD COLUMN venue VARCHAR(32) NOT NULL DEFAULT ''"
                )
                logger.info("Added 'venue' column to endpoint_registry")
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_registry_venue
                ON endpoint_registry(venue)
            """)

    def _row_to_endpoint(self, row) -> RegisteredEndpoint:
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
            variant=row[7],
            created_at=row[8],
            ready_at=row[9],
            last_used_at=row[10],
            expires_at=row[11],
            error_message=row[12],
            consecutive_failures=row[13] or 0,
            metadata=row[14] or {},
            backend_job_id=row[15] or None,
        )

    def save(self, endpoint: RegisteredEndpoint) -> None:
        """Save or update an endpoint (venue-stamped when venue-scoped)."""
        self._refuse_write("save")
        conn = self._get_connection()
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO endpoint_registry (
                    job_id, project_id, model, host, port, status, backend,
                    variant, created_at, ready_at, last_used_at, expires_at,
                    error_message, consecutive_failures, metadata, backend_job_id,
                    venue
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                ON CONFLICT (job_id) DO UPDATE SET
                    project_id = EXCLUDED.project_id,
                    model = EXCLUDED.model,
                    host = EXCLUDED.host,
                    port = EXCLUDED.port,
                    status = EXCLUDED.status,
                    backend = EXCLUDED.backend,
                    variant = EXCLUDED.variant,
                    ready_at = EXCLUDED.ready_at,
                    last_used_at = EXCLUDED.last_used_at,
                    expires_at = EXCLUDED.expires_at,
                    error_message = EXCLUDED.error_message,
                    consecutive_failures = EXCLUDED.consecutive_failures,
                    metadata = EXCLUDED.metadata,
                    backend_job_id = EXCLUDED.backend_job_id,
                    venue = EXCLUDED.venue
                """,
                (
                    endpoint.job_id,
                    endpoint.project_id,
                    endpoint.model,
                    endpoint.host,
                    endpoint.port,
                    endpoint.status.value,
                    endpoint.backend,
                    endpoint.variant,
                    endpoint.created_at,
                    endpoint.ready_at,
                    endpoint.last_used_at,
                    endpoint.expires_at,
                    endpoint.error_message,
                    endpoint.consecutive_failures,
                    json.dumps(endpoint.metadata),
                    endpoint.backend_job_id,
                    self._venue,
                ),
            )
        logger.debug(f"Saved endpoint {endpoint.job_id} to Postgres")

    def load(self, project_id: str, model: str) -> RegisteredEndpoint | None:
        """Load an endpoint by (project_id, model) in this venue."""
        conn = self._get_connection()
        params: list[Any] = [project_id, model]
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT job_id, project_id, model, host, port, status, backend,
                       variant, created_at, ready_at, last_used_at, expires_at,
                       error_message, consecutive_failures, metadata,
                       backend_job_id
                FROM endpoint_registry
                WHERE project_id = %s AND model = %s
                """
                + self._venue_clause(params),
                params,
            )
            row = cur.fetchone()
            return self._row_to_endpoint(row) if row else None

    def load_by_job_id(self, job_id: str) -> RegisteredEndpoint | None:
        """Load an endpoint by job_id in this venue."""
        conn = self._get_connection()
        params: list[Any] = [job_id]
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT job_id, project_id, model, host, port, status, backend,
                       variant, created_at, ready_at, last_used_at, expires_at,
                       error_message, consecutive_failures, metadata,
                       backend_job_id
                FROM endpoint_registry
                WHERE job_id = %s
                """
                + self._venue_clause(params),
                params,
            )
            row = cur.fetchone()
            return self._row_to_endpoint(row) if row else None

    def delete(self, job_id: str) -> bool:
        """Delete an endpoint by job_id in this venue."""
        self._refuse_write("delete")
        conn = self._get_connection()
        params: list[Any] = [job_id]
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM endpoint_registry WHERE job_id = %s"
                + self._venue_clause(params),
                params,
            )
            deleted = cur.rowcount > 0
        if deleted:
            logger.debug(f"Deleted endpoint {job_id} from Postgres")
        return deleted

    def list_all(
        self,
        project_id: str | None = None,
        status_filter: list[str] | None = None,
    ) -> list[RegisteredEndpoint]:
        """List all endpoints with optional filters."""
        conn = self._get_connection()
        # Same 16 columns as load()/load_by_job_id(): _row_to_endpoint reads
        # row[15]. The 15-column list here made every startup load raise
        # IndexError (2026-09-18 → 10-01), so each restart began with an empty
        # registry and the janitor reaped every live endpoint as an orphan.
        params: list[Any] = []
        query = """
            SELECT job_id, project_id, model, host, port, status, backend,
                   variant, created_at, ready_at, last_used_at, expires_at,
                   error_message, consecutive_failures, metadata,
                   backend_job_id
            FROM endpoint_registry WHERE 1=1
        """ + self._venue_clause(params)

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
        """Clear all endpoints (own venue only — never another site's)."""
        self._refuse_write("clear_all")
        conn = self._get_connection()
        params: list[Any] = []
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM endpoint_registry WHERE 1=1" + self._venue_clause(params),
                params,
            )
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


def _postgres_backend() -> PostgresBackend:
    """The Postgres backend: shared hub table (ADR-060 U-1) or the
    per-cluster DSN; a reader (``UMRS_REGISTRY_READ_ONLY``) gets a
    read-only backend scoped by ``_reader_venue``."""
    read_only = _registry_read_only()
    reader: dict[str, Any] = {"read_only": True} if read_only else {}
    if _shared_registry_enabled():
        venue = _reader_venue() if read_only else _shared_registry_venue()
        logger.info(
            "Using shared PostgreSQL registry backend (umrs_db, venue=%r, "
            "read_only=%s) — ADR-060 U-1",
            venue,
            read_only,
        )
        return PostgresBackend(_shared_registry_dsn(), venue=venue, **reader)
    dsn = os.environ.get("UMRS_REGISTRY_POSTGRES_DSN")
    if not dsn:
        raise ValueError("UMRS_REGISTRY_POSTGRES_DSN required for postgres backend")
    logger.info("Using PostgreSQL registry backend (read_only=%s)", read_only)
    return PostgresBackend(dsn, **reader)


def get_backend(backend_type: str | None = None) -> RegistryBackend:
    """
    Get the appropriate registry backend based on configuration.

    Args:
        backend_type: Override backend type. If None, auto-detect from environment.

    Environment variables:
        UMRS_REGISTRY_BACKEND: "memory" | "postgres" | "redis"
        UMRS_REGISTRY_POSTGRES_DSN: PostgreSQL connection string (for postgres backend)
        UMRS_ENDPOINT_REGISTRY_SHARED: the hub table in umrs_db (ADR-060 U-1)
        UMRS_REGISTRY_READ_ONLY: reader of a registry other processes own
        UMRS_REGISTRY_VENUE: a reader's venue scope (default: site.yaml name)
        UMRS_REGISTRY_REDIS_URL: Redis URL (for redis backend, future)

    Returns:
        Configured registry backend.
    """
    backend_type = _resolve_backend_type(backend_type)

    if backend_type == "memory":
        logger.info("Using in-memory registry backend")
        return InMemoryBackend()

    elif backend_type == "postgres":
        return _postgres_backend()

    elif backend_type == "redis":
        raise NotImplementedError("Redis backend not yet implemented")

    else:
        raise ValueError(f"Unknown registry backend: {backend_type}")
