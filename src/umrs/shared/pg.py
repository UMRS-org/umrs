"""Shared synchronous Postgres pool helper.

Lightweight companion to ``umrs.server.job_tracking_handler``'s pool builder:
this helper never creates databases or touches the admin connection — it only
opens a small ``ThreadedConnectionPool`` against an already-provisioned UMRS
database. Foundation-layer modules (research satellites, knowledge) use it so
they do not need to import from ``umrs.server``.

Environment variables (same names as the durable jobs bootstrap):

  DB_HOST                     (default 127.0.0.1)
  DB_PORT / POSTGRES_PORT     (default 5432)
  UMRS_DB / UMRS_JOBS_DB / POSTGRES_DB  (default umrs_db)
  DB_USER / DB_SUPERUSER / POSTGRES_USER      (default postgres)
  DB_PASSWORD / DB_SUPERUSER_PASSWORD / POSTGRES_PASSWORD (default postgres)
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any

logger = logging.getLogger(__name__)

_pool_lock = threading.Lock()
_default_pool: Any = None


def _env_first(*names: str, default: str = "") -> str:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return default


def pg_configured_from_env() -> bool:
    """Return True when the environment points at a Postgres instance.

    Cheap check only — does not open a connection. ``DB_HOST`` must be set
    explicitly so local dev/CI without a database keeps using in-memory
    stores instead of paying connection timeouts at 127.0.0.1.
    """
    return bool(os.environ.get("DB_HOST") or os.environ.get("DATABASE_URL"))


def build_sync_pool_from_env(*, min_conn: int = 1, max_conn: int = 4) -> Any:
    """Build a ThreadedConnectionPool from process environment.

    Raises:
        ImportError: If psycopg2 is unavailable.
        psycopg2.Error: If the database is unreachable.
    """
    from psycopg2 import pool as pg_pool

    dsn = os.environ.get("DATABASE_URL")
    kwargs: dict[str, Any] = {"minconn": min_conn, "maxconn": max_conn}
    if dsn:
        return pg_pool.ThreadedConnectionPool(dsn=dsn, **kwargs)
    return pg_pool.ThreadedConnectionPool(
        host=_env_first("DB_HOST", default="127.0.0.1"),
        port=_env_first("DB_PORT", "POSTGRES_PORT", default="5432"),
        dbname=_env_first("UMRS_DB", "UMRS_JOBS_DB", "POSTGRES_DB", default="umrs_db"),
        user=_env_first("DB_USER", "DB_SUPERUSER", "POSTGRES_USER", default="postgres"),
        password=_env_first(
            "DB_PASSWORD",
            "DB_SUPERUSER_PASSWORD",
            "POSTGRES_PASSWORD",
            default="postgres",
        ),
        # Local container deployments do not support SSL (matches the durable
        # jobs bootstrap which uses sslmode=disable).
        sslmode="disable",
        **kwargs,
    )


def get_default_sync_pool() -> Any:
    """Process-wide lazily-built pool shared by satellite PG stores."""
    global _default_pool
    with _pool_lock:
        if _default_pool is None:
            _default_pool = build_sync_pool_from_env()
    return _default_pool


def reset_default_sync_pool() -> None:
    """Close and drop the shared pool (test teardown / reload hooks)."""
    global _default_pool
    with _pool_lock:
        if _default_pool is not None:
            try:
                _default_pool.closeall()
            except Exception:  # noqa: BLE001 — teardown must not raise
                logger.warning("Failed to close shared PG pool", exc_info=True)
            _default_pool = None


def resolve_satellite_pool(backend: str, db_pool: Any, label: str) -> Any:
    """Shared pool-selection logic for research-satellite store factories.

    Args:
        backend: ``"pg"`` or ``"auto"`` ("memory" is handled by callers).
        db_pool: Explicit pool; when given it always wins.
        label: Log label for fallback warnings.

    Returns:
        A connection pool, or None when the caller should use in-memory.

    Raises:
        Exception: For backend ``"pg"`` when no pool can be built.
    """
    if db_pool is not None:
        return db_pool
    if backend == "pg":
        return get_default_sync_pool()
    if not pg_configured_from_env():
        return None
    try:
        return get_default_sync_pool()
    except Exception:  # noqa: BLE001 — auto mode degrades to in-memory
        logger.warning(
            "%s: Postgres unavailable, falling back to in-memory store",
            label,
            exc_info=True,
        )
        return None
