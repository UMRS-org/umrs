"""Shared lazy asyncpg pool for knowledge-store routes.

asyncpg-based stores (``KnowledgeStore``, PG ``VerificationStore``) call
``pool.acquire()`` and therefore require an asyncpg pool. The shared
``app.state.umrs_db_pool`` is a psycopg2 ``ThreadedConnectionPool`` (durable
jobs) and must not be passed to those stores — doing so surfaces as
``'ThreadedConnectionPool' object has no attribute 'acquire'`` 500s.

The pool is created once from the platform DSN, cached on ``app.state``
(``umrs_knowledge_asyncpg_pool``), and reused across requests.

Codec parity with ``jobs.knowledge_adapters._open_pool``: without symmetric
JSON/JSONB codecs, asyncpg rejects dict parameters for jsonb columns
(DataError) — this silently killed knowledge writes done through this pool
(ADR-027 P1a minting on rag upload is the first write consumer).
"""

import json
import os
from typing import Any

from fastapi import HTTPException, Request

from umrs.api.typed_errors import PlatformError, raise_typed


async def _init_json_codecs(conn: Any) -> None:
    """Register symmetric JSON/JSONB codecs (dict encode/decode)."""
    for typename in ("json", "jsonb"):
        await conn.set_type_codec(
            typename,
            encoder=json.dumps,
            decoder=json.loads,
            schema="pg_catalog",
        )


def _resolve_knowledge_dsn() -> tuple[str, bool]:
    """Resolve asyncpg DSN from env, mirroring the durable-jobs pool chain.

    Returns (dsn, is_local). Precedence: UMRS_DATA_CATALOG_DB / DATABASE_URL
    (full DSN), otherwise the same component vars the jobs pool uses
    (DB_HOST/DB_PORT/UMRS_JOBS_DB/POSTGRES_*/DB_SUPERUSER*).

    Empty-config parity with ``jobs.knowledge_adapters._resolve_knowledge_dsn``:
    returns ``("", False)`` when no DSN and no component var is present, so an
    unconfigured environment hits the designed 503 ("no DSN") instead of
    silently fabricating a localhost admin-DSN that connects to an ambient
    postgres with default credentials.
    """
    for var in ("UMRS_DATA_CATALOG_DB", "DATABASE_URL"):
        dsn = os.environ.get(var, "")
        if dsn:
            return dsn, False
    component_vars = (
        "DB_HOST",
        "DB_PORT",
        "UMRS_JOBS_DB",
        "POSTGRES_PORT",
        "POSTGRES_DB",
        "DB_SUPERUSER",
        "DB_SUPERUSER_PASSWORD",
        "POSTGRES_USER",
        "POSTGRES_PASSWORD",
    )
    if not any(os.environ.get(var) for var in component_vars):
        return "", False
    host = os.environ.get("DB_HOST", "127.0.0.1")
    port = os.environ.get("DB_PORT", os.environ.get("POSTGRES_PORT", "5432"))
    name = os.environ.get("UMRS_JOBS_DB", os.environ.get("POSTGRES_DB", "umrs_db"))
    user = os.environ.get("DB_SUPERUSER", os.environ.get("POSTGRES_USER", "postgres"))
    pw = os.environ.get(
        "DB_SUPERUSER_PASSWORD", os.environ.get("POSTGRES_PASSWORD", "postgres")
    )
    dsn = f"postgresql://{user}:{pw}@{host}:{port}/{name}"
    return dsn, host in ("127.0.0.1", "localhost", "::1")


async def knowledge_asyncpg_pool(request: Request):
    """Return the app-cached asyncpg pool, creating it on first use.

    Raises:
        HTTPException(503): if no DSN is configured or asyncpg is missing.
    """
    pool = getattr(request.app.state, "umrs_knowledge_asyncpg_pool", None)
    if pool is not None:
        return pool

    dsn, is_local = _resolve_knowledge_dsn()
    if not dsn:
        raise HTTPException(status_code=503, detail="Database unavailable (no DSN)")
    try:
        import asyncpg
    except ImportError as exc:
        raise_typed(
            PlatformError.asyncpgpool(
                "unavailable",
                "asyncpg unavailable in server install"
            ),
            cause=exc,
        )

    from umrs.shared.pg_settings import asyncpg_pool_kwargs

    # R-14 session hygiene: application_name + idle-in-transaction reaper.
    kwargs: dict = asyncpg_pool_kwargs(
        "knowledge-asyncpg", dsn=dsn, init=_init_json_codecs
    )
    if is_local:
        kwargs["ssl"] = "disable"  # container-local PG has no TLS (jobs pool parity)
    pool = await asyncpg.create_pool(**kwargs)
    request.app.state.umrs_knowledge_asyncpg_pool = pool
    return pool
