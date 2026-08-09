"""Shared lazy asyncpg pool for knowledge-store routes.

asyncpg-based stores (``KnowledgeStore``, PG ``VerificationStore``) call
``pool.acquire()`` and therefore require an asyncpg pool. The shared
``app.state.umrs_db_pool`` is a psycopg2 ``ThreadedConnectionPool`` (durable
jobs) and must not be passed to those stores — doing so surfaces as
``'ThreadedConnectionPool' object has no attribute 'acquire'`` 500s.

The pool is created once from the platform DSN, cached on ``app.state``
(``umrs_knowledge_asyncpg_pool``), and reused across requests.
"""

import os

from fastapi import HTTPException, Request


async def knowledge_asyncpg_pool(request: Request):
    """Return the app-cached asyncpg pool, creating it on first use.

    Raises:
        HTTPException(503): if no DSN is configured or asyncpg is missing.
    """
    pool = getattr(request.app.state, "umrs_knowledge_asyncpg_pool", None)
    if pool is not None:
        return pool

    conn_str = os.environ.get("UMRS_DATA_CATALOG_DB") or os.environ.get("DATABASE_URL") or ""
    if not conn_str:
        raise HTTPException(status_code=503, detail="Database unavailable (no DSN)")
    try:
        import asyncpg
    except ImportError as exc:
        raise HTTPException(status_code=503, detail="asyncpg unavailable in server install") from exc

    pool = await asyncpg.create_pool(dsn=conn_str)
    request.app.state.umrs_knowledge_asyncpg_pool = pool
    return pool
