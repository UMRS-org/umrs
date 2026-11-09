"""Real-Postgres helpers for the wave C/D tests (skip without a database).

``ledger_pool_or_skip()`` returns a pool over a database migrated to 0136
with 0137/0138 applied on top (both files are idempotent), or skips. Run
the suite with ``DATABASE_URL`` (or ``UMRS_DATA_CATALOG_DB``) naming a
pg15 database so these tests do not pass vacuously.
"""

from __future__ import annotations

from typing import Any

import pytest

from umrs import migrations

NEW_MIGRATIONS = ("0137_hpc_grants.sql", "0138_hpc_usage_ledger.sql")


def _migrated(conn: Any) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT 1 FROM information_schema.columns WHERE table_name = "
            "'workflow_executions' AND column_name = 'intent_id'"
        )
        return cur.fetchone() is not None


def ledger_pool_or_skip() -> Any:
    pytest.importorskip("psycopg2")
    from umrs.api._asyncpg_pool import _resolve_knowledge_dsn
    from umrs.cli.admin.cascade import _ConnPerCallPool

    dsn, _ = _resolve_knowledge_dsn()
    if not dsn:
        pytest.skip("no Postgres configured")
    pool = _ConnPerCallPool(dsn)
    try:
        conn = pool.getconn()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Postgres unavailable: {exc}")
    try:
        if not _migrated(conn):
            pytest.skip("database not migrated to 0136 (python -m umrs.migrations)")
        with conn.cursor() as cur:
            for name in NEW_MIGRATIONS:
                cur.execute((migrations.MIGRATIONS_DIR / name).read_text())
        conn.commit()
    finally:
        pool.putconn(conn)
    return pool


def execute(pool: Any, sql: str, params: tuple[Any, ...] = ()) -> list[tuple[Any, ...]]:
    conn = pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall() if cur.description else []
        conn.commit()
        return list(rows)
    except Exception:
        conn.rollback()
        raise
    finally:
        pool.putconn(conn)


def delete_ledger_rows(pool: Any, venues: list[str]) -> None:
    """Test cleanup: the append-only trigger is disabled for this session only."""
    conn = pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute("SET session_replication_role = replica")
            cur.execute("DELETE FROM hpc_usage_ledger WHERE venue = ANY(%s)", (venues,))
            cur.execute("SET session_replication_role = DEFAULT")
        conn.commit()
    except Exception:  # noqa: BLE001 - not a superuser: the rows stay (unique venues)
        conn.rollback()
    finally:
        pool.putconn(conn)
