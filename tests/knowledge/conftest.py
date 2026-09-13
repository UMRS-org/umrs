"""Shared fixtures for knowledge tests.

Several knowledge test modules are synchronous and drive coroutines via
``asyncio.get_event_loop().run_until_complete``. pytest-asyncio (auto mode)
closes/unsets the thread's event loop after async tests in other modules
run, which breaks ``asyncio.get_event_loop()`` when the full suite runs
together. Provide a fresh, current event loop per test.
"""

from __future__ import annotations

import asyncio

import pytest


@pytest.fixture(autouse=True)
def _fresh_event_loop():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield
    loop.close()
    asyncio.set_event_loop(None)


# Tables the knowledge premise suites read-after-write against. When a
# knowledge DSN resolves (CI self-hosted, or any host with deployment
# creds), leftover rows from previous runs break counts (rehearsal
# 2026-09-13: 100 != 21).
_KNOWLEDGE_PREMISE_TABLES = (
    "knowledge_node_embeddings",
    "knowledge_edges",
    "knowledge_nodes",
    "sat_knowledge_object_versions",
    "sat_knowledge_tombstones",
    "sat_knowledge_objects",
    "scientific_memory_entries",
    "hypothesis_history",
    "hypotheses",
    "evidence_items",
    "experiment_results",
    "experiments",
)


def _truncate_knowledge_premise_tables() -> None:
    """TRUNCATE premise tables when — and only when — a knowledge DSN
    resolves. No-op on hosts without PG (tests then ride in-memory state)."""
    try:
        from umrs.jobs.knowledge_adapters import _resolve_knowledge_dsn

        dsn = _resolve_knowledge_dsn()
        if not dsn:
            return
        import psycopg2

        conn = psycopg2.connect(dsn)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT tablename FROM pg_tables "
                    "WHERE schemaname='public' AND tablename = ANY(%s)",
                    (list(_KNOWLEDGE_PREMISE_TABLES),),
                )
                existing = [row[0] for row in cur.fetchall()]
                for table in existing:
                    cur.execute(f"TRUNCATE TABLE {table} CASCADE")
        finally:
            conn.commit()
            conn.close()
    except Exception:  # noqa: BLE001 — isolation helper must not break collection
        pass


@pytest.fixture(autouse=True, scope="module")
def _clean_knowledge_premise_state():
    """Module-scoped clean slate for PG-resolved knowledge premise suites."""
    _truncate_knowledge_premise_tables()
    yield
