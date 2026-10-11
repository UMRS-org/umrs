"""KNOWLEDGE_EXPORT, KNOWLEDGE_VERIFY and the domain benchmark run on an
initialized pool, and closing it never masks their result.

Found 2026-10-12 (ADR-093 S8): the three backends took the pool from a sync
``_get_pool()`` that returned an un-awaited ``asyncpg.create_pool`` (never
initialized), and their ``finally`` called ``pool.close()`` un-awaited then
``await pool.wait_closed()``, which an asyncpg pool does not have. Every
export failed with ``'Pool' object has no attribute 'wait_closed'``, the
error that hid the real one.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from umrs.jobs import knowledge_adapters as ka


class _AsyncpgLikePool:
    """What ``await asyncpg.create_pool(...)`` returns: ``close`` is a
    coroutine and there is no ``wait_closed``."""

    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


def test_the_created_pool_is_awaited(monkeypatch):
    pool = _AsyncpgLikePool()
    created: list[dict] = []

    async def create_pool(**kwargs):
        created.append(kwargs)
        return pool

    import asyncpg

    monkeypatch.setattr(asyncpg, "create_pool", create_pool)
    monkeypatch.setattr(ka, "_resolve_knowledge_dsn", lambda: "postgresql://x/y")
    assert asyncio.run(ka._created_pool()) is pool
    assert created and created[0]["dsn"] == "postgresql://x/y"


def test_no_dsn_is_no_pool(monkeypatch):
    monkeypatch.setattr(ka, "_resolve_knowledge_dsn", lambda: "")
    assert asyncio.run(ka._created_pool()) is None


@pytest.mark.parametrize("closing", ["coroutine", "sync", "raises"])
def test_closing_never_raises(closing):
    class _Pool:
        def close(self):
            if closing == "raises":
                raise RuntimeError("already closed")
            if closing == "coroutine":
                return asyncio.sleep(0)
            return None

    asyncio.run(ka._close_pool(_Pool()))
    asyncio.run(ka._close_pool(None))


def test_the_export_closes_an_asyncpg_pool_without_wait_closed(monkeypatch):
    pool = _AsyncpgLikePool()

    async def created():
        return pool

    async def export(self, *_a, **_kw):
        return SimpleNamespace(
            artifact_count=1,
            embedding_count=0,
            manifest_hash="h",
            archive_key="",
            degraded=False,
            degraded_reason="",
        )

    monkeypatch.setattr(ka, "_created_pool", created)
    monkeypatch.setattr(ka.KnowledgeExportBackend, "_export_with_integrity", export)
    monkeypatch.setattr(ka.KnowledgeExportBackend, "_publish_archive", lambda *_a: None)
    monkeypatch.setattr(ka, "_write_result", lambda *_a: None)
    job = SimpleNamespace(
        job_id="j-1", project_id="p", spec={"project_id": "p"}, user_id="u"
    )
    attempt = SimpleNamespace(attempt_id="a-1", metrics={})
    ref = asyncio.run(ka.KnowledgeExportBackend().submit_job(job, attempt))
    assert ref.startswith("knowledge_export|project=p|artifacts=1")
    assert pool.closed
