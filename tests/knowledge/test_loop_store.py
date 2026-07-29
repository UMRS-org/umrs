"""Tests for LoopStore (memory mode + Postgres-backed persistence)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from umrs.knowledge.loop import LoopStatus, ResearchLoop
from umrs.knowledge.loop_store import LoopStore


def _make_loop(loop_id: str, status: LoopStatus = LoopStatus.RUNNING) -> ResearchLoop:
    return ResearchLoop(
        loop_id=loop_id,
        project_id="proj-a",
        research_goal="goal-x",
        status=status,
        max_iterations=5,
        current_iteration=1,
        total_hypotheses_tested=1,
        total_experiments_run=1,
        total_findings=2,
        success_rate=0.4,
        created_at="2026-07-30T00:00:00",
        started_at="2026-07-30T00:00:00",
        updated_at="2026-07-30T00:01:00",
        metadata={"tag": "test"},
    )


class _Acquire:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _MockPool:
    def __init__(self, conn):
        self._conn = conn

    def acquire(self):
        return _Acquire(self._conn)


def _mock_pool() -> tuple[_MockPool, MagicMock]:
    conn = MagicMock()
    conn.execute = AsyncMock()
    conn.fetchrow = AsyncMock()
    conn.fetch = AsyncMock()
    return _MockPool(conn), conn


class TestMemoryMode:
    def test_roundtrip(self):
        store = LoopStore.memory()
        loop = _make_loop("loop-1")
        store.save(loop)
        assert store.get("loop-1") is loop
        assert len(store.list_all()) == 1

    def test_missing_returns_none(self):
        store = LoopStore.memory()
        assert store.get("nope") is None

    def test_clear(self):
        store = LoopStore.memory()
        store.save(_make_loop("a"))
        store.clear()
        assert store.list_all() == []

    def test_delete(self):
        store = LoopStore.memory()
        store.save(_make_loop("a"))
        store.delete("a")
        assert store.get("a") is None


class TestPoolMode:
    def test_save_executes_upsert(self):
        pool, conn = _mock_pool()
        store = LoopStore(pool)
        conn.fetchrow.return_value = None  # irrelevant for save
        store.save(_make_loop("loop-2"))
        assert conn.execute.await_count >= 1
        first_sql = conn.execute.await_args_list[0].args[0]
        assert "INSERT INTO research_loops" in first_sql
        assert "ON CONFLICT (loop_id) DO UPDATE" in first_sql

    def test_save_replaces_iterations(self):
        pool, conn = _mock_pool()
        store = LoopStore(pool)
        conn.fetchrow.return_value = {"loop_id": "loop-3"}
        store.save(_make_loop("loop-3"))
        sqls = [c.args[0] for c in conn.execute.await_args_list]
        assert any("DELETE FROM research_loop_iterations" in s for s in sqls)

    def test_get_returns_none_when_absent(self):
        pool, conn = _mock_pool()
        conn.fetchrow.return_value = None
        store = LoopStore(pool)
        assert store.get("ghost") is None

    def test_clear_wipes_tables(self):
        pool, conn = _mock_pool()
        store = LoopStore(pool)
        store.clear()
        sqls = [c.args[0] for c in conn.execute.await_args_list]
        assert any("DELETE FROM research_loops" in s for s in sqls)
        assert any("DELETE FROM research_loop_iterations" in s for s in sqls)

    def test_is_memory_flag(self):
        assert LoopStore.memory().is_memory
        pool, conn = _mock_pool()
        assert not LoopStore(pool).is_memory


class TestStoreBootMigration:
    def test_configure_upgrades_and_carries(self):
        """api.loop.configure_loop_store should carry memory loops to store."""
        import umrs.api.loop as loop_api

        pool, conn = _mock_pool()
        original = loop_api._loop_store
        try:
            loop_api.clear_loop_store()
            loop_api._loop_store.save(_make_loop("legacy"))
            loop_api.configure_loop_store(pool)
            assert conn.execute.await_count >= 1
        finally:
            loop_api._loop_store = original
