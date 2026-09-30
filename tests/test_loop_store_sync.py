"""Sync psycopg2 LoopStore + restart driver re-registration pins.

- sync_pg facade routes through psycopg2 (%s params, Json wrappers, commit).
- facade branch selection prefers sync_pool over async pool/memory.
- reregister_loop_drivers re-registers RUNNING engine loops from the store.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest
from umrs.knowledge.loop import LoopPhase, LoopStatus, ResearchLoop
from umrs.knowledge.loop_store import LoopStore


def _loop(status: LoopStatus = LoopStatus.RUNNING, driver: str = "engine") -> ResearchLoop:
    loop = ResearchLoop(
        loop_id="loop-sync-1",
        project_id="proj-a",
        research_goal="goal",
        status=status,
        phase=LoopPhase.PLANNING,
        max_iterations=2,
        current_iteration=1,
        metadata={"driver": driver},
    )
    loop.iterations = []
    return loop


class _FakeCursor:
    def __init__(self, conn: _FakeConn) -> None:
        self._conn = conn
        self.executed: list[tuple[str, Any]] = []

    def __enter__(self) -> _FakeCursor:
        return self

    def __exit__(self, *exc: Any) -> None:
        return None

    def execute(self, sql: str, params: Any = None) -> None:
        self.executed.append((sql, params))
        self._conn.calls.append((sql, params))

    def fetchone(self):
        return None

    def fetchall(self):
        return []


class _FakeConn:
    is_psycopg2 = True

    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []
        self.commits = 0

    def cursor(self, cursor_factory=None):
        return _FakeCursor(self)

    def commit(self) -> None:
        self.commits += 1


def test_sync_store_facade_uses_psycopg2_branch() -> None:
    conn = _FakeConn()
    store = LoopStore.sync_pg(conn)
    assert not store.is_memory
    store.save(_loop())
    assert conn.commits == 1
    inserts = [c for c in conn.calls if "INSERT INTO research_loops" in c[0]]
    assert inserts, "expected loop insert to reach the connection"


def test_sync_store_get_returns_none_for_missing() -> None:
    conn = _FakeConn()
    store = LoopStore.sync_pg(conn)
    assert store.get("loop-missing") is None


def test_memory_facade_unchanged() -> None:
    store = LoopStore.memory()
    store.save(_loop())
    assert store.get("loop-sync-1") is not None
    assert len(store.list_all()) == 1


class TestDriverReregistration:
    @pytest.fixture(autouse=True)
    def _memory_loop_store(self):
        """Premise store: memory, never whatever pool a prior suite booted
        into the api-level global (self-hosted CI: PG sync pool legitimately
        attached by the sidecar suites → psycopg strictness on fixture rows
        + aborted-tx poisoning of every later test in the process)."""
        from umrs.api import loop as loop_api

        prior = loop_api._loop_store
        loop_api._loop_store = LoopStore.memory()
        try:
            yield
        finally:
            loop_api._loop_store = prior

    def test_engine_loop_restores_driver(self) -> None:
        from umrs.api import loop as loop_api

        loop_api._loop_store.clear()
        loop_api._loop_store.save(_loop(LoopStatus.RUNNING, driver="engine"))

        registered: list[str] = []
        with patch(
            "umrs.knowledge.engine_loop_runner.register",
            side_effect=lambda loop, **kw: registered.append(loop.loop_id),
        ):
            resumed = loop_api.reregister_loop_drivers()
        assert resumed == 1
        assert registered == ["loop-sync-1"]

    def test_terminal_loops_not_restored(self) -> None:
        from umrs.api import loop as loop_api

        loop_api._loop_store.clear()
        loop_api._loop_store.save(_loop(LoopStatus.STOPPED))

        with patch(
            "umrs.knowledge.engine_loop_runner.register",
            side_effect=AssertionError("register must not be called"),
        ):
            assert loop_api.reregister_loop_drivers() == 0


def test_sync_save_serializes_finding_dataclasses_and_artifact_ids() -> None:
    """Engine findings are dataclasses; the sync store must JSON them (WP-05 fix)."""
    from umrs.knowledge.loop import LoopIteration, LoopPhase

    from r4_contracts.research import Finding

    loop = _loop()
    iteration = LoopIteration(
        iteration_id="it-f",
        loop_id=loop.loop_id,
        iteration_number=1,
        phase=LoopPhase.MEMORY_UPDATE,
        status="completed",
        findings=[Finding(finding_id="finding-1", statement="real", confidence=0.7)],
        artifact_ids=["finding-1"],
    )
    loop.iterations.append(iteration)
    conn = _FakeConn()
    LoopStore.sync_pg(conn).save(loop)
    inserts = [q for q in conn.calls if "INSERT INTO research_loop_iterations" in q[0]]
    assert len(inserts) == 1
    params = inserts[0][1]
    findings_param, artifact_param = params[11], params[13]
    assert findings_param.adapted == [
        {
            "finding_id": "finding-1",
            "hypothesis_id": "",
            "statement": "real",
            "evidence_refs": [],
            "workflow_run_id": "",
            "durable_job_id": None,
            "confidence": 0.7,
            "simulated": False,
        }
    ]
    assert artifact_param.adapted == ["finding-1"]
