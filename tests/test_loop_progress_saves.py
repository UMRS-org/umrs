"""H-1 leftover (loop-route durable saves): a control route's save writes its
status transition WITHOUT erasing a concurrent runner's progress — the mirror
of the runner's ``save_keeping_control`` (wave 2026-10-18 (9)).

Memory lane runs everywhere; the psycopg2 lane (same FOR UPDATE + merge used
by runner threads) runs on the CI Postgres leg (`--db-url`). On the old code
these fail: the store has no ``save_preserving_progress`` (AttributeError),
and a plain ``save(loop)`` of a stale route copy keeps the stale counters.
"""

from __future__ import annotations

import pytest
from umrs.knowledge.loop import LoopPhase, LoopStatus, ResearchLoop
from umrs.knowledge.loop_store import LoopStore


def _stored(loop_id: str = "loop-1") -> ResearchLoop:
    """What the store actually holds mid-turn (runner progress + metadata)."""
    loop = ResearchLoop(
        loop_id=loop_id,
        project_id="demo",
        research_goal="alloy X",
        status=LoopStatus.RUNNING,
        phase=LoopPhase.EXPERIMENT_EXECUTION,
        current_iteration=3,
    )
    loop.total_hypotheses_tested = 12
    loop.total_experiments_run = 6
    loop.total_findings = 2
    loop.success_rate = 0.6
    loop.metadata = {"driver": "engine", "last_iteration_id": "iter-3"}
    return loop


def _route_copy(loop_id: str, status: LoopStatus) -> ResearchLoop:
    """The stop/pause route's view: the same loop read earlier, control change
    only, all progress fields stale (before the runner's latest work)."""
    stale = ResearchLoop(loop_id=loop_id, project_id="demo", research_goal="alloy X")
    stale.status = status
    stale.completed_at = "2026-10-19T12:00:00+00:00" if status == LoopStatus.STOPPED else ""
    return stale


class TestMemoryLane:
    def test_stop_preserves_runner_progress(self):
        store = LoopStore.memory()
        store.save(_stored())
        store.save_preserving_progress(_route_copy("loop-1", LoopStatus.STOPPED))
        got = store.get("loop-1")
        assert got is not None
        assert (
            got.status,
            got.completed_at,
            (got.current_iteration, got.phase),
            (
                got.total_hypotheses_tested,
                got.total_experiments_run,
                got.total_findings,
                got.success_rate,
            ),
            got.metadata,
        ) == (
            LoopStatus.STOPPED,
            "2026-10-19T12:00:00+00:00",
            (3, LoopPhase.EXPERIMENT_EXECUTION),
            (12, 6, 2, 0.6),
            {"driver": "engine", "last_iteration_id": "iter-3"},
        )

    def test_pause_preserves_progress_and_drops_no_completion(self):
        store = LoopStore.memory()
        store.save(_stored())
        store.save_preserving_progress(_route_copy("loop-1", LoopStatus.PAUSED))
        got = store.get("loop-1")
        assert got is not None
        assert got.status == LoopStatus.PAUSED
        assert got.completed_at == ""
        assert got.current_iteration == 3
        assert got.metadata["last_iteration_id"] == "iter-3"

    def test_vacuum_row_lands_the_route_record_verbatim(self):
        store = LoopStore.memory()
        stale = _route_copy("ghost", LoopStatus.STOPPED)
        store.save_preserving_progress(stale)
        got = store.get("ghost")
        assert got is not None
        assert got.status == LoopStatus.STOPPED
        assert got.completed_at == "2026-10-19T12:00:00+00:00"
        assert got.current_iteration == 0

    def test_runner_never_resurrects_a_preserved_pause(self):
        """The two lanes compose: after the route pauses, a trailing runner
        save (RUNNING copy) must still honor the stored control status."""
        store = LoopStore.memory()
        store.save(_stored())
        store.save_preserving_progress(_route_copy("loop-1", LoopStatus.PAUSED))
        store.save_keeping_control(_stored())  # the runner's own stale-save
        got = store.get("loop-1")
        assert got is not None
        assert got.status == LoopStatus.PAUSED


class TestPsycopg2Lane:
    @pytest.fixture(autouse=True)
    def _cleanup(self, db_url):
        yield
        import psycopg2

        with psycopg2.connect(db_url) as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM research_loops WHERE loop_id = 'loop-1'")

    def test_stop_preserves_runner_progress_on_pg(self, db_url):
        import psycopg2

        conn = psycopg2.connect(db_url)
        try:
            store = LoopStore(sync_pool=conn)
            store.save(_stored())
            store.save_preserving_progress(_route_copy("loop-1", LoopStatus.STOPPED))
            got = store.get("loop-1")
            assert got is not None
            assert got.status == LoopStatus.STOPPED
            assert got.completed_at == "2026-10-19T12:00:00+00:00"
            assert (got.current_iteration, got.phase) == (3, LoopPhase.EXPERIMENT_EXECUTION)
            assert got.total_hypotheses_tested == 12
            assert got.metadata["last_iteration_id"] == "iter-3"
        finally:
            conn.close()


def test_route_stop_and_pause_preserve_runner_progress(monkeypatch):
    """Route-level: stop and pause land their control transition without
    erasing the runner's progress (the wave's contract; plain save() would
    have kept the route's stale counters instead)."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from umrs.api import loop as loop_api

    app = FastAPI()
    store = LoopStore.memory()
    store.save(_stored("api-1"))
    monkeypatch.setattr(loop_api, "_loop_store", store)
    app.include_router(loop_api.router)
    client = TestClient(app)
    headers = {"X-User-ID": "u1", "X-Project-ID": "demo"}

    resp = client.post("/umrs/loops/api-1/stop", headers=headers)
    assert resp.status_code == 200, resp.text
    got = store.get("api-1")
    assert got is not None
    assert got.status == LoopStatus.STOPPED
    assert got.completed_at != ""
    assert (got.current_iteration, got.phase) == (3, LoopPhase.EXPERIMENT_EXECUTION)
    assert got.total_hypotheses_tested == 12
    assert got.metadata["last_iteration_id"] == "iter-3"
