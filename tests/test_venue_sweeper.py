"""Sweeper lifespan wiring: projection of torn-down sessions + gating."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from umrs.jobs.venue_sweeper import VenueSweeper, build_sweeper
from umrs.venues.frontend_config import FrontendVenueConfig
from umrs.venues.venue_sessions import VenueSessionStore


class _Executor:
    def __init__(self, torn: list[str], errors: list[str] | None = None):
        self._torn = torn
        self._errors = errors or []
        self.calls = 0

    def sweep_once(self, *, actor: str = "sweeper"):
        del actor
        self.calls += 1
        return SimpleNamespace(
            expired=len(self._torn), idled=0, torn_down=list(self._torn), errors=list(self._errors)
        )


def _store_with(sid: str) -> VenueSessionStore:
    store = VenueSessionStore()
    store.upsert_provisioning(
        session_id=sid,
        user_id="alice",
        subnet="172.30.0.0/24",
        source_names=("s",),
        source_ids=("11111111-1111-4111-8111-111111111111",),
        default_source_id="11111111-1111-4111-8111-111111111111",
        default_source_name="s",
        web_policy="public",
    )
    store.mark_running(sid, lease_id="l", egress_ip="i")
    return store


class TestSweepAndProject:
    def test_torn_down_rows_marked_stopped(self):
        store = _store_with("s1")
        sweeper = VenueSweeper(_Executor(["s1"]), store, interval_sec=5)
        result = sweeper.sweep_and_project()
        assert result.torn_down == ["s1"]
        assert store.get("s1").state == "stopped"
        assert "swept" in store.get("s1").reason

    def test_run_loop_stops(self):
        store = _store_with("s1")
        executor = _Executor([])
        sweeper = VenueSweeper(executor, store, interval_sec=5)

        async def _drive():
            task = asyncio.create_task(sweeper.run())
            await asyncio.sleep(0.05)
            sweeper.stop()
            await asyncio.wait_for(task, timeout=2)

        asyncio.run(_drive())
        assert executor.calls >= 1


class TestBuild:
    def test_disabled_returns_none(self):
        app = SimpleNamespace(state=SimpleNamespace(umrs_db_pool=object()))
        cfg = FrontendVenueConfig.from_env({})
        assert build_sweeper(app, cfg) is None

    def test_enabled_without_pool_returns_none(self):
        app = SimpleNamespace(state=SimpleNamespace(umrs_db_pool=None))
        cfg = FrontendVenueConfig.from_env({"UMRS_VENUE_SWEEP_ENABLED": "1"})
        assert build_sweeper(app, cfg) is None
