"""Lifespan wiring for the frontend-venue sweeper (R1-4 tail, wave 2026-09-21).

``VenueRuntimeExecutor.sweep_once`` had no caller in the server process:
expired/idle leases were never torn down unless someone ran the executor
by hand. This module owns one asyncio task on the sidecar loop (janitor
style) that runs the sweep in a worker thread every
``UMRS_VENUE_SWEEP_INTERVAL_SEC`` and projects torn-down sessions onto the
0099 registry rows. Gate: ``UMRS_VENUE_SWEEP_ENABLED=1`` (one API container
per host — teardown is idempotent, so a second sweeper is harmless).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)

_task: asyncio.Task | None = None
_sweeper: VenueSweeper | None = None


class VenueSweeper:
    def __init__(self, executor: Any, store: Any, *, interval_sec: int = 60) -> None:
        self._executor = executor
        self._store = store
        self._interval = max(5, int(interval_sec))
        self._stop = asyncio.Event()

    def sweep_and_project(self) -> Any:
        """One pass (thread-safe, synchronous): sweep + registry projection."""
        result = self._executor.sweep_once()
        for sid in result.torn_down:
            try:
                self._store.mark_stopped(sid, reason="swept: lease expired or idle")
            except Exception:  # noqa: BLE001 — projection is best-effort
                logger.warning("venue sweeper: registry projection failed for %s", sid)
        if result.errors:
            logger.warning(
                "venue sweeper: %d error(s): %s", len(result.errors), result.errors[:3]
            )
        return result

    async def run(self) -> None:
        while not self._stop.is_set():
            try:
                await asyncio.to_thread(self.sweep_and_project)
            except Exception:
                logger.exception("venue sweeper pass failed")
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self._interval)
            except asyncio.TimeoutError:
                continue

    def stop(self) -> None:
        self._stop.set()


def build_sweeper(app: Any, config: Any | None = None) -> VenueSweeper | None:
    """Construct the sweeper from env + app state; None when disabled/unwired."""
    from umrs.api.secret_broker import get_broker
    from umrs.venues.frontend_config import FrontendVenueConfig
    from umrs.venues.frontend_podman import FrontendPodmanProvisioner
    from umrs.venues.venue_sessions import VenueSessionStore

    from .session_venue_runtime import VenueRuntimeExecutor

    cfg = config or FrontendVenueConfig.from_env()
    if not cfg.sweep_enabled:
        return None
    pool = getattr(app.state, "umrs_db_pool", None)
    if pool is None:
        logger.warning("venue sweeper enabled but no shared DB pool — not started")
        return None
    broker = get_broker()
    provisioner = FrontendPodmanProvisioner(broker, session_root=cfg.session_root)
    executor = VenueRuntimeExecutor(broker, provisioner, idle_seconds=cfg.idle_seconds)
    return VenueSweeper(
        executor, VenueSessionStore(pool), interval_sec=cfg.sweep_interval_sec
    )


def start_venue_sweeper(app: Any) -> bool:
    """Start the loop on the running event loop; True when started."""
    global _task, _sweeper
    if _task is not None and not _task.done():
        return True
    sweeper = build_sweeper(app)
    if sweeper is None:
        return False
    _sweeper = sweeper
    _task = asyncio.get_running_loop().create_task(
        sweeper.run(), name="umrs-venue-sweeper"
    )
    return True


def stop_venue_sweeper() -> None:
    global _task, _sweeper
    if _sweeper is not None:
        _sweeper.stop()
    if _task is not None:
        _task.cancel()
    _task = None
    _sweeper = None
