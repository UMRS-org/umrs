"""Periodic scheduler service — wakes ``ConnectorSyncExecutor`` on an interval.

Closes the P2 loop: until this module exists, ``ScheduleStore.get_due`` only
fired when something manually called :meth:`ConnectorSyncExecutor.tick`.
The service runs a daemon thread that ticks at a fixed interval so cron
schedules actually execute in production.

Safety invariants:
    - One full tick never kills the loop (per-tick try/except isolation).
    - ``UMRS_SCHEDULER_ENABLED=false`` disables startup entirely.
    - ``UMRS_SCHEDULER_TICK_SECONDS`` (default 60) bounds loop frequency.
    - Clean shutdown via ``threading.Event.wait`` — no busy-polling.

The module-level default store (:func:`get_default_schedule_store`) is the
single shared ``ScheduleStore`` so API surfaces, the executor, and tests all
observe the same schedule table without import cycles.
"""

from __future__ import annotations

import logging
import os
import threading

from . import ScheduleStore
from .connector_executor import ConnectorSyncExecutor, TickSummary

logger = logging.getLogger(__name__)

_DEFAULT_INTERVAL = 60
_schedule_lock = threading.Lock()


def scheduler_enabled() -> bool:
    """True unless explicitly disabled by env."""
    return os.environ.get("UMRS_SCHEDULER_ENABLED", "true").lower() not in (
        "false",
        "0",
        "no",
    )


def tick_interval_seconds() -> float:
    """Env-configured tick interval (floor of 1s; default 60)."""
    raw = os.environ.get("UMRS_SCHEDULER_TICK_SECONDS", str(_DEFAULT_INTERVAL))
    try:
        return max(1.0, float(raw))
    except ValueError:
        logger.warning("Invalid UMRS_SCHEDULER_TICK_SECONDS=%r — using 60s", raw)
        return float(_DEFAULT_INTERVAL)


_default_store: ScheduleStore | None = None


def get_default_schedule_store() -> ScheduleStore:
    """Shared schedule table for service + API + tests."""
    global _default_store
    with _schedule_lock:
        if _default_store is None:
            _default_store = ScheduleStore()
        return _default_store


def build_executor() -> ConnectorSyncExecutor:
    """Assemble the executor with durable runner defaults."""
    from ..connectors.sync_runner import ConnectorSyncRunner
    from ..connectors.sync_state import build_default_store

    return ConnectorSyncExecutor(
        get_default_schedule_store(), ConnectorSyncRunner(build_default_store())
    )


class SchedulerService:
    """Background loop driving the connector-sync executor tick."""

    def __init__(
        self,
        executor: ConnectorSyncExecutor | None = None,
        interval: float | None = None,
    ) -> None:
        self._executor = executor or build_executor()
        self._interval = interval if interval is not None else tick_interval_seconds()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.last_summary: TickSummary | None = None
        self.tick_count: int = 0
        self.consecutive_errors: int = 0

    def start(self) -> bool:
        """Start the daemon loop. Returns False when already running/disabled."""
        if not scheduler_enabled():
            logger.info("Scheduler service disabled by UMRS_SCHEDULER_ENABLED")
            return False
        if self._thread is not None and self._thread.is_alive():
            return False
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="umrs-scheduler", daemon=True
        )
        self._thread.start()
        logger.info("Scheduler service started (interval=%ss)", self._interval)
        return True

    def stop(self, timeout: float = 5.0) -> None:
        """Signal stop and join (bounded)."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
            logger.info("Scheduler service stopped")

    def _loop(self) -> None:
        """Tick immediately, then once per interval until stopped."""
        while not self._stop.is_set():
            self._tick_once()
            self._stop.wait(self._interval)

    def _tick_once(self) -> None:
        """One isolated tick — executor failures never kill the loop."""
        try:
            self.last_summary = self._executor.tick()
            self.tick_count += 1
            self.consecutive_errors = 0
        except Exception as exc:
            self.consecutive_errors += 1
            logger.exception(
                "Scheduler tick failed (%d consecutive): %s",
                self.consecutive_errors,
                exc,
            )


_service: SchedulerService | None = None
_service_lock = threading.Lock()


def get_scheduler_service() -> SchedulerService:
    """Module singleton (mirrors janitor pattern)."""
    global _service
    with _service_lock:
        if _service is None:
            _service = SchedulerService()
        return _service


def start_scheduler() -> bool:
    """Start the global scheduler service. No-op when disabled."""
    return get_scheduler_service().start()


def stop_scheduler() -> None:
    """Stop the global scheduler service."""
    get_scheduler_service().stop()


def _reset_for_tests() -> None:
    """Test seam: drop singletons so env-dependent tests stay isolated."""
    global _service, _default_store
    with _service_lock:
        _service = None
    with _schedule_lock:
        _default_store = None
