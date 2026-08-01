"""Scheduler service tests — periodic tick wiring (Wave 1).

Covers: disabled env prevents startup, double-start safety, tick execution +
isolation (failing executor never kills the loop), clean stop, and the shared
default-store seam that the API surfaces and tests rely on.
"""

from __future__ import annotations

import pytest
from umrs.scheduler.connector_executor import TickSummary
from umrs.scheduler.service import (
    SchedulerService,
    _reset_for_tests,
    build_executor,
    get_default_schedule_store,
    get_scheduler_service,
    scheduler_enabled,
    start_scheduler,
    stop_scheduler,
    tick_interval_seconds,
)


@pytest.fixture(autouse=True)
def _clean_singletons():
    _reset_for_tests()
    yield
    _reset_for_tests()


class _FakeExecutor:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.calls = 0

    def tick(self, **_kw: object) -> TickSummary:
        self.calls += 1
        if self.fail:
            raise RuntimeError("boom")
        return TickSummary(due=0)


class TestEnvGuards:
    def test_enabled_by_default(self, monkeypatch):
        monkeypatch.delenv("UMRS_SCHEDULER_ENABLED", raising=False)
        assert scheduler_enabled()

    def test_disabled_by_env(self, monkeypatch):
        monkeypatch.setenv("UMRS_SCHEDULER_ENABLED", "false")
        assert not scheduler_enabled()

    def test_interval_default(self, monkeypatch):
        monkeypatch.delenv("UMRS_SCHEDULER_TICK_SECONDS", raising=False)
        assert tick_interval_seconds() == 60

    def test_interval_custom_and_invalid(self, monkeypatch):
        monkeypatch.setenv("UMRS_SCHEDULER_TICK_SECONDS", "5")
        assert tick_interval_seconds() == 5.0
        monkeypatch.setenv("UMRS_SCHEDULER_TICK_SECONDS", "junk")
        assert tick_interval_seconds() == 60


class TestServiceLifecycle:
    def test_disabled_never_starts_thread(self, monkeypatch):
        monkeypatch.setenv("UMRS_SCHEDULER_ENABLED", "false")
        svc = SchedulerService(executor=_FakeExecutor())
        assert svc.start() is False
        assert svc._thread is None

    def test_start_tick_and_stop(self, monkeypatch):
        monkeypatch.delenv("UMRS_SCHEDULER_ENABLED", raising=False)
        fake = _FakeExecutor()
        svc = SchedulerService(executor=fake, interval=3600)
        assert svc.start() is True
        assert svc.start() is False  # double-start guard
        svc._tick_once()
        assert fake.calls >= 1
        assert svc.tick_count >= 1
        assert svc.last_summary is not None
        svc.stop()
        assert svc._thread is None

    def test_failing_executor_never_kills_service(self, monkeypatch):
        monkeypatch.delenv("UMRS_SCHEDULER_ENABLED", raising=False)
        svc = SchedulerService(executor=_FakeExecutor(fail=True), interval=3600)
        svc._tick_once()
        after_manual = svc.consecutive_errors
        assert after_manual == 1
        assert svc.start() is True  # still starts after an exception
        before = svc.consecutive_errors
        svc._tick_once()
        assert svc.consecutive_errors == before + 1
        svc.stop()

    def test_global_start_stop(self, monkeypatch):
        monkeypatch.setenv("UMRS_SCHEDULER_ENABLED", "false")
        assert start_scheduler() is False
        monkeypatch.setenv("UMRS_SCHEDULER_ENABLED", "true")
        svc = get_scheduler_service()
        svc._interval = 3600
        assert start_scheduler() is True
        stop_scheduler()


class TestAssembly:
    def test_default_store_is_shared_singleton(self):
        assert get_default_schedule_store() is get_default_schedule_store()

    def test_build_executor_uses_default_store(self):
        executor = build_executor()
        assert executor._store is get_default_schedule_store()
