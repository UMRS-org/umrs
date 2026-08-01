"""P2 scheduled executor tests — tick behavior + failure isolation."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from umrs.connectors import ConnectorType
from umrs.connectors.sync_runner import ConnectorSyncRunner
from umrs.connectors.sync_state import InMemorySyncStateStore

# Interface-parity signatures: fakes intentionally ignore args.
# ruff: noqa: ARG002
from umrs.scheduler import ScheduleSpec, ScheduleStore
from umrs.scheduler.connector_executor import (
    ConnectorSyncExecutor,
    _build_config,
)


def _store_with(*specs) -> ScheduleStore:
    store = ScheduleStore()
    for s in specs:
        store.add(s)
    return store


def _spec(
    name="fs-daily",
    cron="0 6 * * *",
    job_type="connector_sync",
    job_config=None,
    owner="alice",
    status=None,
    last_run=None,
    run_count=0,
):
    return ScheduleSpec(
        name=name,
        cron_expr=cron,
        job_type=job_type,
        job_config=job_config or {"type": "filesystem"},
        owner_id=owner,
        status=status
        or __import__("umrs.scheduler", fromlist=["ScheduleStatus"]).ScheduleStatus.ACTIVE,
        last_run=last_run,
        run_count=run_count,
    )


class _FSConnector:
    """Registered stub so registry.get('filesystem') resolves in tests."""

    name = "filesystem-stub"

    @property
    def connector_type(self):
        return ConnectorType.FILESYSTEM

    def validate_config(self, config):
        return []

    def fetch(self, config, checkpoint=None, *args, **kwargs):
        return iter([])


@pytest.fixture(autouse=True)
def _fs_registered():
    from umrs.connectors import get_connector_registry

    registry = get_connector_registry()
    if registry.get("filesystem") is None:
        registry.register(_FSConnector())
    yield


class _RecordingRunner:
    def __init__(self):
        self.calls = []

    def run(self, connector, config, ingest=None):
        from umrs.connectors.sync_runner import SyncRunReport

        self.calls.append((config.name, config.connector_type.value))
        return SyncRunReport(connector_name=config.name, status="ok")


class _FailRunner(_RecordingRunner):
    def run(self, connector, config, ingest=None):
        from umrs.connectors.sync_runner import SyncRunReport

        return SyncRunReport(connector_name=config.name, status="error", error="boom offline")


_NOW = datetime(2026, 8, 1, 6, 0, tzinfo=timezone.utc)  # "0 6 * * *" matches (min=0,hour=6)


def test_tick_runs_due_connector_sync_only():
    sched = _spec(cron="0 6 * * *")
    store = _store_with(sched, _spec(name="vfs-off", cron="0 5 * * *"))
    runner = _RecordingRunner()
    executor = ConnectorSyncExecutor(store, runner)
    summary = executor.tick(now=_NOW)
    assert summary.due == 1
    assert summary.syncs_ok == 1
    assert runner.calls == [("fs-daily", "filesystem")]


def test_tick_marks_runs():
    sched = _spec()
    store = _store_with(sched)
    executor = ConnectorSyncExecutor(store, _RecordingRunner())
    executor.tick(now=_NOW)
    assert store.get(sched.schedule_id).run_count == 1
    assert store.get(sched.schedule_id).last_run is not None


def test_tick_skips_non_connector_jobs():
    sched = _spec(job_type="recipe_run", job_config={})
    store = _store_with(sched)
    summary = ConnectorSyncExecutor(store, _RecordingRunner()).tick(now=_NOW)
    assert summary.skipped == 1
    assert summary.syncs_ok == 0


def test_tick_failure_isolated_per_schedule():
    store = _store_with(_spec(name="ok-one"), _spec(name="broken-one"))
    executor = ConnectorSyncExecutor(store, _FailRunner())
    summary = executor.tick(now=_NOW)
    assert summary.due == 2
    assert summary.syncs_ok == 0
    assert summary.syncs_error == 2  # _FailRunner marks both error
    assert summary.any_error


def test_max_runs_guards_tick():
    sched = _spec(run_count=2)
    sched.max_runs = 2
    store = _store_with(sched)
    summary = ConnectorSyncExecutor(store, _RecordingRunner()).tick(now=_NOW)
    # Store side already expires the schedule (due == 0, status -> COMPLETED).
    assert summary.due == 0
    from umrs.scheduler import ScheduleStatus

    assert store.get(sched.schedule_id).status == ScheduleStatus.COMPLETED


def test_build_config_str_fallback_and_rejects_unknown_type():
    sched_wrong = _spec(job_config={"type": "unknowntype"})
    from umrs.connectors import get_connector_registry

    result = _build_config(sched_wrong, get_connector_registry())
    assert isinstance(result, str)
    assert result.startswith("unknown connector type")


def test_build_config_defaults_name_and_owner():
    from umrs.connectors import get_connector_registry

    spec = _spec(name="fallback-name")
    spec.job_config = {"type": "filesystem", "credentials": {"token": "x"}}
    config = _build_config(spec, get_connector_registry())
    assert config.name == "fallback-name"
    assert config.owner_id == "alice"
    assert config.credentials == {"token": "x"}


def test_run_tick_end_to_end_with_runner_durable_record():
    """runner.mark_seen + state survive (post-gate P2 checks out end-to-end)."""
    store = _store_with(_spec())
    sync_state = InMemorySyncStateStore()
    runner = ConnectorSyncRunner(store=sync_state)
    executor = ConnectorSyncExecutor(store, runner)

    class DummyConnector:
        name = "dummy"

        @property
        def connector_type(self):
            return ConnectorType.FILESYSTEM

        def validate_config(self, config):
            return []

        def fetch(self, config, checkpoint=None, *args, **kwargs):
            return iter([])

    from umrs.connectors import get_connector_registry

    registry = get_connector_registry()
    registry.register(DummyConnector())
    summary = executor.tick(now=_NOW)
    assert summary.syncs_ok == 1
    state = sync_state.get("fs-daily")
    assert state is not None
    assert state.last_status == "ok"
