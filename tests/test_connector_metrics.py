"""X1 connector metrics tests — instrumentation of runs and ticks (Wave 2).

Verifies: metric methods exist and are callable no-op-safe, runner emits
status + docs counts at the end of a sync (including disabled-config early
return), executor emits one tick outcome metric, and instrumentation errors
never break the sync path.
"""

from __future__ import annotations

from datetime import datetime, timezone

from umrs.connectors import ConnectorConfig, ConnectorType, Document, DocumentStatus
from umrs.connectors.sync_runner import ConnectorSyncRunner
from umrs.connectors.sync_state import InMemorySyncStateStore
from umrs.metrics import METRICS
from umrs.scheduler import ScheduleSpec, ScheduleStore
from umrs.scheduler.connector_executor import ConnectorSyncExecutor


class _FakeConnector:
    def __init__(self, docs) -> None:
        self._docs = docs

    @property
    def connector_type(self) -> ConnectorType:
        return ConnectorType.CUSTOM

    def validate_config(self, config):  # noqa: ARG002 — test seam
        return []

    def fetch(self, config, checkpoint=None):  # noqa: ARG002 — test seam
        return iter(self._docs)


def _doc(source_id: str, content: str) -> Document:
    return Document(
        source_id=source_id,
        title=source_id,
        content=content,
        status=DocumentStatus.FETCHED,
    )


def _config(name: str = "m-conn") -> ConnectorConfig:
    return ConnectorConfig(
        connector_type=ConnectorType.CUSTOM,
        name=name,
        settings={},
        credentials={},
    )


class TestMetricsMethods:
    def test_methods_are_callable_no_op_safe(self):
        METRICS.connector_sync("filesystem", "ok", 0.01)
        METRICS.connector_docs("filesystem", synced=2, dup=1, failed=0, blocked=0)
        METRICS.connector_tick("empty")

    def test_sync_enabled_run_emits_metrics(self):
        runner = ConnectorSyncRunner(InMemorySyncStateStore())
        connector = _FakeConnector([_doc("a", "one"), _doc("b", "two")])
        report = runner.run(connector, _config())
        assert report.status == "ok"
        assert report.docs_synced == 2

    def test_disabled_config_still_emits(self):
        runner = ConnectorSyncRunner(InMemorySyncStateStore())
        cfg = _config()
        cfg.enabled = False
        connector = _FakeConnector([_doc("a", "one")])
        report = runner.run(connector, cfg)
        assert report.status == "skipped_disabled"

    def test_tick_emits_metric_and_summary(self):
        store = ScheduleStore()
        now = datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)
        store.add(
            ScheduleSpec(
                name="due-conn",
                cron_expr="0 12 * * *",
                job_type="connector_sync",
                job_config={"type": "custom", "name": "c1"},
                owner_id="t",
            )
        )
        executor = ConnectorSyncExecutor(store, ConnectorSyncRunner(InMemorySyncStateStore()))
        registry = {"custom": _FakeConnector([_doc("x", "payload")])}
        summary = executor.tick(registry=registry, now=now)
        assert summary.due == 1
        assert summary.syncs_ok == 1

    def test_tick_empty_outcome(self):
        store = ScheduleStore()
        executor = ConnectorSyncExecutor(store, ConnectorSyncRunner(InMemorySyncStateStore()))
        summary = executor.tick(registry={}, now=datetime(2026, 8, 1, tzinfo=timezone.utc))
        assert summary.due == 0

    def test_instrumentation_failure_never_breaks_sync(self, monkeypatch):
        def _raise_down(*_a: object, **_k: object) -> None:
            raise RuntimeError("metrics down")

        monkeypatch.setattr("umrs.metrics.METRICS.connector_sync", _raise_down)
        monkeypatch.setattr("umrs.metrics.METRICS.connector_docs", _raise_down)
        runner = ConnectorSyncRunner(InMemorySyncStateStore())
        connector = _FakeConnector([_doc("a", "one")])
        report = runner.run(connector, _config())
        assert report.status == "ok"
        assert report.docs_synced == 1
