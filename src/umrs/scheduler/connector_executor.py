"""P2 scheduled executor for expires at ``job_type = "connector_sync"``.

Sequentially peeks due schedules from the cron :class:`ScheduleStore`
(``get_due(now)``), resolves the connector from
:class:`umrs.connectors.ConnectorRegistry`, sets up a
``ConnectorConfig`` from the schedule's ``job_config``, and drives the
:class:`umrs.connectors.sync_runner.ConnectorSyncRunner`.

Each schedule failure returns an executor-level ``ExecutorResult.result``
with ``status="error"``; the runner itself never raises past its own
failure isolation, so one bad connector never stalls the rest of the
clock-tick.

Usage::

    executor = ConnectorSyncExecutor(schedule_store, sync_runner)
    results = executor.run_tick()

Optional: an ``ingest`` hook (knowledge-ingest callback) can be injected
per run; default no-op.

Schedule/job_**cfg****payload shape:
{
  "type": "filesystem"           # connector_type.value
  "name": "nightly-lab-scan"     # connector instance name (durable state key)
  "settings": {"path": "/data"},
  "credentials": {...},          # optional
  "project_id": "...",           # optional
  "owner_id": "...",             # owner from schedule row
}
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from ..connectors import ConnectorConfig, ConnectorType
from ..connectors.sync_runner import SyncRunReport

logger = logging.getLogger(__name__)


@dataclass
class ExecutorResult:
    """One scheduled execution outcome."""

    schedule_id: str
    schedule_name: str
    status: str  # "ok" | "error" | "skipped"
    reason: str = ""
    report: SyncRunReport | None = None


@dataclass
class TickSummary:
    """Aggregate for one executor tick."""

    due: int = 0
    syncs_ok: int = 0
    syncs_error: int = 0
    skipped: int = 0
    results: list[ExecutorResult] = field(default_factory=list)

    @property
    def any_error(self) -> bool:
        return self.syncs_error > 0


class ConnectorSyncExecutor:
    """Runs due connector_sync schedule entries against the registry."""

    def __init__(self, schedule_store, sync_runner) -> None:
        self._store = schedule_store
        self._runner = sync_runner

    def tick(
        self,
        *,
        registry: Any | None = None,
        ingest: Callable | None = None,
        now: Any = None,
    ) -> TickSummary:
        """Execute all due connector_sync schedules once."""
        from ..connectors import get_connector_registry

        reg = registry or get_connector_registry()
        due = self._store.get_due(now)
        summary = TickSummary(due=len(due))
        for sched in due:
            result = self._execute_one(sched, reg, ingest)
            summary.results.append(result)
            if result.status == "ok":
                summary.syncs_ok += 1
            elif result.status == "error":
                summary.syncs_error += 1
            else:
                summary.skipped += 1
        return summary

    # ── internals ─────────────────────────────────────────────────────────

    def _execute_one(self, sched, registry, ingest) -> ExecutorResult:
        """One schedule → one runner invocation."""
        if sched.job_type != "connector_sync":
            return ExecutorResult(
                schedule_id=sched.schedule_id,
                schedule_name=sched.name,
                status="skipped",
                reason=f"job_type={sched.job_type} not connector_sync",
            )
        if sched.max_runs is not None and sched.run_count >= sched.max_runs:
            return ExecutorResult(
                schedule_id=sched.schedule_id,
                schedule_name=sched.name,
                status="skipped",
                reason="max_runs exhausted",
            )

        config = _build_config(sched, registry)
        if isinstance(config, str):  # build failed → error reason
            return ExecutorResult(
                schedule_id=sched.schedule_id,
                schedule_name=sched.name,
                status="error",
                reason=config,
            )

        connector = registry.get(config.connector_type.value)
        if connector is None:
            return ExecutorResult(
                schedule_id=sched.schedule_id,
                schedule_name=sched.name,
                status="error",
                reason=f"connector {config.connector_type.value} not registered",
            )

        report = self._runner.run(connector, config, ingest=ingest)
        reason_of_status = report.status
        status = "ok" if reason_of_status in ("ok", "skipped_disabled") else "error"
        self._store.mark_run(sched.schedule_id)
        return ExecutorResult(
            schedule_id=sched.schedule_id,
            schedule_name=sched.name,
            status=status,
            reason=report.error or "",
            report=report,
        )


def _build_config(sched, registry) -> ConnectorConfig | str:
    """Map schedule job_config onto a ConnectorConfig (name-bearing)."""
    cfg = sched.job_config or {}
    error_or_type = _resolve_type(cfg, registry)
    if isinstance(error_or_type, str):
        return error_or_type
    return _finish_config(sched, cfg, error_or_type)


def _resolve_type(cfg: dict, registry) -> ConnectorType | str:
    """Validate connector type from cfg keys; return error string if bad."""
    raw_type = str(cfg.get("type") or cfg.get("connector") or "")
    try:
        ctype = ConnectorType(raw_type)
    except ValueError:
        return f"unknown connector type {raw_type!r}"
    if ctype.value not in registry:
        return f"connector {ctype.value} not registered"
    return ctype


def _finish_config(sched, cfg: dict, ctype: ConnectorType) -> ConnectorConfig:
    """Compose the final ConnectorConfig with fallbacks."""
    return ConnectorConfig(
        connector_type=ctype,
        name=str(cfg.get("name") or sched.name),
        credentials=dict(cfg.get("credentials") or {}),
        settings=dict(cfg.get("settings") or {}),
        project_id=cfg.get("project_id") or sched.project_id,
        owner_id=cfg.get("owner_id") or sched.owner_id,
        enabled=bool(cfg.get("enabled", True)),
    )


__all__ = ["ConnectorSyncExecutor", "ExecutorResult", "TickSummary"]
