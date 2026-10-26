"""The four umrs-front servers share umrs_db: each monitor and each boot
reconciliation must stay on its own site's attempts, and a transition another
writer won first is a lost race, not a server error.

Ops-watch #587 (bdrt, "Cannot mark attempt … as running") and #588 (ai4s,
"Cannot mark attempt … as failed"), 2026-10-25. 7afb44c1 scoped the claim
only: get_active_attempts() and the boot reconciler still read every site's
SUBMITTED/RUNNING rows, so a front polled (and transitioned) the other
fronts' attempts, and a restarted front marked their live attempts LOST.
"""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from umrs.jobs.models import AttemptStatus
from umrs.jobs.monitor import AttemptCheckResult, AttemptMonitor
from umrs.jobs.reconciler import Reconciler
from umrs.jobs.store import InvalidTransitionError, JobStore


class _Cur:
    def __init__(self, sink, rows):
        self.sink, self.rows = sink, rows

    def execute(self, sql, params=None):
        self.sink.append((sql, params))

    def fetchall(self):
        return self.rows

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


class _Conn:
    def __init__(self, sink, rows=()):
        self.sink, self.rows = sink, list(rows)

    def cursor(self, *_a, **_k):
        return _Cur(self.sink, self.rows)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


def _attempt(status=AttemptStatus.SUBMITTED):
    return SimpleNamespace(attempt_id="att-1", job_id="job-1", status=status)


def _monitor(store, result):
    checker = MagicMock()
    checker.check_status = AsyncMock(return_value=result)
    return AttemptMonitor(job_store=store, status_checker=checker)


def _errors(caplog):
    return [r for r in caplog.records if r.levelno >= logging.ERROR]


def test_lost_running_race_is_not_an_error(caplog):
    store = MagicMock()
    store.mark_attempt_running.side_effect = InvalidTransitionError(
        "Cannot mark attempt att-1 as running"
    )
    monitor = _monitor(store, AttemptCheckResult.running("slurm state RUNNING"))

    with caplog.at_level(logging.INFO, logger="umrs.jobs.monitor"):
        asyncio.run(monitor._check_attempt(_attempt()))

    assert _errors(caplog) == []
    assert "att-1" in caplog.text


def test_lost_failure_race_is_not_an_error(caplog):
    store = MagicMock()
    store.mark_attempt_failed.side_effect = InvalidTransitionError(
        "Cannot mark attempt att-1 as failed"
    )
    monitor = _monitor(store, AttemptCheckResult.failed("slurm state FAILED"))

    with caplog.at_level(logging.INFO, logger="umrs.jobs.monitor"):
        asyncio.run(monitor._check_attempt(_attempt(AttemptStatus.RUNNING)))

    assert _errors(caplog) == []
    assert monitor.stats["attempts_failed"] == 0


def test_other_store_errors_still_log_an_error(caplog):
    store = MagicMock()
    store.mark_attempt_failed.side_effect = RuntimeError("connection closed")
    monitor = _monitor(store, AttemptCheckResult.failed("boom"))

    asyncio.run(monitor._check_attempt(_attempt(AttemptStatus.RUNNING)))

    assert len(_errors(caplog)) == 1


def test_active_attempts_are_scoped_to_this_site():
    sink: list = []
    store = JobStore.__new__(JobStore)
    store.site_name = "bdrt"
    store._get_conn = lambda: _Conn(sink)

    assert store.get_active_attempts() == []
    sql, params = sink[0]
    assert "venue" in sql
    assert params[-1] == "bdrt"


def test_boot_reconcile_takes_only_this_sites_attempts():
    sink: list = []
    pool = SimpleNamespace(getconn=lambda: _Conn(sink), putconn=lambda _c: None)
    reconciler = Reconciler.__new__(Reconciler)
    reconciler.job_store = SimpleNamespace(db_pool=pool, site_name="ai4s")
    reconciler.umrs_instance_id = "new-instance"

    reconciler._find_stale_attempts()
    sql, params = sink[0]
    assert "venue" in sql
    assert "ai4s" in params
    assert params[-1] == "new-instance"
