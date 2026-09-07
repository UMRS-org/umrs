"""Attempt phase-span persistence + observability (WP-12, G-P11-1)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace

from umrs.jobs.store import JobStore

T0 = datetime(2026, 9, 7, 10, 0, 0, tzinfo=timezone.utc)
T1 = datetime(2026, 9, 7, 10, 0, 30, tzinfo=timezone.utc)


class _FakeCursor:
    def __init__(self, rows):
        self._rows = rows

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def execute(self, sql, params):
        if "metrics" in sql.lower() and "job_attempts" in sql.lower():
            payload = json.loads(params[0])
            self._rows[params[1]] = payload


class _FakeConn:
    def __init__(self, rows):
        self._cursor = _FakeCursor(rows)

    def cursor(self, cursor_factory=None):
        return self._cursor

    def commit(self):
        return None


class _FakePool:
    def __init__(self):
        self.rows = {}
        self.conn = _FakeConn(self.rows)

    def getconn(self):
        return self.conn

    def putconn(self, _conn):
        return None


class _TestStore(JobStore):
    def _ensure_schema(self):
        return None


def _store_with_attempt():
    pool = _FakePool()
    store = _TestStore(pool)
    attempt = SimpleNamespace(attempt_id="att-1", backend="slurm", started_at=T0)
    store.get_attempt = lambda attempt_id: attempt
    return store, pool


class TestRecordAttemptTiming:
    def test_queue_wait_persisted_and_observed(self, monkeypatch):
        store, pool = _store_with_attempt()
        observed: list[tuple] = []
        monkeypatch.setattr(
            "umrs.jobs.store.METRICS.attempt_phase_seconds",
            lambda authority, phase, seconds: observed.append((authority, phase, seconds)),
        )
        store._record_attempt_timing("att-1", "running", T1)
        assert pool.rows["att-1"] == {"timing": {"queue_wait_ms": 30000}}
        assert observed == [("slurm", "queue_wait", 30.0)]

    def test_wall_phase(self, monkeypatch):
        store, pool = _store_with_attempt()
        monkeypatch.setattr(
            "umrs.jobs.store.METRICS.attempt_phase_seconds",
            lambda authority, phase, seconds: None,
        )
        end = datetime(2026, 9, 7, 12, 0, 0, tzinfo=timezone.utc)
        store._record_attempt_timing("att-1", "ended", end)
        assert pool.rows["att-1"] == {"timing": {"wall_ms": 7200000}}

    def test_unknown_started_at_persists_nothing_not_silence_bug(self, monkeypatch):
        store, pool = _store_with_attempt()
        attempt = SimpleNamespace(attempt_id="att-2", backend="k8s", started_at=None)
        store.get_attempt = lambda _id: attempt
        monkeypatch.setattr(
            "umrs.jobs.store.METRICS.attempt_phase_seconds",
            lambda authority, phase, seconds: None,
        )
        store._record_attempt_timing("att-2", "running", T1)
        assert "att-2" not in pool.rows  # nothing fabricated

    def test_failure_is_best_effort(self):
        pool = _FakePool()
        store = _TestStore(pool)

        def boom(_id):
            raise RuntimeError("attempt row gone")

        store.get_attempt = boom
        # must not raise — timing never blocks a transition
        store._record_attempt_timing("att-x", "running", T1)


class TestLoopIterationHookMath:
    def test_engine_label_and_duration(self):
        from umrs.jobs.timing import duration_ms

        ms = duration_ms(
            "2026-09-07T10:00:00+00:00", "2026-09-07T10:01:30+00:00"
        )
        assert ms == 90000
