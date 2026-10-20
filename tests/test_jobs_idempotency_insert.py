"""Job creation idempotency insert shape (2026-10-20, ops-watch 2026-10-17).

The store used to race the ``idx_jobs_idempotency`` unique index with a
plain INSERT and catch ``UniqueViolation`` Python-side — a fully handled
path that still logged an ERROR server-side
(``duplicate key value violates unique constraint "idx_jobs_idempotency"``
on umrs-postgres, filed by ops-watch). The insert is now
``ON CONFLICT … DO NOTHING RETURNING`` with the SELECT fallback, so the
handled race logs nothing.
"""

from __future__ import annotations

from typing import Any

import pytest
from umrs.jobs.models import JobType
from umrs.jobs.store import DuplicateJobError, JobStore


class _FakeCursor:
    """Serves fetchone() from a scripted queue and records executed SQL."""

    def __init__(self, fetchone_queue: list[Any]) -> None:
        self.queries: list[str] = []
        self.params: list[tuple] = []
        self._queue = list(fetchone_queue)
        self.rowcount = 0

    def execute(self, sql: str, params: tuple | None = None) -> None:
        self.queries.append(sql)
        self.params.append(params or ())
        self.rowcount = 0

    def fetchone(self):
        return self._queue.pop(0) if self._queue else None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeConn:
    def __init__(self, cur: _FakeCursor) -> None:
        self.cur = cur
        self.commits = 0
        self.rollbacks = 0

    def cursor(self):
        return self.cur

    def commit(self) -> None:
        self.commits += 1

    def rollback(self) -> None:
        self.rollbacks += 1


class _FakePool:
    def __init__(self, conn: _FakeConn) -> None:
        self._conn = conn

    def getconn(self):
        return self._conn

    def putconn(self, conn) -> None:
        pass


def _store(monkeypatch, cur: _FakeCursor) -> JobStore:
    import umrs.jobs.store as store_mod

    monkeypatch.setattr(store_mod, "ensure_schema", lambda _conn: None)
    return JobStore(_FakePool(_FakeConn(cur)))


def test_create_job_insert_uses_on_conflict_do_nothing_returning(monkeypatch):
    cur = _FakeCursor(fetchone_queue=[("job-new",)])
    store = _store(monkeypatch, cur)

    job_id = store.create_job("proj", "user", JobType.WORKFLOW_RUN, {"x": 1})

    assert job_id
    insert = cur.queries[0]
    assert "ON CONFLICT (project_id, idempotency_key) DO NOTHING" in insert
    assert "RETURNING job_id" in insert
    assert len(cur.queries) == 1  # no SELECT fallback on the winning path


def test_create_job_conflict_selects_existing_and_raises_duplicate(monkeypatch):
    # RETURNING empty (race lost), then the fallback SELECT finds the row.
    cur = _FakeCursor(fetchone_queue=[None, ("job-existing",)])
    store = _store(monkeypatch, cur)

    with pytest.raises(DuplicateJobError, match="job-existing"):
        store.create_job("proj", "user", JobType.WORKFLOW_RUN, {"x": 1})

    assert len(cur.queries) == 2
    assert "SELECT job_id FROM jobs" in cur.queries[1]
