"""Boot reconciliation recovers an attempt whose submission never finished.

Found on the stand-alone stack 2026-10-09: Postgres restarted while two jobs
were being submitted; their attempts stayed CREATED ("submitting to backend")
and the jobs RUNNING, because boot reconciliation looked only at
SUBMITTED/RUNNING attempts. Safe only with one durable runtime per process
(test_durable_runtime_singleton.py): a second runtime's boot reconciliation
took live attempts for stale ones (STATE JOBS-20261009-001).
"""

from __future__ import annotations

from types import SimpleNamespace

from umrs.jobs.models import AttemptStatus
from umrs.jobs.reconciler import Reconciler


class _Cur:
    def __init__(self, sink):
        self.sink = sink

    def execute(self, sql, params=None):
        self.sink.append((sql, params))

    def fetchall(self):
        return [("att-1", "job-1", 1, "old-instance")]

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


def test_created_attempts_are_stale_after_a_restart():
    sink: list = []
    conn = SimpleNamespace(cursor=lambda: _Cur(sink))
    pool = SimpleNamespace(getconn=lambda: conn, putconn=lambda _c: None)
    reconciler = Reconciler.__new__(Reconciler)
    reconciler.job_store = SimpleNamespace(db_pool=pool)
    reconciler.umrs_instance_id = "new-instance"

    assert reconciler._find_stale_attempts() == [("att-1", "job-1", 1, "old-instance")]
    _sql, params = sink[0]
    assert AttemptStatus.CREATED.value in params
    assert AttemptStatus.SUBMITTED.value in params and AttemptStatus.RUNNING.value in params
    assert params[-1] == "new-instance"
