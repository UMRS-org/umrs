"""ADR-060 U-1 rollout helper pins: per-cluster table → hub registry copy.

Idempotent (ON CONFLICT DO NOTHING), venue-stamped, honest 0 on a missing
source table, positional column contract (ADR-058) with venue 17th.
"""

from __future__ import annotations

from umrs.migrations.hub_endpoint_registry_copy import (
    SOURCE_COLUMNS,
    copy_endpoints,
)


class _Cur:
    def __init__(self, conn):
        self._conn = conn
        self.rowcount = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self._conn.record.append((" ".join(str(sql).split()), params))
        self.rowcount = 1 if "INSERT" in sql else 0

    def fetchone(self):
        return self._conn.one

    def fetchall(self):
        return self._conn.many


class _Conn:
    def __init__(self, *, exists=True, one=None, many=()):  # noqa: ARG002 — test double mirrors the real signature
        self.record: list = []
        self.one = "endpoint_registry" if exists else None
        self.many = list(many)
        self.committed = False

    def cursor(self):
        return _Cur(self)

    def commit(self):
        self.committed = True


_ROW = (
    "j1", "default", "m", "node", 1, "ready", "slurm-remote", None,
    "t", "t", "t", None, None, 0, {}, "bj1",
)


def test_rows_copied_with_venue_stamped_and_conflict_skipped():
    src = _Conn(many=[_ROW, _ROW])
    dst = _Conn()
    inserted = copy_endpoints(src, dst, venue="rccs-front")
    assert inserted == 2
    select_sql, _ = src.record[1]  # record[0] is the table-exists probe
    assert "SELECT" in select_sql and "FROM endpoint_registry" in select_sql
    for sql, params in dst.record:
        assert "ON CONFLICT (job_id) DO NOTHING" in sql
        assert params[-1] == "rccs-front"
        assert len(params) == len(SOURCE_COLUMNS) + 1
    assert dst.committed


def test_missing_source_table_is_an_honest_zero():
    src = _Conn(exists=False)
    dst = _Conn()
    assert copy_endpoints(src, dst, venue="ai4s") == 0
    assert dst.record == []  # never touches the hub table


def test_source_select_uses_the_positional_contract():
    assert SOURCE_COLUMNS[0] == "job_id"
    assert SOURCE_COLUMNS[-1] == "backend_job_id"
    assert len(SOURCE_COLUMNS) == 16
    src = _Conn(many=[_ROW])
    copy_endpoints(src, _Conn(), venue="octopus-front")
    sql, _ = src.record[1]  # record[0] is the table-exists probe
    for name in SOURCE_COLUMNS:
        assert name in sql
    assert "venue" not in sql  # venue exists only in the hub's INSERT
