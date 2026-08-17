"""Lane table-motion unit tests (stability S9).

SQL shape + preflight discipline pinned against a recording fake — the
live semantics of the same statements are validated by the rollback
drill recorded in docs/operations/lane-migration-runbook.md.
"""

from __future__ import annotations

import pytest
from umrs.shared.lane_motion import (
    TableMotionError,
    move_tables_to_schema,
    rollback_tables_to_public,
)


class _FakeCursor:
    def __init__(self, *, existing_schema: bool, present: list[tuple]):
        self.statements: list[tuple] = []
        self._existing_schema = existing_schema
        self._present = present

    def execute(self, query, params=None):
        self.statements.append((query, params))
        self._last = query

    def fetchone(self):
        if "pg_namespace" in self._last:
            return ("lane_beta",) if self._existing_schema else None
        return None

    def fetchall(self):
        return list(self._present)


class _FakeConn:
    def __init__(self, cursor: _FakeCursor):
        self._cursor = cursor

    def cursor(self):
        return self._cursor


def _conn(*, existing_schema: bool = True, present: tuple = ("alpha", "beta")):
    cur = _FakeCursor(existing_schema=existing_schema, present=[(t,) for t in present])
    return _FakeConn(cur), cur


def test_moves_all_tables_atomically_in_order():
    conn, cur = _conn()
    moved = move_tables_to_schema(conn, ["alpha", "beta"], target_schema="lane_beta")
    assert moved == ["alpha", "beta"]
    alters = [q for q, _ in cur.statements if q.startswith("ALTER TABLE")]
    assert alters == [
        'ALTER TABLE "public"."alpha" SET SCHEMA "lane_beta"',
        'ALTER TABLE "public"."beta" SET SCHEMA "lane_beta"',
    ]
    # Lock timeout is set inside the caller's transaction first.
    assert cur.statements[0][0].startswith("SET LOCAL lock_timeout")


def test_preflight_fails_before_any_alter_when_schema_missing():
    conn, cur = _conn(existing_schema=False)
    with pytest.raises(TableMotionError, match="0072"):
        move_tables_to_schema(conn, ["alpha"], target_schema="lane_beta")
    assert not any(q.startswith("ALTER TABLE") for q, _ in cur.statements)


def test_preflight_fails_before_any_alter_when_table_missing():
    conn, cur = _conn(present=("alpha",))
    with pytest.raises(TableMotionError, match="beta"):
        move_tables_to_schema(conn, ["alpha", "beta"], target_schema="lane_beta")
    assert not any(q.startswith("ALTER TABLE") for q, _ in cur.statements)


@pytest.mark.parametrize(
    "bad",
    ["Lane_Beta", "lane beta", "lane;drop", "1lane", "", "lane-beta", "lane.beta"],
)
def test_identifier_allowlist_rejects_non_system_names(bad):
    conn, _ = _conn()
    with pytest.raises(TableMotionError, match="invalid"):
        move_tables_to_schema(conn, ["alpha"], target_schema=bad)
    with pytest.raises(TableMotionError, match="invalid"):
        move_tables_to_schema(conn, [bad], target_schema="lane_beta")


def test_empty_table_list_is_a_noop():
    conn, cur = _conn()
    assert move_tables_to_schema(conn, [], target_schema="lane_beta") == []
    assert not any(q.startswith("ALTER TABLE") for q, _ in cur.statements)


def test_rollback_is_the_symmetric_inverse_motion():
    conn, cur = _conn()
    moved = rollback_tables_to_public(conn, ["alpha"], current_schema="lane_beta")
    assert moved == ["alpha"]
    alters = [q for q, _ in cur.statements if q.startswith("ALTER TABLE")]
    assert alters == ['ALTER TABLE "lane_beta"."alpha" SET SCHEMA "public"']


def test_custom_source_and_lock_timeout():
    conn, cur = _conn()
    move_tables_to_schema(
        conn,
        ["alpha"],
        target_schema="lane_stable",
        source_schema="lane_beta",
        lock_timeout_ms=1500,
    )
    assert cur.statements[0][0] == "SET LOCAL lock_timeout = '1500'"
    alters = [q for q, _ in cur.statements if q.startswith("ALTER TABLE")]
    assert alters == ['ALTER TABLE "lane_beta"."alpha" SET SCHEMA "lane_stable"']
