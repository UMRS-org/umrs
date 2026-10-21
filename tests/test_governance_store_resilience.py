"""GovernanceStore resilience to dead Postgres connections (2026-09-08).

Live failure class: a server booted while postgres cycled during converge;
its boot-issued conn died once, then every check_quota 500'd with
psycopg2.InterfaceError 'connection already closed' until the server was
restarted. The store now recovers via the pool when one is provided.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import psycopg2
import pytest
from umrs.api.governance import GovernanceStore


def _row(name: str = "default", count: int = 0, max_count: int = 10) -> tuple:
    return (
        name,
        "endpoints",
        max_count,
        count,
        "2026-09-08T00:00:00+00:00",
        0,
    )


def _cursor_with(then: Any, result: Any) -> MagicMock:
    cur = MagicMock(name="cursor")

    def close() -> None:
        return None

    cur.close = close
    if isinstance(then, Exception):
        cur.execute.side_effect = then
    else:
        cur.fetchone.return_value = result
    return cur


class _FakePool:
    def __init__(self, next_conn) -> None:
        self.putconn_seen: list[tuple] = []
        self.close_seen: list[tuple] = []
        self.getconn_calls = 0
        self._next_conn = next_conn

    def putconn(self, conn, close: bool = False) -> None:
        (self.close_seen if close else self.putconn_seen).append(conn)

    def getconn(self):
        self.getconn_calls += 1
        return self._next_conn


def _conn(cursor_return) -> MagicMock:
    conn = MagicMock(name="conn")
    conn.cursor.return_value = cursor_return
    return conn


@pytest.mark.parametrize(
    "dead_exc",
    [
        psycopg2.InterfaceError("connection already closed"),
        psycopg2.OperationalError("server closed the connection unexpectedly"),
    ],
)
def test_get_quota_recovers_after_dead_conn_when_pool_provided(dead_exc):
    """Dead first query -> reconnect once -> fresh conn serves the row."""
    fresh_conn = _conn(_cursor_with(None, _row()))
    pool = _FakePool(fresh_conn)
    store = GovernanceStore(
        _conn(_cursor_with(dead_exc, None)),
        pool=pool,
    )

    result = store.get_quota("default", "endpoints")

    assert result is not None
    assert result["project_id"] == "default"
    assert store.conn is fresh_conn
    assert pool.getconn_calls == 1


def test_get_quota_returns_none_when_reconnect_fails():
    pool = _FakePool(None)
    pool.getconn = MagicMock(side_effect=Exception("pool empty"))
    store = GovernanceStore(
        _conn(_cursor_with(psycopg2.InterfaceError("closed"), None)),
        pool=pool,
    )

    assert store.get_quota("default", "endpoints") is None


def test_get_quota_returns_none_on_non_conn_exception_without_reconnect():
    pool = _FakePool(_conn(_cursor_with(None, _row())))
    store = GovernanceStore(
        _conn(_cursor_with(ValueError("whatever"), None)),
        pool=pool,
    )

    assert store.get_quota("default", "endpoints") is None
    assert pool.getconn_calls == 0


def test_get_quota_without_pool_still_fails_gracefully():
    store = GovernanceStore(
        _conn(_cursor_with(psycopg2.OperationalError("server closed"), None)),
    )

    assert store.get_quota("default", "endpoints") is None


def test_get_quota_success_second_attempt_when_pool_conn_recovers_later():
    """Second loop iter on same call after reconnect serves the row."""
    fresh_conn = MagicMock(name="fresh_conn")
    fresh_conn.cursor.return_value = _cursor_with(None, _row())
    pool = _FakePool(fresh_conn)
    store = GovernanceStore(
        MagicMock(
            name="dead_conn",
            cursor=MagicMock(return_value=_cursor_with(psycopg2.InterfaceError("closed"), None)),
        ),
        pool=pool,
    )

    out = store.get_quota("default", "endpoints")

    assert out is not None
    assert store.conn is fresh_conn


def test_get_quota_recovers_when_cursor_open_raises_on_dead_conn():
    """2026-10 recurrence: ``conn.cursor()`` raising (not ``execute``) skipped
    the Sep-8 retry loop entirely — cursor creation moved inside the try."""
    dead = MagicMock(name="dead_conn")
    dead.cursor.side_effect = psycopg2.InterfaceError("connection already closed")
    fresh_conn = _conn(_cursor_with(None, _row()))
    pool = _FakePool(fresh_conn)
    store = GovernanceStore(dead, pool=pool)

    result = store.get_quota("default", "endpoints")

    assert result is not None
    assert result["project_id"] == "default"
    assert store.conn is fresh_conn
    assert pool.getconn_calls == 1


def test_get_quota_cursor_open_raises_twice_no_pool_fails_gracefully():
    dead = MagicMock(name="dead_conn")
    dead.cursor.side_effect = psycopg2.InterfaceError("connection already closed")
    store = GovernanceStore(dead)  # no pool

    assert store.get_quota("default", "endpoints") is None


def test_get_quota_runs_one_select_on_a_healthy_conn():
    """services#1 (2026-10-21): the retry loop had no break, so every call
    opened a second cursor and re-ran the SELECT."""
    cur = _cursor_with(None, _row())
    conn = _conn(cur)
    store = GovernanceStore(conn)

    result = store.get_quota("default", "endpoints")

    assert result is not None
    assert conn.cursor.call_count == 1
    assert cur.execute.call_count == 1


def test_check_quota_blocks_at_quota_project_when_conn_dies_after_select(
    monkeypatch,
):
    """services#1: a row fetched on attempt 1 must be returned. The old loop
    re-ran the SELECT; when the conn died on that extra pass get_quota
    returned None and check_quota let an at-quota project through."""
    from fastapi import HTTPException
    from umrs.api import governance

    conn = MagicMock(name="conn")
    conn.cursor.side_effect = [
        _cursor_with(None, _row(count=2, max_count=2)),
        psycopg2.InterfaceError("connection already closed"),
    ]
    monkeypatch.setattr(governance, "_governance_store", GovernanceStore(conn))
    monkeypatch.setattr(
        "umrs.api.audit_events.emit_audit", lambda *_a, **_k: None, raising=False
    )

    with pytest.raises(HTTPException) as exc_info:
        governance.check_quota("default", "endpoints")

    assert exc_info.value.status_code == 409
