"""Live rccs 2026-09-24: `r4 project loop list` answered 500 with
psycopg2 "connection already closed". The sync loop store shared one
connection whose reads never ended their transaction, so the server's
idle-in-transaction timeout killed it and nothing reconnected."""

from __future__ import annotations

import psycopg2
import pytest
from umrs.knowledge.loop_store import LoopStore


class _Cursor:
    def __init__(self, conn):
        self._conn = conn

    def __enter__(self):
        if self._conn.fail_next:
            self._conn.fail_next = False
            self._conn.closed = 2
            raise psycopg2.InterfaceError("connection already closed")
        return self

    def __exit__(self, *_a):
        return False

    def execute(self, *_a):
        self._conn.executed += 1

    def fetchall(self):
        return []


class _Conn:
    def __init__(self, fail_next=False):
        self.closed = 0
        self.rollbacks = 0
        self.executed = 0
        self.fail_next = fail_next

    def cursor(self, **_kw):
        return _Cursor(self)

    def rollback(self):
        self.rollbacks += 1

    def commit(self):
        pass


def test_reads_end_their_transaction():
    conn = _Conn()
    store = LoopStore.sync_pg(conn)
    assert store.list_all() == []
    assert conn.rollbacks == 1  # never left idle in transaction


def test_a_closed_connection_is_replaced_before_use():
    dead, fresh = _Conn(), _Conn()
    dead.closed = 1
    store = LoopStore.sync_pg(dead, reconnect=lambda: fresh)
    assert store.list_all() == []
    assert fresh.executed == 1 and dead.executed == 0


def test_a_connection_lost_mid_call_reconnects_and_retries_once():
    broken, fresh = _Conn(fail_next=True), _Conn()
    store = LoopStore.sync_pg(broken, reconnect=lambda: fresh)
    assert store.list_all() == []
    assert fresh.executed == 1


def test_without_a_reconnect_factory_the_error_surfaces():
    store = LoopStore.sync_pg(_Conn(fail_next=True))
    with pytest.raises(psycopg2.InterfaceError):
        store.list_all()


def test_non_connection_errors_are_not_retried():
    calls = []

    class _Boom(_Conn):
        def cursor(self, **_kw):
            calls.append(1)
            raise ValueError("bad sql")

    store = LoopStore.sync_pg(_Boom(), reconnect=_Conn)
    with pytest.raises(ValueError):
        store.list_all()
    assert calls == [1]


def test_durable_pool_default_leaves_room_beside_the_leased_boot_stores(monkeypatch):
    from umrs.shared.pg import durable_pool_max

    monkeypatch.delenv("UMRS_DB_POOL_MAX", raising=False)
    assert durable_pool_max() == 32
    monkeypatch.setenv("UMRS_DB_POOL_MAX", "4")
    assert durable_pool_max() == 10  # floor: 6 boot stores + request traffic
    monkeypatch.setenv("UMRS_DB_POOL_MAX", "x")
    assert durable_pool_max() == 32
