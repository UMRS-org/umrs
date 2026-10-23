"""Which psycopg2 errors mean the loop store's session is gone (review 2026-10-23).

Every OperationalError counted as a lost session: a statement timeout
(QueryCanceledError, an OperationalError subclass) on a healthy connection
dropped it and ran the statement again on a fresh one, and the old
connection was never closed (its socket leaked until GC). Only an
InterfaceError, or an OperationalError that left the connection closed, is a
loss now, and the old connection is closed before the reconnect.
"""

from __future__ import annotations

import psycopg2
import pytest
from psycopg2.extensions import QueryCanceledError
from umrs.knowledge.loop_store import LoopStore


class _Cursor:
    def __init__(self, conn):
        self._conn = conn

    def __enter__(self):
        return self

    def __exit__(self, *_a):
        return False

    def execute(self, *_a):
        self._conn.executed += 1
        if self._conn.raises is not None:
            exc, self._conn.raises = self._conn.raises, None
            if self._conn.close_on_raise:
                self._conn.closed = 2
            raise exc

    def fetchone(self):
        return (1,)


class _Conn:
    def __init__(self, raises=None, *, close_on_raise=False):
        self.closed = 0
        self.raises = raises
        self.close_on_raise = close_on_raise
        self.executed = 0
        self.close_calls = 0

    def cursor(self, **_kw):
        return _Cursor(self)

    def rollback(self):
        pass

    def close(self):
        self.close_calls += 1
        self.closed = 1


def _store(conn):
    made = []

    def _reconnect():
        made.append(_Conn())
        return made[-1]

    return LoopStore.sync_pg(conn, reconnect=_reconnect), made


def test_a_statement_timeout_surfaces_and_keeps_the_connection():
    conn = _Conn(QueryCanceledError("canceling statement due to statement timeout"))
    store, made = _store(conn)
    with pytest.raises(QueryCanceledError):
        store.read_row("SELECT 1", ())
    assert made == [] and conn.executed == 1 and conn.close_calls == 0
    assert store.read_row("SELECT 1", ()) == (1,)  # the same session serves on


def test_an_operational_error_on_an_open_connection_is_not_a_loss():
    conn = _Conn(psycopg2.OperationalError("could not serialize access"))
    store, made = _store(conn)
    with pytest.raises(psycopg2.OperationalError):
        store.read_row("SELECT 1", ())
    assert made == [] and conn.close_calls == 0


def test_a_lost_session_closes_the_old_connection_then_reconnects():
    conn = _Conn(
        psycopg2.OperationalError("server closed the connection unexpectedly"),
        close_on_raise=True,
    )
    store, made = _store(conn)
    assert store.read_row("SELECT 1", ()) == (1,)
    assert conn.close_calls == 1
    assert len(made) == 1 and made[0].executed == 1


def test_a_closed_connection_found_before_use_is_closed_and_replaced():
    conn = _Conn()
    conn.closed = 2
    store, made = _store(conn)
    assert store.read_row("SELECT 1", ()) == (1,)
    assert conn.close_calls == 1 and conn.executed == 0
    assert len(made) == 1
