"""Ops-watch #532 (ai4s 2026-10-21/22): "project budget read failed for
default: connection already closed" on every loop start.

The ADR-067 budget read used the loop store's psycopg2 connection raw: no
store lock, no liveness check, no reconnect, and its SELECT's transaction
was never ended. A loop start refused after the read (400) left that
transaction open, the R-14 idle-in-transaction reaper killed the session
ten minutes later, and every later budget read hit the dead connection
(the loop caps were silently dropped) until some other store call
happened to replace it. The read now goes through ``LoopStore.read_row``.
"""

from __future__ import annotations

import psycopg2
import pytest
from umrs.api import loop as loop_api
from umrs.knowledge.loop_store import LoopStore

_ROW = (2.0, 5.0)


class _Cursor:
    def __init__(self, conn):
        self._conn = conn

    def __enter__(self):
        return self

    def __exit__(self, *_a):
        return False

    def execute(self, *_a):
        if self._conn.reaped:
            # psycopg2 on a session the server already terminated: the first
            # statement discovers it and marks the connection closed.
            self._conn.closed = 2
            raise psycopg2.OperationalError("server closed the connection unexpectedly")
        self._conn.executed += 1

    def fetchone(self):
        return _ROW


class _Conn:
    def __init__(self, *, closed=0, reaped=False):
        self.closed = closed
        self.reaped = reaped
        self.executed = 0
        self.rollbacks = 0

    def cursor(self, **_kw):
        if self.closed:
            raise psycopg2.InterfaceError("connection already closed")
        return _Cursor(self)

    def rollback(self):
        self.rollbacks += 1

    def commit(self):
        pass


def _install(monkeypatch, conn, fresh):
    store = LoopStore.sync_pg(conn, reconnect=lambda: fresh)
    monkeypatch.setattr(loop_api, "_loop_store", store)


def test_a_closed_connection_is_replaced_and_the_budget_is_read(monkeypatch):
    fresh = _Conn()
    _install(monkeypatch, _Conn(closed=2), fresh)
    caps = loop_api._project_budget_caps("default")
    assert caps == {"max_gpu_hours": 2.0, "max_wall_clock_hours": 5.0}
    assert fresh.executed == 1


def test_a_reaped_session_is_detected_and_the_read_retried_once(monkeypatch):
    fresh = _Conn()
    _install(monkeypatch, _Conn(reaped=True), fresh)
    caps = loop_api._project_budget_caps("default")
    assert caps == {"max_gpu_hours": 2.0, "max_wall_clock_hours": 5.0}
    assert fresh.executed == 1


def test_the_budget_read_never_leaves_its_transaction_open(monkeypatch):
    conn = _Conn()
    _install(monkeypatch, conn, _Conn())
    loop_api._project_budget_caps("default")
    assert conn.rollbacks == 1  # the reaper has nothing to kill


def test_a_memory_store_has_no_budget(monkeypatch):
    monkeypatch.setattr(loop_api, "_loop_store", LoopStore.memory())
    assert loop_api._project_budget_caps("default") == {}


def test_read_row_surfaces_a_loss_without_a_reconnect_factory():
    store = LoopStore.sync_pg(_Conn(closed=2))
    with pytest.raises(psycopg2.InterfaceError):
        store.read_row("SELECT 1", ())
