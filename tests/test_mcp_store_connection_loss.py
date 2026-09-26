"""MCPServerStore survives a dropped Postgres session (live bdrt 2026-09-27).

`r4 mcp list --servers` answered 500 on umrs-server-bdrt:
psycopg2.OperationalError "server closed the connection unexpectedly" at
store.py list_servers, then InterfaceError "connection already closed" on
every later call. The store held one boot-leased connection, its reads never
ended their implicit transaction, and the R-14 idle-in-transaction reaper
killed the session; nothing replaced it. The fakes below die the way psycopg2
does (the failing call raises OperationalError and the connection then
reports ``closed``; any later cursor() raises InterfaceError).
"""

from __future__ import annotations

from datetime import datetime, timezone

import psycopg2
import pytest
from psycopg2.extensions import (
    TRANSACTION_STATUS_IDLE,
    TRANSACTION_STATUS_INTRANS,
    TRANSACTION_STATUS_UNKNOWN,
    QueryCanceledError,
)
from umrs.mcp.store import MCPServerNotFoundError, MCPServerStore

_TS = datetime(2026, 9, 27, tzinfo=timezone.utc)
_SERVER_ROW = (
    "srv-1", "lab-scope", "1.0.0", "http://127.0.0.1:9901/mcp", "http",
    "healthy", "scope control", "none", "{}", "{}", "admin", None, "local",
    "local-safe", _TS, _TS,
)  # fmt: skip


class _Cursor:
    def __init__(self, conn: _Conn) -> None:
        self._conn = conn
        self._rows: list[tuple] = []

    def __enter__(self) -> _Cursor:
        return self

    def __exit__(self, *_exc) -> bool:
        return False

    def execute(self, sql, params=None) -> None:
        conn = self._conn
        if conn.die_on_execute:
            conn.die_on_execute = False
            conn.closed = 2
            raise psycopg2.OperationalError("server closed the connection unexpectedly")
        if conn.error_on_execute is not None:
            raise conn.error_on_execute
        text = " ".join(sql.split())
        conn.executed.append(text)
        conn.pending.append(text)
        conn.in_transaction = True  # psycopg2 opens one on the first statement
        self._rows = _rows_for(text, params)

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return list(self._rows)


def _rows_for(text: str, params) -> list[tuple]:
    if "INTO skills" in text:  # SkillStore.register_skill ... RETURNING
        return [("skl-1", _TS, _TS)]
    if "FROM skills" in text:  # SkillStore.get_skill_by_name: not there yet
        return []
    if text.startswith(("UPDATE", "DELETE", "INSERT")):
        return []
    if "COUNT(*)" in text:
        return [(1,)]
    if "FROM mcp_tools t" in text:
        return [("focus", "d", {}, "lab-scope")]
    if "FROM mcp_tools WHERE server_id" in text:
        return []
    return [] if "missing" in (params or ()) else [_SERVER_ROW]


class _Conn:
    def __init__(self, *, die_on_execute=False, die_on_commit=False, error=None):
        self.closed = 0
        self.in_transaction = False
        self.commits = 0
        self.rollbacks = 0
        self.executed: list[str] = []
        self.pending: list[str] = []  # statements of the open transaction
        self.durable: list[str] = []  # statements a commit made permanent
        self.die_on_execute = die_on_execute
        self.die_on_commit = die_on_commit
        self.error_on_execute = error

    def get_transaction_status(self) -> int:
        if self.closed:
            return TRANSACTION_STATUS_UNKNOWN
        return TRANSACTION_STATUS_INTRANS if self.in_transaction else TRANSACTION_STATUS_IDLE

    def cursor(self) -> _Cursor:
        if self.closed:
            raise psycopg2.InterfaceError("connection already closed")
        return _Cursor(self)

    def commit(self) -> None:
        if self.closed:
            raise psycopg2.InterfaceError("connection already closed")
        if self.die_on_commit:
            self.closed = 2
            raise psycopg2.OperationalError("server closed the connection unexpectedly")
        self.commits += 1
        self.durable += self.pending
        self.pending = []
        self.in_transaction = False

    def rollback(self) -> None:
        if self.closed:
            raise psycopg2.InterfaceError("connection already closed")
        self.rollbacks += 1
        self.pending = []
        self.in_transaction = False


class _Pool:
    """ThreadedConnectionPool contract: putconn(close=True) discards."""

    def __init__(self, *conns: _Conn) -> None:
        self.idle = list(conns)
        self.returned: list[tuple[_Conn, bool]] = []
        self.borrowed = 0

    def getconn(self) -> _Conn:
        self.borrowed += 1
        return self.idle.pop(0)

    def putconn(self, conn: _Conn, close: bool = False) -> None:
        self.returned.append((conn, close))
        if not close:
            self.idle.append(conn)


# -- the live defect ------------------------------------------------------


def test_list_servers_recovers_from_a_killed_connection_and_stays_recovered():
    dead, fresh = _Conn(die_on_execute=True), _Conn()
    pool = _Pool(dead, fresh)
    store = MCPServerStore(pool=pool)

    (server,) = store.list_servers()  # was: 500 OperationalError
    assert server.name == "lab-scope"
    assert (dead, True) in pool.returned  # discarded, never reissued
    assert store.list_servers()[0].name == "lab-scope"  # was: InterfaceError
    assert pool.idle == [fresh] and not fresh.in_transaction


@pytest.mark.parametrize(
    ("method", "args"),
    [
        ("list_servers", ()),
        ("get_server", ("srv-1",)),
        ("get_server_by_name", ("lab-scope",)),
        ("get_server_for_tool", ("focus",)),
        ("list_tools", ()),
        ("get_stats", ()),
    ],
)
def test_a_read_leaves_no_open_transaction(method, args):
    conn = _Conn()
    getattr(MCPServerStore(conn), method)(*args)
    assert conn.executed and not conn.in_transaction  # nothing for R-14 to reap


def test_a_failed_read_ends_its_transaction_and_is_not_retried():
    conn = _Conn()
    pool = _Pool(conn, _Conn())
    with pytest.raises(MCPServerNotFoundError):
        MCPServerStore(pool=pool).get_server("missing")
    assert not conn.in_transaction
    assert pool.borrowed == 1 and pool.returned == [(conn, False)]


def test_non_connection_errors_are_not_retried():
    conn = _Conn(error=psycopg2.ProgrammingError("relation does not exist"))
    pool = _Pool(conn, _Conn())
    with pytest.raises(psycopg2.ProgrammingError):
        MCPServerStore(pool=pool).list_servers()
    assert pool.borrowed == 1 and conn.rollbacks == 1


def test_the_retry_is_bounded_to_one():
    first, second = _Conn(die_on_execute=True), _Conn(die_on_execute=True)
    pool = _Pool(first, second)
    with pytest.raises(psycopg2.OperationalError):
        MCPServerStore(pool=pool).list_servers()
    assert pool.returned == [(first, True), (second, True)]


def test_an_operational_error_on_a_live_session_is_not_a_connection_loss(caplog):
    """QueryCanceled, deadlock, serialization, lock timeout ... subclass
    OperationalError but leave the session alive (``conn.closed == 0``):
    no retry, no 'connection lost' warning, and the healthy connection goes
    back to the pool instead of being closed."""
    live = _Conn(error=QueryCanceledError("canceling statement due to user request"))
    pool = _Pool(live, _Conn())
    with pytest.raises(QueryCanceledError):
        MCPServerStore(pool=pool).list_servers()
    assert pool.borrowed == 1 and pool.returned == [(live, False)]
    assert live.rollbacks == 1 and not live.in_transaction
    assert "connection lost" not in caplog.text


# -- conn mode: a caller-owned connection ---------------------------------


def test_a_read_leaves_the_callers_open_transaction_alone():
    conn = _Conn()
    conn.in_transaction = True  # the caller's own uncommitted work
    store = MCPServerStore(conn)
    store.list_servers()
    with pytest.raises(MCPServerNotFoundError):
        store.get_server_by_name("missing")
    assert conn.rollbacks == 0 and conn.in_transaction


def test_restore_skill_stubs_survive_the_mcp_ref_lookup():
    """bundle_restore shares one connection across SkillStore, MCPServerStore
    and ScientificCatalogStore. register_skill never commits; the catalog
    create's commit makes the stubs durable, so an MCP read in between must
    not roll them back (reviewer 2026-09-27: INSERT, SELECT, ROLLBACK,
    COMMIT left skill_refs_imported counting stubs that were never stored)."""
    from umrs.knowledge.bundle_import import _import_skill_refs, _resolve_mcp_refs
    from umrs.skills.store import SkillStore

    conn = _Conn()
    imported, _ = _import_skill_refs(
        SkillStore(conn), [{"ref": "project:fit-peaks", "version_pin": "1.0"}], "p1"
    )
    present, missing = _resolve_mcp_refs(
        MCPServerStore(conn), [{"ref": "lab-scope"}, {"ref": "missing"}]
    )
    conn.commit()  # ScientificCatalogStore.create, same connection

    assert (imported, present, missing) == (1, 1, ["missing"])
    assert any("INTO skills" in stmt for stmt in conn.durable)
    assert conn.rollbacks == 0


# -- writes ---------------------------------------------------------------


def test_a_write_lost_before_its_commit_is_retried_once():
    dead, fresh = _Conn(die_on_execute=True), _Conn()
    pool = _Pool(dead, fresh)
    MCPServerStore(pool=pool).update_status("srv-1", "unhealthy")
    assert fresh.commits == 1 and len(fresh.executed) == 1
    assert (dead, True) in pool.returned


def test_a_write_lost_during_its_commit_is_never_retried():
    committing, spare = _Conn(die_on_commit=True), _Conn()
    pool = _Pool(committing, spare)
    with pytest.raises(psycopg2.OperationalError):
        MCPServerStore(pool=pool).unregister_server("lab-scope")
    assert pool.borrowed == 1 and spare.executed == []  # the commit may have landed
    assert pool.returned == [(committing, True)]


# -- sidecar wiring -------------------------------------------------------


def test_sidecar_borrows_per_call_instead_of_leasing_a_connection(monkeypatch):
    from types import SimpleNamespace

    import umrs.api.mcp as mcp_api
    import umrs.jobs.adapters as adapters
    import umrs.mcp as mcp_pkg
    from umrs.server.http_sidecar import _init_mcp_store

    monkeypatch.setattr(mcp_pkg, "ensure_schema", lambda _conn: 1)
    monkeypatch.setattr(mcp_api, "_store", None)
    monkeypatch.setattr(adapters, "_MCP_REGISTRY", None)
    pool = _Pool(_Conn())

    _init_mcp_store(SimpleNamespace(state=SimpleNamespace()), pool)

    assert mcp_api._store is not None and mcp_api._store._pool is pool
    assert pool.borrowed == len(pool.returned)  # no connection left leased
