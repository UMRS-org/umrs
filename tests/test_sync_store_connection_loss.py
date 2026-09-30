"""Sync Postgres stores: writes commit, reads end their transaction, a lost
session is retried once (umrs.shared.pg_calls; A9 store hygiene 2026-09-27).

Found while fixing the MCP store's dead-connection 500 on live bdrt: the
research-session, companion-trust, approval-inbox and audit stores held a
boot-leased connection the same way. Their reads left it idle in
transaction until R-14's idle_in_transaction_session_timeout reaper killed
it, and the session and trust stores committed only on sqlite, so on
Postgres every write sat in a transaction the reaper rolled back. The fakes
die the way psycopg2 does (the failing call raises OperationalError and the
connection then reports ``closed``; any later cursor() raises
InterfaceError) and keep committed statements apart from pending ones.
"""

from __future__ import annotations

import sqlite3
from types import SimpleNamespace

import psycopg2
import pytest
from psycopg2.extensions import (
    TRANSACTION_STATUS_IDLE,
    TRANSACTION_STATUS_INTRANS,
    TRANSACTION_STATUS_UNKNOWN,
    QueryCanceledError,
)
from umrs.api.audit_store import AuditStore
from umrs.companion_trust.approval_inbox import ApprovalInboxStore
from umrs.companion_trust.store import SessionArchivedError, TrustAllowlistStore
from umrs.research_sessions import ResearchSession, SessionLoopLink
from umrs.research_sessions.store import SessionStore

_TS = "2026-09-27T00:00:00+00:00"
_SESSION_ROW = ("rs-1", "p1", "t", "goal", "active", "umrs", "{}", _TS, _TS, None)
_TRUST_ROW = ("alice", "system", "shell_exec", _TS, _TS, "alice")
_APPROVAL_ROW = (
    "appr-1", "loop-1", 1, "alice", "", "", "{}", "pending", "", "", None, _TS,
)  # fmt: skip
_AUDIT_ROW = ("ev-1", "1", 1.0, "job.submit", "alice", "p1", "", "success", "", {})


class _Cursor:
    def __init__(self, conn: _Conn) -> None:
        self.connection = conn
        self.rowcount = -1
        self._rows: list[tuple] = []

    def __enter__(self) -> _Cursor:
        return self

    def __exit__(self, *_exc) -> bool:
        return False

    def close(self) -> None:
        return None

    def execute(self, sql, params=None) -> None:
        conn = self.connection
        if conn.die_on_execute:
            conn.die_on_execute = False
            conn.closed = 2
            raise psycopg2.OperationalError("server closed the connection unexpectedly")
        if conn.error_on_execute is not None:
            raise conn.error_on_execute
        text = " ".join(sql.split())
        conn.executed.append(text)
        conn.bound.append(params)
        conn.pending.append(text)
        conn.in_transaction = True  # psycopg2 opens one on the first statement
        self.rowcount = 1
        self._rows = _rows_for(conn, text)

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return list(self._rows)


def _rows_for(conn: _Conn, text: str) -> list[tuple]:
    if text.startswith(("UPDATE", "DELETE", "INSERT")):
        return []
    if "FROM companion_session_archives" in text:
        return [(1,)] if conn.tombstoned else []
    table_rows = {
        "COUNT(*)": [(3,)],
        "SELECT 1 FROM companion_trust_allowlist": [(1,)],
        "FROM companion_trust_allowlist": [_TRUST_ROW],
        "FROM research_session_loops": [],
        "FROM research_sessions": [_SESSION_ROW],
        "FROM companion_approval_inbox": [_APPROVAL_ROW],
        "FROM audit_events": [_AUDIT_ROW],
    }
    return next((rows for key, rows in table_rows.items() if key in text), [])


class _Conn:
    def __init__(self, *, die_on_execute=False, die_on_commit=False, tombstoned=False, error=None):
        self.closed = 0
        self.autocommit = False
        self.in_transaction = False
        self.commits = 0
        self.rollbacks = 0
        self.executed: list[str] = []
        self.bound: list = []  # the parameters of each executed statement
        self.pending: list[str] = []  # statements of the open transaction
        self.durable: list[str] = []  # statements a commit made permanent
        self.die_on_execute = die_on_execute
        self.die_on_commit = die_on_commit
        self.tombstoned = tombstoned
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


def _durable(*conns: _Conn) -> list[str]:
    return [stmt for conn in conns for stmt in conn.durable]


def _session() -> ResearchSession:
    return ResearchSession(session_id="rs-1", project_id="p1", research_goal="goal")


def _runner(**kwargs):
    # Imported per test so the store tests below still run (and fail on
    # their own assertions) against a tree that predates the runner.
    from umrs.shared.pg_calls import PgCalls

    return PgCalls(**kwargs)


def _select_one(cur):
    cur.execute("SELECT * FROM research_sessions WHERE session_id = %s", ("rs-1",))
    return cur.fetchone()


# -- the shared runner -----------------------------------------------------


def test_the_retry_is_bounded_to_one():
    first, second = _Conn(die_on_execute=True), _Conn(die_on_execute=True)
    pool = _Pool(first, second)
    with pytest.raises(psycopg2.OperationalError):
        _runner(pool=pool).run(_select_one)
    assert pool.returned == [(first, True), (second, True)]


def test_an_operational_error_on_a_live_session_is_not_a_connection_loss(caplog):
    live = _Conn(error=QueryCanceledError("canceling statement due to user request"))
    pool = _Pool(live, _Conn())
    with pytest.raises(QueryCanceledError):
        _runner(pool=pool).run(_select_one)
    assert pool.borrowed == 1 and pool.returned == [(live, False)]
    assert live.rollbacks == 1 and "connection lost" not in caplog.text


def test_a_read_leaves_the_callers_open_transaction_alone():
    conn = _Conn()
    conn.in_transaction = True  # the caller's own uncommitted work
    assert _runner(conn=conn).run(_select_one) == _SESSION_ROW
    assert conn.rollbacks == 0 and conn.in_transaction


def test_a_sqlite_connection_takes_the_same_path():
    conn = sqlite3.connect(":memory:")
    calls = _runner(conn=conn)
    calls.run(lambda cur: cur.execute("CREATE TABLE t (v INTEGER)"), write=True)
    calls.run(lambda cur: cur.execute("INSERT INTO t VALUES (1)"), write=True)
    conn.execute("INSERT INTO t VALUES (2)")  # the caller's own transaction
    assert calls.run(lambda cur: cur.execute("SELECT COUNT(*) FROM t").fetchone()) == (2,)
    assert conn.in_transaction  # a read never rolls back what it did not open


# -- research sessions: writes commit on Postgres (was: data loss) ---------

_SESSION_WRITES = [
    ("create", lambda s: s.create(_session()), "INSERT INTO research_sessions"),
    ("update", lambda s: s.update("rs-1", title="new"), "UPDATE research_sessions"),
    (
        "attach_loop",
        lambda s: s.attach_loop(SessionLoopLink(session_id="rs-1", loop_id="l1")),
        "INSERT INTO research_session_loops",
    ),
    (
        "detach_loop",
        lambda s: s.detach_loop("rs-1", "l1"),
        "DELETE FROM research_session_loops",
    ),
]


@pytest.mark.parametrize(
    ("call", "statement"),
    [(call, stmt) for _name, call, stmt in _SESSION_WRITES],
    ids=[name for name, _call, _stmt in _SESSION_WRITES],
)
def test_a_session_write_commits_on_postgres(call, statement):
    conn = _Conn()
    call(SessionStore(_Pool(conn)))
    assert any(stmt.startswith(statement) for stmt in conn.durable)
    assert not conn.in_transaction  # nothing left for R-14 to reap


def test_a_session_write_commits_on_a_caller_owned_postgres_connection():
    """The sidecar's pre-fix wiring (a leased conn): nothing ever committed."""
    conn = _Conn()
    SessionStore(conn).create(_session())
    assert conn.commits == 1 and not conn.pending


@pytest.mark.parametrize(
    "read",
    [
        lambda s: s.get("rs-1"),
        lambda s: s.list(project_id="p1"),
        lambda s: s.list_loops("rs-1"),
        lambda s: s.session_for_loop("l1"),
        lambda s: s._aged_session_rows(30),
    ],
    ids=["get", "list", "list_loops", "session_for_loop", "aged_rows"],
)
def test_a_session_read_leaves_no_open_transaction(read):
    conn = _Conn()
    read(SessionStore(conn))
    assert conn.executed and not conn.in_transaction


def test_session_store_recovers_from_a_killed_connection():
    dead, fresh = _Conn(die_on_execute=True), _Conn()
    pool = _Pool(dead, fresh)
    store = SessionStore(pool)

    assert store.get("rs-1").session_id == "rs-1"  # was: OperationalError
    assert (dead, True) in pool.returned  # discarded, never reissued
    store.create(_session())
    assert any(s.startswith("INSERT INTO research_sessions") for s in fresh.durable)


def test_a_session_write_lost_during_its_commit_is_never_retried():
    committing, spare = _Conn(die_on_commit=True), _Conn()
    pool = _Pool(committing, spare)
    with pytest.raises(psycopg2.OperationalError):
        SessionStore(pool).detach_loop("rs-1", "l1")
    assert pool.borrowed == 1 and spare.executed == []  # the commit may have landed
    assert pool.returned == [(committing, True)]


def test_archive_cascade_is_one_committed_transaction_on_one_connection():
    conn = _Conn()
    pool = _Pool(conn)
    trust = TrustAllowlistStore(pool)
    SessionStore(pool).archive_with_trust_cascade("rs-1", trust)
    archive = [s for s in conn.durable if "research_sessions SET status" in s]
    revoke = [s for s in conn.durable if "DELETE FROM companion_trust_allowlist" in s]
    assert archive and revoke  # both landed, in the archive's commit
    assert pool.borrowed == len(pool.returned)


# -- companion trust: writes commit on Postgres ---------------------------


@pytest.mark.parametrize(
    ("call", "statement"),
    [
        (lambda t: t.grant("alice", "shell_exec"), "INSERT INTO companion_trust"),
        (lambda t: t.revoke("alice", "shell_exec"), "DELETE FROM companion_trust"),
        (lambda t: t.revoke_scope("project:p1"), "DELETE FROM companion_trust"),
        (lambda t: t.archive_session("s-1"), "INSERT INTO companion_session_archives"),
    ],
    ids=["grant", "revoke", "revoke_scope", "archive_session"],
)
def test_a_trust_write_commits_on_postgres(call, statement):
    conn = _Conn()
    call(TrustAllowlistStore(_Pool(conn)))
    assert any(stmt.startswith(statement) for stmt in conn.durable)
    assert not conn.in_transaction


@pytest.mark.parametrize(
    "read",
    [
        lambda t: t.check("alice", "shell_exec"),
        lambda t: t.check("alice", "shell_exec", scope="session:s-1"),
        lambda t: t.list("alice"),
    ],
    ids=["check", "check_session_scope", "list"],
)
def test_a_trust_read_leaves_no_open_transaction(read):
    conn = _Conn()
    assert read(TrustAllowlistStore(conn))
    assert conn.executed and not conn.in_transaction


def test_a_refused_grant_ends_its_transaction():
    """The tombstone SELECT ... FOR UPDATE locks the row; the refusal used to
    leave that lock and the transaction open on the shared connection."""
    conn = _Conn(tombstoned=True)
    with pytest.raises(SessionArchivedError):
        TrustAllowlistStore(conn).grant("alice", "shell_exec", scope="session:s-1")
    assert conn.rollbacks == 1 and not conn.in_transaction and not conn.durable


def test_trust_store_recovers_from_a_killed_connection():
    dead, fresh = _Conn(die_on_execute=True), _Conn()
    pool = _Pool(dead, fresh)
    store = TrustAllowlistStore(pool)
    store.grant("alice", "shell_exec")  # was: OperationalError
    assert (dead, True) in pool.returned
    assert any(s.startswith("INSERT INTO companion_trust") for s in fresh.durable)


# -- approval inbox --------------------------------------------------------


@pytest.mark.parametrize(
    "call",
    [
        lambda i: i.request("loop-1", 1, {"tool": "x"}, approval_id="appr-1"),
        lambda i: i.decide("appr-1", "once", "alice"),
        lambda i: i.get("appr-1"),
        lambda i: i.pending(user_id="alice"),
    ],
    ids=["request", "decide", "get", "pending"],
)
def test_an_inbox_call_leaves_no_open_transaction(call):
    """request/decide re-read the row after their commit, which reopened a
    transaction; get/pending never ended theirs."""
    conn = _Conn()
    assert call(ApprovalInboxStore(conn))
    assert conn.executed and not conn.in_transaction


def test_an_inbox_decision_is_committed():
    conn = _Conn()
    ApprovalInboxStore(_Pool(conn)).decide("appr-1", "no", "alice")
    assert any(s.startswith("UPDATE companion_approval_inbox") for s in conn.durable)


def test_inbox_recovers_from_a_killed_connection():
    dead, fresh = _Conn(die_on_execute=True), _Conn()
    pool = _Pool(dead, fresh)
    assert ApprovalInboxStore(pool).pending() != []  # was: OperationalError
    assert (dead, True) in pool.returned and not fresh.in_transaction


# -- audit -------------------------------------------------------------------


@pytest.mark.parametrize(
    "read",
    [lambda a: a.query(actor="alice"), lambda a: a.count()],
    ids=["query", "count"],
)
def test_an_audit_read_leaves_no_open_transaction(read):
    conn = _Conn()
    assert read(AuditStore(conn))
    assert conn.executed and not conn.in_transaction


def test_audit_query_binds_every_filter_in_order():
    conn = _Conn()
    AuditStore(conn).query(
        request_id="r", event="job", actor="a", project_id="p",
        since=1.0, until=2.0, outcome="ok", limit=900,
    )  # fmt: skip
    (stmt,) = conn.executed
    assert (
        "WHERE request_id = %s AND event LIKE %s AND actor = %s AND project_id = %s "
        "AND ts >= %s AND ts <= %s AND outcome = %s ORDER BY ts DESC LIMIT %s"
    ) in stmt
    assert conn.bound == [["r", "%job%", "a", "p", 1.0, 2.0, "ok", 500]]


def test_audit_query_without_filters_matches_everything():
    conn = _Conn()
    AuditStore(conn).query()
    assert "WHERE TRUE ORDER BY" in conn.executed[0] and conn.bound == [[50]]


def test_audit_write_recovers_from_a_killed_connection():
    dead, fresh = _Conn(die_on_execute=True), _Conn()
    pool = _Pool(dead, fresh)
    AuditStore(pool=pool).write({"event_id": "ev-1", "event": "job.submit"})
    assert (dead, True) in pool.returned
    assert any(s.startswith("INSERT INTO audit_events") for s in fresh.durable)


def test_audit_write_on_a_dead_caller_connection_is_dropped_not_raised():
    """The old except-branch rollback raised InterfaceError on a dead
    connection, escaping the store's own 'audit never breaks the caller'."""
    conn = _Conn(die_on_execute=True)
    AuditStore(conn).write({"event_id": "ev-1"})
    assert AuditStore(conn).query() == [] and AuditStore(conn).count() == 0


# -- sidecar wiring --------------------------------------------------------


@pytest.fixture()
def _app():
    return SimpleNamespace(state=SimpleNamespace())


def test_sidecar_session_store_borrows_per_call(monkeypatch, _app):
    import umrs.api.research_sessions as sessions_api
    import umrs.research_sessions as sessions_pkg
    from umrs.server.http_sidecar import _init_research_sessions_store

    monkeypatch.setattr(sessions_pkg, "ensure_schema", lambda _conn: True)
    monkeypatch.setattr(sessions_api, "_store", None)
    pool = _Pool(_Conn())

    _init_research_sessions_store(_app, pool)

    assert sessions_api._store is not None and sessions_api._store._pool is pool
    assert pool.borrowed == len(pool.returned)  # no connection left leased


def test_sidecar_trust_store_and_inbox_borrow_per_call(monkeypatch, _app):
    import umrs.companion_trust as trust_pkg
    import umrs.knowledge.tool_approvals as tool_approvals
    import umrs.shared.trust_store_access as trust_access
    from umrs.server.http_sidecar import _init_companion_trust_store

    monkeypatch.setattr(trust_pkg, "ensure_schema", lambda _conn: True)
    monkeypatch.setattr(trust_access, "_trust_store", None)
    monkeypatch.setattr(tool_approvals, "_STORE", None)
    pool = _Pool(_Conn())

    _init_companion_trust_store(_app, pool)

    assert trust_access._trust_store._pool is pool
    assert tool_approvals._STORE._pool is pool
    assert pool.borrowed == len(pool.returned)


def test_sidecar_audit_store_borrows_per_call(monkeypatch, _app):
    import umrs.api.audit_events as audit_events
    import umrs.api.audit_store as audit_store
    from umrs.server.http_sidecar import _init_audit_store

    monkeypatch.setattr(audit_store, "ensure_audit_schema", lambda _conn: None)
    monkeypatch.setattr(audit_events, "_audit_store", None)
    pool = _Pool(_Conn())

    _init_audit_store(_app, pool)

    audit_events._audit_store.write({"event_id": "ev-1"})
    assert pool.borrowed == len(pool.returned)
    assert any(s.startswith("INSERT INTO audit_events") for s in _durable(*pool.idle))
