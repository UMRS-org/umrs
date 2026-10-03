"""RAG corpus read-sharing (ADR-080).

The corpus is a project namespace with no row of its own; 0116's
``rag_corpus_access`` registry carries its scope list. Reads the tier
denies may pass through the list (user:/group:/project:/system); writes
never consult it. Fail-closed everywhere.
"""

from __future__ import annotations

import json

import pytest
from starlette.requests import Request
from umrs.api.rag import _require_rag_scope
from umrs.rag.corpus_access import (
    CorpusAccessStore,
    set_corpus_access_store,
    shared_corpus_readable,
)


class _Cur:
    def __init__(self, conn):
        self._conn = conn

    def execute(self, sql, params=None):
        self._conn.record.append((" ".join(str(sql).split()), params))
        self._conn._params = params

    def fetchone(self):
        return self._conn._fetchone

    def fetchall(self):
        return self._conn._fetchall


class _Conn:
    def __init__(self, fetchone=None, fetchall=()):
        self._fetchone = fetchone
        self._fetchall = list(fetchall)
        self.record: list = []
        self._params = None
        self.committed = False

    def cursor(self):
        return _Cur(self)

    def commit(self):
        self.committed = True

    def rollback(self):
        pass


class _Pool:
    def __init__(self, conn):
        self._conn = conn

    def getconn(self):
        return self._conn

    def putconn(self, conn):
        pass


def _request(user: str, project: str, admin: bool = False) -> Request:
    headers = [
        (b"x-user-id", user.encode()),
        (b"x-verified-project-id", project.encode()),
    ]
    if admin:
        headers.append((b"x-verified-is-admin", b"true"))
    return Request(
        {"type": "http", "method": "GET", "path": "/", "headers": headers}
    )


@pytest.fixture(autouse=True)
def _reset_store():
    set_corpus_access_store(None)
    yield
    set_corpus_access_store(None)


# ── store SQL ────────────────────────────────────────────────────────────


def test_get_access_reads_the_row():
    conn = _Conn(fetchone=(["user:bob", "group:lab"],))
    store = CorpusAccessStore(lambda: _Pool(conn))
    assert store.get_access("proj-a") == ["user:bob", "group:lab"]
    sql, params = conn.record[0]
    assert "FROM rag_corpus_access" in sql and params == ("proj-a",)


def test_get_access_absent_is_empty_not_shared():
    store = CorpusAccessStore(lambda: _Pool(_Conn(fetchone=None)))
    assert store.get_access("proj-a") == []


def test_set_access_upserts_and_commits():
    conn = _Conn()
    store = CorpusAccessStore(lambda: _Pool(conn))
    store.set_access("proj-a", ["system"], set_by="ops")
    sql, params = conn.record[0]
    assert "ON CONFLICT (project_id) DO UPDATE" in sql
    assert params == ("proj-a", json.dumps(["system"]), "ops")
    assert conn.committed


def test_shared_readable_sql_joins_active_memberships():
    conn = _Conn(fetchone=(1,))
    store = CorpusAccessStore(lambda: _Pool(conn))
    assert store.shared_readable("proj-a", "alice")
    sql, params = conn.record[0]
    assert "group_members" in sql and "g.status = 'active'" in sql
    assert '["system"]' in sql and "jsonb_build_array" in sql
    assert params == ("proj-a", "user:alice", "alice")


def test_shared_readable_fail_closed():
    store = CorpusAccessStore(lambda: _Pool(_Conn(fetchone=None)))
    assert not store.shared_readable("proj-a", "alice")
    assert not shared_corpus_readable("proj-a", "")  # no user → never


# ── guard wiring ─────────────────────────────────────────────────────────


class _FakeAccess:
    def __init__(self, rows):
        self.rows = rows  # {project: access list}

    def get_access(self, project):
        return list(self.rows.get(project, []))

    def set_access(self, project, access, *, set_by):  # noqa: ARG002 — test double mirrors the real signature
        self.rows[project] = list(access)

    def shared_readable(self, project, user_id):
        for entry in self.rows.get(project, []):
            if entry == "system" or entry == f"user:{user_id}":
                return True
        return False


def test_read_denied_without_share():
    set_corpus_access_store(_FakeAccess({}))
    req = _request("bob", "proj-b")
    with pytest.raises(Exception) as excinfo:
        _require_rag_scope(req, "proj-a", write=False)
    assert getattr(excinfo.value, "status_code", None) == 403


def test_read_allowed_through_user_share():
    set_corpus_access_store(_FakeAccess({"proj-a": ["user:bob"]}))
    user_id, project = _require_rag_scope(_request("bob", "proj-b"), "proj-a")
    assert (user_id, project) == ("bob", "proj-b")


def test_read_allowed_through_system_share():
    set_corpus_access_store(_FakeAccess({"proj-a": ["system"]}))
    user_id, _ = _require_rag_scope(_request("carol", "proj-c"), "proj-a")
    assert user_id == "carol"


def test_write_never_consults_the_share():
    set_corpus_access_store(_FakeAccess({"proj-a": ["user:bob", "system"]}))
    req = _request("bob", "proj-b")
    with pytest.raises(Exception) as excinfo:
        _require_rag_scope(req, "proj-a", write=True)
    assert getattr(excinfo.value, "status_code", None) == 403


def test_own_project_never_touches_the_registry():
    set_corpus_access_store(_FakeAccess({}))
    user_id, project = _require_rag_scope(_request("alice", "proj-a"), "proj-a", write=True)
    assert (user_id, project) == ("alice", "proj-a")
