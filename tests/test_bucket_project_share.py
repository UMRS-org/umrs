"""``project:<id>`` entries on a personal bucket's access list (ADR-080).

Until 2026-11-08 the buckets API accepted and stored ``project:<id>`` but
``UserBucketStore.can_access`` matched only ``system``, ``user:`` and
``circle:``: a bucket shared to a project admitted none of its members, so a
member's files-API call and HPC staging were refused while the owner's
worked (found while putting the partner demos into their own projects). A
project entry now admits the project's live members (ADR-097 D-1), the set
the gateway admits an ``X-Project-ID`` from.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from umrs.shared.storage import user_buckets as ub


class _SeqCur:
    def __init__(self, conn):
        self._conn = conn

    def execute(self, sql, params=None):
        self._conn.record.append((" ".join(str(sql).split()), params))

    def fetchone(self):
        return self._conn.rows.pop(0) if self._conn.rows else None

    def fetchall(self):
        return self._conn.all_rows


class _SeqConn:
    """Answers successive ``fetchone`` calls from *rows*, in order."""

    def __init__(self, rows=(), all_rows=()):
        self.rows = list(rows)
        self.all_rows = list(all_rows)
        self.record: list = []

    def cursor(self):
        return _SeqCur(self)

    def commit(self):
        pass

    def rollback(self):
        pass


class _Pool:
    def __init__(self, conn):
        self._conn = conn

    def getconn(self):
        return self._conn

    def putconn(self, conn):
        pass


def _store(conn):
    return ub.UserBucketStore(lambda: _Pool(conn))


def _members(monkeypatch, projects, calls=None):
    def fake(conn, user_id):  # noqa: ARG001 — mirrors _member_projects
        if calls is not None:
            calls.append(user_id)
        return set(projects)

    monkeypatch.setattr(ub, "_member_projects", fake)


def test_a_project_entry_admits_a_live_member(monkeypatch):
    _members(monkeypatch, {"proj-a"})
    conn = _SeqConn(rows=[None, (["project:proj-a"],)])
    assert _store(conn).can_access("alice.proj-a.data-output", "bob")
    assert conn.record[1] == (
        "SELECT access FROM user_buckets WHERE bucket = %s",
        ("alice.proj-a.data",),
    )


def test_a_project_entry_refuses_a_non_member(monkeypatch):
    _members(monkeypatch, {"proj-b"})
    conn = _SeqConn(rows=[None, (json.dumps(["project:proj-a"]),)])
    assert not _store(conn).can_access("alice.proj-a.data", "carol")


def test_membership_is_read_only_when_a_project_is_named(monkeypatch):
    calls: list = []
    _members(monkeypatch, {"proj-a"}, calls)
    conn = _SeqConn(rows=[None, (["user:dave", "circle:x/y"],)])
    assert not _store(conn).can_access("alice.proj-a.data", "bob")
    assert calls == []


def test_owner_user_circle_system_keep_one_query(monkeypatch):
    calls: list = []
    _members(monkeypatch, set(), calls)
    conn = _SeqConn(rows=[(1,)])
    assert _store(conn).can_access("alice.proj-a.data", "alice")
    assert len(conn.record) == 1 and calls == []


def test_an_unreadable_membership_fails_closed(monkeypatch):
    def boom(conn, user_id):  # noqa: ARG001
        raise RuntimeError("users row unreadable")

    monkeypatch.setattr(ub, "_member_projects", boom)
    conn = _SeqConn(rows=[None, (["project:proj-a"],)])
    with pytest.raises(RuntimeError):
        _store(conn).can_access("alice.proj-a.data", "bob")


def test_bucket_owned_by_turns_an_unreadable_membership_into_no(monkeypatch):
    from umrs.server.step_staging import bucket_owned_by

    def boom(conn, user_id):  # noqa: ARG001
        raise RuntimeError("users row unreadable")

    monkeypatch.setattr(ub, "_member_projects", boom)
    ub.set_user_bucket_store(_store(_SeqConn(rows=[None, None, (["project:p"],)])))
    try:
        assert not bucket_owned_by("alice.p.data", "bob")
    finally:
        ub.set_user_bucket_store(None)


def test_list_shared_with_matches_the_callers_projects(monkeypatch):
    _members(monkeypatch, {"proj-a", "proj-c"})
    row = ("alice.proj-a.data", "alice", "proj-a", "data", ["project:proj-a"])
    conn = _SeqConn(all_rows=[row])
    rows = _store(conn).list_shared_with("bob")
    sql, params = conn.record[0]
    assert "jsonb_array_elements_text(b.access)" in sql
    assert params == ("bob", "user:bob", "bob", ["project:proj-a", "project:proj-c"])
    assert [r.bucket for r in rows] == ["alice.proj-a.data"]


def test_list_shared_with_survives_unreadable_memberships(monkeypatch):
    def boom(conn, user_id):  # noqa: ARG001
        raise RuntimeError("users row unreadable")

    monkeypatch.setattr(ub, "_member_projects", boom)
    conn = _SeqConn(all_rows=[])
    assert _store(conn).list_shared_with("bob") == []
    assert conn.record[0][1][-1] == []


def _const(value):
    """A stand-in that ignores its arguments and answers *value*."""

    def answer(*_args):
        return value

    return answer


def test_member_projects_is_the_live_membership_set(monkeypatch):
    from umrs.admin import account_cascade

    record = SimpleNamespace(username="bob", projects=["proj-a"], disabled=False)
    monkeypatch.setattr(account_cascade, "read_record", _const(record))
    monkeypatch.setattr(
        account_cascade, "live_project_ids", _const({"proj-a", "own"})
    )
    assert ub._member_projects(object(), "bob") == {"proj-a", "own"}


@pytest.mark.parametrize(
    "record",
    [None, SimpleNamespace(username="bob", projects=["proj-a"], disabled=True)],
)
def test_unknown_or_disabled_accounts_are_members_of_nothing(monkeypatch, record):
    from umrs.admin import account_cascade

    monkeypatch.setattr(account_cascade, "read_record", _const(record))
    monkeypatch.setattr(
        account_cascade, "live_project_ids", _const({"proj-a"})
    )
    assert ub._member_projects(object(), "bob") == set()


def test_project_ids_parses_stored_lists():
    assert ub._project_ids(["project:a", "user:b", "project:", "circle:a/b"]) == ["a"]
    assert ub._project_ids(json.dumps(["project:x"])) == ["x"]
    assert ub._project_ids(None) == []


def _pg_pool():
    pytest.importorskip("psycopg2")
    from umrs.shared.pg import build_sync_pool_from_env

    try:
        return build_sync_pool_from_env(min_conn=1, max_conn=2)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Postgres unavailable: {exc}")


def _apply_migrations(pool):
    root = Path(ub.__file__).resolve().parents[2] / "migrations"
    conn = pool.getconn()
    try:
        with conn, conn.cursor() as cur:
            for name in ("0115_user_buckets.sql", "0116_resource_access.sql", "0119_circles.sql"):
                cur.execute((root / name).read_text())
            cur.execute("DELETE FROM user_buckets WHERE owner_id = 'pytest-pshare'")
    finally:
        pool.putconn(conn)


def test_project_share_against_postgres(monkeypatch):
    # Env-gated like test_store_round_trip_against_postgres.
    pool = _pg_pool()
    _apply_migrations(pool)
    _members(monkeypatch, {"pshare-proj"})
    store = ub.UserBucketStore(lambda: pool)
    try:
        store.add(ub.UserBucket("pytest-pshare.p.v", "pytest-pshare", "p", "v"))
        assert not store.can_access("pytest-pshare.p.v", "member")
        store.set_access("pytest-pshare.p.v", ["project:pshare-proj"])
        assert store.can_access("pytest-pshare.p.v-output", "member")
        shared = [b.bucket for b in store.list_shared_with("member")]
        assert "pytest-pshare.p.v" in shared
        store.set_access("pytest-pshare.p.v", ["project:other"])
        assert not store.can_access("pytest-pshare.p.v", "member")
        assert "pytest-pshare.p.v" not in [b.bucket for b in store.list_shared_with("member")]
    finally:
        conn = pool.getconn()
        try:
            with conn, conn.cursor() as cur:
                cur.execute("DELETE FROM user_buckets WHERE owner_id = 'pytest-pshare'")
        finally:
            pool.putconn(conn)
        pool.closeall()
