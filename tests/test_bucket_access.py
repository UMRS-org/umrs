"""Bucket object-access lists (ADR-080).

A personal bucket's row carries ``access`` (user:/project:/circle:/system,
several entries; multi-membership admitted): the named callers read and
write OBJECTS — uploads, lists, downloads, HPC staging — while the bucket
lifecycle stays owner-only. ``system`` entries are admin-granted only.
"""

from __future__ import annotations

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from umrs.api.buckets import router as buckets_router
from umrs.server.step_staging import bucket_owned_by
from umrs.shared.storage import user_buckets as ub


class _Cur:
    def __init__(self, conn):
        self._conn = conn
        self.rowcount = 0

    def execute(self, sql, params=None):
        self._conn.record.append((" ".join(str(sql).split()), params))
        self.rowcount = 1

    def fetchone(self):
        return self._conn._fetchone

    def fetchall(self):
        return self._conn._fetchall


class _Conn:
    def __init__(self, fetchone=None, fetchall=()):
        self._fetchone = fetchone
        self._fetchall = list(fetchall)
        self.record: list = []

    def cursor(self):
        return _Cur(self)

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


# ── store SQL ────────────────────────────────────────────────────────────


def test_set_access_updates_and_invalidates_owner_cache():
    conn = _Conn()
    store = ub.UserBucketStore(lambda: _Pool(conn))
    assert store.set_access("alice.proj.data", ["user:bob", "circle:proj-a/lab"])
    sql, params = conn.record[0]
    assert "UPDATE user_buckets SET access" in sql
    assert params == (json.dumps(["user:bob", "circle:proj-a/lab"]), "alice.proj.data")


def test_can_access_sql_covers_user_circle_system():
    conn = _Conn(fetchone=(1,))
    store = ub.UserBucketStore(lambda: _Pool(conn))
    assert store.can_access("alice.proj.data-output", "bob")  # -output → input row
    sql, params = conn.record[0]
    assert '["system"]' in sql and "circle_members" in sql
    assert "g.status = 'active'" in sql
    assert params == ("alice.proj.data", "bob", "user:bob", "bob")


def test_can_access_fail_closed():
    store = ub.UserBucketStore(lambda: _Pool(_Conn(fetchone=None)))
    assert not store.can_access("alice.proj.data", "bob")
    assert not store.can_access("alice.proj.data", "")


def test_list_shared_with_excludes_own_and_unshared():
    conn = _Conn(fetchall=[("bob.proj.pool", "bob", "proj", "pool", ["system"])])
    store = ub.UserBucketStore(lambda: _Pool(conn))
    rows = store.list_shared_with("alice")
    sql, params = conn.record[0]
    assert "b.owner_id != %s" in sql
    assert params[0] == "alice"
    assert [r.access for r in rows] == [("system",)]
    assert rows[0].to_dict()["access"] == ["system"]


# ── ownership rule (files API + HPC staging share it) ────────────────────


class _FakeAccessStore:
    def __init__(self, admitted=()):
        self._admitted = set(admitted)

    def owner_of(self, bucket):  # noqa: ARG002 — test double mirrors the real signature
        return None

    def can_access(self, bucket, user_id):  # noqa: ARG002 — test double mirrors the real signature
        return user_id in self._admitted


@pytest.fixture(autouse=True)
def _reset_bucket_store():
    ub.set_user_bucket_store(None)
    yield
    ub.set_user_bucket_store(None)


def test_bucket_owned_by_via_access_list():
    ub.set_user_bucket_store(_FakeAccessStore(admitted={"bob"}))
    assert bucket_owned_by("alice.proj.data", "bob")
    assert not bucket_owned_by("alice.proj.data", "carol")


def test_bucket_owned_by_store_error_is_fail_closed():
    class _Exploding:
        def can_access(self, bucket, user_id):  # noqa: ARG002 — test double mirrors the real signature
            raise RuntimeError("db down")

        def owner_of(self, bucket):  # noqa: ARG002 — test double mirrors the real signature
            return None

    ub.set_user_bucket_store(_Exploding())
    assert not bucket_owned_by("alice.proj.data", "bob")


# ── routes ───────────────────────────────────────────────────────────────


class _RouteStore:
    def __init__(self):
        self.rows: dict[str, ub.UserBucket] = {}

    def add(self, entry):
        self.rows[entry.bucket] = entry
        return True

    def get(self, bucket):
        return self.rows.get(ub.input_bucket_of(bucket))

    def list_for(self, owner):
        return [r for r in self.rows.values() if r.owner_id == owner]

    def owner_of(self, bucket):
        row = self.get(bucket)
        return row.owner_id if row else None

    def set_access(self, bucket, access):
        row = self.get(bucket)
        if row is None:
            return False
        self.rows[row.bucket] = ub.UserBucket(
            row.bucket, row.owner_id, row.project, row.alias, tuple(access)
        )
        return True

    def can_access(self, bucket, user_id):
        row = self.get(bucket)
        if row is None:
            return False
        if row.owner_id == user_id:
            return True
        return "system" in row.access or f"user:{user_id}" in row.access

    def list_shared_with(self, user_id):
        return [
            r
            for r in self.rows.values()
            if r.owner_id != user_id and ("system" in r.access or f"user:{user_id}" in r.access)
        ]


def _client(store):
    app = FastAPI()
    app.include_router(buckets_router)
    ub.set_user_bucket_store(store)
    return TestClient(app)


def _headers(user: str, project: str = "proj-a", admin: bool = False):
    headers = {"x-user-id": user, "x-verified-project-id": project}
    if admin:
        headers["x-verified-is-admin"] = "true"
    return headers


@pytest.fixture
def store():
    s = _RouteStore()
    s.add(ub.UserBucket("alice.proj-a.videos", "alice", "proj-a", "videos"))
    return s


def test_owner_shares_with_users_and_circles(store):
    client = _client(store)
    r = client.put(
        "/umrs/buckets/alice.proj-a.videos/access",
        json={"access": ["user:bob", "circle:proj-a/lab"]},
        headers=_headers("alice"),
    )
    assert r.status_code == 200, r.text
    assert r.json()["access"] == ["user:bob", "circle:proj-a/lab"]
    assert store.get("alice.proj-a.videos").access == ("user:bob", "circle:proj-a/lab")


def test_non_owner_cannot_set_access(store):
    client = _client(store)
    r = client.put(
        "/umrs/buckets/alice.proj-a.videos/access",
        json={"access": ["user:bob"]},
        headers=_headers("carol", "proj-b"),
    )
    # 404, not 403 (H-1, 2026-10-18): a non-owner who may not use the bucket
    # gets the same answer as for an unknown name (no existence oracle).
    assert r.status_code == 404


def test_system_entry_is_admin_only(store):
    client = _client(store)
    assert (
        client.put(
            "/umrs/buckets/alice.proj-a.videos/access",
            json={"access": ["system"]},
            headers=_headers("alice"),
        ).status_code
        == 403
    )
    assert (
        client.put(
            "/umrs/buckets/alice.proj-a.videos/access",
            json={"access": ["system"]},
            headers=_headers("ops", admin=True),
        ).status_code
        == 200
    )


def test_get_access_visible_to_admitted_callers(store):
    store.set_access("alice.proj-a.videos", ["user:bob"])
    client = _client(store)
    r = client.get("/umrs/buckets/alice.proj-a.videos/access", headers=_headers("bob", "proj-b"))
    assert r.status_code == 200
    assert r.json()["access"] == ["user:bob"]
    assert (
        client.get(
            "/umrs/buckets/alice.proj-a.videos/access", headers=_headers("carol", "proj-b")
        ).status_code
        == 404
    )


def test_list_marks_shared_buckets(store):
    store.set_access("alice.proj-a.videos", ["user:bob"])
    client = _client(store)
    r = client.get("/umrs/buckets", headers=_headers("bob", "proj-b"))
    assert r.status_code == 200
    shared = [b for b in r.json()["buckets"] if b["kind"] == "shared"]
    assert [b["bucket"] for b in shared] == ["alice.proj-a.videos"]
    own = [b for b in r.json()["buckets"] if b["bucket"] == "alice.proj-a.videos"]
    assert own[0]["access"] == ["user:bob"]
