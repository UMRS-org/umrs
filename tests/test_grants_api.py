"""Route tests for /umrs/grants (ADR-041)."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from umrs.api import grants as grants_api
from umrs.api.grants import router
from umrs.grants.store import ASSET_KINDS


class _FakeCursor:
    """In-memory asset_grants table behind the SQL shapes GrantStore emits."""

    def __init__(self, rows):
        self._rows = rows
        self.rowcount = 0
        self._one = None
        self._many = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def execute(self, sql, params):
        lowered = sql.strip().lower()
        if lowered.startswith("insert"):
            (
                grant_id,
                asset_kind,
                asset_id,
                grantee_kind,
                grantee_id,
                version_pin,
                expires_at,
                created_by,
            ) = params
            self._rows[grant_id] = [
                grant_id, asset_kind, asset_id, grantee_kind, grantee_id,
                version_pin, expires_at, None, created_by, "2026-09-07",
            ]
            return
        if lowered.startswith("update"):
            marker, grant_id = params
            row = self._rows.get(grant_id)
            if row is not None and row[7] is None:
                row[7] = marker
                self.rowcount = 1
            else:
                self.rowcount = 0
            return
        if lowered.startswith("select"):
            self._select(lowered, params)

    def _select(self, sql, params):
        self._one = None
        self._many = []
        if "grant_id = %s" in sql:
            row = self._rows.get(params[0])
            self._one = tuple(row) if row else None
            return
        if "asset_kind = %s and asset_id = %s" in sql:
            kind, asset = params
            self._many = [
                tuple(r)
                for r in self._rows.values()
                if r[1] == kind and r[2] == asset
            ]
            return
        if "grantee_kind = %s and grantee_id = %s" in sql:
            kind, gid = params
            self._many = [
                tuple(r)
                for r in self._rows.values()
                if r[3] == kind and r[4] == gid and r[7] is None
            ]

    def fetchone(self):
        return self._one

    def fetchall(self):
        return self._many


class _FakeConn:
    def __init__(self, rows):
        self._cursor = _FakeCursor(rows)

    def cursor(self):
        return self._cursor

    def commit(self):
        return None


class _FakePool:
    def __init__(self):
        self.rows = {}
        self.conn = _FakeConn(self.rows)

    def getconn(self):
        return self.conn

    def putconn(self, _conn):
        return None


def _app(with_pool=True) -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    if with_pool:
        app.state.umrs_db_pool = _FakePool()
    return app


def _auth(client: TestClient, user="carol", project="proj-x", admin=False):
    client.headers["x-user-id"] = user
    client.headers["x-verified-project-id"] = project
    client.headers["x-verified-is-admin"] = "true" if admin else "false"
    return client


@pytest.fixture()
def client():
    with TestClient(_app()) as c:
        yield _auth(c)


@pytest.fixture(autouse=True)
def writable_assets(monkeypatch):
    """Assets exist and are caller-writable unless a test overrides."""
    monkeypatch.setattr(grants_api, "_model_writable", lambda *_: True)
    monkeypatch.setattr(grants_api, "_graph_writable", lambda *_: True)


def test_create_grant_round_trip(client):
    response = client.post(
        "/umrs/grants",
        json={
            "asset_kind": "model",
            "asset_id": "mdl-1",
            "grantee_kind": "user",
            "grantee_id": "alice",
        },
    )
    assert response.status_code == 200
    body = response.json()
    grant = body["grant"]
    assert grant["grant_id"].startswith("grt-")
    assert grant["tombstoned_at"] is None
    assert grant["created_by"] == "carol"
    client.app.state.umrs_db_pool.rows  # row persisted in the fake table
    assert grant["grant_id"] in client.app.state.umrs_db_pool.rows


def test_create_rejects_unknown_asset_kind(client):
    response = client.post(
        "/umrs/grants",
        json={
            "asset_kind": "endpoint",
            "asset_id": "e-1",
            "grantee_kind": "user",
            "grantee_id": "alice",
        },
    )
    assert response.status_code == 400


def test_create_404s_when_asset_not_writable(client, monkeypatch):
    monkeypatch.setattr(grants_api, "_model_writable", lambda *_: False)
    response = client.post(
        "/umrs/grants",
        json={
            "asset_kind": "model",
            "asset_id": "mdl-1",
            "grantee_kind": "user",
            "grantee_id": "alice",
        },
    )
    assert response.status_code == 404


def test_missing_identity_401():
    with TestClient(_app()) as anon:
        assert anon.get("/umrs/grants/mine").status_code == 401


def test_missing_pool_503():
    with TestClient(_app(with_pool=False)) as bare:
        _auth(bare)
        assert bare.get("/umrs/grants/mine").status_code == 503


def test_mine_lists_user_and_project_grants(client):
    rows = client.app.state.umrs_db_pool.rows
    rows["grt-a"] = [
        "grt-a", "model", "mdl-1", "user", "carol",
        None, None, None, "root", "2026-09-07",
    ]
    rows["grt-b"] = [
        "grt-b", "knowledge_graph", "g-1", "project", "proj-x",
        None, None, None, "root", "2026-09-07",
    ]
    rows["grt-c"] = [  # tombstoned → hidden from /mine
        "grt-c", "model", "mdl-2", "user", "carol",
        None, None, datetime(2026, 9, 7, tzinfo=timezone.utc), "root", "2026-09-06",
    ]
    rows["grt-d"] = [  # someone else → hidden
        "grt-d", "model", "mdl-3", "user", "alice",
        None, None, None, "root", "2026-09-07",
    ]
    response = client.get("/umrs/grants/mine")
    ids = {g["grant_id"] for g in response.json()["grants"]}
    assert ids == {"grt-a", "grt-b"}


def test_list_for_asset_includes_tombstones_for_writers(client):
    rows = client.app.state.umrs_db_pool.rows
    rows["grt-a"] = [
        "grt-a", "model", "mdl-1", "user", "alice",
        None, None, None, "carol", "2026-09-07",
    ]
    rows["grt-b"] = [
        "grt-b", "model", "mdl-1", "user", "bob",
        None, None, datetime(2026, 9, 7, tzinfo=timezone.utc), "carol", "2026-09-07",
    ]
    response = client.get("/umrs/grants", params={"asset_kind": "model", "asset_id": "mdl-1"})
    assert response.status_code == 200
    grants = {g["grant_id"]: g for g in response.json()["grants"]}
    assert grants["grt-b"]["tombstoned_at"] is not None  # audit visible


def test_revoke_tombstones_never_deletes(client):
    create = client.post(
        "/umrs/grants",
        json={
            "asset_kind": "model",
            "asset_id": "mdl-1",
            "grantee_kind": "user",
            "grantee_id": "alice",
        },
    )
    grant_id = create.json()["grant"]["grant_id"]
    revoke = client.post(f"/umrs/grants/{grant_id}/revoke")
    assert revoke.status_code == 200
    assert revoke.json()["revoked"] is True
    row = client.app.state.umrs_db_pool.rows[grant_id]
    assert row[7] is not None  # tombstoned_at set — row still present
    again = client.post(f"/umrs/grants/{grant_id}/revoke")
    assert again.json()["revoked"] is False  # idempotent


def test_supported_asset_kinds_documented():
    assert set(ASSET_KINDS) == {"model", "knowledge_graph"}
