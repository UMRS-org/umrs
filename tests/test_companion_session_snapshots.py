"""Route tests for companion session snapshots (ADR-036 F2 first leg)."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from umrs.api.companion_sessions import router


class _FakeCursor:
    def __init__(self, table, recorded):
        self._table = table
        self._recorded = recorded
        self._result = []
        self._row = None
        self.rowcount = 0

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def execute(self, sql, params):
        sql_l = sql.lower().strip()
        self._recorded.append((sql, params))
        if sql_l.startswith("insert"):
            uid, sid, name, payload, marker = params
            existing = self._table.get((uid, sid))
            if existing is None or existing["update_marker"] <= marker:
                self._table[(uid, sid)] = {
                    "name": name,
                    "payload": payload,
                    "update_marker": marker,
                }
                self.rowcount = 1
            else:
                self.rowcount = 0
            return
        if sql_l.startswith("select session_id, name, payload"):
            uid, sid = params
            row = self._table.get((uid, sid))
            if row is None:
                self._row = None
            else:
                self._row = (sid, row["name"], row["payload"], row["update_marker"])
            return
        if sql_l.startswith("select session_id, name, update_marker"):
            uid, limit = params
            rows = [
                (sid, r["name"], r["update_marker"])
                for (u, sid), r in self._table.items()
                if u == uid
            ]
            self._result = rows[:limit]

    def fetchone(self):
        return self._row

    def fetchall(self):
        return self._result


class _FakeConn:
    def __init__(self, table, recorded):
        self._cursor = _FakeCursor(table, recorded)
        self.committed = False

    def cursor(self):
        return self._cursor

    def commit(self):
        self.committed = True


class _FakePool:
    def __init__(self):
        self.table: dict = {}
        self.recorded: list = []
        self.conn = _FakeConn(self.table, self.recorded)

    def getconn(self):
        return self.conn

    def putconn(self, _conn):
        return None


def _client(find_pool: _FakePool) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    app.state.umrs_db_pool = find_pool
    return TestClient(app)


class TestSnapshotUpsert:
    def test_insert_then_stale_marker_not_applied(self):
        pool = _FakePool()
        client = _client(pool)
        body = {"name": "demo", "updated_at": "2026-09-06T10:00:00Z", "payload": "{}"}
        r1 = client.post(
            "/umrs/companion/sessions/sess-1/snapshot",
            json=body,
            headers={"x-user-id": "alice"},
        )
        assert r1.status_code == 200
        assert r1.json()["applied"] is True

        older = {"name": "demo", "updated_at": "2026-09-06T09:00:00Z", "payload": "{}"}
        r2 = client.post(
            "/umrs/companion/sessions/sess-1/snapshot",
            json=older,
            headers={"x-user-id": "alice"},
        )
        assert r2.json()["applied"] is False

    def test_missing_identity_header_is_401(self):
        client = _client(_FakePool())
        body = {"name": "x", "updated_at": "1", "payload": "{}"}
        resp = client.post("/umrs/companion/sessions/s-1/snapshot", json=body)
        assert resp.status_code == 401


class TestSnapshotGet:
    def test_owner_reads_own_row(self):
        pool = _FakePool()
        client = _client(pool)
        body = {"name": "demo", "updated_at": "100", "payload": '{"a": 1}'}
        client.post(
            "/umrs/companion/sessions/sess-1/snapshot",
            json=body,
            headers={"x-user-id": "alice"},
        )
        resp = client.get("/umrs/companion/sessions/sess-1", headers={"x-user-id": "alice"})
        assert resp.status_code == 200
        assert resp.json()["payload"] == '{"a": 1}'

    def test_other_user_gets_404(self):
        pool = _FakePool()
        client = _client(pool)
        body = {"name": "demo", "updated_at": "100", "payload": "{}"}
        client.post(
            "/umrs/companion/sessions/sess-1/snapshot",
            json=body,
            headers={"x-user-id": "alice"},
        )
        resp = client.get("/umrs/companion/sessions/sess-1", headers={"x-user-id": "bob"})
        assert resp.status_code == 404


class TestSnapshotList:
    def test_lists_only_my_summary_rows(self):
        pool = _FakePool()
        client = _client(pool)
        for uid in ("alice", "bob"):
            client.post(
                "/umrs/companion/sessions/s1/snapshot",
                json={"name": "x", "updated_at": "1", "payload": "{}"},
                headers={"x-user-id": uid},
            )
        resp = client.get("/umrs/companion/sessions", headers={"x-user-id": "alice"})
        assert resp.status_code == 200
        rows = resp.json()["snapshots"]
        assert len(rows) == 1
        assert rows[0]["session_id"] == "s1"


class TestDegradation:
    def test_no_pool_is_503(self):
        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)
        body = {"name": "x", "updated_at": "1", "payload": "{}"}
        resp = client.post(
            "/umrs/companion/sessions/s1/snapshot",
            json=body,
            headers={"x-user-id": "alice"},
        )
        assert resp.status_code == 503
