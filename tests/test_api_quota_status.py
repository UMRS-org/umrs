"""Quota status route pins (ADR-051 D-2/D-3 read model leg, 2026-09-16).

The /umrs/quota/units gauge must show exactly the numbers the enforcement
edges enforce (same env knobs, same stores) — verified-identity only, and a
typed 503 when the pool is absent. Nothing in the summary may be fabricated.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient
from umrs.api.quota_status import router


class _Cursor:
    def __init__(self, result=None):
        self._result = result if result is not None else []

    def execute(self, _sql, _params=None):
        pass

    def fetchone(self):
        if not self._result:
            return None
        return self._result.pop(0)

    def __enter__(self):
        return self

    def __exit__(self, *_a):
        return False


class _Conn:
    """Serves the two SQL statements the route's stores issue."""

    def __init__(self, *, disk_bytes: int):
        self._disk = disk_bytes

    def cursor(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *_a):
        return False

    def execute(self, sql, _params=None):
        if "FROM user_disk_usage" in sql:
            self._cur = _Cursor([(self._disk,)])
        elif "COUNT(*) FROM jobs" in sql:
            self._cur = _Cursor([(2,)])
        else:
            self._cur = _Cursor([])
        return self._cur

    def fetchone(self):
        return self._cur.fetchone()

    def commit(self):
        pass


class _Pool:
    def __init__(self, *, disk_bytes: int):
        self._conn = _Conn(disk_bytes=disk_bytes)
        self.returned = 0

    def getconn(self):
        return self._conn

    def putconn(self, _conn):
        self.returned += 1


def _client(pool) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    if pool is not None:
        app.state.umrs_db_pool = pool
    return TestClient(app)


_VERIFIED = {"X-User-ID": "alice", "X-Verified-Project-ID": "proj-a"}


class TestQuotaUnitsRoute:
    def test_units_reflects_same_env_caps_and_store_numbers(self, monkeypatch):
        monkeypatch.setenv("UMRS_AGENT_PROCESSES_PER_USER_MAX", "4")
        monkeypatch.setenv("UMRS_USER_DISK_BYTES_MAX", str(50 * (1 << 30)))
        pool = _Pool(disk_bytes=1234)
        resp = _client(pool).get("/umrs/quota/units", headers=_VERIFIED)
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["user_id"] == "alice"
        assert body["cpu_process"] == {"current": 2, "limit": 4}
        assert body["disk_bytes"] == {"current": 1234, "limit": 50 * (1 << 30)}

    def test_uncapped_poles_show_limit_zero_honestly(self, monkeypatch):
        monkeypatch.setenv("UMRS_AGENT_PROCESSES_PER_USER_MAX", "0")
        monkeypatch.setenv("UMRS_USER_DISK_BYTES_MAX", "0")
        pool = _Pool(disk_bytes=99)
        resp = _client(pool).get("/umrs/quota/units", headers=_VERIFIED)
        assert resp.status_code == 200
        body = resp.json()
        assert body["cpu_process"]["limit"] == 0
        assert body["disk_bytes"] == {"current": 99, "limit": 0}

    def test_anonymous_gets_typed_401(self):
        # Test harness runs mvp auth mode — an anonymous scope reaches the
        # body and must be refused with the canonical identity code, riding
        # the x-platform-error-code response header (additive typed shape).
        resp = _client(_Pool(disk_bytes=0)).get("/umrs/quota/units")
        assert resp.status_code == 401
        assert resp.headers.get("x-platform-error-code") == "scopeidentity.unauthorized"

    def test_pool_absent_is_typed_503(self):
        resp = _client(None).get("/umrs/quota/units", headers=_VERIFIED)
        assert resp.status_code == 503
        assert resp.headers.get("x-platform-error-code") == "quota.unavailable"

    def test_conn_is_returned_to_the_pool(self, monkeypatch):
        # 2026-09-09 pool-exhaustion class: every gauge read must balance
        # getconn with putconn, including the failure paths.
        monkeypatch.setenv("UMRS_AGENT_PROCESSES_PER_USER_MAX", "4")
        monkeypatch.setenv("UMRS_USER_DISK_BYTES_MAX", "1")
        pool = _Pool(disk_bytes=0)
        resp = _client(pool).get("/umrs/quota/units", headers=_VERIFIED)
        assert resp.status_code == 200
        # two stores (agent jobs + disk ledger), two balanced get/put pairs
        assert pool.returned == 2
