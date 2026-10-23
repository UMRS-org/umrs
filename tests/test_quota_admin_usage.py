"""ADR-099 (RS-1 lane Q): who sees remote-session usage, and how much.

- ``GET /umrs/admin/usage``: admin only (RS-0 credential), one row per
  subject and day of the proxy's ``infer_proxy`` rows at a site, with the
  subject's daily token limit from the session policy;
- ``umrs-server admin usage show`` calls it with the admin JWT and prints it;
- ``/umrs/quota/units`` gains the caller's ``sessions`` and ``tokens_today``
  poles (``limit`` 0 = no limit), ``null`` when the policy is unreadable.
"""

from __future__ import annotations

import io
import json
import time
import urllib.request
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from jose import jwt as jose_jwt
from umrs.api import audit_events
from umrs.api.admin_usage import router as usage_router
from umrs.api.quota_status import router as quota_router
from umrs.cli.admin.main import main as admin_main

SECRET = "test-jwt-secret-not-real"


def _bearer(roles):
    claims = {"sub": "root", "roles": roles, "project_id": "ops", "exp": int(time.time()) + 600}
    return {"Authorization": "Bearer " + jose_jwt.encode(claims, SECRET, algorithm="HS256")}


class _Cursor:
    def __init__(self, pool):
        self.pool = pool
        self._result = None

    def execute(self, sql, params=None):
        self.pool.sql.append((sql, params))
        self._result = self.pool.answer(sql)

    def fetchall(self):
        return self._result or []

    def fetchone(self):
        return (self._result or [None])[0]

    def __enter__(self):
        return self

    def __exit__(self, *_a):
        return False


class _Conn:
    def __init__(self, pool):
        self.pool = pool

    def __enter__(self):
        return self

    def __exit__(self, *_a):
        return False

    def cursor(self):
        return _Cursor(self.pool)

    def commit(self):
        pass


class _Pool:
    def __init__(self, rows=(), tokens=0, sessions=0, disk=0):
        self.rows = list(rows)
        self.tokens = tokens
        self.sessions = sessions
        self.disk = disk
        self.sql: list = []

    def answer(self, sql):
        if "GROUP BY" in sql:
            return self.rows
        if "SUM(total_tokens)" in sql:
            return [(self.tokens,)]
        if "job_type = %s" in sql:
            return [(self.sessions,)]
        if "COUNT(*) FROM jobs" in sql:
            return [(1,)]
        if "user_disk_usage" in sql:
            return [(self.disk,)]
        return []

    def getconn(self):
        return _Conn(self)

    def putconn(self, _conn):
        pass


@pytest.fixture
def audit(monkeypatch):
    sink = type("Sink", (), {"rows": [], "write": lambda self, r: self.rows.append(r)})()
    monkeypatch.setattr(audit_events, "_audit_store", sink)
    return sink


def _client(pool, monkeypatch):
    monkeypatch.setenv("JWT_SECRET", SECRET)
    monkeypatch.setattr("umrs.jobs.store.configured_site_name", lambda: "rccs")
    app = FastAPI()
    app.include_router(usage_router)
    app.include_router(quota_router)
    app.state.umrs_db_pool = pool
    return TestClient(app)


def _limits(user_cap=None, project_cap=None, sessions=12):
    def fn(_pool, *, venue, user_id, project_id=""):
        assert venue == "rccs"
        return {
            "budget": {"tokens_per_day_user": user_cap, "tokens_per_day_project": project_cap},
            "sessions": {"concurrent_user": sessions},
        }

    return fn


ROWS = [("alice", "2026-10-23", 100, 50, 150, 3), ("bob", "2026-10-23", 10, 5, 15, 1)]


@pytest.mark.usefixtures("audit")
class TestAdminUsageRoute:
    def test_rows_per_user_and_day_with_limits(self, monkeypatch):
        monkeypatch.setattr("umrs.quotas.enforcer.policy_limits", _limits(user_cap=2000000))
        pool = _Pool(rows=ROWS)
        res = _client(pool, monkeypatch).get("/umrs/admin/usage?days=7", headers=_bearer(["admin"]))
        assert res.status_code == 200, res.text
        body = res.json()
        assert (body["venue"], body["scope"], body["limits"]) == ("rccs", "user", "ok")
        assert body["rows"][0] == {
            "scope": "user",
            "subject": "alice",
            "day": "2026-10-23",
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            "requests": 3,
            "limit": 2000000,
        }
        sql, params = next((s, p) for s, p in pool.sql if "GROUP BY" in s)
        assert "origin = 'infer_proxy'" in sql and params[:2] == ("UTC", "rccs")

    def test_project_alone_groups_by_project(self, monkeypatch):
        monkeypatch.setattr("umrs.quotas.enforcer.policy_limits", _limits(project_cap=None))
        pool = _Pool(rows=[("opossum", "2026-10-23", 1, 1, 2, 1)])
        res = _client(pool, monkeypatch).get(
            "/umrs/admin/usage?project=opossum", headers=_bearer(["admin"])
        )
        body = res.json()
        assert body["scope"] == "project" and body["rows"][0]["limit"] is None
        sql, params = next((s, p) for s, p in pool.sql if "GROUP BY" in s)
        assert sql.lstrip().startswith("SELECT project_id") and params[-1] == "opossum"

    def test_unreadable_policy_still_shows_usage(self, monkeypatch):
        def broken(*_a, **_k):
            raise RuntimeError("no rules")

        monkeypatch.setattr("umrs.quotas.enforcer.policy_limits", broken)
        res = _client(_Pool(rows=ROWS), monkeypatch).get(
            "/umrs/admin/usage", headers=_bearer(["admin"])
        )
        assert res.json()["limits"] == "unavailable" and len(res.json()["rows"]) == 2

    def test_non_admin_is_refused(self, monkeypatch):
        res = _client(_Pool(), monkeypatch).get("/umrs/admin/usage", headers=_bearer(["user"]))
        assert res.status_code == 403

    def test_no_credential_is_refused(self, monkeypatch):
        res = _client(_Pool(), monkeypatch).get("/umrs/admin/usage")
        assert res.status_code == 401


class TestQuotaUnits:
    HEADERS = {"X-User-ID": "alice", "X-Verified-Project-ID": "proj-a"}

    def test_sessions_and_tokens_poles(self, monkeypatch):
        monkeypatch.setattr(
            "umrs.quotas.enforcer.policy_limits", _limits(user_cap=2000000, sessions=12)
        )
        pool = _Pool(tokens=1234, sessions=3)
        body = _client(pool, monkeypatch).get("/umrs/quota/units", headers=self.HEADERS).json()
        assert body["sessions"] == {"current": 3, "limit": 12}
        assert body["tokens_today"] == {"current": 1234, "limit": 2000000}

    def test_unlimited_is_limit_zero(self, monkeypatch):
        monkeypatch.setattr(
            "umrs.quotas.enforcer.policy_limits", _limits(user_cap=None, sessions=None)
        )
        body = _client(_Pool(), monkeypatch).get("/umrs/quota/units", headers=self.HEADERS).json()
        assert body["sessions"]["limit"] == 0 and body["tokens_today"]["limit"] == 0

    def test_unreadable_policy_leaves_the_poles_null(self, monkeypatch):
        def broken(*_a, **_k):
            raise RuntimeError("no rules")

        monkeypatch.setattr("umrs.quotas.enforcer.policy_limits", broken)
        body = _client(_Pool(), monkeypatch).get("/umrs/quota/units", headers=self.HEADERS).json()
        assert body["sessions"] is None and body["tokens_today"] is None
        assert body["cpu_process"]["current"] == 1  # the ADR-051 poles are untouched


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class TestUsageCli:
    def test_show_sends_the_admin_jwt_and_prints_the_rows(
        self, tmp_path: Path, monkeypatch, capsys
    ):
        seen: list = []
        reply = {
            "venue": "rccs",
            "scope": "user",
            "limits": "ok",
            "rows": [
                {
                    "scope": "user",
                    "subject": "alice",
                    "day": "2026-10-23",
                    "prompt_tokens": 1000,
                    "completion_tokens": 500,
                    "total_tokens": 1500,
                    "requests": 3,
                    "limit": 2000000,
                }
            ],
        }

        def fake_urlopen(req, timeout=None):
            seen.append(req)
            return _Resp(json.dumps(reply).encode())

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
        monkeypatch.setenv("UMRS_ADMIN_JWT", "aaaa.bbbb.cccc")
        monkeypatch.setenv("UMRS_BASE_URL", "http://umrs.test:8000")
        rc = admin_main(
            ["--user-files-dir", str(tmp_path), "usage", "show", "--user", "alice", "--days", "7"]
        )
        assert rc == 0
        assert seen[0].full_url == "http://umrs.test:8000/umrs/admin/usage?user=alice&days=7"
        assert seen[0].get_header("Authorization") == "Bearer aaaa.bbbb.cccc"
        out = capsys.readouterr().out
        assert "alice" in out and "1,500" in out and "2,000,000" in out

    def test_show_refuses_without_a_credential(self, tmp_path: Path, monkeypatch):
        monkeypatch.delenv("UMRS_ADMIN_JWT", raising=False)
        monkeypatch.setattr(urllib.request, "urlopen", lambda *_a, **_k: pytest.fail("sent"))
        with pytest.raises(SystemExit) as exc:
            admin_main(["--user-files-dir", str(tmp_path), "usage", "show"])
        assert exc.value.code == 2
