"""Companion trust allowlist (wave T3c, ADR-035): store CRUD + router."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from umrs.api.companion import router, set_trust_store
from umrs.companion_trust.store import DEFAULT_TTL_SEC, TrustAllowlistStore


@pytest.fixture()
def store() -> TrustAllowlistStore:
    return TrustAllowlistStore()


@pytest.fixture()
def api(store: TrustAllowlistStore) -> TestClient:
    set_trust_store(store)
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


class TestTrustStore:
    def test_grant_and_check(self, store: TrustAllowlistStore):
        store.grant("alice", "shell_exec")
        row = store.check("alice", "shell_exec")
        assert row is not None
        assert row.user_id == "alice"
        assert row.scope == "global"

    def test_check_unknown_tool_is_none(self, store: TrustAllowlistStore):
        store.grant("alice", "shell_exec")
        assert store.check("alice", "hpc_submit") is None

    def test_check_unknown_user_is_none(self, store: TrustAllowlistStore):
        store.grant("alice", "shell_exec")
        assert store.check("bob", "shell_exec") is None

    def test_grant_refreshes_ttl(self, store: TrustAllowlistStore):
        store.grant("alice", "shell_exec", ttl_sec=60)
        store.grant("alice", "shell_exec", ttl_sec=3600)
        row = store.check("alice", "shell_exec")
        assert row is not None

    def test_project_scope_isolated(self, store: TrustAllowlistStore):
        store.grant("alice", "shell_exec", scope="project:p1")
        assert store.check("alice", "shell_exec", scope="project:p1") is not None
        assert store.check("alice", "shell_exec", scope="project:p2") is None
        assert store.check("alice", "shell_exec", scope="global") is None

    def test_expired_row_is_garbage_collected(self, store: TrustAllowlistStore):
        # 60s is the floor for grant()... but sqlite lets us write a past row
        # directly; exercise expiry via a 1-second-negative ttl through grant
        # is not possible, so use revokes: grant then manually expire via SQL.
        store.grant("alice", "shell_exec", ttl_sec=60)
        store._conn.execute(
            "UPDATE companion_trust_allowlist SET expires_at = datetime('now', '-1 seconds')"
        )
        store._conn.commit()
        assert store.check("alice", "shell_exec") is None
        # lazy GC deleted the row
        assert store.list("alice") == []

    def test_revoke(self, store: TrustAllowlistStore):
        store.grant("alice", "shell_exec")
        assert store.revoke("alice", "shell_exec") is True
        assert store.check("alice", "shell_exec") is None
        assert store.revoke("alice", "shell_exec") is False

    def test_list_scopes(self, store: TrustAllowlistStore):
        store.grant("alice", "alpha")
        store.grant("alice", "beta", scope="project:p1")
        assert len(store.list("alice")) == 2
        assert len(store.list("alice", scope="global")) == 1
        assert store.list("alice", scope="global")[0].tool_name == "alpha"

    def test_default_ttl_named(self):
        assert DEFAULT_TTL_SEC == 1800


class TestTrustRouter:
    def test_grant_then_check_via_http(self, api: TestClient):
        r = api.put(
            "/umrs/companion/trust",
            json={"tool_name": "shell_exec", "scope": "global", "ttl_sec": 600},
        )
        assert r.status_code == 201, r.text
        c = api.get("/umrs/companion/trust/check", params={"tool_name": "shell_exec"})
        assert c.status_code == 200
        assert c.json()["allowed"] is True

    def test_check_falls_back_to_global_scope(self, api: TestClient):
        api.put("/umrs/companion/trust", json={"tool_name": "shell_exec"})
        c = api.get(
            "/umrs/companion/trust/check",
            params={"tool_name": "shell_exec", "scope": "project:p1"},
        )
        assert c.json()["allowed"] is True

    def test_check_denies_unknown_tool(self, api: TestClient):
        c = api.get("/umrs/companion/trust/check", params={"tool_name": "nope"})
        assert c.status_code == 200
        assert c.json()["allowed"] is False

    def test_list_and_revoke(self, api: TestClient):
        api.put("/umrs/companion/trust", json={"tool_name": "shell_exec"})
        api.put(
            "/umrs/companion/trust",
            json={"tool_name": "hpc_submit", "scope": "project:p1"},
        )
        ls = api.get("/umrs/companion/trust")
        assert ls.status_code == 200
        assert ls.json()["count"] == 2
        d = api.delete("/umrs/companion/trust/shell_exec")
        assert d.status_code == 204
        assert api.get("/umrs/companion/trust/check", params={"tool_name": "shell_exec"}).json()["allowed"] is False

    def test_invalid_scope_rejected(self, api: TestClient):
        r = api.put(
            "/umrs/companion/trust",
            json={"tool_name": "shell_exec", "scope": "weird"},
        )
        assert r.status_code == 422

    def test_ttl_bounds_enforced(self, api: TestClient):
        r = api.put("/umrs/companion/trust", json={"tool_name": "x", "ttl_sec": 10})
        assert r.status_code == 422
