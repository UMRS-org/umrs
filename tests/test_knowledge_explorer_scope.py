"""R2 (2026-08-22): knowledge read routes require identity in strict mode.

2026-10-18 (hardening plan H-10.1): the M3.5 explorer routes this file
pinned (``/dashboard``, ``/objects``, ``/objects/{id}``,
``/graph/{id}/neighbors``, ``/correction``) are retired: they served one
module-global graph to every project. Their retirement and the leak they had
are pinned in ``test_h10_module_state.py``; the strict-mode gate stays pinned
here on the per-project routes that replaced them.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from umrs.api.knowledge import router

    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


@pytest.fixture(autouse=True)
def _clear_auth_mode(monkeypatch):
    # Default mode is strict since B12; the "mvp parity" tests below opt in
    # to the legacy permissive mode explicitly.
    monkeypatch.delenv("UMRS_AUTH_MODE", raising=False)
    yield


class TestReadRoutes:
    def test_mvp_anonymous_reaches_the_scope_check(self, client, monkeypatch):
        monkeypatch.setenv("UMRS_AUTH_MODE", "mvp")
        resp = client.get("/umrs/knowledge/readiness")
        assert resp.status_code == 403  # no project: scope_required

    def test_strict_anonymous_rejected(self, client, monkeypatch):
        monkeypatch.setenv("UMRS_AUTH_MODE", "strict")
        assert client.get("/umrs/knowledge/readiness").status_code == 401
        assert client.get("/umrs/knowledge/findings").status_code == 401

    def test_strict_verified_without_signature_rejected(self, client, monkeypatch):
        """Verified headers WITHOUT the gateway HMAC are rejected (jobs parity)."""
        monkeypatch.setenv("UMRS_AUTH_MODE", "strict")
        resp = client.get(
            "/umrs/knowledge/readiness",
            headers={"x-user-id": "u1", "x-verified-project-id": "lab1"},
        )
        assert resp.status_code == 401


class TestWriteRoutes:
    def test_strict_approval_rejected(self, client, monkeypatch):
        monkeypatch.setenv("UMRS_AUTH_MODE", "strict")
        resp = client.post(
            "/umrs/knowledge/approval",
            json={"approved": True, "approved_by": "stalker"},
        )
        assert resp.status_code == 401
