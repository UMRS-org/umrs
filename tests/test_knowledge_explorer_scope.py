"""R2 (2026-08-22): M3.5 Explorer routes require identity in strict mode.

Module-global explorer controllers can't be tier-scoped per project — the
conver gate is strict-mode authentication (401 unauthenticated) with mvp
parity (anonymous callers allowed, matching knowledge graph routes).
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
    def test_mvp_anonymous_allowed(self, client, monkeypatch):
        monkeypatch.setenv("UMRS_AUTH_MODE", "mvp")
        resp = client.get("/umrs/knowledge/objects")
        assert resp.status_code in (200, 404, 500)

    def test_mvp_verified_allowed(self, client, monkeypatch):
        monkeypatch.setenv("UMRS_AUTH_MODE", "mvp")
        resp = client.get(
            "/umrs/knowledge/dashboard",
            headers={"x-user-id": "u1", "x-verified-project-id": "lab1"},
        )
        assert resp.status_code in (200, 404, 500)

    def test_strict_anonymous_rejected(self, client, monkeypatch):
        monkeypatch.setenv("UMRS_AUTH_MODE", "strict")
        assert client.get("/umrs/knowledge/dashboard").status_code == 401
        assert client.get("/umrs/knowledge/objects").status_code == 401
        assert client.get("/umrs/knowledge/graph/abc/neighbors").status_code == 401
        assert client.get("/umrs/knowledge/readiness").status_code == 401

    def test_strict_verified_without_signature_rejected(self, client, monkeypatch):
        """Verified headers WITHOUT the gateway HMAC are rejected (jobs parity)."""
        monkeypatch.setenv("UMRS_AUTH_MODE", "strict")
        resp = client.get(
            "/umrs/knowledge/objects",
            headers={"x-user-id": "u1", "x-verified-project-id": "lab1"},
        )
        assert resp.status_code == 401


class TestWriteRoutes:
    def test_strict_correction_rejected(self, client, monkeypatch):
        monkeypatch.setenv("UMRS_AUTH_MODE", "strict")
        resp = client.post(
            "/umrs/knowledge/correction",
            json={
                "correction_type": "fact",
                "corrected_by": "stalker",
                "reason": "test",
                "affected_object_ids": [],
                "details": {},
            },
        )
        assert resp.status_code == 401

    def test_strict_approval_rejected(self, client, monkeypatch):
        monkeypatch.setenv("UMRS_AUTH_MODE", "strict")
        resp = client.post(
            "/umrs/knowledge/approval",
            json={"approved": True, "approved_by": "stalker"},
        )
        assert resp.status_code == 401
