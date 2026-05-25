"""Tests for ZMQ session CURVE HTTP API endpoints."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from umrs.api.zmq_sessions import (
    router,
    set_session_curve_provider,
    _require_user_id,
)
from umrs.core.session_curve_provider import SessionCredentialsProvider, SessionCurveEntry

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_app():
    """Create a minimal FastAPI app with the ZMQ session router."""
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(router)
    return app


def _fake_z85(prefix: str = "A") -> str:
    return (prefix * 40)[:40]


@pytest.fixture()
def provider():
    p = SessionCredentialsProvider(max_ttl_seconds=86400, max_per_user=5)
    set_session_curve_provider(p)
    yield p
    set_session_curve_provider(None)


@pytest.fixture()
def client(provider):
    app = _make_app()
    return TestClient(app)


def _auth_header(user_id: str = "alice") -> dict[str, str]:
    """Create a mock Authorization header.

    We patch ``_require_user_id`` via dependency override.
    """
    return {"Authorization": f"Bearer fake-jwt-for-{user_id}"}


def _override_user(user_id: str):
    """Create a FastAPI dependency override for _require_user_id."""
    async def _mock():
        return user_id
    return _mock


@pytest.fixture()
def alice_client(provider):
    app = _make_app()
    app.dependency_overrides[_require_user_id] = _override_user("alice")
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture()
def bob_client(provider):
    app = _make_app()
    app.dependency_overrides[_require_user_id] = _override_user("bob")
    yield TestClient(app)
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Tests: GET /umrs/zmq/server-key
# ---------------------------------------------------------------------------

class TestServerKey:
    def test_returns_key_when_configured(self, client):
        mock_cfg = MagicMock()
        mock_cfg.enabled = True
        mock_cfg.server_public_key = _fake_z85("S")
        with patch("umrs.core.zmq_security.ZMQSecurityConfig.from_env", return_value=mock_cfg):
            resp = client.get("/umrs/zmq/server-key")
        assert resp.status_code == 200
        assert resp.json()["server_public_key"] == _fake_z85("S")

    def test_returns_404_when_disabled(self, client):
        mock_cfg = MagicMock()
        mock_cfg.enabled = False
        with patch("umrs.core.zmq_security.ZMQSecurityConfig.from_env", return_value=mock_cfg):
            resp = client.get("/umrs/zmq/server-key")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Tests: POST /umrs/zmq/session
# ---------------------------------------------------------------------------

class TestRegisterSession:
    def test_register_success(self, alice_client, provider):
        mock_cfg = MagicMock()
        mock_cfg.server_public_key = _fake_z85("S")
        with patch("umrs.core.zmq_security.ZMQSecurityConfig.from_env", return_value=mock_cfg):
            resp = alice_client.post(
                "/umrs/zmq/session",
                json={"client_public_key": _fake_z85("C"), "ttl_seconds": 3600},
            )
        assert resp.status_code == 201
        data = resp.json()
        assert "session_id" in data
        assert data["server_public_key"] == _fake_z85("S")
        assert data["expires_at"] > time.time()
        # Verify the key is registered in the provider
        assert provider.callback("*", _fake_z85("C").encode("ascii")) is True

    def test_register_invalid_key_length(self, alice_client):
        resp = alice_client.post(
            "/umrs/zmq/session",
            json={"client_public_key": "short", "ttl_seconds": 3600},
        )
        assert resp.status_code == 422  # validation error

    def test_register_ttl_too_low(self, alice_client):
        resp = alice_client.post(
            "/umrs/zmq/session",
            json={"client_public_key": _fake_z85(), "ttl_seconds": 10},
        )
        assert resp.status_code == 422

    def test_register_ttl_too_high(self, alice_client):
        resp = alice_client.post(
            "/umrs/zmq/session",
            json={"client_public_key": _fake_z85(), "ttl_seconds": 999999},
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Tests: DELETE /umrs/zmq/session/{session_id}
# ---------------------------------------------------------------------------

class TestRevokeSession:
    def test_revoke_own_session(self, alice_client, provider):
        key = _fake_z85("D")
        entry = provider.register(key, user_id="alice", ttl_seconds=3600)
        resp = alice_client.delete(f"/umrs/zmq/session/{entry.session_id}")
        assert resp.status_code == 204
        assert provider.callback("*", key.encode("ascii")) is False

    def test_revoke_nonexistent(self, alice_client):
        resp = alice_client.delete("/umrs/zmq/session/no-such-id")
        assert resp.status_code == 404

    def test_revoke_other_users_session(self, bob_client, provider):
        key = _fake_z85("E")
        entry = provider.register(key, user_id="alice", ttl_seconds=3600)
        resp = bob_client.delete(f"/umrs/zmq/session/{entry.session_id}")
        assert resp.status_code == 403
        # Session should still be valid
        assert provider.callback("*", key.encode("ascii")) is True


# ---------------------------------------------------------------------------
# Tests: POST /umrs/zmq/session/{session_id}/refresh
# ---------------------------------------------------------------------------

class TestRefreshSession:
    def test_refresh_success(self, alice_client, provider):
        key = _fake_z85("F")
        entry = provider.register(key, user_id="alice", ttl_seconds=60)
        resp = alice_client.post(
            f"/umrs/zmq/session/{entry.session_id}/refresh",
            json={"ttl_seconds": 3600},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ttl_remaining"] > 3500

    def test_refresh_other_users_session(self, bob_client, provider):
        key = _fake_z85("G")
        entry = provider.register(key, user_id="alice", ttl_seconds=3600)
        resp = bob_client.post(
            f"/umrs/zmq/session/{entry.session_id}/refresh",
            json={"ttl_seconds": 3600},
        )
        assert resp.status_code == 403


# ---------------------------------------------------------------------------
# Tests: GET /umrs/zmq/session
# ---------------------------------------------------------------------------

class TestListSessions:
    def test_list_own_sessions(self, alice_client, provider):
        provider.register(_fake_z85("H"), user_id="alice", ttl_seconds=3600)
        provider.register(_fake_z85("I"), user_id="alice", ttl_seconds=3600)
        provider.register(_fake_z85("J"), user_id="bob", ttl_seconds=3600)
        resp = alice_client.get("/umrs/zmq/session")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2
        assert all(s["user_id"] == "alice" for s in data)


# ---------------------------------------------------------------------------
# Tests: 503 when provider is not set
# ---------------------------------------------------------------------------

class TestProviderUnavailable:
    def test_register_503_when_no_provider(self):
        set_session_curve_provider(None)
        app = _make_app()
        app.dependency_overrides[_require_user_id] = _override_user("alice")
        c = TestClient(app)
        resp = c.post(
            "/umrs/zmq/session",
            json={"client_public_key": _fake_z85(), "ttl_seconds": 3600},
        )
        assert resp.status_code == 503
        app.dependency_overrides.clear()
