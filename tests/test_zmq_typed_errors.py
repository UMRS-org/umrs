"""Typed error taxonomy parity (G-P12-5, WP-16).

The zmq_sessions family is the first taxonomy consumer: every error it
surfaces carries the code on ``x-platform-error-code`` while the detail
body stays byte-identical to the legacy stringly form. These pins assert
both halves — the header is typed, the body is unchanged.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from r4_contracts.errors import ERROR_CODE_HEADER

from umrs.api.zmq_sessions import router, set_session_curve_provider
from umrs.core.session_curve_provider import SessionCredentialsProvider


def _app() -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture()
def provider():
    p = SessionCredentialsProvider(max_ttl_seconds=86400, max_per_user=5)
    set_session_curve_provider(p)
    yield p
    set_session_curve_provider(None)


@pytest.fixture()
def client(provider):
    return TestClient(_app())


def _override_user(user_id: str):
    from umrs.api.zmq_sessions import _require_user_id

    async def _mock():
        return user_id

    return _mock


def _override_identity(user_id: str):
    from umrs.api.zmq_sessions import _JWTIdentity

    async def _mock():
        return _JWTIdentity(
            user_id=user_id,
            email=f"{user_id}@test.local",
            projects=["test-project"],
            roles=["user"],
        )

    return _mock


class TestHeaderParity:
    def test_missing_auth_header_is_typed(self):
        # No dependency overrides: the raw _extract path runs.
        resp = TestClient(_app()).get("/umrs/zmq/session")
        assert resp.headers.get(ERROR_CODE_HEADER) == "auth.missing_header"
        assert resp.json()["detail"] == "Missing Authorization header"

    def test_bad_scheme_is_typed(self):
        resp = TestClient(_app()).get(
            "/umrs/zmq/session", headers={"Authorization": "token abc"}
        )
        assert resp.headers.get(ERROR_CODE_HEADER) == "auth.bad_scheme"
        assert resp.json()["detail"] == "Authorization header must be 'Bearer <token>'"

    def test_revoke_unknown_session_is_typed_not_found(self, client, provider):
        app = client.app
        from umrs.api.zmq_sessions import _require_user_id

        app.dependency_overrides[_require_user_id] = _override_user("alice")
        resp = client.delete("/umrs/zmq/session/s-404")
        assert resp.status_code == 404
        assert resp.headers.get(ERROR_CODE_HEADER) == "session.not_found"
        assert resp.json()["detail"] == "Session s-404 not found"
        app.dependency_overrides.clear()

    def test_revoke_other_users_session_is_typed_forbidden(self, client, provider):
        app = client.app
        from umrs.api.zmq_sessions import _extract_jwt_identity, _require_user_id

        # Owner registers, stranger revokes.
        app.dependency_overrides[_extract_jwt_identity] = _override_identity("bob")
        app.dependency_overrides[_require_user_id] = _override_user("bob")
        reg = client.post(
            "/umrs/zmq/session",
            json={
                "client_public_key": "A" * 40,
                "ttl_seconds": 300,
            },
        )
        assert reg.status_code == 201, reg.text
        session_id = reg.json()["session_id"]
        app.dependency_overrides[_require_user_id] = _override_user("alice")
        resp = client.delete(f"/umrs/zmq/session/{session_id}")
        assert resp.status_code == 403
        assert resp.headers.get(ERROR_CODE_HEADER) == "session.forbidden"
        assert resp.json()["detail"] == "Cannot revoke another user's session"
        app.dependency_overrides.clear()

    def test_refresh_unknown_session_is_typed(self, client):
        app = client.app
        from umrs.api.zmq_sessions import _require_user_id

        app.dependency_overrides[_require_user_id] = _override_user("alice")
        resp = client.post(
            "/umrs/zmq/session/s-9/refresh", json={"ttl_seconds": 600}
        )
        assert resp.status_code == 404
        assert resp.headers.get(ERROR_CODE_HEADER) == "session.not_found"
        assert resp.json()["detail"] == "Session s-9 not found"
        app.dependency_overrides.clear()


class TestProviderUnavailable:
    def test_register_503_carries_typed_header(self):
        """Dependency-direct probe: shared provider unset -> typed 503."""
        from fastapi import HTTPException

        from umrs.api.zmq_sessions import get_session_curve_provider

        set_session_curve_provider(None)
        with pytest.raises(HTTPException) as caught:
            get_session_curve_provider()
        exc = caught.value
        assert exc.status_code == 503
        assert exc.headers.get(ERROR_CODE_HEADER) == "provider.unavailable"
        assert "provider not available" in exc.detail


# ---------------------------------------------------------------------------
# WP-16 round 2 — research_sessions family becomes typed (additive header)
# ---------------------------------------------------------------------------


def test_contract_gains_research_session_codes():
    from r4_contracts.errors import known_error_codes

    codes = known_error_codes()
    assert codes["session.store_unavailable"] == 503
    assert codes["session.archive_cascade_failed"] == 500
    assert codes["session.not_found"] == 404


def test_research_session_404_is_typed_and_byte_identical():
    import asyncio
    from unittest.mock import MagicMock

    import pytest

    import umrs.api.research_sessions as rs

    store = MagicMock()
    store.get.return_value = None
    prev_store = rs._store
    rs._store = store
    try:
        with pytest.raises(Exception) as excinfo:
            asyncio.run(rs.get_session("nope-1", caller=None))
    finally:
        rs._store = prev_store

    exc = excinfo.value
    assert exc.status_code == 404
    assert exc.detail == "Session nope-1 not found"  # byte-identical, no drift
    headers = getattr(exc, "headers", {}) or {}
    assert headers.get("x-platform-error-code") == "session.not_found"
