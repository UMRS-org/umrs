"""Regression tests for CURVE session bootstrap and client-side refresh.

Covers the stale-session self-heal fix:
- ``bootstrap_curve_session`` must NOT abort when the unauthenticated/redirected
  ``GET /umrs/zmq/server-key`` fails behind an auth gateway; the authoritative
  server public key comes from the authenticated ``POST /umrs/zmq/session``.
- The client ``_force_refresh_curve_session`` / ``_persist_curve_session``
  helpers persist a fresh session with restrictive permissions.
"""

from __future__ import annotations

import json
import os
import stat
from unittest import mock

import pytest


# ---------------------------------------------------------------------------
# bootstrap_curve_session gateway robustness
# ---------------------------------------------------------------------------


class _FakeResp:
    def __init__(self, status_code=200, json_data=None, raise_on_json=False):
        self.status_code = status_code
        self._json_data = json_data or {}
        self._raise_on_json = raise_on_json

    def json(self):
        if self._raise_on_json:
            raise ValueError("not JSON (login page)")
        return self._json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests

            raise requests.HTTPError(f"status {self.status_code}")


_POST_OK = {
    "session_id": "sess-123",
    "server_public_key": "SERVER_PK_FROM_POST",
    "expires_at": 9_999_999_999.0,
    "ttl_seconds": 3600,
    "grants": {"roles": ["admin"]},
}


def _run_bootstrap(get_resp):
    """Invoke bootstrap_curve_session with a mocked GET response + OK POST."""
    from umrs.client import session_bootstrap

    # Avoid mutating the real process environment across tests.
    saved = {
        k: os.environ.get(k)
        for k in (
            "UMRS_ZMQ_AUTH_TYPE",
            "UMRS_ZMQ_CURVE_SERVER_PUBLIC_KEY",
            "UMRS_ZMQ_CURVE_CLIENT_PUBLIC_KEY",
            "UMRS_ZMQ_CURVE_CLIENT_SECRET_KEY",
        )
    }
    try:
        with (
            mock.patch("requests.get", return_value=get_resp),
            mock.patch("requests.post", return_value=_FakeResp(200, _POST_OK)),
        ):
            return session_bootstrap.bootstrap_curve_session(
                gateway_url="https://gw.example/rccs",
                jwt_token="jwt-token",
                set_env=True,
            )
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_bootstrap_tolerates_server_key_redirect_404():
    """A redirected/404 server-key GET must not abort the bootstrap."""
    info = _run_bootstrap(_FakeResp(status_code=404))
    assert info.session_id == "sess-123"
    assert info.server_public_key == "SERVER_PK_FROM_POST"


def test_bootstrap_tolerates_non_json_server_key():
    """A 200 server-key response that is not JSON (login page) is ignored."""
    info = _run_bootstrap(_FakeResp(status_code=200, raise_on_json=True))
    assert info.server_public_key == "SERVER_PK_FROM_POST"


def test_bootstrap_uses_server_key_when_available():
    """When server-key GET returns JSON, the POST value still wins (authoritative)."""
    info = _run_bootstrap(_FakeResp(status_code=200, json_data={"server_public_key": "FROM_GET"}))
    # POST response carries the authoritative key.
    assert info.server_public_key == "SERVER_PK_FROM_POST"


def test_bootstrap_get_network_error_is_non_fatal():
    """A network error on the server-key GET must not abort the bootstrap."""
    from umrs.client import session_bootstrap

    import requests

    with (
        mock.patch("requests.get", side_effect=requests.ConnectionError("boom")),
        mock.patch("requests.post", return_value=_FakeResp(200, _POST_OK)),
    ):
        info = session_bootstrap.bootstrap_curve_session(
            gateway_url="https://gw.example/rccs",
            jwt_token="jwt-token",
            set_env=False,
        )
    assert info.session_id == "sess-123"
    assert info.server_public_key == "SERVER_PK_FROM_POST"


# ---------------------------------------------------------------------------
# client-side refresh helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def _client_mod():
    import importlib
    import sys

    # client_umrs installed editable alongside umrs (ADR-053).
    return importlib.import_module("client_umrs")


class _Conn:
    def __init__(self, gateway_url=None, zmq_auth_type=None):
        self.gateway_url = gateway_url
        self.zmq_auth_type = zmq_auth_type
        self.profile_path = None
        self.profile_name = None


def test_is_session_mode_requires_gateway_and_auth(_client_mod, monkeypatch):
    monkeypatch.delenv("UMRS_ZMQ_AUTH_TYPE", raising=False)
    assert _client_mod._is_session_mode(_Conn(gateway_url=None, zmq_auth_type="session")) is False
    assert _client_mod._is_session_mode(_Conn(gateway_url="x", zmq_auth_type="none")) is False
    assert _client_mod._is_session_mode(_Conn(gateway_url="x", zmq_auth_type="session")) is True
    # env var supplies auth type when profile field is empty
    monkeypatch.setenv("UMRS_ZMQ_AUTH_TYPE", "session")
    assert _client_mod._is_session_mode(_Conn(gateway_url="x", zmq_auth_type=None)) is True


def test_persist_curve_session_writes_0600(_client_mod, tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))

    class _Info:
        session_id = "sess-xyz"
        server_public_key = "SPK"
        client_public_key = "CPK"
        client_secret_key = "CSK"
        expires_at = 9_999_999_999.0

    _client_mod._persist_curve_session(_Info(), "https://gw.example/rccs/")

    session_file = tmp_path / "r4" / "curve-session.json"
    assert session_file.is_file()
    data = json.loads(session_file.read_text())
    assert data["session_id"] == "sess-xyz"
    assert data["server_public_key"] == "SPK"
    assert data["client_public_key"] == "CPK"
    assert data["client_secret_key"] == "CSK"
    assert data["umrs_base_url"] == "https://gw.example/rccs"
    mode = stat.S_IMODE(os.stat(session_file).st_mode)
    assert mode == 0o600


def test_force_refresh_noop_without_session_mode(_client_mod):
    # No gateway → not session mode → no-op False.
    assert _client_mod._force_refresh_curve_session(_Conn(gateway_url=None)) is False


def test_force_refresh_bootstraps_and_persists(_client_mod, tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setenv("UMRS_ZMQ_AUTH_TYPE", "session")
    monkeypatch.setenv("UMRS_API_KEY", "jwt-token")

    class _Info:
        session_id = "fresh-1"
        server_public_key = "SPK"
        client_public_key = "CPK"
        client_secret_key = "CSK"
        expires_at = 9_999_999_999.0

    with mock.patch(
        "umrs.client.session_bootstrap.bootstrap_curve_session",
        return_value=_Info(),
    ) as m:
        ok = _client_mod._force_refresh_curve_session(
            _Conn(gateway_url="https://gw.example/rccs", zmq_auth_type="session")
        )

    assert ok is True
    m.assert_called_once()
    session_file = tmp_path / "r4" / "curve-session.json"
    assert json.loads(session_file.read_text())["session_id"] == "fresh-1"
