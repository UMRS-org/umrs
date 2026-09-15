"""Tests for session-CURVE server-side wiring.

Covers the glue that was previously missing so that
``UMRS_ZMQ_AUTH_TYPE=session`` actually functions end-to-end:

1. A process-wide :class:`SessionCredentialsProvider` singleton shared by the
   HTTP session API and the ZMQ ZAP authenticators.
2. ``MessageQueue._apply_socket_security`` being a strict no-op when auth is
   disabled (default), and requiring a provider in session mode (server role).
3. ``server_umrs._init_session_curve_provider`` creating/registering the
   provider only when session mode is enabled.
"""

from __future__ import annotations

import types
from unittest.mock import MagicMock

import pytest
from umrs.core.session_curve_provider import (
    SessionCredentialsProvider,
    get_shared_session_provider,
    set_shared_session_provider,
)


@pytest.fixture(autouse=True)
def _clear_shared_provider():
    """Ensure the shared provider singleton is reset around every test."""
    set_shared_session_provider(None)
    yield
    set_shared_session_provider(None)


class TestSharedProviderSingleton:
    def test_roundtrip(self):
        assert get_shared_session_provider() is None
        provider = SessionCredentialsProvider()
        set_shared_session_provider(provider)
        assert get_shared_session_provider() is provider
        set_shared_session_provider(None)
        assert get_shared_session_provider() is None

    def test_zmq_sessions_api_delegates_to_core(self):
        from fastapi import HTTPException
        from umrs.api.zmq_sessions import (
            get_session_curve_provider,
            set_session_curve_provider,
        )

        # Unset → 503 dependency error
        with pytest.raises(HTTPException) as exc:
            get_session_curve_provider()
        assert exc.value.status_code == 503

        provider = SessionCredentialsProvider()
        set_session_curve_provider(provider)
        # API setter writes through to the core singleton
        assert get_shared_session_provider() is provider
        assert get_session_curve_provider() is provider

        set_session_curve_provider(None)
        assert get_shared_session_provider() is None


class TestApplySocketSecurity:
    def _stub(self):
        from umrs.core.MessageQueue import MessageQueue

        stub = types.SimpleNamespace(context=MagicMock(), _zmq_authenticators=[])
        return MessageQueue._apply_socket_security, stub

    def test_noop_when_auth_disabled(self, monkeypatch):
        monkeypatch.delenv("UMRS_ZMQ_AUTH_TYPE", raising=False)
        apply, stub = self._stub()
        sock = MagicMock()
        apply(stub, sock, server=True)
        # No authenticator started, socket left untouched (no curve_server set)
        assert stub._zmq_authenticators == []
        assert not sock.bind.called

    def test_session_server_requires_provider(self, monkeypatch):
        monkeypatch.setenv("UMRS_ZMQ_AUTH_TYPE", "session")
        set_shared_session_provider(None)
        apply, stub = self._stub()
        with pytest.raises(RuntimeError):
            apply(stub, MagicMock(), server=True)


class TestInitSessionCurveProvider:
    def test_noop_when_not_session_mode(self, monkeypatch):
        monkeypatch.delenv("UMRS_ZMQ_AUTH_TYPE", raising=False)
        server_umrs = pytest.importorskip("server_umrs")
        server_umrs._init_session_curve_provider()
        assert get_shared_session_provider() is None

    def test_creates_provider_in_session_mode(self, monkeypatch):
        monkeypatch.setenv("UMRS_ZMQ_AUTH_TYPE", "session")
        server_umrs = pytest.importorskip("server_umrs")
        server_umrs._init_session_curve_provider()
        provider = get_shared_session_provider()
        assert provider is not None
        try:
            provider.stop()
        finally:
            set_shared_session_provider(None)
