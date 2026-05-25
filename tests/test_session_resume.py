"""Tests for session expiry → resume / re-registration flow.

Validates that:
1. ``CurveSessionManager.ensure_session()`` re-registers when a session is
   nearly expired (< 60 s remaining).
2. ``CurveSessionManager.invalidate()`` forces a fresh registration on
   the next ``ensure_session()`` call.
3. ``UMRSTransport.connect()`` retries with a fresh session when the
   first CURVE handshake fails (server lost the session).
4. ``_apply_session_curve()`` always goes through ``ensure_session()``
   so the TTL guard is enforced.
5. Re-registration after server restart works seamlessly.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

# ---------------------------------------------------------------------------
# CurveSessionManager tests
# ---------------------------------------------------------------------------

from r4.curve_session import CurveSession, CurveSessionManager


def _make_session(*, ttl: float = 3600.0, session_id: str = "s1") -> CurveSession:
    """Create a CurveSession with the given remaining TTL."""
    return CurveSession(
        session_id=session_id,
        server_public_key="S" * 40,
        client_public_key="C" * 40,
        client_secret_key="K" * 40,
        expires_at=time.time() + ttl,
        umrs_base_url="http://localhost:8082",
    )


class TestCurveSessionManagerResume:
    """ensure_session() correctly re-registers when sessions expire."""

    def test_reuse_valid_session(self, tmp_path: Path):
        """A session with plenty of TTL is reused, not re-registered."""
        mgr = CurveSessionManager(
            umrs_base_url="http://localhost:8082",
            config_dir=tmp_path,
        )
        good_session = _make_session(ttl=3600)
        mgr._cached_session = good_session

        result = mgr.ensure_session(jwt_token="fake-jwt")
        assert result.session_id == good_session.session_id

    def test_reregister_when_ttl_under_60s(self, tmp_path: Path):
        """A session with < 60 s remaining triggers re-registration."""
        mgr = CurveSessionManager(
            umrs_base_url="http://localhost:8082",
            config_dir=tmp_path,
        )
        # Session with 30 s remaining — below the 60 s threshold
        almost_expired = _make_session(ttl=30, session_id="old")
        mgr._cached_session = almost_expired

        new_session = _make_session(ttl=3600, session_id="fresh")

        with patch.object(mgr, "_register_new_session", return_value=new_session) as mock_reg:
            result = mgr.ensure_session(jwt_token="fake-jwt")

        mock_reg.assert_called_once_with("fake-jwt")
        assert result.session_id == "fresh"

    def test_reregister_when_fully_expired(self, tmp_path: Path):
        """A fully expired session triggers re-registration."""
        mgr = CurveSessionManager(
            umrs_base_url="http://localhost:8082",
            config_dir=tmp_path,
        )
        expired = _make_session(ttl=-10, session_id="dead")
        mgr._cached_session = expired

        new_session = _make_session(ttl=3600, session_id="alive")
        with patch.object(mgr, "_register_new_session", return_value=new_session) as mock_reg:
            result = mgr.ensure_session(jwt_token="fake-jwt")

        mock_reg.assert_called_once()
        assert result.session_id == "alive"

    def test_invalidate_clears_cache_and_file(self, tmp_path: Path):
        """invalidate() discards cache and deletes the session file."""
        mgr = CurveSessionManager(
            umrs_base_url="http://localhost:8082",
            config_dir=tmp_path,
        )
        session = _make_session(ttl=3600)
        mgr._save_session(session)
        mgr._cached_session = session

        assert mgr._session_file.exists()
        mgr.invalidate()

        assert mgr._cached_session is None
        assert not mgr._session_file.exists()

    def test_ensure_after_invalidate_reregisters(self, tmp_path: Path):
        """After invalidate(), the next ensure_session() creates a new session."""
        mgr = CurveSessionManager(
            umrs_base_url="http://localhost:8082",
            config_dir=tmp_path,
        )
        old_session = _make_session(ttl=3600, session_id="old")
        mgr._cached_session = old_session
        mgr._save_session(old_session)

        mgr.invalidate()

        new_session = _make_session(ttl=3600, session_id="new-after-invalidate")
        with patch.object(mgr, "_register_new_session", return_value=new_session) as mock_reg:
            result = mgr.ensure_session(jwt_token="fake-jwt")

        mock_reg.assert_called_once()
        assert result.session_id == "new-after-invalidate"

    def test_load_session_returns_none_after_invalidate(self, tmp_path: Path):
        """load_session() returns None after invalidate()."""
        mgr = CurveSessionManager(
            umrs_base_url="http://localhost:8082",
            config_dir=tmp_path,
        )
        session = _make_session(ttl=3600)
        mgr._save_session(session)
        mgr._cached_session = session

        mgr.invalidate()
        assert mgr.load_session() is None

    def test_get_transport_keys_returns_none_after_invalidate(self, tmp_path: Path):
        """get_transport_keys() returns None after invalidate()."""
        mgr = CurveSessionManager(
            umrs_base_url="http://localhost:8082",
            config_dir=tmp_path,
        )
        session = _make_session(ttl=3600)
        mgr._save_session(session)
        mgr._cached_session = session

        mgr.invalidate()
        assert mgr.get_transport_keys() is None


# ---------------------------------------------------------------------------
# UMRSTransport session-retry tests
# ---------------------------------------------------------------------------


class TestUMRSTransportSessionRetry:
    """connect() retries with a fresh session when the handshake fails."""

    def _make_transport(self, **overrides):
        """Create a UMRSTransport with session auth, mocking zmq."""
        from r4.umrs_transport import UMRSTransport, UMRSTransportConfig

        config = UMRSTransportConfig(
            server_host="127.0.0.1",
            control_port=5555,
            zmq_auth_type="session",
            **overrides,
        )
        return UMRSTransport(config=config)

    @patch("r4.umrs_transport.ZEROMQ_AVAILABLE", True)
    @patch("r4.umrs_transport.zmq")
    def test_connect_retries_on_stale_session(self, mock_zmq):
        """If the first ping fails, connect() invalidates and retries."""
        transport = self._make_transport()

        # Set up a mock CurveSessionManager
        mock_session_mgr = MagicMock()
        first_session = _make_session(ttl=3600, session_id="stale")
        fresh_session = _make_session(ttl=3600, session_id="fresh")
        mock_session_mgr.ensure_session.side_effect = [first_session, fresh_session]
        transport._session_mgr = mock_session_mgr

        # Mock JWT retrieval
        with patch.object(transport, "_get_jwt_for_session", return_value="fake-jwt"):
            # Mock context / socket
            mock_ctx = MagicMock()
            mock_zmq.Context.return_value = mock_ctx
            mock_socket = MagicMock()
            mock_ctx.socket.return_value = mock_socket
            mock_zmq.DEALER = 5
            mock_zmq.RCVTIMEO = 27
            mock_zmq.SNDTIMEO = 28
            mock_zmq.LINGER = 17

            # First ping fails, second succeeds
            ping_results = [False, True]
            with patch.object(transport, "_ping", side_effect=ping_results):
                result = transport.connect()

        assert result is True
        # invalidate() should have been called once
        mock_session_mgr.invalidate.assert_called_once()
        # ensure_session called twice (first attempt + retry)
        assert mock_session_mgr.ensure_session.call_count == 2

    @patch("r4.umrs_transport.ZEROMQ_AVAILABLE", True)
    @patch("r4.umrs_transport.zmq")
    def test_connect_gives_up_after_retry(self, mock_zmq):
        """If the retry also fails, connect() returns False."""
        transport = self._make_transport()

        mock_session_mgr = MagicMock()
        session = _make_session(ttl=3600)
        mock_session_mgr.ensure_session.return_value = session
        transport._session_mgr = mock_session_mgr

        with patch.object(transport, "_get_jwt_for_session", return_value="fake-jwt"):
            mock_ctx = MagicMock()
            mock_zmq.Context.return_value = mock_ctx
            mock_socket = MagicMock()
            mock_ctx.socket.return_value = mock_socket
            mock_zmq.DEALER = 5
            mock_zmq.RCVTIMEO = 27
            mock_zmq.SNDTIMEO = 28
            mock_zmq.LINGER = 17

            # Both pings fail
            with patch.object(transport, "_ping", return_value=False):
                result = transport.connect()

        assert result is False
        mock_session_mgr.invalidate.assert_called_once()

    @patch("r4.umrs_transport.ZEROMQ_AVAILABLE", True)
    @patch("r4.umrs_transport.zmq")
    def test_connect_no_retry_for_non_session_auth(self, mock_zmq):
        """Non-session auth types do NOT retry on ping failure."""
        from r4.umrs_transport import UMRSTransport, UMRSTransportConfig

        config = UMRSTransportConfig(
            server_host="127.0.0.1",
            control_port=5555,
            zmq_auth_type="none",
        )
        transport = UMRSTransport(config=config)

        mock_ctx = MagicMock()
        mock_zmq.Context.return_value = mock_ctx
        mock_socket = MagicMock()
        mock_ctx.socket.return_value = mock_socket
        mock_zmq.DEALER = 5
        mock_zmq.RCVTIMEO = 27
        mock_zmq.SNDTIMEO = 28
        mock_zmq.LINGER = 17

        with patch.object(transport, "_ping", return_value=False):
            result = transport.connect()

        assert result is False

    @patch("r4.umrs_transport.ZEROMQ_AVAILABLE", True)
    @patch("r4.umrs_transport.zmq")
    def test_apply_session_curve_always_uses_ensure_session(self, mock_zmq):
        """_apply_session_curve() goes through ensure_session(), not get_transport_keys()."""
        transport = self._make_transport()

        mock_session_mgr = MagicMock()
        session = _make_session(ttl=3600)
        mock_session_mgr.ensure_session.return_value = session
        transport._session_mgr = mock_session_mgr

        mock_socket = MagicMock()
        with patch.object(transport, "_get_jwt_for_session", return_value="fake-jwt"):
            transport._apply_session_curve(mock_socket)

        # ensure_session must be called, NOT get_transport_keys
        mock_session_mgr.ensure_session.assert_called_once_with("fake-jwt")
        mock_session_mgr.get_transport_keys.assert_not_called()


# ---------------------------------------------------------------------------
# End-to-end: SessionCredentialsProvider re-registration after sweep
# ---------------------------------------------------------------------------


class TestServerSideReRegistration:
    """After the server sweeps an expired session, a new registration works."""

    def test_sweep_then_reregister(self):
        """Provider accepts a new key after the old one was swept."""
        from umrs.core.session_curve_provider import SessionCredentialsProvider

        provider = SessionCredentialsProvider(max_ttl_seconds=86400)

        old_key = "A" * 40
        with patch("umrs.core.session_curve_provider.time") as mt:
            mt.time.return_value = 1000.0
            provider.register(old_key, user_id="alice", ttl_seconds=60)

        # Advance past expiry and sweep
        with patch("umrs.core.session_curve_provider.time") as mt:
            mt.time.return_value = 2000.0
            provider.cleanup_expired()
            assert provider.callback("*", old_key.encode()) is False

        # New key registered after sweep should be accepted
        new_key = "B" * 40
        provider.register(new_key, user_id="alice", ttl_seconds=3600)
        assert provider.callback("*", new_key.encode()) is True

    def test_reregister_same_user_different_key(self):
        """A user can register a new key after their old one expired."""
        from umrs.core.session_curve_provider import SessionCredentialsProvider

        provider = SessionCredentialsProvider(max_ttl_seconds=86400)

        key1 = "C" * 40
        entry1 = provider.register(key1, user_id="bob", ttl_seconds=3600)

        # Revoke old session (simulates logout or server restart)
        provider.revoke(entry1.session_id)
        assert provider.callback("*", key1.encode()) is False

        # New session with different key
        key2 = "D" * 40
        provider.register(key2, user_id="bob", ttl_seconds=3600)
        assert provider.callback("*", key2.encode()) is True
        # Old key still rejected
        assert provider.callback("*", key1.encode()) is False

    def test_server_restart_loses_sessions(self):
        """After server restart (new provider instance), old keys are rejected
        but new registrations work."""
        from umrs.core.session_curve_provider import SessionCredentialsProvider

        provider1 = SessionCredentialsProvider()
        key = "E" * 40
        provider1.register(key, user_id="carol", ttl_seconds=3600)
        assert provider1.callback("*", key.encode()) is True

        # Simulate server restart: new provider instance, no state
        provider2 = SessionCredentialsProvider()
        assert provider2.callback("*", key.encode()) is False

        # Client re-registers with provider2
        new_key = "F" * 40
        provider2.register(new_key, user_id="carol", ttl_seconds=3600)
        assert provider2.callback("*", new_key.encode()) is True
