"""Tests for SessionCredentialsProvider (ephemeral CURVE session backend)."""

from __future__ import annotations

import threading
import time
from unittest.mock import patch

import pytest
from umrs.core.session_curve_provider import (
    CURVE_KEY_LEN,
    SessionCredentialsProvider,
    SessionCurveEntry,
)


def _fake_z85(prefix: str = "A") -> str:
    """Generate a 40-char fake Z85 key string."""
    return (prefix * CURVE_KEY_LEN)[:CURVE_KEY_LEN]


class TestSessionCurveEntry:
    def test_is_expired_future(self):
        e = SessionCurveEntry(
            pubkey_z85=_fake_z85(),
            session_id="s1",
            user_id="alice",
            created_at=time.time(),
            expires_at=time.time() + 3600,
        )
        assert not e.is_expired
        assert e.ttl_remaining > 3500

    def test_is_expired_past(self):
        e = SessionCurveEntry(
            pubkey_z85=_fake_z85(),
            session_id="s1",
            user_id="alice",
            created_at=time.time() - 7200,
            expires_at=time.time() - 1,
        )
        assert e.is_expired
        assert e.ttl_remaining == 0.0


class TestSessionCredentialsProvider:
    def test_register_and_callback_allow(self):
        p = SessionCredentialsProvider()
        key = _fake_z85("B")
        p.register(key, user_id="alice", ttl_seconds=3600)
        assert p.callback("*", key.encode("ascii")) is True

    def test_callback_reject_unknown_key(self):
        p = SessionCredentialsProvider()
        assert p.callback("*", _fake_z85("C").encode("ascii")) is False

    def test_callback_reject_expired(self):
        p = SessionCredentialsProvider()
        key = _fake_z85("D")
        with patch("umrs.core.session_curve_provider.time") as mock_time:
            mock_time.time.return_value = 1000.0
            p.register(key, user_id="alice", ttl_seconds=60)

        # Advance past expiry
        with patch("umrs.core.session_curve_provider.time") as mock_time:
            mock_time.time.return_value = 2000.0
            assert p.callback("*", key.encode("ascii")) is False

    def test_register_invalid_key_length(self):
        p = SessionCredentialsProvider()
        with pytest.raises(ValueError, match="40 Z85 characters"):
            p.register("short", user_id="alice", ttl_seconds=3600)

    def test_register_negative_ttl(self):
        p = SessionCredentialsProvider()
        with pytest.raises(ValueError, match="positive"):
            p.register(_fake_z85(), user_id="alice", ttl_seconds=-1)

    def test_ttl_clamped_to_max(self):
        p = SessionCredentialsProvider(max_ttl_seconds=300)
        key = _fake_z85("E")
        entry = p.register(key, user_id="alice", ttl_seconds=9999)
        assert entry.ttl_remaining <= 301  # allows small timing drift

    def test_revoke_session(self):
        p = SessionCredentialsProvider()
        key = _fake_z85("F")
        entry = p.register(key, user_id="alice", ttl_seconds=3600)
        assert p.revoke(entry.session_id) is True
        assert p.callback("*", key.encode("ascii")) is False

    def test_revoke_nonexistent(self):
        p = SessionCredentialsProvider()
        assert p.revoke("no-such-session") is False

    def test_revoke_user(self):
        p = SessionCredentialsProvider()
        k1 = _fake_z85("G")
        k2 = _fake_z85("H")
        p.register(k1, user_id="bob", ttl_seconds=3600)
        p.register(k2, user_id="bob", ttl_seconds=3600)
        removed = p.revoke_user("bob")
        assert removed == 2
        assert p.callback("*", k1.encode("ascii")) is False
        assert p.callback("*", k2.encode("ascii")) is False

    def test_user_id_for_key(self):
        p = SessionCredentialsProvider()
        key = _fake_z85("I")
        p.register(key, user_id="carol", ttl_seconds=3600)
        assert p.user_id_for_key(key) == "carol"
        assert p.user_id_for_key(_fake_z85("J")) is None

    def test_max_per_user_evicts_oldest(self):
        p = SessionCredentialsProvider(max_per_user=2)
        k1 = _fake_z85("K")
        k2 = _fake_z85("L")
        k3 = _fake_z85("M")
        p.register(k1, user_id="dave", ttl_seconds=3600)
        p.register(k2, user_id="dave", ttl_seconds=3600)
        # k1 should be evicted when k3 is added
        p.register(k3, user_id="dave", ttl_seconds=3600)
        assert p.callback("*", k1.encode("ascii")) is False
        assert p.callback("*", k2.encode("ascii")) is True
        assert p.callback("*", k3.encode("ascii")) is True

    def test_refresh_extends_ttl(self):
        p = SessionCredentialsProvider(max_ttl_seconds=86400)
        key = _fake_z85("N")
        entry = p.register(key, user_id="eve", ttl_seconds=60)
        updated = p.refresh(entry.session_id, ttl_seconds=3600)
        assert updated is not None
        assert updated.ttl_remaining > 3500

    def test_refresh_expired_returns_none(self):
        p = SessionCredentialsProvider()
        key = _fake_z85("O")
        with patch("umrs.core.session_curve_provider.time") as mock_time:
            mock_time.time.return_value = 1000.0
            entry = p.register(key, user_id="frank", ttl_seconds=60)

        with patch("umrs.core.session_curve_provider.time") as mock_time:
            mock_time.time.return_value = 2000.0
            assert p.refresh(entry.session_id, ttl_seconds=3600) is None

    def test_get_session(self):
        p = SessionCredentialsProvider()
        key = _fake_z85("P")
        entry = p.register(key, user_id="grace", ttl_seconds=3600)
        found = p.get_session(entry.session_id)
        assert found is not None
        assert found.user_id == "grace"

    def test_get_session_not_found(self):
        p = SessionCredentialsProvider()
        assert p.get_session("missing") is None

    def test_active_session_count(self):
        p = SessionCredentialsProvider()
        p.register(_fake_z85("Q"), user_id="heidi", ttl_seconds=3600)
        p.register(_fake_z85("R"), user_id="heidi", ttl_seconds=3600)
        p.register(_fake_z85("S"), user_id="ivan", ttl_seconds=3600)
        assert p.active_session_count() == 3
        assert p.active_session_count(user_id="heidi") == 2
        assert p.active_session_count(user_id="ivan") == 1

    def test_list_sessions_sorted(self):
        p = SessionCredentialsProvider()
        p.register(_fake_z85("T"), user_id="judy", ttl_seconds=3600)
        p.register(_fake_z85("U"), user_id="judy", ttl_seconds=3600)
        sessions = p.list_sessions(user_id="judy")
        assert len(sessions) == 2
        assert sessions[0].created_at <= sessions[1].created_at

    def test_cleanup_expired(self):
        p = SessionCredentialsProvider()
        key = _fake_z85("V")
        with patch("umrs.core.session_curve_provider.time") as mock_time:
            mock_time.time.return_value = 1000.0
            p.register(key, user_id="karl", ttl_seconds=60)

        with patch("umrs.core.session_curve_provider.time") as mock_time:
            mock_time.time.return_value = 2000.0
            removed = p.cleanup_expired()
            assert removed == 1
            assert p.active_session_count() == 0

    def test_start_stop(self):
        p = SessionCredentialsProvider(sweep_interval_seconds=0.1)
        p.start()
        assert p._sweep_thread is not None
        assert p._sweep_thread.is_alive()
        p.stop()
        assert not p._sweep_thread.is_alive() if p._sweep_thread else True

    def test_thread_safety(self):
        """Concurrent register/revoke should not raise."""
        p = SessionCredentialsProvider(max_per_user=100)
        errors: list[Exception] = []

        def worker(n: int):
            try:
                for i in range(20):
                    key = _fake_z85(chr(65 + (n * 20 + i) % 26))
                    # Make keys unique
                    k = f"{key[:38]}{n:01d}{i:01d}"
                    if len(k) != 40:
                        k = k.ljust(40, "A")[:40]
                    entry = p.register(k, user_id=f"user{n}", ttl_seconds=3600)
                    p.callback("*", k.encode("ascii"))
                    if i % 3 == 0:
                        p.revoke(entry.session_id)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert errors == [], f"Thread safety errors: {errors}"

    def test_re_register_replaces_old(self):
        p = SessionCredentialsProvider()
        key = _fake_z85("W")
        e1 = p.register(key, user_id="mallory", ttl_seconds=3600)
        e2 = p.register(key, user_id="mallory", ttl_seconds=3600)
        assert e1.session_id != e2.session_id
        assert p.active_session_count(user_id="mallory") == 1
