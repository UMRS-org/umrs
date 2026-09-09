"""R2-3: per-session rate limiter (sliding window, opt-in)."""

from __future__ import annotations

import httpx
import pytest
from httpx import ASGITransport

from umrs.infer_proxy.app import SessionRateLimiter, create_app
from umrs.security.seal import generate_recipient_keypair, seal_credential

from tests.test_infer_proxy import (
    _FakeBroker,
    _FakeStore,
    _lease,
    _make_fetch,
    _request,
    _stream_response,
)


class TestSessionRateLimiter:
    def test_off_by_default(self):
        limiter = SessionRateLimiter(0)
        for _ in range(500):
            assert limiter.allow("sess-x")

    def test_window_enforces_cap(self, monkeypatch):
        now = [100.0]
        monkeypatch.setattr("umrs.infer_proxy.app.time.monotonic", lambda: now[0])
        limiter = SessionRateLimiter(rpm=3)
        assert limiter.allow("sess-x")
        assert limiter.allow("sess-x")
        assert limiter.allow("sess-x")
        assert not limiter.allow("sess-x")
        # Roll out of window.
        now[0] = 161.0
        assert limiter.allow("sess-x")

    def test_sessions_isolated(self):
        limiter = SessionRateLimiter(rpm=1)
        assert limiter.allow("sess-a")
        assert not limiter.allow("sess-a")
        assert limiter.allow("sess-b")


class TestRateGateEnd2End:
    def test_429_rate_denied_with_audit(self, monkeypatch):
        import socket

        monkeypatch.setattr(
            socket,
            "getaddrinfo",
            lambda *_args, **_kw: [(2, 1, 6, "", ("104.18.2.161", 0))],
        )
        priv, pub = generate_recipient_keypair()
        fetch = _make_fetch((pub, priv))
        broker = _FakeBroker()
        monkeypatch.setattr("umrs.infer_proxy.app.lease_by_token", lambda _p, _t: _lease())
        app = create_app(
            umrs_db_pool=None,
            unseal_privkey=priv,
            sealed_store=_FakeStore(fetch),
            broker=broker,
            http_client=httpx.AsyncClient(
                transport=httpx.MockTransport(
                    lambda _r: _stream_response(body=b"ok", headers={"content-type": "text/plain"})
                )
            ),
        )
        app.state.rate_limiter = SessionRateLimiter(rpm=2)

        ok1 = _request(
            app, "GET", "/s/sess-1/v1/models", headers={"authorization": "Bearer tok-session"}
        )
        ok2 = _request(
            app, "GET", "/s/sess-1/v1/models", headers={"authorization": "Bearer tok-session"}
        )
        third = _request(
            app, "GET", "/s/sess-1/v1/models", headers={"authorization": "Bearer tok-session"}
        )
        assert ok1.status_code == 200
        assert ok2.status_code == 200
        assert third.status_code == 429
        assert third.json()["error"] == "rate_denied"
        assert any(op == "deny" and detail == "rate_denied" for op, detail in broker.ops)
