"""R1-2: lease verdicts + end-to-end proxy flow (fake lease/store/upstream)."""

from __future__ import annotations

import asyncio
import socket
from datetime import datetime, timedelta, timezone

import httpx
from httpx import ASGITransport
from umrs.infer_proxy.app import create_app
from umrs.infer_proxy.leases import check_lease, extract_session_token, token_hash
from umrs.jobs.secret_broker import SecretLease
from umrs.security.seal import generate_recipient_keypair, seal_credential


def _stream_response(
    status: int = 200, body: bytes = b"chunk-1\nchunk-2\n", headers: dict | None = None
):
    """httpx.Response whose AsyncByteStream survives aiter_raw in a MockTransport."""

    async def _yield():
        yield body

    from httpx._content import AsyncIteratorByteStream

    stream = AsyncIteratorByteStream(_yield())
    return httpx.Response(
        status,
        headers=headers or {"content-type": "text/event-stream", "x-usage": "12"},
        stream=stream,
    )


def _lease(**over) -> SecretLease:
    base = {
        "lease_id": "lease-x1",
        "job_id": "job-1",
        "secret_name": "inference-token",
        "scope": "inference",
        "issued_at": datetime.now(timezone.utc) - timedelta(minutes=5),
        "expires_at": datetime.now(timezone.utc) + timedelta(minutes=55),
        "session_id": "sess-1",
        "source_ids": ("src-a", "src-b"),
        "default_source_id": "src-a",
        "egress_ip": "127.0.0.1",
        "secret_hash": token_hash("tok-session"),
    }
    base.update(over)
    return SecretLease(**base)


class TestTokenExtract:
    def test_bearer_wins(self):
        assert extract_session_token({"authorization": "Bearer tok-1"}) == "tok-1"

    def test_api_key_fallbacks(self):
        assert extract_session_token({"x-api-key": "k2"}) == "k2"
        assert extract_session_token({"api-key": "k3"}) == "k3"
        assert extract_session_token({"x-goog-api-key": "k4"}) == "k4"

    def test_none_absent(self):
        assert extract_session_token({"content-type": "application/json"}) is None


class TestLeaseVerdict:
    def test_ok(self):
        verdict = check_lease(_lease(), sid="sess-1", client_ip="127.0.0.1", source_override=None)
        assert verdict.ok and verdict.source_id == "src-a"

    def test_unknown_token(self):
        assert (
            check_lease(None, sid="sess-1", client_ip="127.0.0.1", source_override=None).reason
            == "token_invalid"
        )

    def test_expired(self):
        expired = _lease(expires_at=datetime.now(timezone.utc) - timedelta(minutes=1))
        assert (
            check_lease(expired, sid="sess-1", client_ip="127.0.0.1", source_override=None).reason
            == "lease_expired"
        )

    def test_revoked(self):
        assert (
            check_lease(
                _lease(is_revoked=True), sid="sess-1", client_ip="127.0.0.1", source_override=None
            ).reason
            == "lease_revoked"
        )

    def test_sid_mismatch(self):
        assert (
            check_lease(_lease(), sid="sess-X", client_ip="127.0.0.1", source_override=None).reason
            == "sid_mismatch"
        )

    def test_ip_mismatch(self):
        assert (
            check_lease(_lease(), sid="sess-1", client_ip="10.0.0.1", source_override=None).reason
            == "ip_mismatch"
        )

    def test_source_override_allowed_in_set(self):
        verdict = check_lease(
            _lease(), sid="sess-1", client_ip="127.0.0.1", source_override="src-b"
        )
        assert verdict.ok and verdict.source_id == "src-b"

    def test_source_override_rejected_out_of_set(self):
        verdict = check_lease(
            _lease(), sid="sess-1", client_ip="127.0.0.1", source_override="src-z"
        )
        assert verdict.reason == "source_not_in_lease"


# ── End-to-end proxy flow ───────────────────────────────────────────────


class _FakeStore:
    def __init__(self, fetch):
        self._fetch = fetch

    def fetch_for_unseal(self, _source_id):
        return self._fetch


class _FakeBroker:
    def __init__(self):
        self.ops = []

    def _audit_log(self, _lease_id, operation, _actor, detail):
        self.ops.append((operation, detail))


def _make_fetch(priv_pub, *, provider_kind="openai"):
    pub, _priv = priv_pub
    blob = seal_credential(
        "sk-real-provider-key",
        pubkey_b64=pub,
        owner="alice",
        name="openai-main",
        provider_kind=provider_kind,
        allow_host="https://api.example-provider.com",
    )
    return type(
        "Fetch",
        (),
        {
            "id": "src-a",
            "owner": "alice",
            "name": "openai-main",
            "provider_kind": provider_kind,
            "base_url": "https://api.example-provider.com/v1",
            "allow_host": "https://api.example-provider.com",
            "allow_paths": (),
            "key_fingerprint": "sha256:deadbeef",
            "seal_key_id": blob.seal_key_id,
            "dek_wrapped": blob.dek_wrapped,
            "nonce": blob.nonce,
            "ciphertext": blob.ciphertext,
        },
    )()


def _app(monkeypatch, *, lease: SecretLease | None = None, store=None, broker=None):
    if lease is None:
        lease = _lease()
    priv, pub = generate_recipient_keypair()
    fetch = _make_fetch((pub, priv))
    monkeypatch.setenv("UMRS_UNSEAL_PRIVKEY_B64", priv)
    monkeypatch.setattr("umrs.infer_proxy.app.lease_by_token", lambda _pool, _token: lease)
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda _host, _port, _family, _socktype: [(2, 1, 6, "", ("104.18.2.161", 0))],
    )

    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["headers"] = dict(request.headers)
        seen["body"] = request.read()
        return _stream_response()

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    fake_broker = broker or _FakeBroker()
    app = create_app(
        umrs_db_pool=None,
        unseal_privkey=priv,
        sealed_store=store or _FakeStore(fetch),
        broker=fake_broker,
        http_client=http_client,
    )
    return app, seen, fake_broker


def _sync(coro_factory):
    async def runner():
        async with coro_factory() as resp_holder:
            return resp_holder

    return asyncio.run(runner())


def _request(app, method: str, url: str, **kw):
    async def go():
        transport = ASGITransport(app=app, client=("127.0.0.1", 50000))
        async with httpx.AsyncClient(transport=transport, follow_redirects=False) as client:
            return await client.request(method, f"http://testserver{url}", **kw)

    return asyncio.run(go())


class TestProxyEndToEnd:
    def _call(self, monkeypatch, **kw):
        app, seen, broker = _app(monkeypatch, **kw)
        return app, seen, broker

    def test_injected_credential_reaches_upstream_session_token_absent(self, monkeypatch):
        app, seen, _broker = self._call(monkeypatch)
        resp = _request(
            app,
            "POST",
            "/s/sess-1/v1/chat/completions",
            json={"model": "m", "messages": []},
            headers={
                "authorization": "Bearer tok-session",
                "cookie": "sid=abc",
                "x-api-key": "attacker-tries",
            },
        )
        assert resp.status_code == 200
        upstream_headers = seen["headers"]
        assert upstream_headers.get("authorization") == "Bearer sk-real-provider-key"
        assert "tok-session" not in upstream_headers.values()
        assert "attacker-tries" not in upstream_headers.values()
        assert "cookie" not in upstream_headers
        assert seen["url"] == "https://api.example-provider.com/v1/chat/completions"
        assert resp.text == "chunk-1\nchunk-2\n"
        assert resp.headers["x-usage"] == "12"

    def test_streaming_passthrough_order(self, monkeypatch):
        chunks = []

        def handler(request: httpx.Request) -> httpx.Response:
            return _stream_response(
                body=b"a\nb\nc\n", headers={"content-type": "text/event-stream"}
            )

        monkeypatch.setattr(
            socket, "getaddrinfo", lambda _h, _p, _f, _t: [(2, 1, 6, "", ("8.8.4.4", 0))]
        )
        priv, pub = generate_recipient_keypair()
        fetch = _make_fetch((pub, priv))
        monkeypatch.setenv("UMRS_UNSEAL_PRIVKEY_B64", priv)
        monkeypatch.setattr("umrs.infer_proxy.app.lease_by_token", lambda _p, _t: _lease())
        broker = _FakeBroker()
        app = create_app(
            umrs_db_pool=None,
            unseal_privkey=priv,
            sealed_store=_FakeStore(fetch),
            broker=broker,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )
        resp = _request(
            app,
            "POST",
            "/s/sess-1/v1/chat/completions",
            content=b'{"stream":true}',
            headers={"authorization": "Bearer tok-session", "content-type": "application/json"},
        )
        chunks = resp.text.split("\n")
        assert chunks == ["a", "b", "c", ""]
        assert any(op == "infer" for op, _ in broker.ops)

    def test_path_denied_audits_and_403s(self, monkeypatch):
        app, _seen, broker = self._call(monkeypatch)
        resp = _request(
            app,
            "GET",
            "/s/sess-1/v1/organization/users",
            headers={"authorization": "Bearer tok-session"},
        )
        assert resp.status_code == 403
        assert resp.json()["error"] == "path_denied"
        assert any(op == "deny" and "path_denied" in detail for op, detail in broker.ops)

    def test_missing_token_401(self, monkeypatch):
        app, _seen, _broker = self._call(monkeypatch)
        assert _request(app, "GET", "/s/sess-1/v1/models").status_code == 401

    def test_source_override_in_set_passes_proxy(self, monkeypatch):
        app, _seen, _broker = self._call(monkeypatch)
        resp = _request(
            app,
            "POST",
            "/s/sess-1/v1/chat/completions",
            json={},
            headers={"authorization": "Bearer tok-session", "x-r4-source": "src-b"},
        )
        # src-b is in the lease's source_ids → no source_not_in_lease verdict.
        assert resp.status_code == 200

    def test_source_override_out_of_set_403(self, monkeypatch):
        app, _seen, _broker = self._call(monkeypatch)
        resp = _request(
            app,
            "POST",
            "/s/sess-1/v1/chat/completions",
            json={},
            headers={"authorization": "Bearer tok-session", "x-r4-source": "src-z"},
        )
        assert resp.status_code == 403
        assert resp.json()["error"] == "source_not_in_lease"

    def test_offhost_redirect_502(self, monkeypatch):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(302, headers={"location": "https://evil.example.org/x"})

        monkeypatch.setattr(
            socket, "getaddrinfo", lambda _h, _p, _f, _t: [(2, 1, 6, "", ("8.8.4.4", 0))]
        )
        priv, pub = generate_recipient_keypair()
        fetch = _make_fetch((pub, priv))
        monkeypatch.setenv("UMRS_UNSEAL_PRIVKEY_B64", priv)
        monkeypatch.setattr("umrs.infer_proxy.app.lease_by_token", lambda _p, _t: _lease())
        app = create_app(
            umrs_db_pool=None,
            unseal_privkey=priv,
            sealed_store=_FakeStore(fetch),
            broker=_FakeBroker(),
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )
        resp = _request(
            app, "GET", "/s/sess-1/v1/models", headers={"authorization": "Bearer tok-session"}
        )
        assert resp.status_code == 502
        assert resp.json()["error"] == "redirect_offhost"

    def test_stale_source_410(self, monkeypatch):
        priv, pub = generate_recipient_keypair()
        monkeypatch.setenv("UMRS_UNSEAL_PRIVKEY_B64", priv)
        monkeypatch.setattr("umrs.infer_proxy.app.lease_by_token", lambda _p, _t: _lease())
        app = create_app(
            umrs_db_pool=None,
            unseal_privkey=priv,
            sealed_store=_FakeStore(None),
            broker=_FakeBroker(),
            http_client=httpx.AsyncClient(
                transport=httpx.MockTransport(lambda _request: httpx.Response(200))
            ),
        )
        resp = _request(
            app, "GET", "/s/sess-1/v1/models", headers={"authorization": "Bearer tok-session"}
        )
        assert resp.status_code == 410
