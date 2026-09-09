"""R2-2: TOCTOU IP pin — one resolution per request, pinned IP + original Host/SNI."""

from __future__ import annotations

import socket

import httpx
from tests.test_infer_proxy import (
    _FakeBroker,
    _FakeStore,
    _lease,
    _make_fetch,
    _request,
    _stream_response,
)
from umrs.infer_proxy.app import create_app, resolve_checked_ips
from umrs.security.seal import generate_recipient_keypair


class TestResolveCheckedIps:
    def test_literal_public_ip_pinned(self):
        assert resolve_checked_ips("104.18.2.161") == ["104.18.2.161"]

    def test_literal_private_ip_rejected(self):
        assert resolve_checked_ips("10.0.0.5") == []

    def test_dns_sorted_and_infra_filtered(self, monkeypatch):
        monkeypatch.setattr(
            socket,
            "getaddrinfo",
            lambda *_args, **_kw: [
                (2, 1, 6, "", ("104.18.3.161", 0)),
                (2, 1, 6, "", ("104.18.2.161", 0)),
            ],
        )
        assert resolve_checked_ips("api.example.org") == ["104.18.2.161", "104.18.3.161"]

    def test_dns_mixed_set_rejected(self, monkeypatch):
        monkeypatch.setattr(
            socket,
            "getaddrinfo",
            lambda *_args, **_kw: [
                (2, 1, 6, "", ("104.18.3.161", 0)),
                (2, 1, 6, "", ("10.0.0.9", 0)),
            ],
        )
        assert resolve_checked_ips("api.example.org") == []

    def test_resolve_failure_rejected(self, monkeypatch):
        monkeypatch.setattr(
            socket, "getaddrinfo", lambda *_args, **_kw: (_ for _ in ()).throw(socket.gaierror())
        )
        assert resolve_checked_ips("api.example.org") == []


class TestToctouPin:
    def test_single_resolution_per_request_and_pinned_ip(self, monkeypatch):
        """One getaddrinfo call per request; the dial uses only that answer."""
        calls: list[str] = []
        pairs = [("104.18.3.161", 0), ("104.18.2.161", 0)]

        def fake_gai(host, port, family, socktype):
            calls.append(host)
            return [(2, 1, 6, "", pair) for pair in pairs]

        monkeypatch.setattr(socket, "getaddrinfo", fake_gai)

        seen: dict[str, object] = {}
        priv, pub = generate_recipient_keypair()
        fetch = _make_fetch((pub, priv))
        monkeypatch.setenv("UMRS_UNSEAL_PRIVKEY_B64", priv)
        monkeypatch.setattr("umrs.infer_proxy.app.lease_by_token", lambda _p, _t: _lease())

        def handler(request: httpx.Request) -> httpx.Response:
            seen["url"] = str(request.url)
            seen["sni"] = (request.extensions or {}).get("sni_hostname")
            seen["host"] = request.headers.get("host")
            return _stream_response(body=b"ok", headers={"content-type": "text/plain"})

        app = create_app(
            umrs_db_pool=None,
            unseal_privkey=priv,
            sealed_store=_FakeStore(fetch),
            broker=_FakeBroker(),
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )
        resp = _request(
            app,
            "POST",
            "/s/sess-1/v1/chat/completions",
            json={},
            headers={"authorization": "Bearer tok-session"},
        )
        assert resp.status_code == 200
        assert calls == ["api.example-provider.com"]  # one resolution total
        assert seen["url"].startswith("https://104.18.2.161")
        assert seen["host"] == "api.example-provider.com"
        assert seen["sni"] == "api.example-provider.com"

    def test_second_resolution_hash_change_still_uses_first_answer(self, monkeypatch):
        """Rebinding attempt: resolution flips AFTER validation — pin holds the checked answer."""
        answers = [["104.18.2.161"], ["10.0.0.66"]]  # check-time answer, attack-time answer
        answered: list[list[str]] = []

        def fake_gai(host, port, family, socktype):
            answered.append(answers[0])
            return [(2, 1, 6, "", (answers.pop(0)[0], 0))]

        monkeypatch.setattr(socket, "getaddrinfo", fake_gai)
        seen: dict[str, object] = {}
        priv, pub = generate_recipient_keypair()
        fetch = _make_fetch((pub, priv))
        monkeypatch.setenv("UMRS_UNSEAL_PRIVKEY_B64", priv)
        monkeypatch.setattr("umrs.infer_proxy.app.lease_by_token", lambda _p, _t: _lease())

        def handler(request: httpx.Request) -> httpx.Response:
            seen["url"] = str(request.url)
            return _stream_response(body=b"ok", headers={"content-type": "text/plain"})

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
        assert resp.status_code == 200
        # Never re-resolved for the dial (connector consumed the pinned answer).
        assert answered == [["104.18.2.161"]]
        assert seen["url"].startswith("https://104.18.2.161")

    def test_denied_when_all_resolution_results_infra(self, monkeypatch):
        monkeypatch.setattr(
            socket,
            "getaddrinfo",
            lambda *_args, **_kw: [(2, 1, 6, "", ("127.0.0.1", 0))],
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
            http_client=httpx.AsyncClient(
                transport=httpx.MockTransport(
                    lambda _r: _stream_response(body=b"ok", headers={"content-type": "text/plain"})
                )
            ),
        )
        resp = _request(
            app, "GET", "/s/sess-1/v1/models", headers={"authorization": "Bearer tok-session"}
        )
        assert resp.status_code == 470
        assert resp.json()["error"] == "host_denied"
