"""S-D37-3 (ADR-062 amendment 2, DEC-037): workstation leases and the
``/w/`` relay that the edge's ``/gw/infer/`` reaches.

Before this lane umrs-infer had no workstation lease kind and no ``/w/``
route, and ``POST /umrs/infer/leases`` did not exist: every test here fails
on the old code (404 on the route or the import of the new names).

The proxy half runs ``create_app`` with the proxy tests' fakes (fixed
lease, MockTransport upstream). The issuance half drives the real router
with the real in-memory ``SecretBroker`` and hands the token it minted to
the proxy, so the two halves are tested against each other.
"""

from __future__ import annotations

import socket
from dataclasses import dataclass

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from test_infer_proxy import (
    _FakeBroker,
    _FakeStore,
    _lease,
    _make_fetch,
    _request,
    _stream_response,
)
from umrs.api import infer_leases
from umrs.api._scope_identity import get_caller_scope
from umrs.infer_proxy.app import create_app
from umrs.infer_proxy.leases import WORKSTATION_KIND, check_route_lease
from umrs.jobs.secret_broker import SecretBroker
from umrs.knowledge.access_scope import CallerScope
from umrs.quotas.enforcer import QuotaDenial
from umrs.security.seal import generate_recipient_keypair

CHAT = "/w/ws-1/v1/chat/completions"
TOKEN = {"authorization": "Bearer tok-session"}
PROVIDER = "https://api.example-provider.com"
INSTITUTION = "https://inference.example-institution.org"


@pytest.fixture(autouse=True)
def _site_declares_one_institution(monkeypatch):
    monkeypatch.setenv("UMRS_INSTITUTIONAL_PROVIDER_HOSTS", "inference.example-institution.org")


def _ws_lease(**over):
    base = {
        "tier": WORKSTATION_KIND,
        "secret_name": "workstation-inference",
        "session_id": "ws-1",
        "source_ids": ("src-a",),
        "default_source_id": "src-a",
        "egress_ip": "",
        "owner_id": "alice",
        "project": "proj-open",
    }
    base.update(over)
    return _lease(**base)


class _Harness:
    """The proxy with a recording upstream, data-class table and usage log."""

    def __init__(self, monkeypatch, lease, *, allow_host=PROVIDER, owner="alice", **opts):
        self.upstream: list[httpx.Request] = []
        self.usage: list[dict] = []
        self.classes = {"proj-open": "non_sensitive", "proj-secret": "sensitive"}
        priv, pub = generate_recipient_keypair()
        fetch = _make_fetch((pub, priv), allow_host=allow_host)
        fetch.owner = owner
        fetch.id = opts.get("source_id", "src-a")
        self.broker = _FakeBroker()
        monkeypatch.setattr("umrs.infer_proxy.app.lease_by_token", lambda _p, _t: lease)
        monkeypatch.setattr(
            "umrs.infer_proxy.app.data_class_for_project",
            lambda _pool, project: self.classes.get(project, ""),
        )
        monkeypatch.setattr(
            "umrs.infer_proxy.app.record_proxy_usage",
            lambda _pool, **row: self.usage.append(row),
        )
        monkeypatch.setattr(socket, "getaddrinfo", lambda *_a: [(2, 1, 6, "", ("104.18.2.161", 0))])
        self.app = create_app(
            umrs_db_pool=None,
            unseal_privkey=priv,
            sealed_store=_FakeStore(fetch),
            broker=self.broker,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(self._upstream)),
            quota_enforcer=opts.get("enforcer"),
        )

    def _upstream(self, request: httpx.Request) -> httpx.Response:
        request.read()
        self.upstream.append(request)
        return _stream_response(body=b'{"usage":{"prompt_tokens":3,"completion_tokens":2}}')

    def call(self, path=CHAT, headers=TOKEN, method="POST"):
        return _request(self.app, method, path, json={"model": "m"}, headers=headers)


# ── the relay route (/w/) ──────────────────────────────────────────────────


class TestRelayRefusals:
    def test_no_lease_token_is_401(self, monkeypatch):
        h = _Harness(monkeypatch, _ws_lease())
        resp = h.call(headers={})
        assert (resp.status_code, resp.json()["error"]) == (401, "token_invalid")
        assert h.upstream == []

    def test_unknown_token_is_401(self, monkeypatch):
        h = _Harness(monkeypatch, None)
        assert h.call().status_code == 401
        assert h.upstream == []

    def test_sandbox_lease_refused_on_the_workstation_route(self, monkeypatch):
        h = _Harness(monkeypatch, _lease(session_id="ws-1", egress_ip=""))
        resp = h.call()
        assert (resp.status_code, resp.json()["error"]) == (401, "lease_kind_mismatch")

    def test_workstation_lease_refused_on_the_sandbox_route(self, monkeypatch):
        h = _Harness(monkeypatch, _ws_lease(session_id="sess-1"))
        resp = h.call(path="/s/sess-1/v1/chat/completions")
        assert (resp.status_code, resp.json()["error"]) == (401, "lease_kind_mismatch")
        site = h.call(path="/s/sess-1/site/v1/chat/completions")
        assert site.status_code in (401, 403)
        assert h.upstream == []

    def test_lease_without_project_is_unbound(self, monkeypatch):
        h = _Harness(monkeypatch, _ws_lease(project=""))
        assert h.call().json()["error"] == "lease_unbound"

    def test_sensitive_project_refused_a_public_source(self, monkeypatch):
        h = _Harness(monkeypatch, _ws_lease(project="proj-secret"))
        resp = h.call()
        assert (resp.status_code, resp.json()["error"]) == (403, "source_class_refused")
        assert h.upstream == [] and h.usage == []

    def test_unknown_class_fails_closed(self, monkeypatch):
        h = _Harness(monkeypatch, _ws_lease(project="proj-unlisted"))
        assert h.call().json()["error"] == "source_class_refused"

    def test_another_users_source_is_refused(self, monkeypatch):
        h = _Harness(monkeypatch, _ws_lease(), owner="mallory")
        resp = h.call()
        assert (resp.status_code, resp.json()["error"]) == (403, "source_not_owned")
        assert h.upstream == []

    def test_site_source_not_served_on_the_workstation_route(self, monkeypatch):
        h = _Harness(monkeypatch, _ws_lease())
        resp = h.call(path="/w/ws-1/site/v1/chat/completions")
        assert (resp.status_code, resp.json()["error"]) == (403, "path_denied")

    def test_quota_over_cap_is_429_before_the_provider(self, monkeypatch):
        denial = QuotaDenial(
            code="quota.rpm_exceeded",
            status=429,
            scope="user",
            subject="alice",
            limit=1,
            used=1,
            window="minute",
            resets_at="",
            retry_after_s=60,
            message="over",
        )
        enforcer = type("E", (), {"check": lambda _self, _lease: denial})()
        h = _Harness(monkeypatch, _ws_lease(), enforcer=enforcer)
        resp = h.call()
        assert resp.status_code == 429
        assert h.upstream == [] and h.usage == []


class TestRelayAdmits:
    def test_non_sensitive_relays_with_the_server_side_key(self, monkeypatch):
        h = _Harness(monkeypatch, _ws_lease())
        resp = h.call(
            path=CHAT + "?key=leaked-client-key&api-version=1",
            headers={
                "authorization": "Bearer tok-session",
                "x-api-key": "client-provider-key",
                "x-forwarded-for": "198.51.100.7",
                "x-real-ip": "198.51.100.7",
                "x-verified-hmac": "forged",
                "x-user-id": "bob",
                "x-r4-source": "src-a",
                "cookie": "session=1",
            },
        )
        assert resp.status_code == 200
        (sent,) = h.upstream
        assert sent.headers["authorization"] == "Bearer sk-real-provider-key"
        leaked = {"tok-session", "client-provider-key", "198.51.100.7", "forged", "bob"}
        assert not leaked & set(sent.headers.values())
        assert not {"x-forwarded-for", "x-r4-source", "cookie"} & set(sent.headers)
        assert dict(sent.url.params) == {"api-version": "1"}
        assert [(u["session_id"], u["lease"].project) for u in h.usage] == [("ws-1", "proj-open")]

    def test_institutional_host_admitted_for_a_sensitive_project(self, monkeypatch):
        h = _Harness(monkeypatch, _ws_lease(project="proj-secret"), allow_host=INSTITUTION)
        assert h.call().status_code == 200
        assert len(h.upstream) == 1


def test_route_verdict_names_the_kind():
    sandbox = check_route_lease(
        _lease(), workstation=True, sid="sess-1", client_ip="127.0.0.1", source_override=None
    )
    assert sandbox.reason == "lease_kind_mismatch"
    ws = check_route_lease(
        _ws_lease(), workstation=True, sid="ws-1", client_ip="10.9.9.9", source_override=None
    )
    assert ws.ok and ws.source_id == "src-a"


# ── issuance (POST/DELETE /umrs/infer/leases) ──────────────────────────────


@dataclass
class _Meta:
    id: str
    owner: str
    name: str
    provider_kind: str = "openai"
    allow_host: str = PROVIDER


class _Store:
    def __init__(self, *metas: _Meta):
        self._rows = {(m.owner, m.name): m for m in metas}

    def get_meta(self, owner, name):
        return self._rows.get((owner, name))


class _Api:
    """The real router over the in-memory broker and a fake sealed store."""

    def __init__(self, monkeypatch, caller: CallerScope, *metas: _Meta):
        self.broker = SecretBroker(db_pool=None, venue="site-a")
        store = _Store(*metas)
        classes = {"proj-open": "non_sensitive", "proj-secret": "sensitive"}
        monkeypatch.setattr(infer_leases, "_broker", lambda: self.broker)
        monkeypatch.setattr(infer_leases, "_sealed_store", lambda _r: store)
        monkeypatch.setattr(
            infer_leases, "data_class_for_project", lambda _p, project: classes.get(project, "")
        )
        app = FastAPI()
        app.include_router(infer_leases.router)
        app.dependency_overrides[get_caller_scope] = lambda: caller
        self.client = TestClient(app)

    def issue(self, **body):
        body.setdefault("sources", ["openai-main"])
        return self.client.post("/umrs/infer/leases", json=body)


ALICE = CallerScope(user_id="alice", project_id="proj-open")
ALICE_SECRET = CallerScope(user_id="alice", project_id="proj-secret")
OWN = _Meta(id="11111111-1111-4111-8111-111111111111", owner="alice", name="openai-main")
THEIRS = _Meta(id="22222222-2222-4222-8222-222222222222", owner="mallory", name="openai-main")


class TestIssue:
    def test_issues_a_short_workstation_lease_bound_to_the_caller(self, monkeypatch):
        api = _Api(monkeypatch, ALICE, OWN, THEIRS)
        resp = api.issue(project="proj-open", ttl_seconds=600, owner_id="mallory")
        assert resp.status_code == 201, resp.text
        body = resp.json()
        lease = api.broker.lease_by_plaintext_token(body["token"])
        bound = (lease.tier, lease.scope, lease.owner_id, lease.project, lease.source_ids)
        assert bound == (WORKSTATION_KIND, "inference", "alice", "proj-open", (OWN.id,))
        assert (lease.egress_ip, lease.max_lifetime_at) == ("", lease.expires_at)
        assert body["relay_path"] == f"/gw/infer/{lease.session_id}"
        assert not api.broker.renew_lease(lease.lease_id, 60)

    def test_another_project_is_refused(self, monkeypatch):
        resp = _Api(monkeypatch, ALICE, OWN).issue(project="proj-other")
        assert resp.status_code == 403

    def test_a_source_the_caller_does_not_own_is_404(self, monkeypatch):
        resp = _Api(monkeypatch, ALICE, THEIRS).issue(project="proj-open")
        assert resp.status_code == 404

    def test_sensitive_project_refused_a_public_source_at_issue(self, monkeypatch):
        api = _Api(monkeypatch, ALICE_SECRET, OWN)
        resp = api.issue()
        assert resp.status_code == 403
        assert "source_class_refused" in resp.text
        assert api.broker.list_leases() == []

    def test_anonymous_caller_is_refused(self, monkeypatch):
        resp = _Api(monkeypatch, CallerScope(anonymous=True), OWN).issue(project="proj-open")
        assert resp.status_code == 401

    def test_ttl_above_the_cap_is_rejected(self, monkeypatch):
        resp = _Api(monkeypatch, ALICE, OWN).issue(ttl_seconds=7200)
        assert resp.status_code == 422


class TestIssueThenRelay:
    def _minted(self, monkeypatch):
        api = _Api(monkeypatch, ALICE, OWN)
        body = api.issue(project="proj-open").json()
        lease = api.broker.lease_by_plaintext_token(body["token"])
        h = _Harness(monkeypatch, None, source_id=lease.source_ids[0])
        monkeypatch.setattr(
            "umrs.infer_proxy.app.lease_by_token",
            lambda _p, token: api.broker.lease_by_plaintext_token(token),
        )
        return api, body, h

    def test_minted_token_relays_then_revocation_denies_the_next_call(self, monkeypatch):
        api, body, h = self._minted(monkeypatch)
        path = f"/w/{body['session_id']}/v1/chat/completions"
        auth = {"authorization": f"Bearer {body['token']}"}
        assert h.call(path=path, headers=auth).status_code == 200
        assert api.client.delete(f"/umrs/infer/leases/{body['lease_id']}").status_code == 204
        resp = h.call(path=path, headers=auth)
        assert (resp.status_code, resp.json()["error"]) == (401, "lease_revoked")
        assert len(h.upstream) == 1

    def test_another_users_lease_cannot_be_revoked(self, monkeypatch):
        api, body, _h = self._minted(monkeypatch)
        api.client.app.dependency_overrides[get_caller_scope] = lambda: CallerScope(
            user_id="mallory", project_id="proj-open"
        )
        assert api.client.delete(f"/umrs/infer/leases/{body['lease_id']}").status_code == 404
