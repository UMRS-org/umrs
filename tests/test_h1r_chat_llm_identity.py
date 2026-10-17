"""H-1 remainder, chat-llm lane (hardening plan H-1, 2026-10-18).

- ``chat._get_caller_identity`` and ``llm._get_caller_identity`` resolved the
  gateway tuple without checking its HMAC: a direct sidecar caller could
  claim any user and project (and, on /umrs/llm, the admin flag) in strict
  mode.
- ``llm._endpoint_caller`` read the raw ``X-Verified-Is-Admin`` header, so the
  mvp legacy tier (``X-Project-ID`` only) could still claim admin.
- ``llm._build_upstream_headers`` forwarded the client's gateway identity
  headers (``X-Verified-*``, a replayable HMAC among them) to the upstream
  LiteLLM, which may be another project's shared endpoint (ADR-078); only the
  verified user and project go upstream now.
"""

from __future__ import annotations

from unittest.mock import patch

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from umrs.orchestrator.endpoint_registry import EndpointRegistry, get_registry

from r4_contracts.handshake_hmac import sign_claims

SECRET = "h1r-chat-llm-secret"

LLM_ROUTES = (
    ("POST", "/umrs/llm/v1/chat/completions"),
    ("POST", "/umrs/llm/v1/completions"),
    ("POST", "/umrs/llm/v1/embeddings"),
    ("GET", "/umrs/llm/v1/models"),
)
CHAT_ROUTES = (
    ("POST", "/umrs/chat/v1/chat/completions"),
    ("GET", "/umrs/chat/v1/models"),
    ("GET", "/umrs/chat/manifest"),
)


def _signed(user: str, project: str, admin: bool = False) -> dict[str, str]:
    return {
        "X-User-ID": user,
        "X-Verified-Project-ID": project,
        "X-Verified-Is-Admin": "true" if admin else "false",
        "X-Verified-HMAC": sign_claims(SECRET, user, project, admin),
    }


def _forged() -> dict[str, str]:
    """A foreign project with the admin flag and a bogus signature."""
    return {
        "X-User-ID": "mallory",
        "X-Verified-Project-ID": "proj-victim",
        "X-Verified-Is-Admin": "true",
        "X-Verified-HMAC": "0" * 64,
    }


@pytest.fixture
def strict(monkeypatch):
    monkeypatch.setenv("UMRS_AUTH_MODE", "strict")
    monkeypatch.setenv("UMRS_HANDSHAKE_HMAC_SECRET", SECRET)


def _registry(project: str) -> EndpointRegistry:
    registry = EndpointRegistry()
    registry.register(
        project_id=project,
        model="m1",
        job_id=f"job-{project}",
        host="compute-01",
        port=4000,
        backend="slurm",
        metadata={},
    )
    registry.mark_ready(f"job-{project}")
    return registry


def _client(module_name: str, registry: EndpointRegistry) -> TestClient:
    from importlib import import_module

    app = FastAPI()
    app.include_router(import_module(f"umrs.api.{module_name}").router)
    app.dependency_overrides[get_registry] = lambda: registry
    return TestClient(app)


def _call(client: TestClient, method: str, path: str, headers: dict[str, str]):
    body = {"model": "m1", "messages": [], "input": "x", "prompt": "x"}
    if method == "GET":
        return client.get(path, headers=headers)
    return client.post(path, json=body, headers=headers)


class _Upstream:
    """Records every upstream call (httpx client methods) and answers 200."""

    def __init__(self) -> None:
        self.headers: list[dict[str, str]] = []

    def _answer(self, headers) -> httpx.Response:
        self.headers.append(dict(headers or {}))
        request = httpx.Request("GET", "http://upstream.test/")
        return httpx.Response(200, json={"data": [], "choices": []}, request=request)

    def patch(self):
        async def fake(_client, *_args, **kwargs):
            return self._answer(kwargs.get("headers"))

        return patch.multiple(httpx.AsyncClient, request=fake, post=fake, get=fake)


# ── forged tuple: 401 everywhere, nothing reaches upstream ──────────────────


@pytest.mark.parametrize(("method", "path"), LLM_ROUTES + CHAT_ROUTES)
def test_forged_tuple_is_unauthorized(strict, method, path):
    module = "llm" if path.startswith("/umrs/llm") else "chat"
    upstream = _Upstream()
    with upstream.patch():
        resp = _call(_client(module, _registry("proj-victim")), method, path, _forged())
    assert resp.status_code == 401
    assert upstream.headers == []


@pytest.mark.parametrize(("method", "path"), LLM_ROUTES + CHAT_ROUTES)
def test_unsigned_tuple_is_unauthorized(strict, method, path):
    module = "llm" if path.startswith("/umrs/llm") else "chat"
    headers = _signed("alice", "proj-a")
    del headers["X-Verified-HMAC"]
    upstream = _Upstream()
    with upstream.patch():
        resp = _call(_client(module, _registry("proj-a")), method, path, headers)
    assert resp.status_code == 401
    assert upstream.headers == []


# ── signed tuple: works, and only the verified identity goes upstream ───────


@pytest.mark.parametrize(("method", "path"), LLM_ROUTES + CHAT_ROUTES)
def test_signed_tuple_reaches_upstream(strict, method, path):
    module = "llm" if path.startswith("/umrs/llm") else "chat"
    upstream = _Upstream()
    with upstream.patch():
        resp = _call(_client(module, _registry("proj-a")), method, path, _signed("alice", "proj-a"))
    assert resp.status_code == 200
    assert upstream.headers


def _lower(headers: dict[str, str]) -> dict[str, str]:
    return {k.lower(): v for k, v in headers.items()}


@pytest.mark.parametrize(("method", "path"), LLM_ROUTES[:3])
def test_llm_forwards_only_the_verified_identity(strict, method, path):
    headers = {
        **_signed("alice", "proj-a"),
        "X-Project-ID": "proj-attacker",
        "X-Verified-Project-Memberships": "proj-a,proj-b",
        "Cookie": "r4_jwt=caller-session-token",
    }
    upstream = _Upstream()
    with upstream.patch():
        resp = _call(_client("llm", _registry("proj-a")), method, path, headers)
    assert resp.status_code == 200
    sent = _lower(upstream.headers[0])
    assert sent["x-user-id"] == "alice"
    assert sent["x-project-id"] == "proj-a"
    assert not [k for k in sent if k.startswith("x-verified")]
    assert "cookie" not in sent


def test_chat_forwards_the_verified_identity(strict):
    headers = {**_signed("alice", "proj-a"), "X-Project-ID": "proj-attacker"}
    upstream = _Upstream()
    with upstream.patch():
        resp = _call(_client("chat", _registry("proj-a")), "GET", "/umrs/chat/v1/models", headers)
    assert resp.status_code == 200
    sent = _lower(upstream.headers[0])
    assert (sent["x-user-id"], sent["x-project-id"]) == ("alice", "proj-a")


# ── admin flag: only the verified one counts ───────────────────────────────


def _routing_caller(monkeypatch, headers: dict[str, str]):
    """The EndpointCaller the llm route hands to discovery."""
    from umrs.api import llm

    seen = []

    def fake_discover(_registry, _project, _model, caller=None):
        seen.append(caller)
        llm.raise_typed(llm.PlatformError.llm("unavailable", "stop here"))

    monkeypatch.setattr(llm, "_discover_endpoint", fake_discover)
    _call(_client("llm", _registry("proj-a")), "POST", LLM_ROUTES[0][1], headers)
    return seen[0]


def test_signed_admin_flag_is_honoured(strict, monkeypatch):
    caller = _routing_caller(monkeypatch, _signed("root", "proj-a", admin=True))
    assert (caller.user_id, caller.project_id, caller.is_admin) == ("root", "proj-a", True)


def test_legacy_tier_cannot_claim_admin(monkeypatch):
    """mvp legacy tier (X-Project-ID only): no verified tuple, so no admin."""
    monkeypatch.setenv("UMRS_AUTH_MODE", "mvp")
    headers = {"X-User-ID": "bob", "X-Project-ID": "proj-a", "X-Verified-Is-Admin": "true"}
    caller = _routing_caller(monkeypatch, headers)
    assert (caller.user_id, caller.project_id, caller.is_admin) == ("bob", "proj-a", False)


def test_mvp_verified_tuple_still_trusted(monkeypatch):
    """Dev (mvp) behaviour kept: the unsigned verified tuple is trusted."""
    monkeypatch.setenv("UMRS_AUTH_MODE", "mvp")
    headers = {"X-User-ID": "bob", "X-Verified-Project-ID": "proj-a", "X-Verified-Is-Admin": "true"}
    caller = _routing_caller(monkeypatch, headers)
    assert (caller.user_id, caller.project_id, caller.is_admin) == ("bob", "proj-a", True)


# ── replayed signature with the admin header flipped ───────────────────────


def _flipped(user: str, project: str, admin_header: str) -> dict[str, str]:
    """The caller's own valid non-admin signature, admin header rewritten."""
    return {**_signed(user, project), "X-Verified-Is-Admin": admin_header}


@pytest.mark.parametrize(("method", "path"), LLM_ROUTES + CHAT_ROUTES)
def test_flipped_admin_flag_is_unauthorized(strict, method, path):
    """'true' is parsed as admin, so the non-admin signature no longer matches."""
    module = "llm" if path.startswith("/umrs/llm") else "chat"
    upstream = _Upstream()
    with upstream.patch():
        resp = _call(
            _client(module, _registry("proj-a")), method, path, _flipped("alice", "proj-a", "true")
        )
    assert resp.status_code == 401
    assert upstream.headers == []


def test_case_variant_admin_flag_is_not_admin(strict, monkeypatch):
    """'TRUE' is not an admin value: the non-admin signature verifies, as non-admin."""
    caller = _routing_caller(monkeypatch, _flipped("alice", "proj-a", "TRUE"))
    assert (caller.user_id, caller.project_id, caller.is_admin) == ("alice", "proj-a", False)


def test_chat_case_variant_admin_cannot_read_foreign_corpus(strict, monkeypatch):
    """The RAG corpus check on chat completions must agree with the verified
    admin flag: a non-admin signature plus 'X-Verified-Is-Admin: TRUE' must
    not open another project's corpus (needs rag._require_rag_scope to read
    the signed flag, H-1 rag lane; the raw-header reader granted it)."""
    monkeypatch.setattr("umrs.api.rag.shared_corpus_readable", lambda *_a: False)
    body = {"model": "m1", "messages": [], "project_id": "proj-victim"}
    upstream = _Upstream()
    with upstream.patch():
        resp = _client("chat", _registry("proj-a")).post(
            CHAT_ROUTES[0][1], json=body, headers=_flipped("mallory", "proj-a", "TRUE")
        )
    assert resp.status_code == 403
    assert upstream.headers == []
