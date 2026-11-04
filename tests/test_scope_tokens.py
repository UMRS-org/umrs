"""ADR-108 D-3 (W2b): signed scope tokens of work levels.

A loop or teamup acts for a user for hours; the only agent mark before was
the self-declared ``x-r4-actor`` header. A token is issued on the caller's
verified identity, narrows (never widens) what the caller may do, and can be
revoked with everything beneath it.
"""

from __future__ import annotations

import uuid

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from umrs.api.scope_tokens import require_grant, scope_of
from umrs.api.scope_tokens import router as scopes_router
from umrs.scopes import tokens as t

from r4_contracts.handshake_hmac import sign_claims

SECRET = "test-handshake-secret"


def _issue(path: str = "s-1", grants=("ingest",), **kw):
    return t.issue(user="alice", project="proj", path=path, grants=grants, secret=SECRET, **kw)


# ── the token core ─────────────────────────────────────────────────────────


def test_issue_and_decode_round_trip() -> None:
    token, claims = _issue(repo="github.com/lab/mof", branch="r4/x", commit="abc1234")
    back = t.decode(token, SECRET)
    assert (back.user, back.path, back.grants, back.commit) == (
        "alice",
        "s-1",
        ("ingest",),
        "abc1234",
    )
    assert back == claims


@pytest.mark.parametrize("damage", ["payload", "signature", "secret"])
def test_a_changed_token_or_another_key_is_invalid(damage: str) -> None:
    token, _ = _issue()
    body, sig = token.split(".")
    if damage == "payload":
        token = f"{body[:-2]}xx.{sig}"
    elif damage == "signature":
        token = f"{body}.{sig[:-2]}xx"
    secret = "other" if damage == "secret" else SECRET
    with pytest.raises(t.ScopeError) as err:
        t.decode(token, secret)
    assert err.value.code == "invalid"


def test_expiry_and_ttl_bounds() -> None:
    token, claims = _issue(ttl=60, now=1_000)
    with pytest.raises(t.ScopeError):
        t.decode(token, SECRET, now=claims.exp)
    for ttl in (0, t.MAX_TTL_SEC + 1):
        with pytest.raises(t.ScopeError) as err:
            _issue(ttl=ttl)
        assert err.value.code == "bad_request"


def test_paths_and_grants_are_checked() -> None:
    for path in ("", "a//b", "../x", "/".join(["s"] * (t.MAX_DEPTH + 1))):
        with pytest.raises(t.ScopeError):
            _issue(path=path)
    with pytest.raises(t.ScopeError):
        _issue(grants=("root",))
    assert _issue(grants=("workflow:*", "workflow:fit-v2", "web"))[1].grants == (
        "workflow:*",
        "workflow:fit-v2",
        "web",
    )


def test_a_child_narrows_its_parent() -> None:
    _, parent = _issue(path="s-1", grants=("ingest", "workflow:*"), ttl=600, now=1_000)
    _, child = _issue(
        path="s-1/loop-7", grants=("workflow:fit",), parent=parent, ttl=6000, now=1_000
    )
    assert (child.parent, child.exp) == ("s-1", parent.exp)  # never outlives the parent
    cases = [
        {"path": "s-2/loop-7", "grants": ("ingest",)},  # not beneath
        {"path": "s-1/loop-8", "grants": ("upload",)},  # a grant the parent lacks
    ]
    for case in cases:
        with pytest.raises(t.ScopeError) as err:
            _issue(parent=parent, **case)
        assert err.value.code == "forbidden"
    with pytest.raises(t.ScopeError):
        t.issue(user="bob", project="proj", path="s-1/x", grants=(), secret=SECRET, parent=parent)


def test_grants_and_revocation_prefixes() -> None:
    assert (t.allows(("workflow:*",), "workflow:any"), t.allows(("ingest",), "workflow:x")) == (
        True,
        False,
    )
    assert (
        t.revoked_by("s-1/loop-7", "s-1"),
        t.revoked_by("s-10", "s-1"),
        t.revoked_by("s-1", "s-1"),
    ) == (
        True,
        False,
        True,
    )


def test_the_handshake_signature_is_never_a_scope_signature() -> None:
    assert t.derive_key(SECRET) != SECRET.encode()
    with pytest.raises(t.ScopeError) as err:
        t.derive_key("")
    assert err.value.code == "unavailable"


# ── the HTTP routes ────────────────────────────────────────────────────────


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("UMRS_HANDSHAKE_HMAC_SECRET", SECRET)
    app = FastAPI()
    app.include_router(scopes_router)

    @app.post("/probe/{wanted}")
    def probe(wanted: str, request: Request) -> dict:
        claims = scope_of(
            request,
            user=request.headers.get("x-user-id", ""),
            project=request.headers.get("x-verified-project-id", ""),
        )
        require_grant(claims, wanted)
        return {"path": claims.path if claims else ""}

    return TestClient(app)


def _who(user: str = "alice", project: str = "proj") -> dict[str, str]:
    return {
        "X-User-ID": user,
        "X-Verified-Project-ID": project,
        "X-Verified-Is-Admin": "false",
        "X-Verified-HMAC": sign_claims(SECRET, user, project, False),
    }


def _code(response) -> str:
    return response.headers.get("x-platform-error-code", "")


def test_issue_a_root_and_a_child_then_use_them(client: TestClient) -> None:
    root = f"s-{uuid.uuid4().hex[:6]}"
    made = client.post(
        "/umrs/scopes", json={"path": root, "grants": ["ingest", "workflow:*"]}, headers=_who()
    )
    assert made.status_code == 200, made.text
    child = client.post(
        "/umrs/scopes",
        json={"path": f"{root}/loop-1", "grants": ["ingest"], "parent_token": made.json()["token"]},
        headers=_who(),
    )
    assert child.json()["scope"]["parent"] == root
    scoped = {**_who(), "x-r4-scope": child.json()["token"]}
    assert client.post("/probe/ingest", headers=scoped).json() == {"path": f"{root}/loop-1"}
    refused = client.post("/probe/workflow:fit", headers=scoped)
    assert (refused.status_code, _code(refused)) == (403, "scopes.grant_missing")
    assert client.post("/probe/ingest", headers=_who()).json() == {
        "path": ""
    }  # no header: unchanged


def test_a_token_is_bound_to_its_user_and_project(client: TestClient) -> None:
    token = client.post(
        "/umrs/scopes", json={"path": "s-bound", "grants": ["ingest"]}, headers=_who()
    ).json()["token"]
    other = client.post("/probe/ingest", headers={**_who("bob"), "x-r4-scope": token})
    assert (other.status_code, _code(other)) == (403, "scopes.forbidden")
    junk = client.post("/probe/ingest", headers={**_who(), "x-r4-scope": "not.a-token"})
    assert (junk.status_code, _code(junk)) == (401, "scopes.invalid")


def test_revoking_a_level_revokes_its_children(client: TestClient) -> None:
    root = f"s-{uuid.uuid4().hex[:6]}"
    parent = client.post(
        "/umrs/scopes", json={"path": root, "grants": ["ingest"]}, headers=_who()
    ).json()["token"]
    child = client.post(
        "/umrs/scopes",
        json={"path": f"{root}/team-1/ED", "grants": ["ingest"], "parent_token": parent},
        headers=_who(),
    ).json()["token"]
    assert (
        client.post("/umrs/scopes/revoke", json={"path": root}, headers=_who()).json()["revoked"]
        == root
    )
    for token in (parent, child):
        refused = client.post("/probe/ingest", headers={**_who(), "x-r4-scope": token})
        assert (refused.status_code, _code(refused)) == (401, "scopes.revoked")
    late = client.post(
        "/umrs/scopes", json={"path": f"{root}/late", "parent_token": parent}, headers=_who()
    )
    assert _code(late) == "scopes.revoked"


def test_a_child_beyond_its_parent_is_refused_by_name(client: TestClient) -> None:
    parent = client.post(
        "/umrs/scopes", json={"path": "s-narrow", "grants": ["ingest"]}, headers=_who()
    ).json()["token"]
    wide = client.post(
        "/umrs/scopes",
        json={"path": "s-narrow/x", "grants": ["workflow:*"], "parent_token": parent},
        headers=_who(),
    )
    assert (wide.status_code, _code(wide)) == (403, "scopes.forbidden")


# ── the routes a level calls ───────────────────────────────────────────────


def test_ingest_and_workflow_checks_use_the_scope(client: TestClient, monkeypatch) -> None:
    from types import SimpleNamespace

    from umrs.api import knowledge_v2, workflows

    token = client.post(
        "/umrs/scopes", json={"path": "s-gate", "grants": ["workflow:fit"]}, headers=_who()
    ).json()["token"]
    app = client.app

    @app.post("/probe-ingest")
    def probe_ingest(request: Request) -> dict:
        scope = knowledge_v2._ingest_scope(request, "proj")
        return {
            "actor": knowledge_v2._scoped_actor(request, scope),
            **knowledge_v2._scope_fields(scope),
        }

    @app.post("/probe-trigger/{name}")
    def probe_trigger(name: str, request: Request) -> dict:
        resolved = SimpleNamespace(user_id="alice", project_id="proj")
        workflows._refuse_unscoped_trigger(request, name, resolved)  # type: ignore[arg-type]
        return {"ok": True}

    scoped = {**_who(), "x-r4-scope": token}
    no_ingest = client.post("/probe-ingest", headers=scoped)
    assert (no_ingest.status_code, _code(no_ingest)) == (403, "scopes.grant_missing")
    assert client.post("/probe-ingest", headers=_who()).json() == {"actor": "user"}
    assert client.post("/probe-trigger/fit", headers=scoped).json() == {"ok": True}
    other = client.post("/probe-trigger/other", headers=scoped)
    assert _code(other) == "scopes.grant_missing"
    ingest_token = client.post(
        "/umrs/scopes", json={"path": "s-ing", "grants": ["ingest"]}, headers=_who()
    ).json()["token"]
    allowed = client.post("/probe-ingest", headers={**_who(), "x-r4-scope": ingest_token}).json()
    assert allowed == {"actor": "agent", "scope_path": "s-ing"}


def test_a_scoped_ingest_reads_the_user_only_through_the_hmac(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stolen token beside a spoofed, unsigned user header is refused
    (strict mode, the production default)."""
    from umrs.api import knowledge_v2

    monkeypatch.setenv("UMRS_AUTH_MODE", "strict")

    token = client.post(
        "/umrs/scopes", json={"path": "s-hmac", "grants": ["ingest"]}, headers=_who()
    ).json()["token"]

    @client.app.post("/probe-hmac")
    def probe_hmac(request: Request) -> dict:
        return {"path": knowledge_v2._ingest_scope(request, "proj").path}

    spoofed = {"X-User-ID": "alice", "X-Verified-Project-ID": "proj", "x-r4-scope": token}
    refused = client.post("/probe-hmac", headers=spoofed)
    assert (refused.status_code, _code(refused)) == (401, "scopeidentity.unauthorized")
    signed = client.post("/probe-hmac", headers={**_who(), "x-r4-scope": token})
    assert signed.json() == {"path": "s-hmac"}
