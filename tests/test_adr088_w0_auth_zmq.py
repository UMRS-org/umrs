"""ADR-088 W0: ``auth.failed`` (UMRS) and ``zmq.session_*`` rows.

Each test asserts the row is written and that the credential, the key or
the full path the request carried is not in it.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from umrs.api import audit_events
from umrs.core.session_curve_provider import SessionCredentialsProvider, SessionGrants

JWT = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJtYWxsb3J5In0.Zm9yZ2Vk"
PUBKEY = "Q" * 40


class _Capture:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def write(self, record: dict[str, Any]) -> None:
        self.rows.append(record)

    def of(self, event: str) -> list[dict[str, Any]]:
        return [r for r in self.rows if r["event"] == event]

    def text(self) -> str:
        return json.dumps(self.rows, default=str)


@pytest.fixture
def captured(monkeypatch: pytest.MonkeyPatch) -> _Capture:
    sink = _Capture()
    monkeypatch.setattr(audit_events, "_audit_store", sink)
    return sink


# -- auth.failed: UMRS identity resolver ------------------------------------


def _request(path: str, headers: dict[str, str]) -> Any:
    return SimpleNamespace(url=SimpleNamespace(path=path), headers=headers)


def test_strict_missing_identity_writes_auth_failed(
    monkeypatch: pytest.MonkeyPatch, captured: _Capture
) -> None:
    from umrs.api._identity import request_identity

    monkeypatch.setenv("UMRS_AUTH_MODE", "strict")
    request = _request("/umrs/files/download/thesis-draft.pdf", {"x-user-id": "mallory"})
    with pytest.raises(HTTPException) as exc:
        request_identity(request)
    assert exc.value.status_code == 401
    (row,) = captured.of("auth.failed")
    assert row["details"]["surface"] == "umrs"
    assert row["details"]["reason"] == "missing_verified_identity"
    assert row["details"]["route"] == "/umrs/files"
    assert "thesis-draft" not in captured.text()


def test_bad_gateway_signature_writes_auth_failed(
    monkeypatch: pytest.MonkeyPatch, captured: _Capture
) -> None:
    from umrs.api import _verified_headers
    from umrs.api._identity import request_identity

    monkeypatch.setattr(_verified_headers, "verified_headers_trusted", lambda **_k: False)
    request = _request(
        "/umrs/jobs",
        {"x-verified-project-id": "p1", "x-user-id": "u1", "x-verified-hmac": "deadbeef"},
    )
    with pytest.raises(HTTPException):
        request_identity(request, verify_hmac=True)
    (row,) = captured.of("auth.failed")
    assert row["details"]["reason"] == "bad_signature"
    assert "deadbeef" not in captured.text()


# -- auth.failed: the ZMQ session route's own JWT check ----------------------


@pytest.fixture
def provider() -> Any:
    from umrs.api.zmq_sessions import set_session_curve_provider

    p = SessionCredentialsProvider(max_ttl_seconds=86400, max_per_user=5)
    set_session_curve_provider(p)
    yield p
    set_session_curve_provider(None)


def _client() -> TestClient:
    from umrs.api.zmq_sessions import router

    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_zmq_session_bad_token_writes_auth_failed_without_the_token(
    provider: Any, captured: _Capture
) -> None:
    resp = _client().post(
        "/umrs/zmq/session",
        json={"client_public_key": PUBKEY},
        headers={"Authorization": f"Bearer {JWT}"},
    )
    assert resp.status_code == 401
    (row,) = captured.of("auth.failed")
    assert row["details"] == {
        "surface": "zmq_session",
        "reason": "bad_token",
        "route": "/umrs/zmq",
        "security_schema": "1",
    }
    assert JWT not in captured.text()


def test_zmq_session_missing_header_writes_auth_failed(provider: Any, captured: _Capture) -> None:
    resp = _client().get("/umrs/zmq/session")
    assert resp.status_code == 401
    assert captured.of("auth.failed")[0]["details"]["reason"] == "missing_header"


# -- zmq.session_minted / zmq.session_revoked --------------------------------


def test_register_writes_minted_without_key_or_email(captured: _Capture) -> None:
    p = SessionCredentialsProvider()
    grants = SessionGrants(email="erin@example.org", max_gpus=2)
    entry = p.register(PUBKEY, "erin", 600, grants=grants)
    (row,) = captured.of("zmq.session_minted")
    assert row["actor"] == "erin"
    assert row["scope"] == "user"
    assert row["details"]["session_id"] == entry.session_id
    assert row["details"]["kind"] == "session"
    assert row["details"]["ttl_seconds"] == 600
    assert row["details"]["max_gpus"] == 2
    assert PUBKEY not in captured.text()
    assert "erin@example.org" not in captured.text()


def test_revoke_writes_revoked_with_actor_and_cause(captured: _Capture) -> None:
    p = SessionCredentialsProvider()
    entry = p.register(PUBKEY, "erin", 600)
    assert p.revoke(entry.session_id, actor="watch", cause="t1_contain")
    (row,) = captured.of("zmq.session_revoked")
    assert row["actor"] == "watch"
    assert row["details"]["session_id"] == entry.session_id
    assert row["details"]["cause"] == "t1_contain"
    assert PUBKEY not in captured.text()


def test_revoke_of_unknown_session_writes_nothing(captured: _Capture) -> None:
    assert not SessionCredentialsProvider().revoke("nope")
    assert captured.of("zmq.session_revoked") == []


def test_revoke_user_writes_one_row_with_the_count(captured: _Capture) -> None:
    p = SessionCredentialsProvider()
    p.register("A" * 40, "frank", 600)
    p.register("B" * 40, "frank", 600)
    assert p.revoke_user("frank", actor="admin1") == 2
    (row,) = captured.of("zmq.session_revoked")
    assert row["details"]["cause"] == "revoke_user"
    assert row["details"]["count"] == 2
    assert row["actor"] == "admin1"


def test_account_sync_disable_writes_revoked_rows(captured: _Capture) -> None:
    p = SessionCredentialsProvider()
    p.register("C" * 40, "gina", 600)
    record = SimpleNamespace(
        username="gina", curve_public_key="", disabled=True, curve_key_revoked_at=""
    )
    p.sync_account_keys([record])
    (row,) = captured.of("zmq.session_revoked")
    assert row["details"]["cause"] == "account_disabled"
    assert "C" * 40 not in captured.text()


def test_owner_delete_route_names_owner_cause(
    provider: Any, captured: _Capture
) -> None:
    from umrs.api.zmq_sessions import _require_user_id

    entry = provider.register(PUBKEY, "hana", 600)
    client = _client()
    client.app.dependency_overrides[_require_user_id] = lambda: "hana"
    resp = client.delete(f"/umrs/zmq/session/{entry.session_id}")
    assert resp.status_code == 204
    (row,) = captured.of("zmq.session_revoked")
    assert row["actor"] == "hana"
    assert row["details"]["cause"] == "owner"
