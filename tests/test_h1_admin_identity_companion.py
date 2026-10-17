"""H-1 (hardening plan, 2026-10-17), item 7: companion session, approval and
venue owner routes took the owner from a bare ``X-User-ID`` header.

In strict mode the owner must be the HMAC-verified gateway caller; a caller
that names another user (with no signature, or with its own signature over a
different user) is refused. Mvp mode keeps the raw header (dev contract).
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from r4_contracts.handshake_hmac import sign_claims

SECRET = "h1-admin-identity-secret"


def _headers(user: str, project: str, admin: bool = False) -> dict[str, str]:
    return {
        "X-User-ID": user,
        "X-Verified-Project-ID": project,
        "X-Verified-Is-Admin": "true" if admin else "false",
        "X-Verified-HMAC": sign_claims(SECRET, user, project, admin),
    }


def _impersonating(victim: str) -> dict[str, str]:
    """Mallory's own valid signature, with the user header swapped."""
    headers = _headers("mallory", "proj-m")
    headers["X-User-ID"] = victim
    return headers


@pytest.fixture
def strict(monkeypatch):
    monkeypatch.setenv("UMRS_AUTH_MODE", "strict")
    monkeypatch.setenv("UMRS_HANDSHAKE_HMAC_SECRET", SECRET)


class _VenueService:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def status(self, *, owner: str, session_id: str):
        self.calls.append(("status", owner))
        return {"owner": owner, "session_id": session_id} if owner == "bob" else None

    def stop(self, *, owner: str, session_id: str):
        self.calls.append(("stop", owner))
        return {"owner": owner, "session_id": session_id, "stopped": True}


@pytest.fixture
def companion():
    from umrs.api.companion_chat_approvals import router as approvals_router
    from umrs.api.companion_sessions import router as sessions_router
    from umrs.api.companion_venue import router as venue_router

    app = FastAPI()
    app.include_router(venue_router)
    app.include_router(approvals_router)
    app.include_router(sessions_router)
    service = _VenueService()
    app.state.companion_venue_service = service
    return TestClient(app), service


# ── strict: the owner is the verified caller ─────────────────────────────────


def test_unsigned_owner_header_cannot_read_a_venue(strict, companion):
    client, service = companion
    resp = client.get("/umrs/companion/sessions/s1/venue", headers={"X-User-ID": "bob"})
    assert resp.status_code == 401
    assert not service.calls


def test_foreign_signature_cannot_stop_anothers_venue(strict, companion):
    client, service = companion
    resp = client.post("/umrs/companion/sessions/s1/venue/stop", headers=_impersonating("bob"))
    assert resp.status_code == 401
    assert not service.calls


def test_signed_owner_reads_own_venue(strict, companion):
    client, service = companion
    resp = client.get("/umrs/companion/sessions/s1/venue", headers=_headers("bob", "proj-a"))
    assert resp.status_code == 200, resp.text
    assert service.calls == [("status", "bob")]


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("get", "/umrs/companion/sessions"),
        ("get", "/umrs/companion/sessions/s1"),
        ("delete", "/umrs/companion/sessions/s1"),
        ("get", "/umrs/companion/sessions/s1/turns"),
        ("get", "/umrs/companion/sessions/s1/approvals"),
    ],
)
def test_impersonated_owner_is_refused_on_session_routes(strict, companion, method, path):
    client, _service = companion
    resp = getattr(client, method)(path, headers=_impersonating("bob"))
    assert resp.status_code == 401


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("get", "/umrs/companion/sessions/s1"),
        ("get", "/umrs/companion/sessions/s1/approvals"),
    ],
)
def test_signed_owner_passes_identity_on_session_routes(strict, companion, method, path):
    """Identity passes; the bare app has no DB pool, so the route answers 503."""
    client, _service = companion
    resp = getattr(client, method)(path, headers=_headers("bob", "proj-a"))
    assert resp.status_code == 503


# ── mvp: raw header keeps working (dev contract) ─────────────────────────────


def test_mvp_mode_keeps_the_raw_owner_header(monkeypatch, companion):
    monkeypatch.setenv("UMRS_AUTH_MODE", "mvp")
    client, service = companion
    resp = client.get("/umrs/companion/sessions/s1/venue", headers={"X-User-ID": "bob"})
    assert resp.status_code == 200, resp.text
    assert service.calls == [("status", "bob")]
