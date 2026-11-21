"""UMRS Companion HTTP surface tests (wave T3b m5)."""

from __future__ import annotations

import pytest

# splice removed (ADR-053): pytest auto-inserts tests/ for sibling helpers; umrs editable
from fake_pg_pool import FakePool
from fastapi import FastAPI
from fastapi.testclient import TestClient
from umrs.api.companion import router

from umrs_protocol.handshake_hmac import sign_claims

SECRET = "test-surface-key"


def _headers(user: str = "alice", project: str = "proj-a", admin: bool = False):
    return {
        "X-User-ID": user,
        "X-Verified-Project-ID": project,
        "X-Verified-Is-Admin": "true" if admin else "false",
        "X-Verified-HMAC": sign_claims(SECRET, user, project, admin),
    }


HEADERS = _headers("alice")
OTHER_HEADERS = _headers("bob")


def _app(pool, *, actor: str = "alice") -> TestClient:
    from umrs.api.companion import _strict_headers

    app = FastAPI()
    app.include_router(router)
    app.state.umrs_db_pool = pool
    # Strict mode is live locally; inject the actor via dependency_overrides
    # so routes exercise themselves (the sidecar gate stays honest and no
    # process-env mutation leaks into unrelated suites).
    app.dependency_overrides[_strict_headers] = lambda: actor
    return TestClient(app)


def _worker_row(**overrides):
    row = {
        "registry_key": "alice@default",
        "owner_user_id": "alice",
        "project_id": "default",
        "job_id": "job-w1",
        "session_token_fingerprint": "",
        "state": "active",
        "attempt_index": 0,
        "attempts_max": 6,
        "last_heartbeat_at": None,
        "last_active_at": None,
        "created_at": None,
        "updated_at": None,
    }
    row.update(overrides)
    return row


def _turn_row(**overrides):
    row = {
        "attempt_id": "at-1",
        "job_id": "job-w1",
        "registry_key": "alice@default",
        "session_id": "s-1",
        "request_token": "rt-1",
        "user_message_hash": "h",
        "state": "in_flight",
        "committed_response_ref": None,
        "parent_attempt_id": None,
        "research_session_id": None,
    }
    row.update(overrides)
    return row


class TestApprovalSurface:
    @pytest.fixture(autouse=True)
    def _clean_tool_approvals(self):
        from umrs.knowledge import tool_approvals

        tool_approvals.clear_all()
        yield
        tool_approvals.clear_all()

    def test_pending_approvals_empty_reports_durability(self):
        client = _app(FakePool([]))
        resp = client.get("/umrs/companion/approvals/pending", headers=HEADERS)
        assert resp.status_code == 200
        assert resp.json()["pending"] == []
        # Unconfigured process → the inbox says so (ADR-036 (a) honesty rule).
        assert resp.json()["durability"] == "process"

    def test_decision_roundtrip_records_intent(self):
        """Real registry contract (ADR-028): gate intent → decision unblocks."""
        from umrs.knowledge import tool_approvals

        # Owned by the caller: an ownerless intent belongs to its loop's
        # project only (H-1; covered in test_h1_projects_sessions_scope).
        tool_approvals.request_approval(
            "loop-x", 1, {"kind": "workflow_trigger"}, approval_id="ap-1", user_id="alice"
        )
        client = _app(FakePool([]))
        resp = client.post(
            "/umrs/companion/approvals/ap-1/decision",
            headers=HEADERS,
            json={"decision": "once"},
        )
        assert resp.status_code == 200
        assert resp.json()["approval_id"] == "ap-1"
        assert resp.json()["decision"] == "once"
        assert resp.json()["status"] == "approved"
        assert resp.json()["loop_id"] == "loop-x"

    def test_decision_unknown_approval_404s(self):
        client = _app(FakePool([]))
        resp = client.post(
            "/umrs/companion/approvals/ap-missing/decision",
            headers=HEADERS,
            json={"decision": "once"},
        )
        assert resp.status_code == 404

    def test_pending_surfaces_registered_intents(self):
        from umrs.knowledge import tool_approvals

        tool_approvals.request_approval(
            "loop-y", 1, {"kind": "workflow_trigger"}, approval_id="ap-2", user_id="alice"
        )
        client = _app(FakePool([]))
        resp = client.get("/umrs/companion/approvals/pending", headers=HEADERS)
        assert resp.status_code == 200
        assert resp.json()["pending"][0]["id"] == "ap-2"


class TestRetiredWorkerSurface:
    """U3 S4a (D6): /workers, /chat and /turns are gone, not 401/500."""

    def test_worker_and_turn_routes_are_not_served(self):
        client = _app(object())
        assert client.post("/umrs/companion/workers", json={}, headers=HEADERS).status_code == 404
        assert client.get("/umrs/companion/workers/a@b", headers=HEADERS).status_code == 404
        assert client.post("/umrs/companion/chat", json={}, headers=HEADERS).status_code == 404
        assert client.get("/umrs/companion/turns/at-1", headers=HEADERS).status_code == 404
