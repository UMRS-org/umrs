"""UMRS Companion HTTP surface tests (wave T3b m5)."""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fake_pg_pool import FakePool  # noqa: E402
from umrs.api.companion import router  # noqa: E402

from r4_contracts.handshake_hmac import sign_claims  # noqa: E402

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


def _app(pool) -> TestClient:
    import os

    os.environ.setdefault("UMRS_AUTH_MODE", "strict")
    os.environ.setdefault("UMRS_HANDSHAKE_HMAC_SECRET", SECRET)
    app = FastAPI()
    app.include_router(router)
    app.state.umrs_db_pool = pool
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


class TestWorkerRoutes:
    def test_register_worker_persists_and_returns_row(self):
        client = _app(FakePool([None, _worker_row()]))
        resp = client.post(
            "/umrs/companion/workers",
            headers=HEADERS,
            json={"owner_user_id": "alice", "project_id": "default", "job_id": "job-w1"},
        )
        assert resp.status_code == 201
        assert resp.json()["registry_key"] == "alice@default"
        assert resp.json()["state"] == "active"

    def test_get_worker_404(self):
        client = _app(FakePool([None]))
        resp = client.get("/umrs/companion/workers/ghost@default", headers=HEADERS)
        assert resp.status_code == 404

    def test_get_worker_ownership_mismatch_rejected(self):
        client = _app(FakePool([_worker_row()]))
        resp = client.get("/umrs/companion/workers/alice@default", headers=OTHER_HEADERS)
        assert resp.status_code == 401

    def test_get_worker_returns_row(self):
        client = _app(FakePool([_worker_row()]))
        resp = client.get("/umrs/companion/workers/alice@default", headers=HEADERS)
        assert resp.status_code == 200
        assert resp.json()["attempts_max"] == 6


class TestTurnRoutes:
    def test_begin_turn_attempt_returns_accepted_naive(self):
        pool = FakePool([_worker_row(), None, _turn_row()])
        client = _app(pool)
        resp = client.post(
            "/umrs/companion/chat",
            headers=HEADERS,
            json={
                "registry_key": "alice@default",
                "message": "hello world",
                "request_token": "rt-1",
            },
        )
        assert resp.status_code == 202
        assert resp.json()["attempt_id"] == "at-1"
        assert resp.json()["state"] == "in_flight"

    def test_begin_turn_attempt_idempotent_by_token(self):
        pool = FakePool([_worker_row(), None, _turn_row()])
        client = _app(pool)
        resp = client.post(
            "/umrs/companion/chat",
            headers=HEADERS,
            json={"registry_key": "alice@default", "message": "re-send", "request_token": "rt-1"},
        )
        assert resp.status_code == 202
        assert resp.json()["attempt_id"] == "at-1"

    def test_get_turn_attempt_404(self):
        client = _app(FakePool([None]))
        resp = client.get("/umrs/companion/turns/at-unknown", headers=HEADERS)
        assert resp.status_code == 404

    def test_get_turn_attempt_returns_row(self):
        client = _app(FakePool([_turn_row()]))
        resp = client.get("/umrs/companion/turns/at-1", headers=HEADERS)
        assert resp.status_code == 200
        assert resp.json()["request_token"] == "rt-1"


class TestTurnEvents:
    def test_committed_turn_streams_result(self):
        rows = [
            _turn_row(state="committed", committed_response_ref="resp-1"),
        ]
        client = _app(FakePool(list(rows)))
        resp = client.get("/umrs/companion/turns/at-1/events", headers=HEADERS)
        assert resp.status_code == 200
        assert '"type": "result"' in resp.text
        assert "[DONE]" in resp.text

    def test_failed_turn_emits_terminal_error(self):
        client = _app(FakePool([_turn_row(state="failed")]))
        resp = client.get("/umrs/companion/turns/at-1/events", headers=HEADERS)
        assert '"type": "attempt_failed"' in resp.text

    def test_in_flight_polls_then_times_out_cleanly(self):
        rows = [_turn_row(state="in_flight")] * 90  # 25s timeout reaches
        client = _app(FakePool(rows))
        resp = client.get("/umrs/companion/turns/at-1/events", headers=HEADERS)
        assert '"type": "in_flight"' in resp.text


class TestApprovalSurface:
    def test_pending_approvals_scaffold_is_empty(self):
        client = _app(FakePool([]))
        resp = client.get("/umrs/companion/approvals/pending", headers=HEADERS)
        assert resp.status_code == 200
        assert resp.json() == {"pending": []}

    def test_decision_scaffold_echoes_status(self):
        client = _app(FakePool([]))
        resp = client.post(
            "/umrs/companion/approvals/ap-1/decision",
            headers=HEADERS,
            json={"decision": "once"},
        )
        assert resp.json()["approval_id"] == "ap-1"
        assert resp.json()["decision"] == "once"


class TestAuthEnvelope:
    def test_missing_identity_rejected(self):
        client = _app(FakePool([]))
        resp = client.get("/umrs/companion/workers/x@y", headers={})
        assert resp.status_code in (401, 403)
