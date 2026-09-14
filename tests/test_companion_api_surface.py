"""UMRS Companion HTTP surface tests (wave T3b m5)."""

from __future__ import annotations

import pytest

# splice removed (ADR-053): pytest auto-inserts tests/ for sibling helpers; umrs editable
from fake_pg_pool import FakePool  # noqa: E402
from fastapi import FastAPI
from fastapi.testclient import TestClient
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

    def test_register_worker_submits_venue_job_when_job_id_omitted(self):
        pool = FakePool([None, _worker_row(job_id="job-cw-1")])
        client = _app(pool)
        submitted: list[tuple] = []

        async def _submitter(owner: str, project: str) -> str:
            submitted.append((owner, project))
            return "job-cw-1"

        client.app.state.companion_job_submitter = _submitter
        resp = client.post(
            "/umrs/companion/workers",
            headers=HEADERS,
            json={"owner_user_id": "alice", "project_id": "default"},
        )
        assert resp.status_code == 201
        assert resp.json()["job_id"] == "job-cw-1"
        assert submitted == [("alice", "default")]

    def test_get_worker_404(self):
        client = _app(FakePool([None]))
        resp = client.get("/umrs/companion/workers/ghost@default", headers=HEADERS)
        assert resp.status_code == 404

    def test_get_worker_ownership_mismatch_rejected(self):
        client = _app(FakePool([_worker_row()]), actor="bob")
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

        tool_approvals.request_approval(
            "loop-x", 1, {"kind": "workflow_trigger"}, approval_id="ap-1"
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
            "loop-y", 1, {"kind": "workflow_trigger"}, approval_id="ap-2"
        )
        client = _app(FakePool([]))
        resp = client.get("/umrs/companion/approvals/pending", headers=HEADERS)
        assert resp.status_code == 200
        assert resp.json()["pending"][0]["id"] == "ap-2"


class TestTurnRoundtripSurface:
    """T3c: mode capture, executor kickoff seam, and replay content shape."""

    def test_begin_rejects_invalid_mode(self):
        client = _app(FakePool([_worker_row()]))
        resp = client.post(
            "/umrs/companion/chat",
            headers=HEADERS,
            json={"registry_key": "alice@default", "message": "hi", "mode": "yolo"},
        )
        assert resp.status_code == 422

    def test_begin_accepts_mode_and_kicks_executor(self):
        pool = FakePool([_worker_row(), None, _turn_row(mode="auto")])
        client = _app(pool)
        client.app.state.companion_turn_chat = lambda *_a, **_k: "pong"
        resp = client.post(
            "/umrs/companion/chat",
            headers=HEADERS,
            json={"registry_key": "alice@default", "message": "ping", "mode": "auto"},
        )
        assert resp.status_code == 202
        insert_params = next(
            p for s, p in pool.record if s.startswith("INSERT INTO companion_turns")
        )
        assert "ping" in insert_params
        assert "auto" in insert_params

    def test_committed_turn_stream_carries_response_content(self):
        client = _app(
            FakePool(
                [
                    _turn_row(
                        state="committed",
                        committed_response_ref="attempt:at-1",
                        response_content="the answer",
                    )
                ]
            )
        )
        resp = client.get("/umrs/companion/turns/at-1/events", headers=HEADERS)
        assert '"type": "result"' in resp.text
        assert "the answer" in resp.text
        assert "[DONE]" in resp.text

    def test_turn_row_response_includes_content_and_mode(self):
        client = _app(FakePool([_turn_row(mode="auto", response_content="body")]))
        resp = client.get("/umrs/companion/turns/at-1", headers=HEADERS)
        assert resp.status_code == 200
        assert resp.json()["mode"] == "auto"
        assert resp.json()["response_content"] == "body"

    def test_stale_in_flight_attempt_reaped_at_stream_open(self, monkeypatch):
        from umrs.api import companion as api_companion

        monkeypatch.setattr(
            api_companion.CompanionTurnAttemptRegistry,
            "fail_if_stale",
            lambda *_a, **_k: True,
        )
        client = _app(FakePool([_turn_row(state="in_flight")]))
        resp = client.get("/umrs/companion/turns/at-1/events", headers=HEADERS)
        assert "stale_reaped" in resp.text
        assert '"type": "attempt_failed"' in resp.text


# Ownership enforcement stays full-strength (covered in TestWorkerRoutes:
# cross-user ownership read => 401 via _enforce_ownership, independent of
# the sidecar header gate which is exercised by production strict mode).
