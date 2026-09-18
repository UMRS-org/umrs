"""Pins for durable chat-turn approvals (ADR-057 D-3 leg 3)."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from umrs.api.companion_chat_approvals import router
from umrs.companion_trust.chat_turn_approvals import ChatTurnApprovalsStore


class TestStore:
    def test_request_get_pending_cycle(self):
        store = ChatTurnApprovalsStore()
        rec = store.request("s1", "t2", {"tool": "shell_exec", "cmd": "ls"}, user_id="alice")
        assert rec["status"] == "pending"
        got = store.get(rec["approval_id"])
        assert got["action"]["tool"] == "shell_exec"
        pend = store.pending(session_id="s1", user_id="alice")
        assert [p["approval_id"] for p in pend] == [rec["approval_id"]]

    def test_request_is_idempotent_with_given_id(self):
        store = ChatTurnApprovalsStore()
        r1 = store.request("s1", "t1", {"x": 1}, approval_id="appr-fixed")
        r2 = store.request("s1", "t1", {"x": 1}, approval_id="appr-fixed")
        assert r1["approval_id"] == r2["approval_id"]

    def test_decide_once_is_terminal(self):
        store = ChatTurnApprovalsStore()
        rec = store.request("s1", "t1", {}, user_id="alice")
        decided = store.decide(rec["approval_id"], "once", decided_by="alice")
        assert decided["status"] == "approved"
        assert decided["decision"] == "once"
        with pytest.raises(ValueError, match="already decided"):
            store.decide(rec["approval_id"], "always", decided_by="alice")

    def test_decide_reject_path(self):
        store = ChatTurnApprovalsStore()
        rec = store.request("s1", "t1", {}, user_id="alice")
        decided = store.decide(rec["approval_id"], "reject", decided_by="alice")
        assert decided["status"] == "rejected"
        decided_rows = store.pending_decided(session_id="s1", status="rejected")
        assert len(decided_rows) == 1

    def test_decide_unknown_raises_named_error(self):
        store = ChatTurnApprovalsStore()
        with pytest.raises(KeyError, match="appr-nope"):
            store.decide("appr-nope", "once", decided_by="alice")

    def test_invalid_decision_raises(self):
        store = ChatTurnApprovalsStore()
        rec = store.request("s1", "t1", {}, user_id="alice")
        with pytest.raises(ValueError, match="decision must be one of"):
            store.decide(rec["approval_id"], "dunno", decided_by="alice")


# ── Route layer (owner-scoped via snapshot presence; sqlite store) ──────────


class _OwnerConn:
    """Answers only the owner-check query; everything else unused by tests."""

    def __init__(self, owners: dict[str, str]):
        self._owners = owners

    def cursor(self):
        return self

    def execute(self, sql, params):
        sql_l = sql.lower()
        assert sql_l.startswith("select 1 from companion_session_snapshots")
        sid, uid = params
        self._hit = (sid, uid) if self._owners.get(sid) == uid else None

    def fetchone(self):
        return self._hit


class _OwnerPool:
    def __init__(self, owners: dict[str, str]):
        self._owners = owners

    def getconn(self):
        return _OwnerConn(self._owners)

    def putconn(self, _conn):
        return None


def _client(owners=None, store=None) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    app.state.umrs_db_pool = _OwnerPool(owners or {"s1": "alice"})
    monkey_client = TestClient(app)
    monkey_client.store = store or ChatTurnApprovalsStore()
    return monkey_client


@pytest.fixture(autouse=True)
def _sqlite_store(monkeypatch):
    """One shared sqlite store per test (the route layer constructs it per
    request; :memory: would otherwise reset between calls)."""
    shared = ChatTurnApprovalsStore()

    def _factory(conn=None):
        return shared

    monkeypatch.setattr(
        "umrs.api.companion_chat_approvals.ChatTurnApprovalsStore", _factory
    )
    return {"shared": shared}


def _raise(client, sid="s1", uid="alice", **over):
    body = {"turn_id": "t3", "action": {"tool": "file_write"}}
    body.update(over)
    return client.post(
        f"/umrs/companion/sessions/{sid}/approvals", json=body, headers={"x-user-id": uid}
    )


class TestRouteOwnerScope:
    def test_raise_then_owner_lists_pending(self, _sqlite_store):
        client = _client()
        r = _raise(client)
        assert r.status_code == 200
        aid = r.json()["approval"]["approval_id"]
        resp = client.get(
            "/umrs/companion/sessions/s1/approvals", headers={"x-user-id": "alice"}
        )
        rows = resp.json()["approvals"]
        assert [row["approval_id"] for row in rows] == [aid]

    def test_non_session_owner_cannot_raise(self):
        client = _client()
        resp = _raise(client, uid="bob")
        assert resp.status_code == 404

    def test_decide_once_then_list_empty_pending(self, _sqlite_store):
        client = _client()
        r = _raise(client)
        aid = r.json()["approval"]["approval_id"]
        d = client.post(
            f"/umrs/companion/sessions/s1/approvals/{aid}/decision",
            json={"decision": "once"},
            headers={"x-user-id": "alice"},
        )
        assert d.status_code == 200
        assert d.json()["approval"]["status"] == "approved"
        pend = client.get(
            "/umrs/companion/sessions/s1/approvals", headers={"x-user-id": "alice"}
        )
        assert pend.json()["count"] == 0

    def test_invalid_decision_is_400(self, _sqlite_store):
        client = _client()
        r = _raise(client)
        aid = r.json()["approval"]["approval_id"]
        resp = client.post(
            f"/umrs/companion/sessions/s1/approvals/{aid}/decision",
            json={"decision": "always-ish"},
            headers={"x-user-id": "alice"},
        )
        assert resp.status_code == 400

    def test_decision_already_taken_is_409(self, _sqlite_store):
        client = _client()
        r = _raise(client)
        aid = r.json()["approval"]["approval_id"]
        for decision in ("once", "reject"):
            resp = client.post(
                f"/umrs/companion/sessions/s1/approvals/{aid}/decision",
                json={"decision": decision},
                headers={"x-user-id": "alice"},
            )
        assert resp.status_code == 409
