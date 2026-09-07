"""Durable approval inbox (ADR-036 amendment (a), WP-05 round 1; G-P8-1).

- store semantics: idempotent request, terminal decisions, owner filter
- durability: a decision recorded by one process instance is read by another
- facade: ``tool_approvals`` delegates to the configured store, reports durability
- routes: owner-scoped pending list, 401 for non-owners, ``always`` → mode flip
  + session-scope trust row, durability surfaced
- migration 0081 shape
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from umrs.companion_trust.approval_inbox import ApprovalInboxStore
from umrs.companion_trust.store import TrustAllowlistStore
from umrs.knowledge import tool_approvals

ACTION = {"kind": "workflow_trigger", "workflow_name": "wf-x", "params": {"a": 1}}


@pytest.fixture(autouse=True)
def _reset_facade():
    tool_approvals.configure(None)
    yield
    tool_approvals.configure(None)


class TestStore:
    def test_request_is_idempotent_and_carries_owner_job_attempt(self):
        store = ApprovalInboxStore()
        first = store.request(
            "loop-1", 2, ACTION, approval_id="appr-1", user_id="alice",
            job_id="job-1", attempt_id="att-1",
        )
        second = store.request("loop-1", 2, {"other": True}, approval_id="appr-1")
        assert first["id"] == second["id"] == "appr-1"
        assert second["action"] == ACTION  # first write wins
        assert (second["user_id"], second["job_id"], second["attempt_id"]) == (
            "alice", "job-1", "att-1",
        )
        assert second["status"] == "pending"
        assert len(store.pending()) == 1

    def test_decide_is_terminal(self):
        store = ApprovalInboxStore()
        store.request("loop-1", 1, ACTION, approval_id="appr-1")
        decided = store.decide("appr-1", "once", "alice")
        assert decided["status"] == "approved"
        assert decided["decision"] == "once"
        assert decided["decided_by"] == "alice"
        assert decided["decided_at"]
        with pytest.raises(ValueError, match="already decided"):
            store.decide("appr-1", "no", "alice")
        with pytest.raises(KeyError):
            store.decide("appr-missing", "once", "alice")
        with pytest.raises(ValueError, match="decision"):
            store.decide("appr-1", "bogus", "alice")
        assert store.pending() == []
        # rejection path
        store.request("loop-1", 2, ACTION, approval_id="appr-2")
        assert store.decide("appr-2", "no", "alice")["status"] == "rejected"

    def test_pending_is_owner_scoped_with_ownerless_visible(self):
        store = ApprovalInboxStore()
        store.request("loop-a", 1, ACTION, approval_id="a-alice", user_id="alice")
        store.request("loop-b", 1, ACTION, approval_id="b-bob", user_id="bob")
        store.request("loop-c", 1, ACTION, approval_id="c-nobody")
        assert {r["id"] for r in store.pending(user_id="alice")} == {"a-alice", "c-nobody"}
        assert {r["id"] for r in store.pending(user_id="bob")} == {"b-bob", "c-nobody"}
        assert {r["id"] for r in store.pending()} == {"a-alice", "b-bob", "c-nobody"}
        assert [r["id"] for r in store.pending(loop_id="loop-b")] == ["b-bob"]

    def test_decision_survives_process_restart(self, tmp_path: Path):
        path = str(tmp_path / "inbox.sqlite")
        first_process = ApprovalInboxStore(path=path)
        first_process.request("loop-1", 3, ACTION, approval_id="appr-1", user_id="alice")
        # "another client" decides while the first process is gone
        other_client = ApprovalInboxStore(path=path)
        other_client.decide("appr-1", "once", "alice")
        # "restarted" process reads the decision — no re-ask needed
        restarted = ApprovalInboxStore(path=path)
        record = restarted.request("loop-1", 3, ACTION, approval_id="appr-1")
        assert record["status"] == "approved"
        assert restarted.pending() == []

    def test_durability_labels(self, tmp_path: Path):
        assert ApprovalInboxStore().durability == "process"
        assert ApprovalInboxStore(path=str(tmp_path / "x.sqlite")).durability == "sqlite"


class TestFacade:
    def test_unconfigured_facade_is_process_local_and_says_so(self):
        assert tool_approvals.durability() == "process"
        tool_approvals.request_approval("loop-1", 1, ACTION, approval_id="appr-1")
        assert tool_approvals.get("appr-1")["status"] == "pending"

    def test_configure_switches_store(self, tmp_path: Path):
        store = ApprovalInboxStore(path=str(tmp_path / "inbox.sqlite"))
        tool_approvals.configure(store)
        assert tool_approvals.durability() == "sqlite"
        tool_approvals.request_approval(
            "loop-1", 1, ACTION, approval_id="appr-1", user_id="alice", job_id="job-9"
        )
        assert store.get("appr-1")["job_id"] == "job-9"
        tool_approvals.configure(None)
        assert tool_approvals.get("appr-1") is None  # fresh process-local store


def _migration(name: str) -> str:
    root = Path(__file__).resolve().parents[1] / "src" / "umrs" / "migrations"
    return (root / name).read_text(encoding="utf-8")


def test_migration_0081_shape():
    sql = _migration("0081_companion_approval_inbox.sql")
    assert "CREATE TABLE IF NOT EXISTS companion_approval_inbox" in sql
    assert "CHECK (status IN ('pending', 'approved', 'rejected'))" in sql
    assert "idx_companion_approval_inbox_pending" in sql
    for column in ("loop_id", "iteration", "user_id", "job_id", "attempt_id", "action"):
        assert column in sql


def test_schema_module_mirrors_migration_and_bumps_version():
    from umrs.companion_trust.schema import (
        COMPANION_TRUST_SCHEMA_VERSION,
        CREATE_APPROVAL_INBOX_TABLE,
    )

    assert COMPANION_TRUST_SCHEMA_VERSION == 3
    assert "companion_approval_inbox" in CREATE_APPROVAL_INBOX_TABLE


# ── routes ──────────────────────────────────────────────────────────────────


def _app(actor: str) -> TestClient:
    from umrs.api.companion import _strict_headers, router

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[_strict_headers] = lambda: actor
    return TestClient(app)


@pytest.fixture()
def trust_store():
    from umrs.api.companion import set_trust_store

    store = TrustAllowlistStore()
    set_trust_store(store)
    yield store
    set_trust_store(None)


class TestRoutes:
    def test_pending_is_owner_scoped(self):
        tool_approvals.request_approval("loop-a", 1, ACTION, approval_id="a-alice", user_id="alice")
        tool_approvals.request_approval("loop-b", 1, ACTION, approval_id="b-bob", user_id="bob")
        tool_approvals.request_approval("loop-c", 1, ACTION, approval_id="c-nobody")
        body = _app("alice").get("/umrs/companion/approvals/pending").json()
        assert {r["id"] for r in body["pending"]} == {"a-alice", "c-nobody"}
        assert body["durability"] == "process"
        narrowed = _app("alice").get("/umrs/companion/approvals/pending?loop_id=loop-a").json()
        assert [r["id"] for r in narrowed["pending"]] == ["a-alice"]

    def test_decision_by_non_owner_is_401(self):
        tool_approvals.request_approval("loop-a", 1, ACTION, approval_id="a-alice", user_id="alice")
        resp = _app("bob").post(
            "/umrs/companion/approvals/a-alice/decision", json={"decision": "once"}
        )
        assert resp.status_code == 401
        assert tool_approvals.get("a-alice")["status"] == "pending"

    def test_decision_unknown_404_and_repeat_409(self):
        client = _app("alice")
        assert client.post(
            "/umrs/companion/approvals/nope/decision", json={"decision": "once"}
        ).status_code == 404
        tool_approvals.request_approval("loop-a", 1, ACTION, approval_id="a-1", user_id="alice")
        first = client.post("/umrs/companion/approvals/a-1/decision", json={"decision": "no"})
        assert first.status_code == 200
        assert first.json()["status"] == "rejected"
        assert first.json()["durability"] == "process"
        again = client.post("/umrs/companion/approvals/a-1/decision", json={"decision": "once"})
        assert again.status_code == 409

    def test_always_flips_loop_to_auto_and_mints_session_trust(self, trust_store):
        from umrs.api import loop as loop_api
        from umrs.knowledge.loop import LoopPhase, LoopStatus, ResearchLoop

        loop_api.clear_loop_store()
        loop_api._loop_store.save(
            ResearchLoop(
                loop_id="loop-always",
                project_id="proj-a",
                research_goal="g",
                status=LoopStatus.RUNNING,
                phase=LoopPhase.EXPERIMENT_EXECUTION,
                metadata={"mode": "ask", "owner_user_id": "alice"},
            )
        )
        tool_approvals.request_approval(
            "loop-always", 1, ACTION, approval_id="a-1", user_id="alice"
        )
        resp = _app("alice").post(
            "/umrs/companion/approvals/a-1/decision", json={"decision": "always"}
        )
        assert resp.status_code == 200
        assert loop_api._loop_store.get("loop-always").metadata["mode"] == "auto"
        row = trust_store.check("alice", "wf-x", scope="session:loop-always")
        assert row is not None
        assert row.created_by == "alice"
        loop_api.clear_loop_store()
