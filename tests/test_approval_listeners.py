"""U3 S3e: approval listeners through ``add_approval_listener`` (decision D7).

The core decision route flips an ``always`` loop to auto (D1: loops are core)
and calls the registered listeners; the in-tree r4 part registers the
``session:<loop_id>`` trust mint (ADR-036 §3).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from umrs.extensions import load_extensions, loader, reset_extension_context
from umrs.knowledge import tool_approvals

APPROVALS_SRC = Path(__file__).resolve().parents[1] / "src" / "umrs" / "api" / "approvals.py"
ACTION = {"kind": "workflow_trigger", "workflow_name": "wf-x"}


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    monkeypatch.delenv(loader.ENV, raising=False)
    reset_extension_context()
    tool_approvals.configure(None)
    yield
    tool_approvals.configure(None)
    reset_extension_context()


def _client(actor: str) -> TestClient:
    from umrs.api.companion import _strict_headers, router

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[_strict_headers] = lambda: actor
    return TestClient(app)


def test_core_decision_route_no_longer_knows_the_trust_store():
    # Fails on the pre-S3e route, which minted the trust row itself.
    assert "trust_store_access" not in APPROVALS_SRC.read_text(encoding="utf-8")


def test_r4_registers_the_trust_mint():
    from umrs_r4.trust.approval_listener import mint_session_trust_on_always

    assert load_extensions(("r4",)).approval_listeners == [mint_session_trust_on_always]
    assert load_extensions(()).approval_listeners == []


def test_listeners_hear_every_decision_and_one_failure_is_isolated(monkeypatch):
    heard: list[tuple[str, str, str]] = []

    def boom(_record, _decision, _user):
        raise RuntimeError("listener down")

    ctx = load_extensions(())
    ctx.approval_listeners.extend([boom, lambda r, d, u: heard.append((r["id"], d, u))])
    reset_extension_context(ctx)
    tool_approvals.request_approval("loop-a", 1, ACTION, approval_id="a-1", user_id="alice")
    resp = _client("alice").post("/umrs/companion/approvals/a-1/decision", json={"decision": "no"})
    assert resp.status_code == 200
    assert heard == [("a-1", "no", "alice")]


def test_without_r4_always_mints_no_trust_row(monkeypatch):
    from umrs_r4.trust.access import set_trust_store
    from umrs_r4.trust.store import TrustAllowlistStore

    monkeypatch.setenv(loader.ENV, "none")
    reset_extension_context()
    trust = TrustAllowlistStore()
    set_trust_store(trust)
    try:
        tool_approvals.request_approval("loop-n", 1, ACTION, approval_id="a-2", user_id="alice")
        resp = _client("alice").post(
            "/umrs/companion/approvals/a-2/decision", json={"decision": "always"}
        )
        assert resp.status_code == 200
        assert trust.check("alice", "wf-x", scope="session:loop-n") is None
    finally:
        set_trust_store(None)


def test_the_trust_mint_ignores_other_decisions():
    from umrs_r4.trust.approval_listener import mint_session_trust_on_always

    # No trust store, no loop: must simply return for once/no.
    mint_session_trust_on_always({"loop_id": "l"}, "once", "alice")
    mint_session_trust_on_always({}, "always", "alice")
