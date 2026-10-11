"""ADR-092 slice 6 (D-29 tail, D-33): a team-plan run is an ask-policy
action classed ``cost`` before any loop runs one, and a server-hosted loop
refuses a plan iteration by name until slice 8's server venue."""

from __future__ import annotations

import pytest
from test_ask_policy_emitters import _loop, _store

from umrs.knowledge import ask_policy as ap
from umrs.knowledge.engine_loop_runner import EngineLoopRunner

# ── the action class ─────────────────────────────────────────────────────────


def test_ask_policy_reads_the_shared_action_table():
    # The table is pinned once, in r4-contracts (test_ask_policy.py).
    from r4_contracts.ask_policy import ACTION_KIND_CLASSES, PLAN_RUN

    assert ap.PLAN_RUN == PLAN_RUN
    assert set(ap.COST_ACTION_KINDS) == set(ACTION_KIND_CLASSES)
    for kind, classes in ACTION_KIND_CLASSES.items():
        assert ap.action_trigger_classes({"kind": kind}) == classes
        assert ap.action_trigger_classes({"kind": kind, "physical": True}) == (
            classes | {"physical_action"}
        )
    assert ap.action_trigger_classes({"kind": "plan_gate"}) == frozenset()


# ── the engine driver's gate parks at a plan run ─────────────────────────────


def _gate(monkeypatch, decided=None, **meta):
    """The runner's gate over a fake inbox; ``requested`` records every park."""
    requested: list[dict] = []

    def _request(_loop_id, _iteration, action, approval_id=None, **_kw):
        requested.append(action)
        return {"id": approval_id or "appr-plan"}

    monkeypatch.setattr("umrs.knowledge.tool_approvals.request_approval", _request)
    monkeypatch.setattr(
        "umrs.knowledge.tool_approvals.get",
        lambda _id: {"status": decided} if decided else None,
    )
    loop = _loop(**meta)
    runner = EngineLoopRunner(
        loop, engine_factory=lambda: None, store=_store({loop.loop_id: loop}), poll_seconds=0.01
    )
    return runner._make_approval_gate(loop), requested


def test_ask_mode_parks_at_a_plan_run_until_a_human_decides(monkeypatch):
    gate, requested = _gate(monkeypatch, decided="approved", mode="ask")
    assert gate({"kind": "plan_run", "template": "research-round"}) is True
    assert [a["kind"] for a in requested] == ["plan_run"]


def test_ask_mode_plan_run_rejected_by_the_human(monkeypatch):
    gate, requested = _gate(monkeypatch, decided="rejected", mode="ask")
    assert gate({"kind": "plan_run"}) is False
    assert len(requested) == 1


def test_a_cost_profile_asks_about_a_plan_run(monkeypatch):
    gate, requested = _gate(monkeypatch, mode="profile", pause_profile={"triggers": ["cost"]})
    assert gate({"kind": "plan_run"}) == "deferred"
    assert len(requested) == 1


def test_a_profile_without_cost_lets_a_plan_run_go(monkeypatch):
    gate, requested = _gate(
        monkeypatch, mode="profile", pause_profile={"triggers": ["physical_action"]}
    )
    assert gate({"kind": "plan_run"}) is True
    assert requested == []


def test_auto_mode_never_asks(monkeypatch):
    gate, requested = _gate(monkeypatch, mode="auto")
    assert gate({"kind": "plan_run"}) is True
    assert requested == []


# ── the server venue refuses a plan iteration by name ────────────────────────


def test_the_hosting_job_spec_refuses_a_plan_iteration():
    from umrs.jobs.research_loop_backend import (
        PLAN_ITERATION_REFUSAL,
        validate_research_loop_spec,
    )

    with pytest.raises(ValueError) as exc:
        validate_research_loop_spec({"loop_id": "L", "iteration_plan": "research-round"})
    assert str(exc.value) == PLAN_ITERATION_REFUSAL
    assert "ADR-092 slice 8" in PLAN_ITERATION_REFUSAL
    assert "plan_iteration_needs_server_venue" in PLAN_ITERATION_REFUSAL
    assert validate_research_loop_spec({"loop_id": "L", "iteration_plan": ""}) == {
        "loop_id": "L",
        "driver": "engine",
    }


@pytest.fixture
def _api():
    pytest.importorskip("fastapi")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from umrs.api import loop as loop_api
    from umrs.api.loop import clear_loop_store, router
    from umrs.knowledge.loop import AutonomousResearchLoop

    clear_loop_store()
    previous = loop_api._loop_engine
    loop_api._loop_engine = AutonomousResearchLoop(simulated=True)
    app = FastAPI()
    app.include_router(router)
    yield TestClient(app), loop_api
    loop_api._loop_engine = previous
    clear_loop_store()


def test_the_start_route_refuses_a_plan_iteration_before_creating_anything(_api):
    client, loop_api = _api
    resp = client.post(
        "/umrs/loops",
        json={
            "project_id": "proj-plan",
            "research_goal": "a plan per iteration",
            "iteration_plan": "research-round",
        },
    )
    assert resp.status_code == 422, resp.text
    body = resp.text
    assert "plan_iteration_needs_server_venue" in body
    assert "ADR-092 slice 8" in body
    assert loop_api._loop_store.list_all() == []


def test_a_start_without_a_plan_is_unchanged(_api):
    client, _loop_api = _api
    resp = client.post(
        "/umrs/loops",
        json={"project_id": "proj-plain", "research_goal": "plain", "iteration_plan": ""},
    )
    assert resp.status_code == 201, resp.text
