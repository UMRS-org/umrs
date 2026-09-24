"""ADR-066 amendment 3 (D-33): the reserved emitters go live and maturity
stages become named presets of the persisted policy."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from test_api_projects import ProjectStore, _client, _Conn, _headers
from umrs.knowledge import ask_policy as ap
from umrs.knowledge.engine_loop_runner import EngineLoopRunner
from umrs.knowledge.loop import AutonomousResearchLoop, LoopIteration, LoopStatus, ResearchLoop
from umrs.knowledge.planner import ResearchExperiment, ResearchHypothesis, ResearchPlan

# ── classes per action ───────────────────────────────────────────────────────


def test_physical_workflow_trigger_is_cost_and_physical():
    assert ap.action_trigger_classes({"kind": "workflow_trigger"}) == {"cost"}
    assert ap.action_trigger_classes({"kind": "workflow_trigger", "physical": True}) == {
        "cost",
        "physical_action",
    }
    assert ap.action_trigger_classes({"kind": "direction_change"}) == {"direction_change"}


def _store(records):
    class _S:
        def get(self, loop_id):
            return records.get(loop_id)

        def save(self, loop):
            records[loop.loop_id] = loop

    return _S()


def _loop(n_hyp=2, **meta) -> ResearchLoop:
    loop = ResearchLoop(
        loop_id="L-a",
        project_id="p",
        research_goal="g",
        max_iterations=4,
        status=LoopStatus.RUNNING,
        plan=ResearchPlan(
            hypotheses=[
                ResearchHypothesis(hypothesis_id=f"H{i}", statement=f"h{i}") for i in range(n_hyp)
            ],
            experiments=[ResearchExperiment(experiment_id="E0")],
        ),
        started_at=datetime.now(timezone.utc).isoformat(),
    )
    loop.metadata.update(meta)
    return loop


def _profile_gate(monkeypatch, triggers):
    """A profile-mode runner gate whose inbox never decides (so a matching
    trigger DEFERS — observable without blocking)."""
    monkeypatch.setattr(
        "umrs.knowledge.tool_approvals.request_approval",
        lambda *_a, approval_id=None, **_k: {"id": approval_id or "appr-1"},
    )
    monkeypatch.setattr("umrs.knowledge.tool_approvals.get", lambda _id: None)
    loop = _loop(mode="profile", pause_profile={"triggers": triggers})
    runner = EngineLoopRunner(
        loop, engine_factory=lambda: None, store=_store({loop.loop_id: loop}), poll_seconds=0.01
    )
    return runner._make_approval_gate(loop)


def test_physical_only_profile_lets_plain_cost_run_and_pauses_hardware(monkeypatch):
    gate = _profile_gate(monkeypatch, ["physical_action"])
    assert gate({"kind": "workflow_trigger", "physical": False}) is True
    assert gate({"kind": "workflow_trigger", "physical": True}) == "deferred"


def test_cost_profile_still_pauses_a_physical_action(monkeypatch):
    gate = _profile_gate(monkeypatch, ["cost"])
    assert gate({"kind": "workflow_trigger", "physical": True}) == "deferred"


# ── direction_change emitter (single-branch loop) ────────────────────────────


class _Gate:
    def __init__(self, verdict):
        self.actions = []
        self._verdict = verdict

    def __call__(self, action):
        self.actions.append(action)
        return self._verdict if action["kind"] == "direction_change" else True


def _with_previous(loop, hypothesis_id):
    loop.iterations.append(
        LoopIteration(iteration_id="prev", status="completed", hypothesis_id=hypothesis_id)
    )
    loop.current_iteration = 2  # simulated rotation selects plan.hypotheses[1]


def _select(verdict, previous="H0"):
    loop = _loop()
    if previous:
        _with_previous(loop, previous)
    gate = _Gate(verdict)
    engine = AutonomousResearchLoop(simulated=True)
    engine._approval_gate = gate
    iteration = LoopIteration(iteration_id="now", loop_id=loop.loop_id)
    ok = engine._select_and_gate_hypothesis(loop, iteration, [])
    return ok, iteration, gate


def test_switching_branch_asks_and_approved_switches():
    ok, it, gate = _select(True)
    changes = [a for a in gate.actions if a["kind"] == "direction_change"]
    assert changes == [{"kind": "direction_change", "from": "H0", "to": "H1", "iteration": 2}]
    assert (ok, it.hypothesis_id) == (True, "H1")


def test_vetoed_switch_stays_on_the_previous_branch():
    ok, it, _ = _select(False)
    assert (ok, it.hypothesis_id, it.results["direction_vetoed"]) == (True, "H0", "H1")


def test_deferred_switch_finalises_the_iteration():
    ok, it, _ = _select("deferred")
    assert (ok, it.status, it.results["deferred_gate"]) == (
        False,
        "awaiting_decision",
        "direction-change",
    )


def test_no_previous_branch_no_question():
    ok, it, gate = _select(False, previous="")
    assert not [a for a in gate.actions if a["kind"] == "direction_change"]
    assert ok is True


def test_same_branch_no_question():
    ok, _, gate = _select(False, previous="H1")
    assert ok is True
    assert not [a for a in gate.actions if a["kind"] == "direction_change"]


def test_physical_experiment_flag_reaches_the_gate():
    seen = []

    class _Port:
        def trigger(self, *_a):
            raise AssertionError("must not trigger after a stopped gate")

        def cancel(self, _e):
            pass

    engine = AutonomousResearchLoop(simulated=False, workflow_port=_Port())
    engine._approval_gate = lambda action: seen.append(action) or False
    engine.request_cancel()
    loop = _loop(workflow_name="wf")
    exp = ResearchExperiment(experiment_id="E-robot", metadata={"physical_action": True})
    engine._execute_experiment(loop, LoopIteration(iteration_id="i"), exp)
    assert seen[-1]["physical"] is True


# ── maturity-stage presets ───────────────────────────────────────────────────


@pytest.mark.parametrize("stage", list(ap.STAGE_PRESETS))
def test_every_preset_round_trips(stage):
    mode, triggers = ap.stage_preset(stage)
    assert ap.stage_for(mode, triggers) == stage


def test_physical_action_is_kept_on_every_profile_preset():
    for mode, triggers in ap.STAGE_PRESETS.values():
        assert mode == "ask" or "physical_action" in triggers


def test_custom_and_unset_policies():
    assert ap.stage_for("profile", ["cost"]) == "custom"
    assert ap.stage_for("", []) == ""
    with pytest.raises(KeyError, match="unknown maturity stage"):
        ap.stage_preset("expert")


def _project():
    store = ProjectStore(_Conn())
    store.create("proj-m", owner_id="alice")
    return _client(store)


def test_route_stage_expands_to_the_preset_policy():
    r = _project().patch(
        "/umrs/projects/proj-m",
        json={"action": "set_ask_policy", "maturity_stage": "competent"},
        headers=_headers("alice"),
    )
    assert r.status_code == 200, r.text
    project = r.json()["project"]
    assert (project["ask_mode"], project["maturity_stage"]) == ("profile", "competent")
    assert set(project["pause_triggers"]) == {
        "competing_hypotheses",
        "direction_change",
        "physical_action",
    }


@pytest.mark.parametrize(
    "body",
    [
        {"action": "set_ask_policy", "maturity_stage": "competent", "ask_mode": "auto"},
        {"action": "set_ask_policy", "maturity_stage": "expert"},
    ],
)
def test_route_refuses_ambiguous_or_unknown_stage(body):
    r = _project().patch("/umrs/projects/proj-m", json=body, headers=_headers("alice"))
    assert r.status_code == 400
