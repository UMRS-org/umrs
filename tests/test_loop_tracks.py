"""D-34 second leg (ADR-066 amendment 2): parallel hypothesis tracks.

Iterations serve active tracks round-robin; the competing-hypotheses
question is raised only when there is a choice; a rejected track closes
without failing the loop; per-track approval slots and deferral ledgers
never interfere; a stopped loop at the gate never triggers its workflow.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from umrs.knowledge import loop_tracks
from umrs.knowledge.engine_loop_runner import (
    EngineLoopRunner,
    _awaiting_get,
    _awaiting_set,
)
from umrs.knowledge.loop import (
    AutonomousResearchLoop,
    LoopIteration,
    LoopStatus,
    ResearchLoop,
)
from umrs.knowledge.planner import ResearchExperiment, ResearchHypothesis, ResearchPlan


def _loop(n_hyp=3, n_exp=2, **meta) -> ResearchLoop:
    plan = ResearchPlan(
        plan_id="P",
        hypotheses=[
            ResearchHypothesis(hypothesis_id=f"H{i}", statement=f"hypothesis {i}")
            for i in range(n_hyp)
        ],
        experiments=[
            ResearchExperiment(experiment_id=f"E{i}", parameters={}) for i in range(n_exp)
        ],
    )
    loop = ResearchLoop(
        loop_id="L-t",
        project_id="proj-t",
        research_goal="g",
        max_iterations=6,
        status=LoopStatus.RUNNING,
        plan=plan,
        started_at=datetime.now(timezone.utc).isoformat(),
    )
    loop.metadata.update(meta)
    return loop


def _engine(gate=None) -> AutonomousResearchLoop:
    engine = AutonomousResearchLoop(simulated=True)
    engine._approval_gate = gate
    return engine


class _Gate:
    def __init__(self, verdict=True):
        self.actions: list[dict] = []
        self._verdict = verdict

    def __call__(self, action):
        self.actions.append(action)
        return self._verdict(action) if callable(self._verdict) else self._verdict


# ── scheduling ───────────────────────────────────────────────────────────────


def test_max_tracks_one_is_the_single_branch_loop():
    loop = _loop()
    assert not loop_tracks.tracks_enabled(loop)
    _engine(_Gate()).run(loop)
    assert "tracks" not in loop.metadata
    assert all("track_id" not in it.results for it in loop.iterations)


def test_iterations_serve_tracks_round_robin_with_own_experiment_sequences():
    loop = _loop(n_hyp=2, n_exp=2, max_tracks=2)
    _engine(_Gate()).run(loop)
    served = [
        (it.results["track_id"], it.hypothesis_id, it.experiment_id) for it in loop.iterations
    ]
    assert served == [
        ("track-1", "H0", "E0"),
        ("track-2", "H1", "E0"),
        ("track-1", "H0", "E1"),
        ("track-2", "H1", "E1"),
        ("track-1", "H0", "E0"),
        ("track-2", "H1", "E0"),
    ]
    assert [t["turns"] for t in loop.metadata["tracks"]] == [3, 3]


def test_no_competing_question_when_every_candidate_gets_a_track():
    gate = _Gate()
    _engine(gate).run(_loop(n_hyp=2, max_tracks=2))
    assert not [a for a in gate.actions if a["kind"] == "competing_hypotheses"]


def test_competing_question_asked_once_when_there_is_a_choice():
    gate = _Gate()
    loop = _loop(n_hyp=3, max_tracks=2)
    _engine(gate).run(loop)
    asked = [a for a in gate.actions if a["kind"] == "competing_hypotheses"]
    assert len(asked) == 1 and asked[0]["candidates"] == 3
    assert [t["hypothesis_id"] for t in loop.metadata["tracks"]] == ["H0", "H1"]


def test_ranking_prefers_evidence_overlap():
    loop = _loop(n_hyp=3)
    loop.plan.hypotheses[2].statement = "carbon capture uptake"
    engine = AutonomousResearchLoop(simulated=False)
    ranked = engine._rank_hypotheses(loop, ["measured carbon capture uptake rose"])
    assert ranked[0].hypothesis_id == "H2"


def test_track_budget_is_clamped():
    assert loop_tracks.max_tracks(_loop(max_tracks=9)) == loop_tracks.MAX_TRACKS_LIMIT
    assert loop_tracks.max_tracks(_loop(max_tracks="x")) == 1


# ── rejection closes a track, not the loop ────────────────────────────────────


def test_rejected_track_closes_and_the_loop_continues_then_completes():
    loop = _loop(n_hyp=2, max_tracks=2)
    engine = _engine(_Gate())
    loop_tracks.open_tracks(loop, list(loop.plan.hypotheses))
    engine._track_id = "track-1"
    iteration = LoopIteration(iteration_id="i1", loop_id=loop.loop_id, iteration_number=1)
    engine._gate_denied(loop, iteration, "wf")  # decided "no" on track-1
    assert iteration.status == "track_rejected"
    assert [t["status"] for t in loop.metadata["tracks"]] == ["rejected", "active"]
    assert loop.status == LoopStatus.RUNNING

    engine._track_id = "track-2"
    engine._gate_denied(loop, LoopIteration(iteration_id="i2", loop_id="L-t"), "wf")
    ok, track = engine._hypothesis_phase(loop, LoopIteration(iteration_id="i3"), [])
    assert (ok, track) == (False, None)
    assert loop.metadata["tracks_exhausted"] is True
    assert loop.status == LoopStatus.STOPPED


def test_single_branch_denial_still_fails_the_loop():
    loop = _loop()
    engine = _engine(_Gate())
    with pytest.raises(RuntimeError, match="operator rejected"):
        engine._gate_denied(loop, LoopIteration(iteration_id="i"), "wf")


# ── stop at the gate never triggers the workflow (pre-existing defect) ──────


class _Port:
    def __init__(self):
        self.triggered = []

    def trigger(self, name, _params):
        self.triggered.append(name)
        return "exec-1"

    def cancel(self, _execution_id):
        pass


def test_stopped_loop_at_the_gate_never_triggers():
    port = _Port()
    engine = AutonomousResearchLoop(simulated=False, workflow_port=port)
    engine._approval_gate = _Gate(verdict=False)
    engine.request_cancel()
    loop = _loop(workflow_name="wf")
    iteration = LoopIteration(iteration_id="i", loop_id=loop.loop_id)
    outcome = engine._execute_experiment(loop, iteration, loop.plan.experiments[0])
    assert outcome == "halted"
    assert port.triggered == []
    assert loop.status == LoopStatus.STOPPED


def test_workflow_gate_action_carries_the_track():
    gate = _Gate(verdict=False)
    port = _Port()
    engine = AutonomousResearchLoop(simulated=False, workflow_port=port)
    engine._approval_gate = gate
    loop = _loop(n_hyp=2, max_tracks=2, workflow_name="wf")
    loop_tracks.open_tracks(loop, list(loop.plan.hypotheses))
    engine._track_id = "track-2"
    iteration = LoopIteration(iteration_id="i", loop_id=loop.loop_id)
    assert engine._execute_experiment(loop, iteration, loop.plan.experiments[0]) == "halted"
    assert gate.actions[-1]["track_id"] == "track-2"
    assert port.triggered == []  # rejected track: no trigger
    assert iteration.status == "track_rejected"


# ── per-track approval slots and ledgers ─────────────────────────────────────


def _store(records):
    class _S:
        def get(self, loop_id):
            return records.get(loop_id)

        def save(self, loop):
            records[loop.loop_id] = loop

    return _S()


def test_awaiting_slots_are_per_track_and_legacy_keys_unchanged():
    loop = _loop()
    _awaiting_set(loop, "", "appr-legacy")
    _awaiting_set(loop, "track-1", "appr-1")
    _awaiting_set(loop, "track-2", "appr-2")
    assert loop.metadata["awaiting_approval"] == "appr-legacy"
    assert (_awaiting_get(loop, "track-1"), _awaiting_get(loop, "track-2")) == ("appr-1", "appr-2")
    _awaiting_set(loop, "track-1", None)
    assert _awaiting_get(loop, "track-1") is None
    assert _awaiting_get(loop, "track-2") == "appr-2"


def test_interleaved_tracks_never_reset_each_others_strikes():
    loop = _loop(n_hyp=2, max_tracks=2)
    runner = EngineLoopRunner(
        loop, engine_factory=lambda: None, store=_store({loop.loop_id: loop}), poll_seconds=0.01
    )
    a, b = {"id": "appr-a"}, {"id": "appr-b"}
    parked = [
        runner._defer_strike(loop, a, "track-1"),
        runner._defer_strike(loop, b, "track-2"),
        runner._defer_strike(loop, a, "track-1"),
        runner._defer_strike(loop, b, "track-2"),
        runner._defer_strike(loop, a, "track-1"),
    ]
    # track-1's third consecutive deferral parks despite track-2 in between
    assert parked == [False, False, False, False, True]
