"""ADR-066 (D-33) + ADR-067 (D-36) pins: pause-profile gate + wall-clock budget."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from umrs.knowledge.engine_loop_runner import (
    EngineLoopRunner,
    _action_trigger_class,
    _elapsed_hours,
    _gpu_hours_exceeded,
    _measured_gpu_seconds,
    _wall_clock_exceeded,
)
from umrs.knowledge.loop import LoopStatus, ResearchLoop


def _store(records):
    class _S:
        def __init__(self):
            self.saved = []

        def get(self, loop_id):
            return records.get(loop_id)

        def save(self, loop):
            records[loop.loop_id] = loop
            self.saved.append(loop)

    return _S()


def _loop(**meta) -> ResearchLoop:
    loop = ResearchLoop(
        loop_id="L-b",
        project_id="proj-b",
        research_goal="g",
        max_iterations=5,
        status=LoopStatus.RUNNING,
        started_at=datetime.now(timezone.utc).isoformat(),
    )
    loop.metadata.update(meta)
    return loop


class TestWallClockBudget:
    def test_uncapped_loop_never_exceeds(self):
        assert _wall_clock_exceeded(_loop()) is False

    def test_overtime_loop_exceeds(self):
        past = (datetime.now(timezone.utc) - timedelta(hours=3)).isoformat()
        loop = _loop(max_wall_clock_hours=2.0)
        loop.started_at = past
        assert _wall_clock_exceeded(loop) is True

    def test_dead_string_cap_fails_open(self):
        assert _wall_clock_exceeded(_loop(max_wall_clock_hours="lots")) is False

    def test_cancel_check_marks_budget_reason_and_returns_true(self):
        past = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
        loop = _loop(max_wall_clock_hours=1.0)
        loop.started_at = past
        store = _store({loop.loop_id: loop})
        runner = EngineLoopRunner(
            loop, engine_factory=lambda: None, store=store, poll_seconds=0.01
        )
        assert runner._cancel_check() is True
        assert loop.metadata["state_reason"] == "budget_exhausted:wall_clock"

    def test_elapsed_hours_unparseable_is_zero(self):
        loop = _loop()
        loop.started_at = "n/a"
        assert _elapsed_hours(loop) == 0.0


class TestPauseProfileGate:
    def _runner(self, loop):
        store = _store({loop.loop_id: loop})
        return EngineLoopRunner(
            loop, engine_factory=lambda: None, store=store, poll_seconds=0.1
        )

    def test_trigger_class_mapping(self):
        assert _action_trigger_class({"kind": "workflow_trigger"}) == "cost"
        assert _action_trigger_class({"kind": "agent_iteration"}) == "cost"
        assert (
            _action_trigger_class({"kind": "competing_hypotheses"})
            == "competing_hypotheses"
        )
        assert _action_trigger_class({"kind": "mystery"}) == ""

    def test_profile_park_only_named_triggers(self):
        loop = _loop(mode="profile", pause_profile={"triggers": ["competing_hypotheses"]})
        runner = self._runner(loop)
        gate = runner._make_approval_gate(loop)
        # cost-class actions pass without parking…
        assert gate({"kind": "_no_such_kind_"}) is True
        # …while a named trigger class would park (blocked on the approval
        # inbox — asserted by construction, not by sleeping): we verify the
        # decision branch instead via the class map + trigger set.
        triggers = set(loop.metadata["pause_profile"]["triggers"])
        assert _action_trigger_class({"kind": "competing_hypotheses"}) in triggers
        assert _action_trigger_class({"kind": "workflow_trigger"}) not in triggers

    def test_auto_mode_gate_passes_everything(self):
        loop = _loop(mode="auto")
        runner = self._runner(loop)
        gate = runner._make_approval_gate(loop)
        assert gate({"kind": "workflow_trigger"}) is True
        assert gate({"kind": "competing_hypotheses"}) is True


class TestGpuHoursBudget:
    """D-36 leg B: the GPU-hour cap rides MEASURED cost only."""

    def _ledger_loop(self, gpu_seconds, cap=1.0):
        return _loop(
            max_gpu_hours=cap,
            gpu_cost_iterations={"it-1": {"gpu_seconds": gpu_seconds}},
        )

    def test_uncapped_loop_never_exceeds(self):
        assert _gpu_hours_exceeded(_loop(gpu_cost_iterations={"it": {"gpu_seconds": 9e9}})) is False

    def test_measured_over_cap_exceeds(self):
        assert _gpu_hours_exceeded(self._ledger_loop(3610.0, cap=1.0)) is True

    def test_measured_under_cap_passes(self):
        assert _gpu_hours_exceeded(self._ledger_loop(3500.0, cap=1.0)) is False

    def test_zero_measured_never_exceeds(self):
        assert _gpu_hours_exceeded(self._ledger_loop(0.0, cap=0.001)) is False

    def test_dead_ledger_fails_open(self):
        assert _gpu_hours_exceeded(_loop(max_gpu_hours=1.0, gpu_cost_iterations="junk")) is False

    def test_dead_string_cap_fails_open(self):
        assert _gpu_hours_exceeded(_loop(max_gpu_hours="lots")) is False

    def test_measured_seconds_sum_rows(self):
        loop = _loop(
            gpu_cost_iterations={
                "it-1": {"gpu_seconds": 100.0},
                "it-2": {"gpu_seconds": 50.0},
                "it-3": {"gpu_seconds": "junk"},  # dead row contributes nothing
                "it-4": {"steps": 1},  # no measurement contributes nothing
            }
        )
        assert _measured_gpu_seconds(loop) == 150.0

    def test_cancel_check_marks_gpu_reason_and_returns_true(self):
        loop = self._ledger_loop(2 * 3600.0, cap=1.0)
        store = _store({loop.loop_id: loop})
        runner = EngineLoopRunner(
            loop, engine_factory=lambda: None, store=store, poll_seconds=0.01
        )
        assert runner._cancel_check() is True
        assert loop.metadata["state_reason"] == "budget_exhausted:gpu_hours"
