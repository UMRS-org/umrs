"""ADR-064 pins: AgenticEngine — the agent executor under the engine driver."""

from __future__ import annotations

from umrs.knowledge.agentic_engine import AgenticEngine
from umrs.knowledge.loop import LoopStatus, ResearchLoop


def _loop(max_iterations: int = 2) -> ResearchLoop:
    return ResearchLoop(
        loop_id="L-agentic",
        project_id="proj-a",
        research_goal="goal",
        max_iterations=max_iterations,
        status=LoopStatus.RUNNING,
    )


def test_runs_to_completion_with_hook_per_iteration():
    hooks: list[str] = []
    engine = AgenticEngine(
        executor=lambda _lid, n, _goal: {"hypothesis": f"h{n}", "llm": True}
    )
    engine._iteration_hook = hooks.append
    loop = engine.run(_loop(max_iterations=2))
    assert loop.current_iteration == 2
    assert len(loop.iterations) == 2
    assert loop.iterations[0].hypothesis == "h1"
    assert hooks == [loop.iterations[0], loop.iterations[1]]
    assert loop.status == LoopStatus.STOPPED
    assert loop.total_hypotheses_tested == 2
    assert loop.total_experiments_run == 2


def test_cancel_check_stops_midway():
    calls: list[int] = []

    def _cancel() -> bool:
        return loop.current_iteration >= 1

    loop = _loop(max_iterations=5)
    engine = AgenticEngine(executor=lambda _l, n, _g: calls.append(n) or {"llm": True})
    engine._cancel_check = _cancel
    out = engine.run(loop)
    assert calls == [1]
    assert out.status == LoopStatus.RUNNING  # left RUNNING for cooperative stop


def test_approval_gate_rejection_fails_the_iteration():
    engine = AgenticEngine(executor=lambda *_a: {"llm": True})
    engine._approval_gate = lambda action: False
    loop = _loop()
    try:
        engine.run(loop)
        assert False, "rejection must raise"
    except RuntimeError as exc:
        assert "rejected by the approval gate" in str(exc)
    assert loop.current_iteration == 1
    assert loop.iterations == []
