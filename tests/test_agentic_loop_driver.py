"""Wave 9: UMRS-side agentic loop driver — executor + runner semantics."""

from __future__ import annotations

import json
import sys
from types import ModuleType
from unittest.mock import MagicMock

from umrs.knowledge.agent_executor import (
    UmrsAgentIterationExecutor,
    normalize_iteration_requires,
)
from umrs.knowledge.agent_loop_runner import AgentLoopRunner, kick
from umrs.knowledge.loop import LoopPhase, LoopStatus, ResearchLoop


def _loop(loop_id: str = "loop-a", max_iterations: int = 3) -> ResearchLoop:
    return ResearchLoop(
        loop_id=loop_id,
        project_id="proj-1",
        research_goal="Develop a digital filter for MERMAID seismic peak extraction",
        status=LoopStatus.RUNNING,
        phase=LoopPhase.PLANNING,
        max_iterations=max_iterations,
        current_iteration=0,
        created_at="",
        started_at="",
        updated_at="",
    )


class _Store(dict):
    def get(self, loop_id):
        return dict.get(self, loop_id)

    def save(self, loop):
        dict.__setitem__(self, loop.loop_id, loop)


def test_run_agentic_advances_to_completion():
    store_body = _Store()

    def executor(loop_id: str, iteration: int, goal: str) -> dict:
        assert iteration >= 1
        return {
            "llm": True,
            "hypothesis": f"hypothesis {iteration}",
            "evaluation": f"progress {iteration}",
            "next_step": "next",
            "requires": {"gpu": 1} if iteration == 2 else {},
            "graph_nodes": 2,
        }

    finished = kick(_loop(), executor=executor, store=store_body)

    assert finished.current_iteration == 3
    assert finished.phase == LoopPhase.COMPLETED
    assert finished.status == LoopStatus.STOPPED
    assert finished.total_hypotheses_tested == 3
    assert len(finished.iterations) == 3
    iter2 = finished.iterations[1]
    assert iter2.results["requires"] == {"gpu": 1}
    assert store_body["loop-a"].current_iteration == 3


def test_run_agentic_honors_stop_from_store():
    store_body = _Store()

    def executor(loop_id: str, iteration: int, goal: str) -> dict:
        # After the first iteration someone marks the loop STOPPED in the store.
        if iteration == 1:
            stored = dict.get(store_body, "loop-a")
            stored.status = LoopStatus.STOPPED
        return {"llm": True, "hypothesis": "h", "evaluation": "e"}

    loop = _loop()
    store_body["loop-a"] = loop
    finished = kick(loop, executor=executor, store=store_body)

    assert finished.status == LoopStatus.STOPPED
    assert finished.current_iteration == 1


def test_wait_resumed_skips_paused_until_running():
    """The pause wait holds until the store flips RUNNING (or STOPPED)."""
    loop = _loop()
    loop.status = LoopStatus.PAUSED
    store = _FlakyStore(loop)
    runner = AgentLoopRunner(loop, executor=lambda *_args: {}, store=store)
    runner._wait_resumed("loop-a")
    calls = list(store.calls)
    assert calls[0] == "paused"
    assert len(calls) == 3
    assert loop.status == LoopStatus.RUNNING


class _FlakyStore:
    """Status sequence: PAUSED → PAUSED → RUNNING for the wait-resume loop."""

    def __init__(self, loop):
        self._loop = loop
        self.calls: list[str] = []

    def get(self, _loop_id):
        self.calls.append(self._loop.status.value)
        if len(self.calls) >= 3:
            self._loop.status = LoopStatus.RUNNING
        return self._loop

    def save(self, loop):
        pass


def test_agentic_runner_thread_is_daemon():
    runner = AgentLoopRunner(_loop(), executor=lambda *_args: {"llm": True}, store=_Store())
    assert runner.daemon is True


class TestApiStartWiring:
    def test_start_loop_agentic_registers_runner(self, monkeypatch):
        import umrs.api.loop as loop_mod
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        fake_runner_module = ModuleType("umrs.knowledge.agent_loop_runner")
        recorded: dict = {}

        def _fake_register(loop, *, executor, store, poll_seconds=0.5):
            recorded["loop_id"] = loop.loop_id
            recorded["store"] = store
            return MagicMock()

        fake_runner_module.register = _fake_register  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "umrs.knowledge.agent_loop_runner", fake_runner_module)

        fake_project_module = ModuleType("umrs.knowledge.project")

        class _RP:
            def __init__(self, project_id):
                self.project_id = project_id

        fake_project_module.ResearchProject = _RP  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "umrs.knowledge.project", fake_project_module)

        def _fake_executor_for(project):
            recorded["project_id"] = project.project_id
            return lambda *_args: {"llm": True, "hypothesis": "x"}

        monkeypatch.setattr(loop_mod, "_agent_executor_for", _fake_executor_for)

        app = FastAPI()
        app.include_router(loop_mod.router)
        client = TestClient(app)

        r = client.post(
            "/umrs/loops",
            json={
                "project_id": "proj-ag",
                "research_goal": "goal",
                "max_iterations": 1,
                "agentic": True,
            },
        )
        assert r.status_code == 201, r.text
        payload = r.json()
        assert recorded["loop_id"] == payload["loop_id"]
        assert recorded["project_id"] == "proj-ag"
        assert recorded["store"] is not None

    def test_start_loop_default_no_runner(self, monkeypatch):
        import umrs.api.loop as loop_mod
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        # The module engine is real-mode here: since WP-07 (G-P1-3) a phased loop
        # resolves its workflow at start — one registered workflow is the default.
        monkeypatch.setattr(loop_mod, "_registered_workflow_names", lambda owner: ["wf-only"])
        app = FastAPI()
        app.include_router(loop_mod.router)
        client = TestClient(app)

        r = client.post(
            "/umrs/loops",
            json={
                "project_id": "proj-plain",
                "research_goal": "goal",
                "max_iterations": 1,
            },
        )
        assert r.status_code == 201, r.text
        assert r.json()["workflow_name"] == "wf-only"


class TestUmrsAgentIterationExecutor:
    def test_iteration_records_llm_decision(self):
        chat = MagicMock(
            return_value=json.dumps(
                {
                    "hypothesis": "bandpass 0.5-2Hz isolates P-wave",
                    "evidence_ids": ["n1"],
                    "evaluation": "progress",
                    "next_step": "synthesize PSD",
                    "requires": {"gpu": 1},
                }
            )
        )
        executor = UmrsAgentIterationExecutor(object(), chat=chat)
        summary = executor("loop-x", 1, "goal")

        assert summary["llm"] is True
        assert summary["hypothesis"].startswith("bandpass")
        assert summary["requires"] == {"gpu": 1}
        chat.assert_called_once()

    def test_failed_chat_falls_back_digest(self):
        executor = UmrsAgentIterationExecutor(
            object(), chat=MagicMock(side_effect=RuntimeError("offline"))
        )
        summary = executor("loop-x", 1, "goal")
        assert summary["llm"] is False
        assert "offline" in summary["note"]

    def test_hypotheses_journaling_skips_unusable_replies(self):
        chat = MagicMock(return_value="not json at all")
        executor = UmrsAgentIterationExecutor(object(), chat=chat)
        executor("loop-x", 1, "goal")
        assert executor.hypotheses == []

    def test_normalize_iteration_requires_shapes(self):
        assert normalize_iteration_requires({"gpu": 1, "notes": "x"}) == {"gpu": 1}
        assert normalize_iteration_requires({"gpu": 0, "cluster": False}) == {}
        assert normalize_iteration_requires({"slurm": True}) == {"slurm": True}
        assert normalize_iteration_requires("") == {}
