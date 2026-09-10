"""``research_loop_job`` durable kind (ADR-036 D1/D2/(b), WP-05 round 1; G-P4-1).

- spec validation + dispatch wiring (enum, backoff, factory, validator, guard)
- backend submit hosts the loop via loop_hosting and returns a typed ref
- LOST → requeue resumes the campaign at its last iteration (never restarts)
- per-iteration checkpoints land in the CheckpointStore for the hosting attempt
- terminal loops are not re-driven; missing loops fail the attempt clearly
- status checker maps the loop record onto attempt states
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import pytest

from umrs.jobs.checkpoint_store import CheckpointStore
from umrs.jobs.models import DEFAULT_BACKOFFS, JobType
from umrs.jobs.research_loop_backend import (
    ResearchLoopBackend,
    ResearchLoopStatusChecker,
    _result_for,
    validate_research_loop_spec,
)
from umrs.knowledge import engine_loop_runner, loop_hosting
from umrs.knowledge.finding_ingest import InMemoryFindingSink
from umrs.knowledge.loop import (
    AutonomousResearchLoop,
    LoopIteration,
    LoopPhase,
    LoopStatus,
    ResearchLoop,
)
from umrs.knowledge.loop_store import LoopStore


def _wire(store: LoopStore, ckpt: CheckpointStore | None = None, sink=None) -> None:
    loop_hosting.configure(
        loop_hosting.HostingWiring(
            store=lambda: store,
            engine_factory=lambda: lambda: AutonomousResearchLoop(simulated=True),
            agent_executor=lambda project_id: lambda *a: {},
            checkpoint_store=lambda: ckpt,
            finding_sink=lambda: sink,
        )
    )


@pytest.fixture(autouse=True)
def _isolated_hosting():
    engine_loop_runner.stop_all()
    yield
    engine_loop_runner.stop_all()
    from umrs.api import loop as loop_api

    loop_api._configure_hosting()  # restore the REST wiring for other suites


def _loop(
    loop_id: str = "loop-job-1",
    *,
    status=LoopStatus.RUNNING,
    done: int = 0,
    max_iterations: int = 4,
) -> ResearchLoop:
    loop = ResearchLoop(
        loop_id=loop_id,
        project_id="proj-a",
        research_goal="goal",
        status=status,
        phase=LoopPhase.PLANNING,
        max_iterations=max_iterations,
        current_iteration=done,
        metadata={"simulated": True, "mode": "auto"},
    )
    for n in range(1, done + 1):
        loop.iterations.append(
            LoopIteration(
                iteration_id=f"it-old-{n}",
                loop_id=loop_id,
                iteration_number=n,
                phase=LoopPhase.MEMORY_UPDATE,
                status="completed",
                simulated=True,
            )
        )
    return loop


def _job(spec: dict, job_id: str = "job-1"):
    return SimpleNamespace(job_id=job_id, spec=spec, project_id="proj-a", user_id="alice")


def _attempt(attempt_id: str = "att-1", ref: str | None = None):
    return SimpleNamespace(attempt_id=attempt_id, backend_ref=ref)


def _wait_finished(loop_id: str, timeout: float = 5.0) -> None:
    runner = engine_loop_runner.get(loop_id)
    if runner is not None:
        runner.join(timeout=timeout)
    deadline = time.monotonic() + timeout
    while engine_loop_runner.get(loop_id) is not None and time.monotonic() < deadline:
        time.sleep(0.01)


class TestSpecAndWiring:
    def test_spec_validation(self):
        assert validate_research_loop_spec({"loop_id": "loop-1"}) == {
            "loop_id": "loop-1",
            "driver": "engine",
        }
        assert validate_research_loop_spec({"loop_id": "l", "driver": "agent"})["driver"] == "agent"
        # Venue field (WP-05 round 2): frontend default, cluster:<name> engine-only.
        assert validate_research_loop_spec({"loop_id": "l", "venue": "frontend"}) == {
            "loop_id": "l",
            "driver": "engine",
            "venue": "frontend",
        }
        assert validate_research_loop_spec({"loop_id": "l", "venue": "cluster:rccs"}) == {
            "loop_id": "l",
            "driver": "engine",
            "venue": "cluster:rccs",
            "cluster_site": "rccs",
        }
        with pytest.raises(ValueError, match="cluster venue accepts driver 'engine' only"):
            validate_research_loop_spec(
                {"loop_id": "l", "driver": "agent", "venue": "cluster:rccs"}
            )
        for bad in (
            {},
            {"loop_id": " "},
            {"loop_id": "l", "driver": "rocket"},
            {"loop_id": "l", "venue": "cluster-bad-prefix"},
        ):
            with pytest.raises(ValueError):
                validate_research_loop_spec(bad)

    def test_job_kind_registered_everywhere(self):
        from umrs.api.jobs import _UNDISPATCHABLE_JOB_TYPES
        from umrs.jobs.adapters import create_backend_factory
        from umrs.server.job_tracking_handler import _JOB_SPEC_VALIDATORS

        assert JobType.RESEARCH_LOOP.value == "research_loop_job"
        assert DEFAULT_BACKOFFS[JobType.RESEARCH_LOOP]["max_attempts"] >= 2
        assert JobType.RESEARCH_LOOP in _JOB_SPEC_VALIDATORS
        assert JobType.RESEARCH_LOOP not in _UNDISPATCHABLE_JOB_TYPES
        factory = create_backend_factory(
            orchestrator=SimpleNamespace(),
            registry=SimpleNamespace(),
            port_manager=SimpleNamespace(),
        )
        assert isinstance(factory(JobType.RESEARCH_LOOP), ResearchLoopBackend)


class TestHosting:
    def test_submit_hosts_loop_and_returns_typed_ref(self):
        store = LoopStore.memory()
        loop = _loop()
        store.save(loop)
        _wire(store)

        ref = asyncio.run(
            ResearchLoopBackend().submit_job(_job({"loop_id": loop.loop_id}), _attempt())
        )
        assert ref == f"research_loop_job|loop_id={loop.loop_id}|driver=engine"
        _wait_finished(loop.loop_id)
        hosted = store.get(loop.loop_id)
        assert hosted.metadata["hosting"] == "job"
        assert hosted.metadata["durable_job_id"] == "job-1"
        assert hosted.metadata["attempt_id"] == "att-1"
        assert hosted.status == LoopStatus.STOPPED
        assert hosted.phase == LoopPhase.COMPLETED
        assert [it.iteration_number for it in hosted.iterations] == [1, 2, 3, 4]

    @staticmethod
    def _resume_lost_loop():
        """A loop at iteration 2/4 whose attempt was LOST → host a new attempt, wait."""
        store = LoopStore.memory()
        loop = _loop(done=2, max_iterations=4)
        loop.metadata.update(hosting="job", durable_job_id="job-1", attempt_id="att-1")
        store.save(loop)
        ckpt = CheckpointStore()
        _wire(store, ckpt)
        kind = loop_hosting.host(loop.loop_id, job_id="job-1", attempt_id="att-2")
        _wait_finished(loop.loop_id)
        return kind, store.get(loop.loop_id), ckpt

    def test_lost_attempt_resumes_at_last_iteration(self):
        """ADR-036 (b): requeue after LOST continues the campaign, never restarts it."""
        kind, resumed, _ = self._resume_lost_loop()
        assert kind == "engine"
        assert resumed.status == LoopStatus.STOPPED
        assert resumed.current_iteration == 4
        # the two old iterations were kept, two new ones ran
        assert [it.iteration_number for it in resumed.iterations] == [1, 2, 3, 4]
        assert [it.iteration_id for it in resumed.iterations[:2]] == ["it-old-1", "it-old-2"]
        assert resumed.metadata["attempt_id"] == "att-2"

    def test_resumed_attempt_checkpoints_only_its_own_iterations(self):
        _, resumed, ckpt = self._resume_lost_loop()
        rows = ckpt.list_checkpoints("job-1")
        assert [c.iteration for c in rows] == [3, 4]
        assert {c.attempt_id for c in rows} == {"att-2"}
        assert {c.step_name for c in rows} == {"iteration"}
        assert rows[-1].progress == 1.0
        assert rows[-1].state_inline["loop_id"] == resumed.loop_id

    def test_resume_redoes_the_interrupted_iteration(self):
        """Live finding (WP-07): a loop killed mid-iteration must redo it, not skip it."""
        store = LoopStore.memory()
        loop = _loop("loop-mid", done=1, max_iterations=2)
        loop.current_iteration = (
            2  # iteration 2 had started (checkpointed mid-flight) when the process died
        )
        loop.metadata.update(
            hosting="job", durable_job_id="job-1", attempt_id="att-1", awaiting_approval="appr-x"
        )
        store.save(loop)
        _wire(store)
        assert (
            loop_hosting.rewind_interrupted_iteration(_loop("x", done=2)) is None
        )  # nothing to rewind
        loop_hosting.host(loop.loop_id, job_id="job-1", attempt_id="att-2")
        _wait_finished(loop.loop_id)
        resumed = store.get(loop.loop_id)
        assert resumed.metadata["resume_rewind"] == {"from": 2, "to": 1}
        assert resumed.metadata["awaiting_approval"] == "appr-x"  # marker kept for the re-run gate
        assert [it.iteration_number for it in resumed.iterations] == [1, 2]
        assert resumed.iterations[0].iteration_id == "it-old-1"
        assert resumed.status == LoopStatus.STOPPED and resumed.phase == LoopPhase.COMPLETED

    def test_terminal_loop_is_not_redriven(self):
        store = LoopStore.memory()
        stopped = _loop("loop-done", status=LoopStatus.STOPPED, done=4)
        store.save(stopped)
        _wire(store)
        assert loop_hosting.host("loop-done", job_id="job-2", attempt_id="att-9") == "terminal"
        assert engine_loop_runner.get("loop-done") is None
        assert "durable_job_id" not in store.get("loop-done").metadata

    def test_missing_loop_raises_lookup_error(self):
        _wire(LoopStore.memory())
        with pytest.raises(LookupError):
            loop_hosting.host("loop-ghost", job_id="job-3", attempt_id="att-3")

    def test_iteration_hook_mints_findings_into_sink(self):
        from r4_contracts.research import EvidenceRef, EvidenceSource, Finding

        store = LoopStore.memory()
        loop = _loop("loop-mint", max_iterations=1)
        store.save(loop)
        sink = InMemoryFindingSink()
        _wire(store, None, sink)
        hook = loop_hosting.iteration_hook(job_id="job-m", attempt_id="att-m")
        iteration = LoopIteration(
            iteration_id="it-1",
            loop_id=loop.loop_id,
            iteration_number=1,
            phase=LoopPhase.MEMORY_UPDATE,
            status="completed",
            findings=[
                Finding(
                    finding_id="finding-abc",
                    statement="real",
                    evidence_refs=[
                        EvidenceRef(source=EvidenceSource.CATALOG, excerpt="e", artifact_id="art-1")
                    ],
                    confidence=0.7,
                )
            ],
        )
        loop.iterations.append(iteration)
        hook(loop, iteration)
        assert "finding-abc" in sink.artifacts
        assert iteration.artifact_ids == ["finding-abc"]
        assert store.get(loop.loop_id).metadata["finding_ingest"]["minted"] == 1


class TestChecker:
    def test_snapshot_mapping(self):
        base = {
            "phase": "planning",
            "current_iteration": 2,
            "max_iterations": 4,
            "total_findings": 1,
            "error": "",
        }
        running = _result_for({**base, "status": "running", "driver_alive": True})
        assert running.status == "running"
        assert "2/4" in (running.status_detail or "")
        dead = _result_for({**base, "status": "running", "driver_alive": False})
        assert dead.status == "failed"
        done = _result_for(
            {**base, "status": "stopped", "phase": "completed", "driver_alive": False}
        )
        assert done.status == "succeeded"
        assert done.metrics == {"loop_phase": "completed", "iterations": 2, "findings": 1}
        err = _result_for({**base, "status": "error", "error": "boom", "driver_alive": False})
        assert err.status == "failed"
        assert err.exit_reason == "boom"

    def test_check_status_reads_loop_record(self):
        store = LoopStore.memory()
        store.save(_loop("loop-c", status=LoopStatus.STOPPED, done=4))
        _wire(store)
        checker = ResearchLoopStatusChecker()
        assert checker.job_type == JobType.RESEARCH_LOOP
        ok = asyncio.run(
            checker.check_status(
                _attempt(ref="research_loop_job|loop_id=loop-c|driver=engine"), _job({})
            )
        )
        assert ok.status == "succeeded"
        missing = asyncio.run(
            checker.check_status(_attempt(ref="research_loop_job|loop_id=loop-zzz"), _job({}))
        )
        assert missing.status == "failed"
        malformed = asyncio.run(checker.check_status(_attempt(ref="custom|foo=bar"), _job({})))
        assert malformed.status == "failed"
        no_ref = asyncio.run(checker.check_status(_attempt(ref=None), _job({})))
        assert no_ref.status == "failed"
