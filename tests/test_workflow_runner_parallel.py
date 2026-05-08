"""Parallel DAG executor tests for ``run_workflow_record``.

Covers:
* Independent steps in the same wave run concurrently.
* Fan-in steps see results from all parents.
* A step failure does not cancel sibling steps; dependents are skipped.
* The ``_routing`` log records every step that started (succeeded,
  failed, or skipped).
* ``UMRS_RUNNER_MAX_PARALLEL_STEPS`` caps the pool.
* ``cancel_requested`` short-circuits between waves.
* Concurrent blob writes do not lose data.
"""

from __future__ import annotations

import sys
import threading
import time
import types
from typing import Any, Dict, List

import pytest

from umrs.pipeline_sdk.step import Step, StepContext, StepResult
from umrs.server.workflow_execution_registry import (
    get_execution_registry,
    reset_execution_registry_for_tests,
)
from umrs.server.step_router import reset_feedback_store
from umrs.server.workflow_runner import run_workflow_record


@pytest.fixture(autouse=True)
def _reset_registry():
    reset_execution_registry_for_tests()
    reset_feedback_store()
    yield
    reset_execution_registry_for_tests()
    reset_feedback_store()


@pytest.fixture
def probe_module():
    """A fresh sys.modules entry that hosts dynamically-defined step classes.

    Tests append step classes to this module and reference them via the
    entrypoint ``"_runner_parallel_probe:ClassName"``.
    """
    mod_name = "_runner_parallel_probe"
    mod = types.ModuleType(mod_name)
    sys.modules[mod_name] = mod
    yield mod
    sys.modules.pop(mod_name, None)


def _entry(class_name: str) -> str:
    return f"_runner_parallel_probe:{class_name}"


# ---------------------------------------------------------------------------
# 1. Independent steps run concurrently.
# ---------------------------------------------------------------------------


def test_independent_steps_run_concurrently(probe_module, tmp_path):
    barrier = threading.Barrier(2, timeout=5.0)
    thread_ids: List[int] = []
    lock = threading.Lock()

    class SleeperA(Step):
        name = "a"

        def run(self, ctx: StepContext) -> StepResult:
            with lock:
                thread_ids.append(threading.get_ident())
            barrier.wait()  # both threads must arrive concurrently
            time.sleep(0.05)
            return StepResult(extra={"who": "a"})

    class SleeperB(Step):
        name = "b"

        def run(self, ctx: StepContext) -> StepResult:
            with lock:
                thread_ids.append(threading.get_ident())
            barrier.wait()
            time.sleep(0.05)
            return StepResult(extra={"who": "b"})

    probe_module.SleeperA = SleeperA  # type: ignore[attr-defined]
    probe_module.SleeperB = SleeperB  # type: ignore[attr-defined]

    registry = get_execution_registry()
    record = registry.create(workflow_name="par", user_id="u")

    started = time.monotonic()
    run_workflow_record(
        record,
        source_path=str(tmp_path),
        workflow_json={
            "name": "par",
            "steps": [
                {"name": "a", "entrypoint": _entry("SleeperA"), "depends_on": []},
                {"name": "b", "entrypoint": _entry("SleeperB"), "depends_on": []},
            ],
        },
    )
    duration = time.monotonic() - started

    assert record.status == "succeeded", record.error
    # Both barriers fired -> truly concurrent. If serial, the first
    # ``barrier.wait`` would have timed out.
    assert len(set(thread_ids)) == 2
    # Two steps that each sleep 50ms should run in <150ms when parallel,
    # not >100ms when serial. Generous bound to avoid CI flake.
    assert duration < 0.5, f"workflow took {duration:.2f}s; expected parallel"


# ---------------------------------------------------------------------------
# 2. Fan-in step sees all parents' results.
# ---------------------------------------------------------------------------


def test_fan_in_step_sees_all_parent_results(probe_module, tmp_path):
    captured: Dict[str, Any] = {}

    class A(Step):
        name = "a"

        def run(self, ctx: StepContext) -> StepResult:
            return StepResult(extra={"value": "from-a"})

    class B(Step):
        name = "b"

        def run(self, ctx: StepContext) -> StepResult:
            return StepResult(extra={"value": "from-b"})

    class Merge(Step):
        name = "merge"

        def run(self, ctx: StepContext) -> StepResult:
            captured["parents"] = sorted(ctx.parent_results.keys())
            captured["a"] = ctx.parent_results["a"].extra["value"]
            captured["b"] = ctx.parent_results["b"].extra["value"]
            return StepResult(extra={"merged": True})

    probe_module.A = A  # type: ignore[attr-defined]
    probe_module.B = B  # type: ignore[attr-defined]
    probe_module.Merge = Merge  # type: ignore[attr-defined]

    registry = get_execution_registry()
    record = registry.create(workflow_name="diamond", user_id="u")
    run_workflow_record(
        record,
        source_path=str(tmp_path),
        workflow_json={
            "name": "diamond",
            "steps": [
                {"name": "a", "entrypoint": _entry("A"), "depends_on": []},
                {"name": "b", "entrypoint": _entry("B"), "depends_on": []},
                {
                    "name": "merge",
                    "entrypoint": _entry("Merge"),
                    "depends_on": ["a", "b"],
                },
            ],
        },
    )

    assert record.status == "succeeded", record.error
    assert captured["parents"] == ["a", "b"]
    assert captured["a"] == "from-a"
    assert captured["b"] == "from-b"


# ---------------------------------------------------------------------------
# 3. Sibling completion + dependent skip on failure.
# ---------------------------------------------------------------------------


def test_sibling_finishes_when_other_branch_fails(probe_module, tmp_path):
    sibling_completed = threading.Event()

    class Boom(Step):
        name = "boom"

        def run(self, ctx: StepContext) -> StepResult:
            raise RuntimeError("nope")

    class SlowOk(Step):
        name = "slow_ok"

        def run(self, ctx: StepContext) -> StepResult:
            time.sleep(0.1)
            sibling_completed.set()
            return StepResult(extra={"ok": True})

    class Dependent(Step):
        name = "dependent"

        def run(self, ctx: StepContext) -> StepResult:  # pragma: no cover
            raise AssertionError("dependent must not run after parent failed")

    probe_module.Boom = Boom  # type: ignore[attr-defined]
    probe_module.SlowOk = SlowOk  # type: ignore[attr-defined]
    probe_module.Dependent = Dependent  # type: ignore[attr-defined]

    registry = get_execution_registry()
    record = registry.create(workflow_name="failmix", user_id="u")
    run_workflow_record(
        record,
        source_path=str(tmp_path),
        workflow_json={
            "name": "failmix",
            "steps": [
                {"name": "boom", "entrypoint": _entry("Boom"), "depends_on": []},
                {
                    "name": "slow_ok",
                    "entrypoint": _entry("SlowOk"),
                    "depends_on": [],
                },
                {
                    "name": "dependent",
                    "entrypoint": _entry("Dependent"),
                    "depends_on": ["boom"],
                },
            ],
        },
    )

    assert record.status == "failed"
    assert "boom" in (record.error or "")
    # Sibling was allowed to complete despite the failure in its wave.
    assert sibling_completed.is_set()
    assert record.steps["boom"].status == "failed"
    assert record.steps["slow_ok"].status == "succeeded"
    assert record.steps["dependent"].status == "skipped"


# ---------------------------------------------------------------------------
# 4. _routing log records every step that the runner saw.
# ---------------------------------------------------------------------------


def test_routing_log_records_every_step_in_parallel_run(probe_module, tmp_path):
    class A(Step):
        name = "a"

        def run(self, ctx: StepContext) -> StepResult:
            return StepResult()

    class B(Step):
        name = "b"

        def run(self, ctx: StepContext) -> StepResult:
            return StepResult()

    class C(Step):
        name = "c"

        def run(self, ctx: StepContext) -> StepResult:
            return StepResult()

    probe_module.A = A  # type: ignore[attr-defined]
    probe_module.B = B  # type: ignore[attr-defined]
    probe_module.C = C  # type: ignore[attr-defined]

    registry = get_execution_registry()
    record = registry.create(workflow_name="three-par", user_id="u")
    run_workflow_record(
        record,
        source_path=str(tmp_path),
        workflow_json={
            "name": "three-par",
            "steps": [
                {"name": "a", "entrypoint": _entry("A"), "depends_on": []},
                {"name": "b", "entrypoint": _entry("B"), "depends_on": []},
                {"name": "c", "entrypoint": _entry("C"), "depends_on": []},
            ],
        },
    )

    assert record.status == "succeeded", record.error
    routing = record.results["_routing"]
    assert set(routing.keys()) == {"a", "b", "c"}
    # Every entry has wave_index=0 (all in the same wave) and was
    # resolved via the env fallback (no router hooks configured).
    for entry in routing.values():
        assert entry["wave_index"] == 0


# ---------------------------------------------------------------------------
# 5. UMRS_RUNNER_MAX_PARALLEL_STEPS caps pool size.
# ---------------------------------------------------------------------------


def test_max_parallel_steps_env_caps_pool_size(probe_module, tmp_path, monkeypatch):
    monkeypatch.setenv("UMRS_RUNNER_MAX_PARALLEL_STEPS", "1")

    in_flight = 0
    max_in_flight = 0
    lock = threading.Lock()

    class Counter(Step):
        name = "counter"

        def run(self, ctx: StepContext) -> StepResult:
            nonlocal in_flight, max_in_flight
            with lock:
                in_flight += 1
                if in_flight > max_in_flight:
                    max_in_flight = in_flight
            time.sleep(0.05)
            with lock:
                in_flight -= 1
            return StepResult()

    probe_module.Counter = Counter  # type: ignore[attr-defined]

    registry = get_execution_registry()
    record = registry.create(workflow_name="capped", user_id="u")
    run_workflow_record(
        record,
        source_path=str(tmp_path),
        workflow_json={
            "name": "capped",
            "steps": [
                {"name": f"s{i}", "entrypoint": _entry("Counter"), "depends_on": []}
                for i in range(4)
            ],
        },
    )

    assert record.status == "succeeded", record.error
    # With cap=1 the runner must serialise the wave: only one step in
    # flight at any moment.
    assert max_in_flight == 1


# ---------------------------------------------------------------------------
# 6. cancel_requested between waves short-circuits.
# ---------------------------------------------------------------------------


def test_cancel_between_waves_short_circuits(probe_module, tmp_path):
    record_holder: Dict[str, Any] = {}

    class First(Step):
        name = "first"

        def run(self, ctx: StepContext) -> StepResult:
            # Mark cancel after wave 0 completes.
            record_holder["record"].cancel_requested = True
            return StepResult()

    class Second(Step):
        name = "second"

        def run(self, ctx: StepContext) -> StepResult:  # pragma: no cover
            raise AssertionError("second must not run after cancel")

    probe_module.First = First  # type: ignore[attr-defined]
    probe_module.Second = Second  # type: ignore[attr-defined]

    registry = get_execution_registry()
    record = registry.create(workflow_name="cancellable", user_id="u")
    record_holder["record"] = record

    run_workflow_record(
        record,
        source_path=str(tmp_path),
        workflow_json={
            "name": "cancellable",
            "steps": [
                {"name": "first", "entrypoint": _entry("First"), "depends_on": []},
                {
                    "name": "second",
                    "entrypoint": _entry("Second"),
                    "depends_on": ["first"],
                },
            ],
        },
    )

    assert record.status == "cancelled"
    assert record.steps["first"].status == "succeeded"
    # Second must not have been launched.
    assert "second" not in record.steps or record.steps["second"].status == "pending"


# ---------------------------------------------------------------------------
# 7. Concurrent blob writes are not lost.
# ---------------------------------------------------------------------------


def test_concurrent_blob_writes_do_not_lose_data(probe_module, tmp_path):
    class Writer(Step):
        name = "writer"

        def run(self, ctx: StepContext) -> StepResult:
            key = f"writes/{ctx.config['step_name']}.json"
            ctx.s3.put_object(
                Bucket=ctx.output_bucket,
                Key=key,
                Body=ctx.config["step_name"].encode("utf-8"),
            )
            return StepResult(output_keys=[key])

    probe_module.Writer = Writer  # type: ignore[attr-defined]

    registry = get_execution_registry()
    record = registry.create(workflow_name="writers", user_id="u")
    run_workflow_record(
        record,
        source_path=str(tmp_path),
        workflow_json={
            "name": "writers",
            "steps": [
                {
                    "name": f"w{i}",
                    "entrypoint": _entry("Writer"),
                    "depends_on": [],
                }
                for i in range(8)
            ],
        },
    )

    assert record.status == "succeeded", record.error
    blobs = record.results["blobs"]
    for i in range(8):
        full_key = f"wf-runner/writes/w{i}.json"
        assert full_key in blobs, f"missing blob for w{i}"
        assert blobs[full_key] == f"w{i}".encode("utf-8")
