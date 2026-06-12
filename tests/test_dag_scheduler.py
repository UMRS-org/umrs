"""
Tests for Phase 4 — DAG scheduling and parallel step execution.

Covers:
  - DAGScheduler: linear fallback, topological ordering, cycle detection
  - Parallel execution within levels via ThreadPoolExecutor
  - Fan-in/fan-out with merged keys
  - depends_on validation (unknown step, self-dependency)
  - Retry behaviour (max_retries, retry_delay)
  - StepContext.parent_results accessibility
  - StepContext.report_progress()
  - _aggregate_dag_results helper
  - WorkflowDefinition.validate() DAG checks
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import numpy as np
import pytest

from umrs.pipeline_sdk.dag import DAGScheduler, DAGValidationError, execute_dag
from umrs.pipeline_sdk.step import Step, StepContext, StepResult
from umrs.pipeline_sdk.workflow import (
    StepSpec,
    WorkflowDefinition,
    WorkflowPipeline,
)


# =========================================================================
# Helper step classes
# =========================================================================


class PassthroughStep(Step):
    name = "passthrough"

    def run(self, ctx: StepContext) -> StepResult:
        return StepResult(output_keys=list(ctx.input_keys))


class AppendSuffixStep(Step):
    """Appends a configurable suffix to each key."""

    name = "append_suffix"

    def setup(self, config: dict) -> None:
        self.suffix = config.get("suffix", "_out")

    def run(self, ctx: StepContext) -> StepResult:
        return StepResult(
            output_keys=[k + self.suffix for k in ctx.input_keys],
        )


class EmbeddingStep(Step):
    name = "embed"

    def run(self, ctx: StepContext) -> StepResult:
        n = len(ctx.input_keys)
        return StepResult(
            output_keys=ctx.input_keys,
            embeddings=np.ones((n, 4), dtype=np.float32),
            checksums=["abc"] * n,
            metadata=[{"step": "embed"}] * n,
        )


class RecordThreadStep(Step):
    """Records the thread it ran on — used to verify parallel execution."""

    name = "record_thread"

    def __init__(self):
        super().__init__()
        self.thread_id: Optional[int] = None

    def run(self, ctx: StepContext) -> StepResult:
        self.thread_id = threading.current_thread().ident
        time.sleep(0.05)  # small sleep to enable real concurrency
        return StepResult(output_keys=list(ctx.input_keys))


class FailThenSucceedStep(Step):
    """Fails N times then succeeds."""

    name = "fail_then_succeed"

    def __init__(self, fail_count: int = 1):
        super().__init__()
        self.fail_count = fail_count
        self.attempt = 0

    def run(self, ctx: StepContext) -> StepResult:
        self.attempt += 1
        if self.attempt <= self.fail_count:
            raise RuntimeError(f"fail attempt {self.attempt}")
        return StepResult(output_keys=list(ctx.input_keys))


class FanInStep(Step):
    """Stores parent_results for inspection."""

    name = "fan_in"

    def __init__(self):
        super().__init__()
        self.received_parent_results: Dict[str, StepResult] = {}

    def run(self, ctx: StepContext) -> StepResult:
        self.received_parent_results = dict(ctx.parent_results)
        return StepResult(output_keys=list(ctx.input_keys))


# =========================================================================
# Helpers
# =========================================================================


def _make_specs(
    names: List[str],
    deps: Optional[Dict[str, List[str]]] = None,
    **overrides,
) -> List[StepSpec]:
    """Build StepSpec list with optional depends_on map."""
    specs = []
    for n in names:
        kw: Dict[str, Any] = {
            "name": n,
            "entrypoint": f"{n}_mod:{n.capitalize()}Step",
            "config": {},
        }
        if deps and n in deps:
            kw["depends_on"] = deps[n]
        kw.update(overrides)
        specs.append(StepSpec(**kw))
    return specs


def _mock_s3() -> MagicMock:
    store: Dict[str, bytes] = {}
    s3 = MagicMock()

    def put_object(Bucket, Key, Body, **kw):
        store[f"{Bucket}/{Key}"] = Body if isinstance(Body, bytes) else Body.encode()

    def get_object(Bucket, Key):
        full = f"{Bucket}/{Key}"
        if full not in store:
            raise Exception(f"NoSuchKey: {full}")
        body = MagicMock()
        body.read.return_value = store[full]
        return {"Body": body, "ETag": '"abc"', "ContentLength": len(store[full]), "Metadata": {}}

    def head_object(Bucket, Key):
        full = f"{Bucket}/{Key}"
        if full not in store:
            raise Exception(f"NoSuchKey: {full}")
        return {"ETag": '"abc"', "ContentLength": len(store[full]), "Metadata": {}}

    s3.put_object = MagicMock(side_effect=put_object)
    s3.get_object = MagicMock(side_effect=get_object)
    s3.head_object = MagicMock(side_effect=head_object)
    s3._store = store
    return s3


# =========================================================================
# DAGScheduler — graph construction
# =========================================================================


class TestDAGSchedulerConstruction:
    def test_linear_fallback_no_depends_on(self):
        """When no step declares depends_on, use linear chain."""
        specs = _make_specs(["a", "b", "c"])
        sch = DAGScheduler(specs)
        levels = sch.execution_levels()
        assert levels == [[0], [1], [2]], "Expected linear chain"
        assert sch.is_linear()

    def test_single_step(self):
        specs = _make_specs(["only"])
        sch = DAGScheduler(specs)
        levels = sch.execution_levels()
        assert levels == [[0]]
        assert sch.is_linear()

    def test_explicit_deps_diamond(self):
        """
        Diamond: A → B, A → C, B+C → D
        """
        specs = _make_specs(
            ["A", "B", "C", "D"],
            deps={"A": [], "B": ["A"], "C": ["A"], "D": ["B", "C"]},
        )
        sch = DAGScheduler(specs)
        levels = sch.execution_levels()
        # Level 0: A(0), Level 1: B(1)+C(2) parallel, Level 2: D(3)
        assert levels[0] == [0]
        assert sorted(levels[1]) == [1, 2]
        assert levels[2] == [3]
        assert not sch.is_linear()

    def test_multiple_roots(self):
        """Two independent roots: A, B (no deps)."""
        specs = _make_specs(
            ["A", "B"],
            deps={"A": [], "B": []},
        )
        sch = DAGScheduler(specs)
        levels = sch.execution_levels()
        assert levels == [[0, 1]]

    def test_topological_order_linear(self):
        specs = _make_specs(["x", "y", "z"])
        sch = DAGScheduler(specs)
        assert sch.topological_order() == [0, 1, 2]

    def test_topological_order_diamond(self):
        specs = _make_specs(
            ["A", "B", "C", "D"],
            deps={"A": [], "B": ["A"], "C": ["A"], "D": ["B", "C"]},
        )
        sch = DAGScheduler(specs)
        order = sch.topological_order()
        assert order[0] == 0  # A first
        assert order[-1] == 3  # D last
        # B and C in the middle (either order)
        assert set(order[1:3]) == {1, 2}

    def test_parent_names(self):
        specs = _make_specs(
            ["A", "B", "C"],
            deps={"A": [], "B": ["A"], "C": ["A", "B"]},
        )
        sch = DAGScheduler(specs)
        assert sch.parent_names("A") == set()
        assert sch.parent_names("B") == {"A"}
        assert sch.parent_names("C") == {"A", "B"}


# =========================================================================
# DAGScheduler — validation
# =========================================================================


class TestDAGSchedulerValidation:
    def test_cycle_detected(self):
        """A → B → C → A is a cycle."""
        specs = _make_specs(
            ["A", "B", "C"],
            deps={"A": ["C"], "B": ["A"], "C": ["B"]},
        )
        sch = DAGScheduler(specs)
        with pytest.raises(DAGValidationError, match="Cycle"):
            sch.validate()

    def test_self_dependency_in_dag(self):
        """A step depending on itself."""
        specs = _make_specs(
            ["A", "B"],
            deps={"A": [], "B": ["B"]},
        )
        sch = DAGScheduler(specs)
        with pytest.raises(DAGValidationError, match="Cycle"):
            sch.validate()

    def test_unknown_dependency_raises(self):
        """depends_on referencing non-existent step raises immediately."""
        with pytest.raises(DAGValidationError, match="unknown step"):
            _make_specs(
                ["A", "B"],
                deps={"A": [], "B": ["GHOST"]},
            )
            DAGScheduler(
                _make_specs(
                    ["A", "B"],
                    deps={"A": [], "B": ["GHOST"]},
                )
            )

    def test_valid_linear_does_not_raise(self):
        specs = _make_specs(["a", "b", "c"])
        sch = DAGScheduler(specs)
        sch.validate()  # should not raise


# =========================================================================
# WorkflowDefinition.validate() — DAG integration
# =========================================================================


class TestWorkflowDefinitionDAGValidation:
    def _make_wf(self, steps_list):
        return WorkflowDefinition.from_dict(
            {
                "name": "dag-test",
                "source": {"type": "local_path", "path": "/tmp"},
                "steps": steps_list,
            }
        )

    def test_validate_unknown_depends_on(self):
        wf = self._make_wf(
            [
                {"name": "s1", "entrypoint": "a:B", "depends_on": ["nope"]},
            ]
        )
        with pytest.raises(ValueError, match="unknown step.*nope"):
            wf.validate()

    def test_validate_self_dependency(self):
        wf = self._make_wf(
            [
                {"name": "s1", "entrypoint": "a:B", "depends_on": ["s1"]},
            ]
        )
        with pytest.raises(ValueError, match="depends on itself"):
            wf.validate()

    def test_validate_cycle(self):
        wf = self._make_wf(
            [
                {"name": "s1", "entrypoint": "a:B", "depends_on": ["s2"]},
                {"name": "s2", "entrypoint": "c:D", "depends_on": ["s1"]},
            ]
        )
        with pytest.raises(ValueError, match="[Cc]ycle"):
            wf.validate()

    def test_validate_valid_dag(self):
        wf = self._make_wf(
            [
                {"name": "s1", "entrypoint": "a:B", "depends_on": []},
                {"name": "s2", "entrypoint": "c:D", "depends_on": ["s1"]},
            ]
        )
        wf.validate()  # should not raise


# =========================================================================
# StepSpec — new fields
# =========================================================================


class TestStepSpecDAGFields:
    def test_depends_on_from_dict(self):
        spec = StepSpec.from_dict(
            {
                "name": "s",
                "entrypoint": "a:B",
                "depends_on": ["parent1", "parent2"],
            }
        )
        assert spec.depends_on == ["parent1", "parent2"]

    def test_depends_on_default_none(self):
        spec = StepSpec.from_dict({"name": "s", "entrypoint": "a:B"})
        assert spec.depends_on is None

    def test_max_retries_from_dict(self):
        spec = StepSpec.from_dict(
            {
                "name": "s",
                "entrypoint": "a:B",
                "max_retries": 3,
                "retry_delay": 0.5,
            }
        )
        assert spec.max_retries == 3
        assert spec.retry_delay == 0.5

    def test_max_retries_default(self):
        spec = StepSpec.from_dict({"name": "s", "entrypoint": "a:B"})
        assert spec.max_retries == 0
        assert spec.retry_delay == 1.0

    def test_to_dict_includes_depends_on(self):
        spec = StepSpec(
            name="s",
            entrypoint="a:B",
            config={},
            depends_on=["p1"],
            max_retries=2,
        )
        d = spec.to_dict()
        assert d["depends_on"] == ["p1"]
        assert d["max_retries"] == 2

    def test_to_dict_omits_defaults(self):
        spec = StepSpec(name="s", entrypoint="a:B", config={})
        d = spec.to_dict()
        assert "depends_on" not in d
        assert "max_retries" not in d


# =========================================================================
# execute_dag — linear execution
# =========================================================================


class TestExecuteDAGLinear:
    def test_linear_three_steps(self):
        """Three PassthroughSteps in a linear chain."""
        specs = _make_specs(["a", "b", "c"])
        steps = [PassthroughStep(), PassthroughStep(), PassthroughStep()]
        for st, sp in zip(steps, specs):
            st.setup(sp.config)

        sch = DAGScheduler(specs)
        results = execute_dag(
            sch,
            steps,
            specs,
            s3=_mock_s3(),
            input_bucket="src",
            input_keys=["k1", "k2"],
            output_bucket="dst",
            run_id="r1",
            shared_metadata={},
        )
        assert set(results.keys()) == {"a", "b", "c"}
        for r in results.values():
            assert r.output_keys == ["k1", "k2"]

    def test_linear_embedding_step(self):
        specs = _make_specs(["embed"])
        steps = [EmbeddingStep()]
        steps[0].setup({})
        sch = DAGScheduler(specs)
        results = execute_dag(
            sch,
            steps,
            specs,
            s3=_mock_s3(),
            input_bucket="b",
            input_keys=["k1"],
            output_bucket="out",
            run_id="r2",
            shared_metadata={},
        )
        assert results["embed"].embeddings is not None
        assert results["embed"].checksums == ["abc"]


# =========================================================================
# execute_dag — parallel execution
# =========================================================================


class TestExecuteDAGParallel:
    def test_parallel_steps_run_concurrently(self):
        """Diamond DAG: root → (B, C) parallel → merge."""
        specs = _make_specs(
            ["root", "B", "C", "merge"],
            deps={"root": [], "B": ["root"], "C": ["root"], "merge": ["B", "C"]},
        )
        step_b = RecordThreadStep()
        step_c = RecordThreadStep()
        steps = [PassthroughStep(), step_b, step_c, PassthroughStep()]
        for st, sp in zip(steps, specs):
            st.setup(sp.config)

        sch = DAGScheduler(specs)
        results = execute_dag(
            sch,
            steps,
            specs,
            s3=_mock_s3(),
            input_bucket="b",
            input_keys=["k1"],
            output_bucket="out",
            run_id="r3",
            shared_metadata={},
        )
        assert len(results) == 4
        # B and C should have run on different threads (parallel level)
        assert step_b.thread_id is not None
        assert step_c.thread_id is not None
        assert step_b.thread_id != step_c.thread_id

    def test_fan_in_merges_keys(self):
        """Fan-in step receives merged output_keys from both parents."""
        specs = _make_specs(
            ["A", "B", "merge"],
            deps={"A": [], "B": [], "merge": ["A", "B"]},
        )

        class PrefixStep(Step):
            name = "prefix"

            def __init__(self, prefix: str):
                super().__init__()
                self._prefix = prefix

            def run(self, ctx: StepContext) -> StepResult:
                return StepResult(
                    output_keys=[f"{self._prefix}_{k}" for k in ctx.input_keys],
                )

        step_a = PrefixStep("A")
        step_b = PrefixStep("B")
        merge = FanInStep()
        steps = [step_a, step_b, merge]
        for st, sp in zip(steps, specs):
            st.setup(sp.config)

        sch = DAGScheduler(specs)
        results = execute_dag(
            sch,
            steps,
            specs,
            s3=_mock_s3(),
            input_bucket="b",
            input_keys=["k1"],
            output_bucket="out",
            run_id="r4",
            shared_metadata={},
        )
        # merge step should see keys from both parents
        merge_input_expected = {"A_k1", "B_k1"}
        # The merge step received merged keys — verify via its output
        assert set(results["merge"].output_keys) == merge_input_expected
        # parent_results should contain both A and B
        assert "A" in merge.received_parent_results
        assert "B" in merge.received_parent_results

    def test_fan_in_parent_results(self):
        """StepContext.parent_results is accessible in fan-in step."""
        specs = _make_specs(
            ["A", "B", "C"],
            deps={"A": [], "B": [], "C": ["A", "B"]},
        )
        fan_in = FanInStep()
        steps = [PassthroughStep(), PassthroughStep(), fan_in]
        for st, sp in zip(steps, specs):
            st.setup(sp.config)

        sch = DAGScheduler(specs)
        execute_dag(
            sch,
            steps,
            specs,
            s3=_mock_s3(),
            input_bucket="b",
            input_keys=["k"],
            output_bucket="out",
            run_id="r5",
            shared_metadata={},
        )
        assert set(fan_in.received_parent_results.keys()) == {"A", "B"}


# =========================================================================
# execute_dag — retry
# =========================================================================


class TestExecuteDAGRetry:
    def test_retry_succeeds_after_failures(self):
        specs = [
            StepSpec(
                name="flaky",
                entrypoint="a:B",
                config={},
                max_retries=2,
                retry_delay=0.0,
            )
        ]
        step = FailThenSucceedStep(fail_count=2)
        step.setup({})

        sch = DAGScheduler(specs)
        results = execute_dag(
            sch,
            [step],
            specs,
            s3=_mock_s3(),
            input_bucket="b",
            input_keys=["k"],
            output_bucket="out",
            run_id="r6",
            shared_metadata={},
        )
        assert "flaky" in results
        assert step.attempt == 3  # 2 failures + 1 success

    def test_retry_exhausted_raises(self):
        specs = [
            StepSpec(
                name="always_fail",
                entrypoint="a:B",
                config={},
                max_retries=1,
                retry_delay=0.0,
            )
        ]
        step = FailThenSucceedStep(fail_count=999)
        step.setup({})

        sch = DAGScheduler(specs)
        with pytest.raises(RuntimeError, match="fail attempt"):
            execute_dag(
                sch,
                [step],
                specs,
                s3=_mock_s3(),
                input_bucket="b",
                input_keys=["k"],
                output_bucket="out",
                run_id="r7",
                shared_metadata={},
            )

    def test_no_retry_by_default(self):
        specs = [
            StepSpec(
                name="fail_once",
                entrypoint="a:B",
                config={},
            )
        ]
        step = FailThenSucceedStep(fail_count=1)
        step.setup({})

        sch = DAGScheduler(specs)
        with pytest.raises(RuntimeError):
            execute_dag(
                sch,
                [step],
                specs,
                s3=_mock_s3(),
                input_bucket="b",
                input_keys=["k"],
                output_bucket="out",
                run_id="r8",
                shared_metadata={},
            )


# =========================================================================
# StepContext.report_progress
# =========================================================================


class TestStepContextReportProgress:
    def test_report_progress_logs(self, caplog):
        ctx = StepContext(
            s3=MagicMock(),
            input_bucket="b",
            input_keys=["k"],
            output_bucket="out",
            config={},
            metadata={},
            run_id="r9",
        )
        with caplog.at_level("INFO"):
            ctx.report_progress(50.0, "halfway there")
        assert any("50.0%" in r.message for r in caplog.records)
        assert any("halfway" in r.message for r in caplog.records)


# =========================================================================
# _aggregate_dag_results
# =========================================================================


class TestAggregateDAGResults:
    def test_linear_last_step_wins(self):
        specs = _make_specs(["a", "b"])
        sch = DAGScheduler(specs)
        step_results = {
            "a": StepResult(output_keys=["k_a"]),
            "b": StepResult(
                output_keys=["k_b"],
                embeddings=np.zeros((1, 2)),
                checksums=["ck"],
                metadata=[{"m": 1}],
            ),
        }
        emb, ck, meta, keys = WorkflowPipeline._aggregate_dag_results(
            step_results,
            sch,
            ["orig"],
        )
        assert keys == ["k_b"]
        assert ck == ["ck"]
        assert meta == [{"m": 1}]
        assert emb is not None

    def test_empty_results_return_input_keys(self):
        specs = _make_specs(["a"])
        sch = DAGScheduler(specs)
        step_results = {
            "a": StepResult(output_keys=[]),
        }
        emb, ck, meta, keys = WorkflowPipeline._aggregate_dag_results(
            step_results,
            sch,
            ["orig"],
        )
        assert keys == ["orig"]
        assert emb is None
        assert ck == []
        assert meta == []

    def test_diamond_last_level_wins(self):
        """In a diamond, the merge step's output takes precedence."""
        specs = _make_specs(
            ["A", "B", "C", "D"],
            deps={"A": [], "B": ["A"], "C": ["A"], "D": ["B", "C"]},
        )
        sch = DAGScheduler(specs)
        step_results = {
            "A": StepResult(output_keys=["a_k"]),
            "B": StepResult(output_keys=["b_k"], checksums=["ck_b"]),
            "C": StepResult(output_keys=["c_k"], checksums=["ck_c"]),
            "D": StepResult(
                output_keys=["d_k"],
                embeddings=np.ones((1, 3)),
                checksums=["ck_d"],
                metadata=[{"final": True}],
            ),
        }
        emb, ck, meta, keys = WorkflowPipeline._aggregate_dag_results(
            step_results,
            sch,
            ["orig"],
        )
        assert keys == ["d_k"]
        assert ck == ["ck_d"]
        assert meta == [{"final": True}]
        assert emb is not None
