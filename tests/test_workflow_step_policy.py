"""W3 tests — per-step retry + soft-deadline policy (ADR-021).

Covers:

- ``StepSpec.timeout_seconds`` field round-trip (default-omitted = byte-stable)
- ``call_with_deadline`` passthrough / timeout semantics
- Server runner (``run_workflow_record``) retry enforcement:
  success-after-retries, exhaustion, default-no-retry
- Server runner soft deadline: fast-fail + retriable timeout
"""

from __future__ import annotations

import time

import pytest
from umrs.pipeline_sdk.step import StepResult, call_with_deadline
from umrs.pipeline_sdk.workflow import StepSpec, WorkflowDefinition
from umrs.server.step_executor import StepExecutionError
from umrs.server.step_router import reset_feedback_store
from umrs.server.workflow_execution_registry import (
    get_execution_registry,
    reset_execution_registry_for_tests,
)
from umrs.server.workflow_runner import run_workflow_record


@pytest.fixture(autouse=True)
def _reset_registry():
    reset_execution_registry_for_tests()
    reset_feedback_store()
    yield
    reset_execution_registry_for_tests()
    reset_feedback_store()


# ── Helpers ────────────────────────────────────────────────────────────────

_ENTRYPOINT = "umrs.pipeline_sdk.templates.llm_step:LLMStep"


def _step(**extra):
    spec = {"name": "s", "entrypoint": _ENTRYPOINT, "config": {}}
    spec.update(extra)
    return spec


def _run(tmp_path, *, executor, step_dict):
    registry = get_execution_registry()
    record = registry.create(
        workflow_name="policy-test",
        user_id="alice",
        params={},
    )
    run_workflow_record(
        record,
        source_path=str(tmp_path),
        workflow_json={"name": "policy-test", "version": "0.1", "steps": [step_dict]},
        executor=executor,
    )
    return record


class _FlakyExecutor:
    """Fails ``failures`` times with run-phase errors, then succeeds."""

    def __init__(self, failures: int) -> None:
        self._failures = failures
        self.calls = 0
        self.seen: list[tuple[str, str, str]] = []

    def execute(self, *, entrypoint, config, ctx):
        self.calls += 1
        self.seen.append((entrypoint, config.get("step_name", ""), ctx.run_id))
        if self.calls <= self._failures:
            raise StepExecutionError("run", f"boom {self.calls}")
        return StepResult(output_keys=["bucket/out.json"])


class _SlowExecutor:
    """Sleeps ``sleep_s`` per call — used to trip the soft deadline.

    ``track_completion`` (round-6 flake fix): records whether the slow
    working piece actually ran to completion. The deadline test asserts its
    NEGATIVE — a deadline breach marks failure before completion and the
    late result is discarded, deterministically and without a wall-clock
    budget to flake under suite load.
    """

    def __init__(self, sleep_s: float, track_completion: bool = False) -> None:
        self._sleep_s = sleep_s
        self.calls = 0
        self.seen: list[tuple[str, str, str]] = []
        self._track = track_completion
        self.completed = False

    def execute(self, *, entrypoint, config, ctx):
        self.calls += 1
        self.seen.append((entrypoint, config.get("step_name", ""), ctx.run_id))
        time.sleep(self._sleep_s)
        if self._track:
            self.completed = True
        return StepResult(output_keys=["bucket/out.json"])


# ── StepSpec.timeout_seconds field ──────────────────────────────────────────


class TestStepSpecTimeoutField:
    def test_default_omitted_in_to_dict(self):
        spec = StepSpec(name="a", entrypoint="pkg.mod:Cls")
        assert "timeout_seconds" not in spec.to_dict()

    def test_set_included_in_to_dict(self):
        spec = StepSpec(name="a", entrypoint="pkg.mod:Cls", timeout_seconds=30.0)
        assert spec.to_dict()["timeout_seconds"] == 30.0

    def test_from_dict_round_trip(self):
        spec = StepSpec(name="a", entrypoint="pkg.mod:Cls", timeout_seconds=12.5)
        assert StepSpec.from_dict(spec.to_dict()).timeout_seconds == 12.5

    def test_from_dict_absent_defaults_none(self):
        spec = StepSpec.from_dict({"name": "a", "entrypoint": "pkg.mod:Cls"})
        assert spec.timeout_seconds is None


# ── call_with_deadline ───────────────────────────────────────────────────────


class TestCallWithDeadline:
    def test_inline_when_no_timeout(self):
        assert call_with_deadline(lambda: 42, timeout_seconds=None) == 42

    def test_inline_when_non_positive_timeout(self):
        assert call_with_deadline(lambda: 42, timeout_seconds=0) == 42

    def test_result_returned_within_deadline(self):
        out = call_with_deadline(lambda: "ok", timeout_seconds=5.0, label="t")
        assert out == "ok"

    def test_timeout_raises_fast(self):
        started = time.monotonic()
        with pytest.raises(TimeoutError, match="exceeded deadline"):
            call_with_deadline(lambda: time.sleep(1.0), timeout_seconds=0.1, label="slow-step")
        assert time.monotonic() - started < 0.9


# ── Runner retry enforcement ────────────────────────────────────────────────


class TestRunnerRetry:
    def test_success_after_retries(self, tmp_path):
        executor = _FlakyExecutor(failures=2)
        record = _run(
            tmp_path,
            executor=executor,
            step_dict=_step(max_retries=2, retry_delay=0.01),
        )
        assert record.status == "succeeded", record.error
        assert executor.calls == 3
        assert record.steps["s"].extra.get("attempts") == 3

    def test_retry_exhausted_fails(self, tmp_path):
        executor = _FlakyExecutor(failures=5)
        record = _run(
            tmp_path,
            executor=executor,
            step_dict=_step(max_retries=1, retry_delay=0.01),
        )
        assert record.status == "failed"
        assert executor.calls == 2
        assert "after 2 attempts" in (record.steps["s"].error or "")

    def test_default_no_retry(self, tmp_path):
        executor = _FlakyExecutor(failures=1)
        record = _run(tmp_path, executor=executor, step_dict=_step())
        assert record.status == "failed"
        assert executor.calls == 1

    def test_invalid_retry_fields_fall_back_to_no_retry(self, tmp_path):
        executor = _FlakyExecutor(failures=1)
        record = _run(
            tmp_path,
            executor=executor,
            step_dict=_step(max_retries="banana", retry_delay="nope"),
        )
        assert record.status == "failed"
        assert executor.calls == 1


# ── Runner soft deadline ────────────────────────────────────────────────────


class TestRunnerDeadline:
    def test_timeout_marks_step_failed_fast(self, tmp_path):
        # Deterministic under suite load (r6 flake fix): the deadline breach
        # must occur BEFORE the step's own long sleep completes, and the late
        # result must be discarded. The absolute elapsed budget went flaky
        # in parallel suites (observed 1.57s against the old 0.9s budget —
        # threadpool startup + GIL scheduling pushing a 0.1s deadline past
        # its own timeout). Comparing against the step's own sleep duration
        # is the structurally-robust property the test means to assert.
        executor = _SlowExecutor(sleep_s=5.0, track_completion=True)
        registry = get_execution_registry()
        record = registry.create(workflow_name="policy-test", user_id="alice", params={})
        # Time ONLY the runner call — the registry's one-off pool resolution
        # (PG connect attempt when unavailable) must not pollute the deadline
        # assertion window (suite-load flake observed at 1.57s before).
        started = time.monotonic()
        run_workflow_record(
            record,
            source_path=str(tmp_path),
            workflow_json={
                "name": "policy-test",
                "version": "0.1",
                "steps": [_step(timeout_seconds=0.1)],
            },
            executor=executor,
        )
        elapsed = time.monotonic() - started
        assert record.status == "failed"
        assert "exceeded deadline" in (record.steps["s"].error or "")
        assert not executor.completed  # late result discarded (advisory timeout)
        assert elapsed < 5.0  # breached during the step's own sleep, not after

    def test_timeout_is_retriable(self, tmp_path):
        executor = _SlowExecutor(sleep_s=1.0)
        record = _run(
            tmp_path,
            executor=executor,
            step_dict=_step(timeout_seconds=0.1, max_retries=1, retry_delay=0.01),
        )
        assert record.status == "failed"
        assert executor.calls == 2
        assert "after 2 attempts" in (record.steps["s"].error or "")

    def test_no_deadline_runs_to_completion(self, tmp_path):
        executor = _SlowExecutor(sleep_s=0.05)
        record = _run(tmp_path, executor=executor, step_dict=_step())
        assert record.status == "succeeded", record.error
        assert executor.calls == 1


# ── StepPolicy facade (W2 code facade) ───────────────────────────────────────


class TestStepPolicyFacade:
    def test_from_mapping_typed_values(self):
        from umrs.pipeline_sdk.step import StepPolicy

        policy = StepPolicy.from_mapping(
            {"max_retries": 3, "retry_delay": 2.5, "timeout_seconds": 30}
        )
        assert policy.max_retries == 3
        assert policy.retry_delay == 2.5
        assert policy.timeout_seconds == 30.0

    def test_from_mapping_defaults_when_absent(self):
        from umrs.pipeline_sdk.step import StepPolicy

        policy = StepPolicy.from_mapping({})
        assert policy.max_retries == 0
        assert policy.retry_delay == 1.0
        assert policy.timeout_seconds is None

    def test_from_mapping_invalid_values_fall_back(self):
        from umrs.pipeline_sdk.step import StepPolicy

        policy = StepPolicy.from_mapping(
            {
                "max_retries": "banana",
                "retry_delay": "nope",
                "timeout_seconds": "bogus",
            }
        )
        assert policy.max_retries == 0
        assert policy.retry_delay == 1.0
        assert policy.timeout_seconds is None

    def test_from_mapping_non_positive_timeout_ignored(self):
        from umrs.pipeline_sdk.step import StepPolicy

        policy = StepPolicy.from_mapping({"timeout_seconds": 0})
        assert policy.timeout_seconds is None

    def test_step_spec_policy_property(self):
        from umrs.pipeline_sdk.step import StepPolicy

        spec = StepSpec(
            name="a",
            entrypoint="pkg.mod:Cls",
            max_retries=2,
            retry_delay=3.0,
            timeout_seconds=15.0,
        )
        assert isinstance(spec.policy, StepPolicy)
        assert spec.policy.max_retries == 2
        assert spec.policy.retry_delay == 3.0
        assert spec.policy.timeout_seconds == 15.0

    def test_runner_uses_facade_semantics(self, tmp_path):
        """Runner behavior is unchanged when reading policy via the facade."""
        executor = _FlakyExecutor(failures=1)
        record = _run(
            tmp_path,
            executor=executor,
            step_dict=_step(max_retries="banana", retry_delay="bad", timeout_seconds="bad"),
        )
        assert record.status == "failed"
        assert executor.calls == 1


# ── X1 workflow-run metrics emission ─────────────────────────────────────────


class TestWorkflowRunMetrics:
    def test_success_emits_executor_and_outcome(self, tmp_path, monkeypatch):
        calls: list[tuple[str, str]] = []
        monkeypatch.setattr(
            "umrs.server.workflow_runner.METRICS.workflow_run",
            lambda executor, outcome, _seconds: calls.append((executor, outcome)),
        )
        record = _run(tmp_path, executor=_SlowExecutor(0.01), step_dict=_step())
        assert record.status == "succeeded"
        assert calls == [("_SlowExecutor", "succeeded")]

    def test_failure_emits_failed_outcome(self, tmp_path, monkeypatch):
        calls: list[tuple[str, str]] = []
        monkeypatch.setattr(
            "umrs.server.workflow_runner.METRICS.workflow_run",
            lambda executor, outcome, _seconds: calls.append((executor, outcome)),
        )
        record = _run(tmp_path, executor=_FlakyExecutor(1), step_dict=_step())
        assert record.status == "failed"
        assert calls == [("_FlakyExecutor", "failed")]


# ── WorkflowDefinition round-trip stability ─────────────────────────────────


class TestWorkflowDefinitionStability:
    def test_existing_fixture_dict_unchanged_without_policy_fields(self):
        """Legacy fixture dicts must survive a parse/serialise cycle unchanged."""
        legacy = {
            "name": "wf",
            "version": "1.0",
            "source": {"type": "local_path", "path": "/tmp/x"},
            "steps": [{"name": "a", "entrypoint": "pkg.mod:Cls", "config": {}}],
        }
        wf = WorkflowDefinition.from_dict(legacy)
        assert wf.to_dict()["steps"] == legacy["steps"]
