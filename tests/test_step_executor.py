"""Tests for the pluggable step execution boundary (Slice 8).

The runner uses an :class:`umrs.server.step_executor.StepExecutor` to
drive each step's ``setup → run → teardown`` lifecycle. These tests
cover both shipped backends:

* :class:`InProcessExecutor` — preserves today's semantics.
* :class:`SubprocessExecutor` — spawns a real Python subprocess and
  pickles results back. Slow-ish but exercises the full path.
"""

from __future__ import annotations

import pickle
import sys
import textwrap

import pytest

from umrs.pipeline_sdk.step import Step, StepContext, StepResult
from umrs.server.step_executor import (
    InProcessExecutor,
    StepExecutionError,
    SubprocessExecutor,
    build_step_executor,
)


# ----------------------------------------------------------------------
# Sample step classes used by the in-process tests
# ----------------------------------------------------------------------


class _OkStep(Step):
    name = "ok"

    def run(self, ctx: StepContext) -> StepResult:  # noqa: D401
        return StepResult(extra={"hello": ctx.config.get("greeting", "world")})


class _SetupFailStep(Step):
    name = "setup_fail"

    def setup(self, config: dict) -> None:
        raise ValueError("boom in setup")

    def run(self, ctx: StepContext) -> StepResult:
        return StepResult()


class _RunFailStep(Step):
    name = "run_fail"

    def run(self, ctx: StepContext) -> StepResult:
        raise RuntimeError("boom in run")


class _BadReturnStep(Step):
    name = "bad_return"

    def run(self, ctx: StepContext) -> StepResult:  # type: ignore[override]
        return "not a StepResult"  # type: ignore[return-value]


def _make_ctx() -> StepContext:
    return StepContext(
        s3=None,
        input_bucket="b",
        input_keys=[],
        output_bucket="b",
        config={},
        run_id="t",
    )


# ----------------------------------------------------------------------
# In-process executor
# ----------------------------------------------------------------------


def test_inproc_executor_runs_step():
    executor = InProcessExecutor()
    config = {"greeting": "hi"}
    ctx = _make_ctx()
    ctx.config = dict(config)  # runner passes the same dict to setup() and ctx
    result = executor.execute(
        entrypoint=f"{__name__}:_OkStep",
        config=config,
        ctx=ctx,
    )
    assert isinstance(result, StepResult)
    assert result.extra == {"hello": "hi"}


def test_inproc_executor_setup_failure_classified():
    executor = InProcessExecutor()
    with pytest.raises(StepExecutionError) as excinfo:
        executor.execute(
            entrypoint=f"{__name__}:_SetupFailStep",
            config={},
            ctx=_make_ctx(),
        )
    assert excinfo.value.phase == "setup"
    assert "boom in setup" in excinfo.value.message


def test_inproc_executor_run_failure_classified():
    executor = InProcessExecutor()
    with pytest.raises(StepExecutionError) as excinfo:
        executor.execute(
            entrypoint=f"{__name__}:_RunFailStep",
            config={},
            ctx=_make_ctx(),
        )
    assert excinfo.value.phase == "run"
    assert "boom in run" in excinfo.value.message


def test_inproc_executor_bad_return_classified():
    executor = InProcessExecutor()
    with pytest.raises(StepExecutionError) as excinfo:
        executor.execute(
            entrypoint=f"{__name__}:_BadReturnStep",
            config={},
            ctx=_make_ctx(),
        )
    assert excinfo.value.phase == "run"


def test_inproc_executor_invalid_entrypoint():
    executor = InProcessExecutor()
    with pytest.raises(StepExecutionError):
        executor.execute(
            entrypoint="not_a_real_module:Cls",
            config={},
            ctx=_make_ctx(),
        )


# ----------------------------------------------------------------------
# Factory
# ----------------------------------------------------------------------


def test_build_executor_default(monkeypatch):
    monkeypatch.delenv("UMRS_STEP_EXECUTOR", raising=False)
    assert isinstance(build_step_executor(), InProcessExecutor)


def test_build_executor_subprocess(monkeypatch):
    monkeypatch.setenv("UMRS_STEP_EXECUTOR", "subprocess")
    assert isinstance(build_step_executor(), SubprocessExecutor)


def test_build_executor_unknown_falls_back(monkeypatch):
    monkeypatch.setenv("UMRS_STEP_EXECUTOR", "no_such_backend")
    # Must not raise; falls back to inproc with a logged warning.
    assert isinstance(build_step_executor(), InProcessExecutor)


def test_build_executor_explicit_name_wins_over_env(monkeypatch):
    monkeypatch.setenv("UMRS_STEP_EXECUTOR", "subprocess")
    assert isinstance(build_step_executor("inproc"), InProcessExecutor)


# ----------------------------------------------------------------------
# Subprocess executor (real subprocess; slow-ish)
# ----------------------------------------------------------------------


@pytest.fixture
def step_module(tmp_path, monkeypatch):
    """Create a tiny user-module on disk that the subprocess can import.

    The subprocess does NOT inherit pytest's collection of locally-defined
    classes, so we materialise the test fixtures into a real module the
    child can ``import``.
    """
    pkg = tmp_path / "_slice8_steps"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "steps.py").write_text(
        textwrap.dedent(
            '''
            from umrs.pipeline_sdk.step import Step, StepContext, StepResult


            class EchoStep(Step):
                name = "echo"

                def run(self, ctx):
                    return StepResult(extra={"echoed": ctx.config.get("msg", "")})


            class CrashStep(Step):
                name = "crash"

                def run(self, ctx):
                    raise RuntimeError("subprocess boom")
            '''
        ).strip()
        + "\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    # Forward the prepended path into the subprocess so ``-m
    # umrs.server.step_worker`` can import the test module.
    monkeypatch.setenv("PYTHONPATH", str(tmp_path) + ":" + (sys.path[0] or ""))
    yield "_slice8_steps.steps"


def test_subprocess_executor_runs_step(step_module):
    executor = SubprocessExecutor(timeout_seconds=30)
    config = {"msg": "ping"}
    ctx = _make_ctx()
    ctx.config = dict(config)
    result = executor.execute(
        entrypoint=f"{step_module}:EchoStep",
        config=config,
        ctx=ctx,
    )
    assert isinstance(result, StepResult)
    assert result.extra == {"echoed": "ping"}


def test_subprocess_executor_crash_classified(step_module):
    executor = SubprocessExecutor(timeout_seconds=30)
    with pytest.raises(StepExecutionError) as excinfo:
        executor.execute(
            entrypoint=f"{step_module}:CrashStep",
            config={},
            ctx=_make_ctx(),
        )
    assert excinfo.value.phase == "run"
    assert "subprocess boom" in excinfo.value.message


def test_subprocess_executor_invalid_entrypoint():
    executor = SubprocessExecutor(timeout_seconds=30)
    with pytest.raises(StepExecutionError) as excinfo:
        executor.execute(
            entrypoint="no_such_module:Nope",
            config={},
            ctx=_make_ctx(),
        )
    assert excinfo.value.phase == "setup"


def test_subprocess_executor_timeout(step_module, monkeypatch, tmp_path):
    """A hung worker is killed and surfaces as a worker-phase failure."""
    # Reuse the package layout but add a sleeping step.
    pkg = tmp_path / "_slice8_slow"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "slow.py").write_text(
        textwrap.dedent(
            '''
            import time
            from umrs.pipeline_sdk.step import Step, StepResult


            class SlowStep(Step):
                name = "slow"

                def run(self, ctx):
                    time.sleep(10)
                    return StepResult()
            '''
        ).strip()
        + "\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv("PYTHONPATH", str(tmp_path) + ":" + (sys.path[0] or ""))

    executor = SubprocessExecutor(timeout_seconds=0.5)
    with pytest.raises(StepExecutionError) as excinfo:
        executor.execute(
            entrypoint="_slice8_slow.slow:SlowStep",
            config={},
            ctx=_make_ctx(),
        )
    assert excinfo.value.phase == "worker"
    assert "timed out" in excinfo.value.message


# ----------------------------------------------------------------------
# Result is picklable across the boundary
# ----------------------------------------------------------------------


def test_step_result_round_trips_through_pickle():
    """Sanity check: StepResult survives the pickle hop the worker uses."""
    r = StepResult(output_keys=["a"], extra={"k": 1})
    blob = pickle.dumps(r)
    r2 = pickle.loads(blob)
    assert isinstance(r2, StepResult)
    assert r2.output_keys == ["a"]
    assert r2.extra == {"k": 1}


# ----------------------------------------------------------------------
# Runner integration — proves the executor swap doesn't break the DAG
# ----------------------------------------------------------------------


def test_runner_uses_supplied_executor(monkeypatch):
    """``run_workflow_record`` must dispatch through the supplied executor."""
    from umrs.server.workflow_execution_registry import (
        get_execution_registry,
        reset_execution_registry_for_tests,
    )
    from umrs.server.workflow_runner import run_workflow_record

    reset_execution_registry_for_tests()

    calls = []

    class _Recording:
        name = "recording"

        def execute(self, *, entrypoint, config, ctx):
            calls.append((entrypoint, dict(config), ctx.run_id))
            return StepResult(extra={"executed": True})

    registry = get_execution_registry()
    record = registry.create(workflow_name="exec-test", user_id="u")
    run_workflow_record(
        record,
        source_path="/tmp",
        workflow_json={
            "name": "exec-test",
            "steps": [
                {
                    "name": "only",
                    "entrypoint": f"{__name__}:_OkStep",
                    "config": {"greeting": "via-executor"},
                },
            ],
        },
        executor=_Recording(),
    )
    reset_execution_registry_for_tests()

    assert record.status == "succeeded", record.error
    assert len(calls) == 1
    entrypoint, cfg, _run_id = calls[0]
    assert entrypoint == f"{__name__}:_OkStep"
    assert cfg["greeting"] == "via-executor"
