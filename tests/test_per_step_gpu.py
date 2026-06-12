"""Slice 12 tests: per-step GPU override.

Tests cover both executor-level resolution and runner-level plumbing.
"""

from __future__ import annotations

import os
import subprocess
import sys
import types
from typing import Any, Dict, List

import pytest

from umrs.pipeline_sdk.step import Step, StepContext, StepResult
from umrs.server.step_executor import (
    EXECUTOR_GPU_METADATA_KEY,
    EXECUTOR_IMAGE_METADATA_KEY,
    PodmanExecutor,
    StepExecutionError,
    StepExecutor,
)
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


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _make_ctx(**metadata_overrides) -> StepContext:
    return StepContext(
        s3=None,
        input_bucket="b",
        input_keys=[],
        output_bucket="b",
        config={},
        metadata=dict(metadata_overrides),
        run_id="t",
    )


class _FakePodman:
    """Captures the podman invocation and writes a synthetic reply."""

    def __init__(self, reply, *, write_output=True, returncode=0):
        import pickle as _pkl

        self._reply = reply
        self._write_output = write_output
        self._returncode = returncode
        self.calls: List[List[str]] = []
        self._pkl = _pkl

    def __call__(self, cmd, **kwargs):
        self.calls.append(list(cmd))
        workdir = None
        for arg in cmd:
            if ":/umrs-step" in arg:
                workdir = arg.split(":")[0]
                break
        if workdir and self._write_output:
            import os as _os

            out_path = _os.path.join(workdir, "output.pkl")
            with open(out_path, "wb") as fh:
                fh.write(self._pkl.dumps(self._reply))
        return subprocess.CompletedProcess(cmd, self._returncode, stdout=b"", stderr=b"")


class _OkStep(Step):
    name = "probe"

    def setup(self, config):
        pass

    def run(self, ctx: StepContext) -> StepResult:
        return StepResult(extra={"ok": True})


_PROBE_MODULE = types.ModuleType("umrs_test_per_step_gpu_module")
_PROBE_MODULE._OkStep = _OkStep  # type: ignore[attr-defined]
sys.modules["umrs_test_per_step_gpu_module"] = _PROBE_MODULE


class _RecordingExecutor(StepExecutor):
    """Captures every execute() call so we can assert on metadata."""

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def execute(self, *, entrypoint, config, ctx) -> StepResult:
        self.calls.append({"metadata": dict(ctx.metadata)})
        return StepResult(extra={"ok": True})


def _workflow(routing_hints=None):
    step: Dict[str, Any] = {
        "name": "probe",
        "entrypoint": "umrs_test_per_step_gpu_module:_OkStep",
        "config": {},
    }
    if routing_hints is not None:
        step["routing_hints"] = routing_hints
    return {"name": "wf", "steps": [step]}


# ======================================================================
# Executor-level tests (PodmanExecutor._resolve_gpu)
# ======================================================================


class TestResolveGpu:
    def test_absent_falls_back_to_instance_default_false(self, monkeypatch):
        """No GPU hint → uses self.gpu (False by default)."""
        fake = _FakePodman({"status": "ok", "result": StepResult()})
        monkeypatch.setattr(subprocess, "run", fake)
        PodmanExecutor(image="img", gpu=False).execute(
            entrypoint="pkg.mod:Cls", config={}, ctx=_make_ctx()
        )
        cmd = fake.calls[0]
        assert "--device" not in cmd

    def test_absent_falls_back_to_instance_default_true(self, monkeypatch):
        """No GPU hint → uses self.gpu (True when UMRS_STEP_GPU=1)."""
        fake = _FakePodman({"status": "ok", "result": StepResult()})
        monkeypatch.setattr(subprocess, "run", fake)
        PodmanExecutor(image="img", gpu=True).execute(
            entrypoint="pkg.mod:Cls", config={}, ctx=_make_ctx()
        )
        cmd = fake.calls[0]
        assert "--device" in cmd
        assert cmd[cmd.index("--device") + 1] == "nvidia.com/gpu=all"

    def test_hint_true_enables_gpu(self, monkeypatch):
        """routing_hints.gpu=True overrides self.gpu=False."""
        fake = _FakePodman({"status": "ok", "result": StepResult()})
        monkeypatch.setattr(subprocess, "run", fake)
        PodmanExecutor(image="img", gpu=False).execute(
            entrypoint="pkg.mod:Cls",
            config={},
            ctx=_make_ctx(**{EXECUTOR_GPU_METADATA_KEY: True}),
        )
        cmd = fake.calls[0]
        assert "--device" in cmd
        assert cmd[cmd.index("--device") + 1] == "nvidia.com/gpu=all"

    def test_hint_false_disables_gpu(self, monkeypatch):
        """routing_hints.gpu=False overrides self.gpu=True (critical edge case)."""
        fake = _FakePodman({"status": "ok", "result": StepResult()})
        monkeypatch.setattr(subprocess, "run", fake)
        PodmanExecutor(image="img", gpu=True).execute(
            entrypoint="pkg.mod:Cls",
            config={},
            ctx=_make_ctx(**{EXECUTOR_GPU_METADATA_KEY: False}),
        )
        cmd = fake.calls[0]
        assert "--device" not in cmd

    def test_non_bool_raises_setup_error(self, monkeypatch):
        """routing_hints.gpu='yes' → setup-phase error."""
        fake = _FakePodman({"status": "ok", "result": StepResult()})
        monkeypatch.setattr(subprocess, "run", fake)
        with pytest.raises(StepExecutionError) as exc_info:
            PodmanExecutor(image="img").execute(
                entrypoint="pkg.mod:Cls",
                config={},
                ctx=_make_ctx(**{EXECUTOR_GPU_METADATA_KEY: "yes"}),
            )
        assert exc_info.value.phase == "setup"
        assert "bool" in exc_info.value.message.lower()

    def test_int_one_raises_setup_error(self, monkeypatch):
        """routing_hints.gpu=1 (int, not bool) → setup-phase error."""
        fake = _FakePodman({"status": "ok", "result": StepResult()})
        monkeypatch.setattr(subprocess, "run", fake)
        with pytest.raises(StepExecutionError) as exc_info:
            PodmanExecutor(image="img").execute(
                entrypoint="pkg.mod:Cls",
                config={},
                ctx=_make_ctx(**{EXECUTOR_GPU_METADATA_KEY: 1}),
            )
        assert exc_info.value.phase == "setup"

    def test_gpu_key_stripped_from_worker_metadata(self, monkeypatch):
        """The dunder key must NOT leak to user step metadata."""
        fake = _FakePodman({"status": "ok", "result": StepResult()})
        monkeypatch.setattr(subprocess, "run", fake)
        import pickle

        PodmanExecutor(image="img", keep_workdir=True).execute(
            entrypoint="pkg.mod:Cls",
            config={},
            ctx=_make_ctx(**{EXECUTOR_GPU_METADATA_KEY: True}),
        )
        # Parse the input.pkl written by the executor to check metadata.
        cmd = fake.calls[0]
        workdir = None
        for arg in cmd:
            if ":/umrs-step" in arg:
                workdir = arg.split(":")[0]
                break
        assert workdir is not None
        input_path = os.path.join(workdir, "input.pkl")
        with open(input_path, "rb") as fh:
            payload = pickle.loads(fh.read())
        assert EXECUTOR_GPU_METADATA_KEY not in payload["ctx"]["metadata"]


# ======================================================================
# Runner-level tests (routing_hints.gpu → ctx.metadata → routing log)
# ======================================================================


class TestRunnerGpuHint:
    def test_gpu_hint_threaded_into_metadata(self, tmp_path):
        """routing_hints.gpu=True lands in ctx.metadata under the
        executor-internal key."""
        executor = _RecordingExecutor()
        record = get_execution_registry().create(workflow_name="wf", user_id="alice")
        run_workflow_record(
            record,
            source_path=str(tmp_path),
            workflow_json=_workflow({"gpu": True}),
            executor=executor,
        )
        assert record.status == "succeeded", record.error
        assert executor.calls[0]["metadata"].get(EXECUTOR_GPU_METADATA_KEY) is True

    def test_gpu_hint_false_threaded(self, tmp_path):
        """routing_hints.gpu=False also threads (not just truthy values)."""
        executor = _RecordingExecutor()
        record = get_execution_registry().create(workflow_name="wf", user_id="alice")
        run_workflow_record(
            record,
            source_path=str(tmp_path),
            workflow_json=_workflow({"gpu": False}),
            executor=executor,
        )
        assert record.status == "succeeded", record.error
        assert executor.calls[0]["metadata"].get(EXECUTOR_GPU_METADATA_KEY) is False

    def test_no_gpu_hint_no_metadata_key(self, tmp_path):
        """Absence of routing_hints.gpu leaves the dunder key out."""
        executor = _RecordingExecutor()
        record = get_execution_registry().create(workflow_name="wf", user_id="alice")
        run_workflow_record(
            record,
            source_path=str(tmp_path),
            workflow_json=_workflow(),
            executor=executor,
        )
        assert record.status == "succeeded", record.error
        assert EXECUTOR_GPU_METADATA_KEY not in executor.calls[0]["metadata"]

    def test_routing_log_records_resolved_gpu_from_hint(self, tmp_path):
        """When gpu hint is present, routing log shows the hint value."""
        executor = _RecordingExecutor()
        record = get_execution_registry().create(workflow_name="wf", user_id="alice")
        run_workflow_record(
            record,
            source_path=str(tmp_path),
            workflow_json=_workflow({"gpu": True}),
            executor=executor,
        )
        routing = record.results.get("_routing", {})
        assert routing["probe"]["gpu"] is True

    def test_routing_log_records_default_gpu_when_absent(self, tmp_path, monkeypatch):
        """When gpu hint is absent the routing log shows the
        deployment-wide default (UMRS_STEP_GPU)."""
        monkeypatch.setenv("UMRS_STEP_GPU", "1")
        executor = _RecordingExecutor()
        record = get_execution_registry().create(workflow_name="wf", user_id="alice")
        run_workflow_record(
            record,
            source_path=str(tmp_path),
            workflow_json=_workflow(),
            executor=executor,
        )
        routing = record.results.get("_routing", {})
        assert routing["probe"]["gpu"] is True

    def test_routing_log_shows_false_default_when_env_unset(self, tmp_path, monkeypatch):
        """Default GPU is False when UMRS_STEP_GPU is unset."""
        monkeypatch.delenv("UMRS_STEP_GPU", raising=False)
        executor = _RecordingExecutor()
        record = get_execution_registry().create(workflow_name="wf", user_id="alice")
        run_workflow_record(
            record,
            source_path=str(tmp_path),
            workflow_json=_workflow(),
            executor=executor,
        )
        routing = record.results.get("_routing", {})
        assert routing["probe"]["gpu"] is False

    def test_override_precedence_env_true_hint_false(self, tmp_path, monkeypatch):
        """The most important edge case: env GPU=true, hint=false → GPU is
        NOT used. The hint must override, not merge."""
        monkeypatch.setenv("UMRS_STEP_GPU", "1")
        executor = _RecordingExecutor()
        record = get_execution_registry().create(workflow_name="wf", user_id="alice")
        run_workflow_record(
            record,
            source_path=str(tmp_path),
            workflow_json=_workflow({"gpu": False}),
            executor=executor,
        )
        assert record.status == "succeeded", record.error
        # Metadata should carry False (overriding the env default).
        assert executor.calls[0]["metadata"].get(EXECUTOR_GPU_METADATA_KEY) is False
        # Routing log should also show False.
        routing = record.results.get("_routing", {})
        assert routing["probe"]["gpu"] is False
