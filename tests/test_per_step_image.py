"""Slice 10 runner-side tests: per-step image hint plumbing.

The runner is a thin transport: it lifts ``routing_hints.image`` from
the StepSpec into ``StepContext.metadata`` under
``EXECUTOR_IMAGE_METADATA_KEY`` and records the resolved image in the
``_routing`` log. Validation lives in the executor.
"""

from __future__ import annotations

import sys
import types
from typing import Any, Dict, List

import pytest

from umrs.pipeline_sdk.step import Step, StepContext, StepResult
from umrs.server.step_executor import (
    EXECUTOR_IMAGE_METADATA_KEY,
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
# A trivial step + module shim, mirroring test_workflow_runner.py.
# ----------------------------------------------------------------------


class _OkStep(Step):
    name = "probe"

    def setup(self, config):
        pass

    def run(self, ctx: StepContext) -> StepResult:
        return StepResult(extra={"ok": True})


_PROBE_MODULE = types.ModuleType("umrs_test_per_step_image_module")
_PROBE_MODULE._OkStep = _OkStep  # type: ignore[attr-defined]
sys.modules["umrs_test_per_step_image_module"] = _PROBE_MODULE


class _RecordingExecutor(StepExecutor):
    """Captures every executor.execute() call so we can assert on the
    metadata the runner threaded through."""

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def execute(self, *, entrypoint, config, ctx) -> StepResult:
        self.calls.append(
            {
                "entrypoint": entrypoint,
                "metadata": dict(ctx.metadata),
            }
        )
        return StepResult(extra={"ok": True})


class _RaisingExecutor(StepExecutor):
    """Raises a setup-phase StepExecutionError, mimicking PodmanExecutor's
    validation when the runner threads a bad image hint."""

    def execute(self, *, entrypoint, config, ctx) -> StepResult:
        # Mimic PodmanExecutor._resolve_image() exactly.
        if EXECUTOR_IMAGE_METADATA_KEY in ctx.metadata:
            override = ctx.metadata[EXECUTOR_IMAGE_METADATA_KEY]
            if not isinstance(override, str):
                raise StepExecutionError(
                    "setup",
                    f"per-step image override must be a string, got "
                    f"{type(override).__name__}",
                )
            if not override.strip():
                raise StepExecutionError("setup", "override is empty")
        return StepResult(extra={"ok": True})


def _workflow(routing_hints: Dict[str, Any] | None = None) -> Dict[str, Any]:
    step: Dict[str, Any] = {
        "name": "probe",
        "entrypoint": "umrs_test_per_step_image_module:_OkStep",
        "config": {},
    }
    if routing_hints is not None:
        step["routing_hints"] = routing_hints
    return {"name": "wf", "steps": [step]}


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------


def test_routing_hints_image_threaded_into_metadata(tmp_path):
    """A string override on the spec lands in ctx.metadata under the
    executor-internal key."""
    executor = _RecordingExecutor()
    record = get_execution_registry().create(workflow_name="wf", user_id="alice")
    run_workflow_record(
        record,
        source_path=str(tmp_path),
        workflow_json=_workflow({"image": "vlm:cuda"}),
        executor=executor,
    )
    assert record.status == "succeeded", record.error
    assert len(executor.calls) == 1
    assert (
        executor.calls[0]["metadata"].get(EXECUTOR_IMAGE_METADATA_KEY)
        == "vlm:cuda"
    )


def test_no_routing_hints_no_metadata_key(tmp_path):
    """Absence of routing_hints.image leaves the dunder key out of
    ctx.metadata so executors can fall back to their default."""
    executor = _RecordingExecutor()
    record = get_execution_registry().create(workflow_name="wf", user_id="alice")
    run_workflow_record(
        record,
        source_path=str(tmp_path),
        workflow_json=_workflow(),  # no routing_hints
        executor=executor,
    )
    assert record.status == "succeeded", record.error
    assert (
        EXECUTOR_IMAGE_METADATA_KEY not in executor.calls[0]["metadata"]
    )


def test_non_string_image_hint_surfaces_setup_error(tmp_path):
    """A non-string ``routing_hints.image`` is forwarded as-is and the
    executor raises a setup-phase error, which the runner must surface
    as a step failure."""
    record = get_execution_registry().create(workflow_name="wf", user_id="alice")
    run_workflow_record(
        record,
        source_path=str(tmp_path),
        workflow_json=_workflow({"image": 42}),
        executor=_RaisingExecutor(),
    )
    assert record.status == "failed"
    step = record.steps["probe"]
    assert step.status == "failed"
    assert "setup" in (step.error or "")
    assert "string" in (step.error or "").lower()


def test_resolved_image_recorded_in_routing_log(tmp_path):
    """The resolved image must appear in record.results['_routing'][step]."""
    executor = _RecordingExecutor()
    record = get_execution_registry().create(workflow_name="wf", user_id="alice")
    run_workflow_record(
        record,
        source_path=str(tmp_path),
        workflow_json=_workflow({"image": "vlm:cuda"}),
        executor=executor,
    )
    assert record.status == "succeeded", record.error
    routing = record.results.get("_routing", {})
    assert routing["probe"]["image"] == "vlm:cuda"


def test_routing_log_falls_back_to_default_image(tmp_path, monkeypatch):
    """When routing_hints.image is absent the routing log shows the
    deployment-wide default (UMRS_STEP_IMAGE), so callers can always
    see which image a step ran in."""
    monkeypatch.setenv("UMRS_STEP_IMAGE", "default-img:latest")
    executor = _RecordingExecutor()
    record = get_execution_registry().create(workflow_name="wf", user_id="alice")
    run_workflow_record(
        record,
        source_path=str(tmp_path),
        workflow_json=_workflow(),
        executor=executor,
    )
    assert record.status == "succeeded", record.error
    routing = record.results.get("_routing", {})
    assert routing["probe"]["image"] == "default-img:latest"
