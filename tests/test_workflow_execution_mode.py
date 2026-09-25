"""Regression tests: workflow execution_mode is authoritative.

Live bug (Phase 2E): ``SiteConfig.inject_env()`` plants the provisioning-level
``UMRS_STEP_EXECUTOR=slurm`` into the long-lived server process. Later, an
in-process user workflow built ``SlurmStepExecutor`` (via env fallback) and
crashed in step setup without an image. The fix threads the workflow's own
``execution_mode`` into ``build_step_executor(name=...)``.
"""

from __future__ import annotations

import pytest
from umrs.server import workflow_runner as wr
from umrs.server.slurm_step_executor import SlurmStepExecutor
from umrs.server.step_executor import InProcessExecutor
from umrs.server.workflow_execution_registry import ExecutionRecord

_WF_IN_PROCESS = {
    "execution_mode": "in_process",
    "steps": [{"name": "noop", "entrypoint": "mod:NoopStep", "config": {}}],
}
_WF_SLURM = {
    "execution_mode": "slurm",
    "steps": [{"name": "noop", "entrypoint": "mod:Noop", "config": {}}],
}
_WF_NO_MODE = {"steps": [{"name": "noop", "entrypoint": "mod:Noop", "config": {}}]}

_NOOP_STEP_SRC = """\
from umrs.pipeline_sdk import Step, StepResult


class NoopStep(Step):
    name = "noop"

    def setup(self, config):
        self.config = config or {}

    def teardown(self):
        pass

    def run(self, ctx):
        return StepResult(output_keys=[], metadata=[{"ok": True}])
"""


def _record() -> ExecutionRecord:
    return ExecutionRecord(
        execution_id="wf-mode",
        workflow_name="wf",
        user_id="alice",
    )


def _sources(tmp_path) -> str:  # noqa: ANN001
    (tmp_path / "mod.py").write_text(_NOOP_STEP_SRC)
    return str(tmp_path)


# ── Test A: explicit in_process overrides poisoned environment ───────────────


def test_in_process_overrides_slurm_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Historical poisoning: env says slurm, workflow says in_process → inproc."""
    monkeypatch.setenv("UMRS_STEP_EXECUTOR", "slurm")
    name = wr._executor_name_from_execution_mode("in_process")
    assert name == "inproc"
    _, executor = wr._resolve_runner_defaults(None, None, name=name)
    assert isinstance(executor, InProcessExecutor)
    assert not isinstance(executor, SlurmStepExecutor)


# ── Test B: explicit slurm wins over opposing environment ────────────────────


def test_slurm_mode_overrides_opposing_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UMRS_STEP_EXECUTOR", "inproc")
    name = wr._executor_name_from_execution_mode("slurm")
    assert name == "slurm"
    _, executor = wr._resolve_runner_defaults(None, None, name=name)
    assert isinstance(executor, SlurmStepExecutor)


# ── Test C: absent mode preserves fallback ───────────────────────────────────


def test_absent_mode_keeps_env_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UMRS_STEP_EXECUTOR", "slurm")
    name = wr._executor_name_from_execution_mode(None)
    assert name is None
    _, executor = wr._resolve_runner_defaults(None, None, name=name)
    assert isinstance(executor, SlurmStepExecutor)  # env fallback preserved


# ── Test D: run_workflow_record propagates the declared mode ─────────────────


def test_run_workflow_record_propagates_execution_mode(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Full path: run_workflow_record → _resolve_runner_defaults → build(name)."""
    monkeypatch.setenv("UMRS_STEP_EXECUTOR", "slurm")  # poisoning condition
    captured: list[object] = []

    def _capture(*, name=None):  # noqa: ANN001, ANN202
        captured.append(name)
        return InProcessExecutor()

    monkeypatch.setattr(wr, "build_step_executor", _capture)
    record = _record()
    wr.run_workflow_record(
        record,
        source_path=_sources(tmp_path),
        workflow_json=_WF_IN_PROCESS,
    )
    assert captured == ["inproc"], f"executor name not propagated: {captured}"
    assert record.status == "succeeded"
