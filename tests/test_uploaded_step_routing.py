"""Uploaded step code never runs in the UMRS server (ADR-074 D-2, Amendment 1).

Normal users upload their own workflows to run on connected HPCs. The
server must not import that code (an import runs the module's top-level
statements with every platform credential in reach) nor run it in-process:
it goes to the site's HPC lane, or the run is refused by name.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from umrs.pipeline_sdk.step import StepResult
from umrs.server import workflow_runner as wr
from umrs.server.step_trust import is_uploaded_step, uploaded_code_lane
from umrs.server.workflow_execution_registry import ExecutionRecord


def _upload(root: Path, marker: Path) -> str:
    root.mkdir(parents=True, exist_ok=True)
    (root / "usermod.py").write_text(
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('imported in this process')\n"
        "from umrs.pipeline_sdk import Step, StepResult\n\n"
        "class UserStep(Step):\n"
        "    name = 'user'\n"
        "    def setup(self, config): pass\n"
        "    def teardown(self): pass\n"
        "    def run(self, ctx): return StepResult(output_keys=['ran-in-server'])\n",
        encoding="utf-8",
    )
    return str(root)


_WF = {
    "execution_mode": "in_process",
    "steps": [{"name": "s1", "entrypoint": "usermod:UserStep", "config": {}}],
}


class _FakeHpc:
    """Stands in for the Slurm lane; records what it was handed."""

    name = "slurm"

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def execute(self, *, entrypoint, config, ctx):
        self.calls.append({"entrypoint": entrypoint, "metadata": dict(ctx.metadata)})
        return StepResult(output_keys=["ran-on-hpc"])


@pytest.fixture(autouse=True)
def _no_leaked_module():
    saved_path = list(sys.path)
    yield
    sys.path[:] = saved_path
    sys.modules.pop("usermod", None)


def test_uploaded_code_is_refused_by_name_without_an_hpc_lane(tmp_path, monkeypatch):
    monkeypatch.setattr("umrs.server.step_trust.hpc_lane_available", lambda: False)
    marker = tmp_path / "marker"
    record = ExecutionRecord(execution_id="e1", workflow_name="w", user_id="u")
    wr.run_workflow_record(record, source_path=_upload(tmp_path / "wf", marker), workflow_json=_WF)
    assert record.status == "failed"
    assert "connected HPC" in (record.error or "")
    assert not marker.exists(), "the uploaded module ran in the server"
    assert "usermod" not in sys.modules


def test_uploaded_code_runs_on_the_hpc_lane_never_in_the_server(tmp_path, monkeypatch):
    monkeypatch.setattr("umrs.server.step_trust.hpc_lane_available", lambda: True)
    hpc = _FakeHpc()
    names: list = []

    def _build(*, name=None):
        names.append(name)
        return hpc

    monkeypatch.setattr(wr, "build_step_executor", _build)
    marker = tmp_path / "marker"
    source = _upload(tmp_path / "wf", marker)
    record = ExecutionRecord(execution_id="e2", workflow_name="w", user_id="u")
    wr.run_workflow_record(record, source_path=source, workflow_json=_WF)
    assert record.status == "succeeded", record.error
    assert names == ["slurm"]  # the manifest's in_process does not stand
    assert hpc.calls and hpc.calls[0]["entrypoint"] == "usermod:UserStep"
    assert source in hpc.calls[0]["metadata"].values()
    assert not marker.exists(), "the uploaded module ran in the server"
    assert source not in sys.path


def test_classification():
    assert is_uploaded_step("umrs.pipeline_sdk.templates.shell_step:ShellStep", None)
    assert is_uploaded_step("umrs.pipeline_sdk.templates.video_step:VideoStep", None)
    assert not is_uploaded_step("umrs.pipeline_sdk.templates.llm_step:LLMStep", None)
    assert not is_uploaded_step("", "/")  # a gate step is no code


def test_lane_choice(monkeypatch):
    monkeypatch.delenv("UMRS_STEP_EXECUTOR", raising=False)
    monkeypatch.setattr("umrs.server.step_trust.hpc_lane_available", lambda: True)
    assert uploaded_code_lane("inproc") == "slurm"
    assert uploaded_code_lane("slurm-remote") == "slurm-remote"
    monkeypatch.setattr("umrs.server.step_trust.hpc_lane_available", lambda: False)
    assert uploaded_code_lane(None) is None
