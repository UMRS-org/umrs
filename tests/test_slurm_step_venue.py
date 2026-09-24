"""Slurm step executor venue portability pins (RCCS ng-dgx-m*).

Live constraints (deploy/ansible rccs-cloud group_vars + site.yaml):
the ng-dgx-m* partitions reject ``--gres``/``--gpus`` and ``--mem``
requests, the submit host's local /tmp is not visible from compute
nodes, and uploaded workflow step modules do not exist inside the step
container image. These tests pin the executor behaviours that make an
uploaded bundle runnable on such a venue:

* empty ``memory`` omits ``#SBATCH --mem`` entirely;
* ``gpus=0`` omits ``#SBATCH --gres`` even when the step wants a GPU
  (podman CDI passthrough is independent of the Slurm request);
* ``UMRS_SLURM_WORKDIR_ROOT`` relocates the payload workdir onto a
  shared filesystem and is validated at setup time;
* the runner hands the uploaded source tree to the executor via
  ``__executor_source_dir__`` metadata; the executor bind-mounts it
  read-only and exports ``UMRS_STEP_SOURCE_DIR``; the worker puts it
  on ``sys.path`` (and drops stale modules when the dir changes).
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from umrs.pipeline_sdk.step import StepContext, StepResult
from umrs.server import step_worker
from umrs.server.slurm_step_executor import SlurmStepExecutor
from umrs.server.step_executor import (
    EXECUTOR_GPU_METADATA_KEY,
    EXECUTOR_SOURCE_DIR_METADATA_KEY,
    StepExecutionError,
)


def _ctx(metadata: dict | None = None) -> StepContext:
    return StepContext(
        s3=None,
        input_bucket="in",
        input_keys=[],
        output_bucket="out",
        metadata=metadata or {},
        run_id="run-venue",
    )


def _run_with_fake_sbatch(executor: SlurmStepExecutor, ctx: StepContext):
    captured: dict = {}

    def fake_submit(sbatch_path, workdir):
        captured["workdir"] = workdir
        with open(Path(workdir) / "input.pkl", "rb") as handle:
            captured["payload"] = pickle.load(handle)
        with open(Path(workdir) / "output.pkl", "wb") as handle:
            pickle.dump({"status": "ok", "result": StepResult(output_keys=[])}, handle)
        return "12345"

    with (
        patch.object(executor, "_submit_sbatch", side_effect=fake_submit),
        patch.object(executor, "_check_job_status", return_value="COMPLETED"),
    ):
        executor.execute(entrypoint="mod:Noop", config={}, ctx=ctx)
    return captured


def _script(captured: dict) -> str:
    return Path(captured["workdir"], "run_step.sbatch").read_text(encoding="utf-8")


# ── sbatch directive shaping ────────────────────────────────────────────────


def test_empty_memory_omits_mem_directive():
    executor = SlurmStepExecutor(image="img:latest", keep_workdir=True, memory="")
    captured = _run_with_fake_sbatch(executor, _ctx())
    script = _script(captured)
    assert "#SBATCH --mem" not in script


def test_non_empty_memory_keeps_mem_directive():
    executor = SlurmStepExecutor(image="img:latest", keep_workdir=True, memory="16G")
    captured = _run_with_fake_sbatch(executor, _ctx())
    assert "#SBATCH --mem=16G" in _script(captured)


def test_zero_gpus_omits_gres_but_keeps_podman_device():
    ctx = _ctx({EXECUTOR_GPU_METADATA_KEY: True})
    executor = SlurmStepExecutor(image="img:latest", keep_workdir=True, gpus=0)
    captured = _run_with_fake_sbatch(executor, ctx)
    script = _script(captured)
    assert "#SBATCH --gres" not in script
    # Node GPU is still exposed to the container via CDI — the partition
    # allocates the whole node, only the explicit Slurm request is rejected.
    assert "--device nvidia.com/gpu=all" in script


def test_positive_gpus_keeps_gres():
    ctx = _ctx({EXECUTOR_GPU_METADATA_KEY: True})
    executor = SlurmStepExecutor(image="img:latest", keep_workdir=True, gpus=2)
    captured = _run_with_fake_sbatch(executor, ctx)
    assert "#SBATCH --gres=gpu:2" in _script(captured)


# ── workdir root knob ────────────────────────────────────────────────────────


def test_workdir_root_relocates_payload_dir(tmp_path):
    executor = SlurmStepExecutor(image="img:latest", keep_workdir=True, workdir_root=str(tmp_path))
    captured = _run_with_fake_sbatch(executor, _ctx())
    assert Path(captured["workdir"]).parent == tmp_path


def test_workdir_root_missing_fails_fast():
    executor = SlurmStepExecutor(image="img:latest", workdir_root="/nonexistent/umrs-test-root")
    with pytest.raises(StepExecutionError, match="UMRS_SLURM_WORKDIR_ROOT"):
        executor.execute(entrypoint="mod:Noop", config={}, ctx=_ctx())


# ── source-dir transport ─────────────────────────────────────────────────────


def test_source_dir_mounted_and_exported(tmp_path):
    metadata = {EXECUTOR_SOURCE_DIR_METADATA_KEY: str(tmp_path), "user_key": 1}
    executor = SlurmStepExecutor(image="img:latest", keep_workdir=True)
    captured = _run_with_fake_sbatch(executor, _ctx(metadata))
    script = _script(captured)
    assert f"-v {tmp_path}:{tmp_path}:ro" in script
    assert f"UMRS_STEP_SOURCE_DIR={tmp_path}" in script
    # Internal key must not leak into the worker-visible metadata.
    assert EXECUTOR_SOURCE_DIR_METADATA_KEY not in captured["payload"]["ctx"]["metadata"]
    assert captured["payload"]["ctx"]["metadata"]["user_key"] == 1


def test_source_dir_relative_rejected():
    ctx = _ctx({EXECUTOR_SOURCE_DIR_METADATA_KEY: "relative/path"})
    executor = SlurmStepExecutor(image="img:latest")
    with pytest.raises(StepExecutionError, match="absolute path"):
        executor.execute(entrypoint="mod:Noop", config={}, ctx=ctx)


def test_source_dir_missing_rejected():
    ctx = _ctx({EXECUTOR_SOURCE_DIR_METADATA_KEY: "/nonexistent/wf-source"})
    executor = SlurmStepExecutor(image="img:latest")
    with pytest.raises(StepExecutionError, match="does not exist"):
        executor.execute(entrypoint="mod:Noop", config={}, ctx=ctx)


def test_source_dir_absent_means_no_mount():
    executor = SlurmStepExecutor(image="img:latest", keep_workdir=True)
    captured = _run_with_fake_sbatch(executor, _ctx())
    assert "UMRS_STEP_SOURCE_DIR" not in _script(captured)


# ── worker-side sys.path preparation ─────────────────────────────────────────


@pytest.fixture
def _clean_worker_source_state():
    saved_dirs = list(step_worker._WORKER_SOURCE_DIRS)
    saved_path = list(sys.path)
    yield
    step_worker._WORKER_SOURCE_DIRS[:] = saved_dirs
    sys.path[:] = saved_path
    for name in ("wfmod",):
        sys.modules.pop(name, None)


def _write_wfmod(root: Path, value: int) -> None:
    (root / "wfmod.py").write_text(f"VALUE = {value}\n", encoding="utf-8")


def test_worker_puts_source_dir_on_sys_path(monkeypatch, tmp_path, _clean_worker_source_state):
    _write_wfmod(tmp_path, 1)
    monkeypatch.setenv("UMRS_STEP_SOURCE_DIR", str(tmp_path))
    step_worker._prepare_worker_source_dir()
    assert sys.path[0] == str(tmp_path)
    sys.modules.pop("wfmod", None)
    import wfmod  # noqa: E402

    assert wfmod.VALUE == 1


def test_worker_purge_on_source_dir_change(monkeypatch, tmp_path, _clean_worker_source_state):
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()
    _write_wfmod(dir_a, 10)
    _write_wfmod(dir_b, 20)

    monkeypatch.setenv("UMRS_STEP_SOURCE_DIR", str(dir_a))
    step_worker._prepare_worker_source_dir()
    sys.modules.pop("wfmod", None)
    import wfmod  # noqa: E402

    assert wfmod.VALUE == 10

    # Switch to a different workflow source: the previously imported
    # same-named module must be dropped so the new code loads.
    monkeypatch.setenv("UMRS_STEP_SOURCE_DIR", str(dir_b))
    step_worker._prepare_worker_source_dir()
    assert "wfmod" not in sys.modules
    import wfmod as wfmod_b  # noqa: E402

    assert wfmod_b.VALUE == 20


def test_worker_missing_dir_is_noop(monkeypatch, _clean_worker_source_state):
    monkeypatch.setenv("UMRS_STEP_SOURCE_DIR", "/nonexistent/worker-source")
    before = list(sys.path)
    step_worker._prepare_worker_source_dir()
    assert sys.path == before


# ── runner → executor threading ──────────────────────────────────────────────


@pytest.fixture
def _clean_runner_source_state():
    """run_workflow_record mutates process-global import state (sys.path,
    sys.modules, _KNOWN_SOURCE_ROOTS); restore it so neighbouring test
    files that ship same-named modules are not poisoned."""
    from umrs.server import workflow_runner as wr

    saved_roots = list(wr._KNOWN_SOURCE_ROOTS)
    saved_path = list(sys.path)
    saved_mods = set(sys.modules)
    yield
    wr._KNOWN_SOURCE_ROOTS[:] = saved_roots
    sys.path[:] = saved_path
    for name in set(sys.modules) - saved_mods:
        sys.modules.pop(name, None)


def _write_noop_mod(root: Path, marker: str) -> None:
    (root / "mod.py").write_text(
        "from umrs.pipeline_sdk import Step, StepResult\n\n"
        "class Noop(Step):\n"
        "    name = 'noop'\n"
        "    def setup(self, config): pass\n"
        "    def teardown(self): pass\n"
        f"    def run(self, ctx): return StepResult(output_keys=['{marker}'])\n",
        encoding="utf-8",
    )


def _workflow_json() -> dict:
    return {
        "execution_mode": "in_process",
        "steps": [{"name": "s1", "entrypoint": "mod:Noop", "config": {}}],
    }


def test_runner_threads_source_path_into_step_metadata(tmp_path, _clean_runner_source_state):
    from umrs.server import workflow_runner as wr
    from umrs.server.workflow_execution_registry import ExecutionRecord

    _write_noop_mod(tmp_path, "x")

    captured: dict = {}

    class _CaptureExecutor:
        def execute(self, *, entrypoint, config, ctx):
            del entrypoint, config
            captured["metadata"] = dict(ctx.metadata)
            return StepResult(output_keys=[])

    record = ExecutionRecord(execution_id="wf-src", workflow_name="wf", user_id="u")
    wr.run_workflow_record(
        record,
        source_path=str(tmp_path),
        workflow_json=_workflow_json(),
        executor=_CaptureExecutor(),
    )
    assert record.status == "succeeded"
    assert captured["metadata"][EXECUTOR_SOURCE_DIR_METADATA_KEY] == str(tmp_path)


def test_runner_purge_covers_foreign_workflow_trees(tmp_path, _clean_runner_source_state):
    """Two registered workflows shipping a same-named ``mod`` package:
    the second run must see its own code, not the first run's cached
    module (bundle-variant hazard: pipeline vs -nogate sibling)."""
    from umrs.server import workflow_runner as wr
    from umrs.server.workflow_execution_registry import ExecutionRecord

    dir_a = tmp_path / "wf_a"
    dir_b = tmp_path / "wf_b"
    dir_a.mkdir()
    dir_b.mkdir()
    _write_noop_mod(dir_a, "from-a")
    _write_noop_mod(dir_b, "from-b")

    rec_a = ExecutionRecord(execution_id="wf-a", workflow_name="wf-a", user_id="u")
    wr.run_workflow_record(rec_a, source_path=str(dir_a), workflow_json=_workflow_json())
    assert rec_a.status == "succeeded"
    assert rec_a.steps["s1"].extra["output_keys"] == ["from-a"]

    rec_b = ExecutionRecord(execution_id="wf-b", workflow_name="wf-b", user_id="u")
    wr.run_workflow_record(rec_b, source_path=str(dir_b), workflow_json=_workflow_json())
    assert rec_b.status == "succeeded"
    assert rec_b.steps["s1"].extra["output_keys"] == ["from-b"]
