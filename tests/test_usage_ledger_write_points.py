"""The usage ledger's write points on the platform-login lanes (ADR-112 wave
D; plan §6 C/D.0 D2): the service staged step, ``hpc_script`` and cluster
loops. Each writes pending before the submit, submitted after the job id,
and (staged step, ``hpc_script``) settles at the terminal ``sacct`` row; a
pending refusal (budget, ledger) submits nothing. (The gated lane's write
points: ``test_gated_grant_lane.py``.)
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest
from test_remote_slurm_step_executor import (
    RUN,
    _no_bucket_registry,  # noqa: F401 — autouse fixture
    _video_run,
)
from test_remote_slurm_step_executor import cluster as _lane_cluster  # noqa: F401 — fixture
from test_remote_slurm_step_executor import tree as _lane_tree  # noqa: F401 — fixture
from umrs.hpc_accounts.usage_ledger import LedgerRefused, get_usage_ledger
from umrs.jobs import hpc_ledger
from umrs.orchestrator.hpc_identity import HpcIdentity, Lane
from umrs.server.step_ledger import LEDGER_METADATA_KEY


@pytest.fixture
def cluster(request):
    return request.getfixturevalue("_lane_cluster")


@pytest.fixture
def tree(request):
    return request.getfixturevalue("_lane_tree")


IDENTITY = HpcIdentity("site-x", Lane.USER_WORK, "service_account", "op", "grp-a", False, True)


def _rows() -> list[dict[str, Any]]:
    return get_usage_ledger().tables["rows"]


# ── the service staged step ──────────────────────────────────────────────────


def test_a_service_step_is_pending_submitted_then_settled_from_sacct(cluster, tree, monkeypatch):
    import test_remote_slurm_step_executor as lane_tests

    original = lane_tests._ctx

    def with_owner(*a: Any, **k: Any) -> Any:
        k.setdefault("metadata", {})[LEDGER_METADATA_KEY] = {"person": "alice", "project_id": "p1"}
        return original(*a, **k)

    monkeypatch.setattr(lane_tests, "_ctx", with_owner)
    result, _s3, _recorded = _video_run(cluster, tree)
    assert [r["state"] for r in _rows()] == ["pending", "submitted", "COMPLETED"]
    pending, submitted, settled = _rows()
    assert (pending["person"], pending["project_id"], pending["lane"]) == (
        "alice",
        "p1",
        "staged_step",
    )
    assert (pending["access_mode"], pending["method"], pending["source"]) == (
        "service_account",
        "platform",
        "estimate",
    )
    assert submitted["scheduler_job_id"] == result.extra["slurm_job"]["job_id"]
    assert (settled["source"], settled["tag"]) == ("sacct", f"umrs-step-{RUN}-s1")
    assert settled["attempt_id"] == submitted["attempt_id"] == pending["attempt_id"]
    assert settled["attempt_id"].startswith(f"{RUN}/s1/1/")


def test_a_budget_refusal_on_the_service_lane_submits_nothing(cluster, tree):
    """K14: the pending write binds; nothing is submitted."""
    import test_remote_slurm_step_executor as lane_tests
    from step_lane_harness import FakeS3
    from umrs.server.step_executor import EXECUTOR_GPU_COUNT_METADATA_KEY, StepExecutionError

    get_usage_ledger().tables["budgets"]["p1"] = 0.001  # 3.6 s: less than one GPU-minute
    ex = cluster.executor()
    ex._needs_gpu = lambda _ctx: True  # type: ignore[method-assign] — the step asks for a GPU
    owner = {
        LEDGER_METADATA_KEY: {"person": "alice", "project_id": "p1"},
        EXECUTOR_GPU_COUNT_METADATA_KEY: 1,
    }
    ctx = lane_tests._ctx(FakeS3({}), tree, metadata=owner)
    ex.bind_run(RUN)
    with pytest.raises(StepExecutionError) as info:
        ex.execute(entrypoint="steps.lane:Upper", config={"step_name": "s1"}, ctx=ctx)
    assert "hpcaccess.budget_exceeded" in info.value.message
    assert cluster.calls("sbatch") == [] and _rows() == []


def test_a_budget_refusal_fails_the_service_step_once_without_its_retries(cluster, tree):
    """K14 through the runner: the step's ``max_retries`` does not apply to a
    ledger refusal; one attempt, nothing submitted, the run stops."""
    from umrs.hpc_accounts.usage_ledger import LedgerUnit, Resources
    from umrs.server import workflow_runner as wr
    from umrs.server.workflow_execution_registry import ExecutionRecord

    elsewhere = LedgerUnit(
        venue="v2",
        attempt_id="wf-x/s/1",
        person="bob",
        lane="staged_step",
        tag="t",
        project_id="p1",
    )
    get_usage_ledger().pending(
        elsewhere, Resources(1, 1, 3600)
    )  # a GPU-hour reserved at another venue
    get_usage_ledger().tables["budgets"]["p1"] = 0.5
    record = ExecutionRecord(
        execution_id=RUN, workflow_name="lane", user_id="alice", params={"project_id": "p1"}
    )
    step = {
        "name": "s1",
        "entrypoint": "steps.lane:Upper",
        "config": {},
        "max_retries": 2,
        "retry_delay": 0,
    }
    wr.run_workflow_record(
        record, source_path=str(tree), workflow_json={"steps": [step]}, executor=cluster.executor()
    )
    assert record.status == "failed"
    error = str(record.steps["s1"].error)
    assert "hpcaccess.budget_exceeded" in error and "attempts" not in error, error
    assert cluster.calls("sbatch") == [] and [r["attempt_id"] for r in _rows()] == ["wf-x/s/1"]


# ── the local Slurm staged lane ──────────────────────────────────────────────


def _local_ctx():
    from umrs.pipeline_sdk.step import StepContext

    return StepContext(
        s3=None, input_bucket="data", input_keys=[], output_bucket="out",
        config={}, metadata={LEDGER_METADATA_KEY: {"person": "alice", "project_id": "p1"}},
        run_id=RUN,
    )  # fmt: skip


def _accept_local_submit(monkeypatch, executor):
    import pickle

    from umrs.pipeline_sdk.step import StepResult

    def submit(_path, workdir):
        with open(f"{workdir}/output.pkl", "wb") as handle:
            pickle.dump({"status": "ok", "result": StepResult(output_keys=[])}, handle)
        return "777"

    monkeypatch.setattr(executor, "_submit_sbatch", submit)
    monkeypatch.setattr(executor, "_check_job_status", lambda _job_id: "COMPLETED")
    monkeypatch.setattr(
        "umrs.server.slurm_step_executor.measure_slurm_accounting",
        lambda _job_id, **_: {"elapsed_seconds": 60.0, "gpus": 1, "gpu_seconds": 60.0},
    )


def test_the_local_slurm_step_writes_pending_submitted_and_settled(monkeypatch, tmp_path):
    from umrs.server.slurm_step_executor import SlurmStepExecutor

    executor = SlurmStepExecutor(image="img:1", poll_interval=0, workdir_root=str(tmp_path))
    monkeypatch.setenv("UMRS_STEP_GPU", "1")
    _accept_local_submit(monkeypatch, executor)
    ctx = _local_ctx()
    executor.execute(entrypoint="m:S", config={"step_name": "s1"}, ctx=ctx)
    assert [r["state"] for r in _rows()] == ["pending", "submitted", "COMPLETED"]
    pending, submitted, settled = _rows()
    assert (pending["alloc_gpus"], pending["elapsed_s"]) == (1, 3600)
    assert (
        pending["person"],
        pending["project_id"],
        pending["tag"].startswith("umrs-step-"),
        pending["tag"] != "umrs-step",
    ) == ("alice", "p1", True, True)
    assert pending["attempt_id"].startswith(f"{RUN}/s1/1/")
    assert submitted["scheduler_job_id"] == "777" and settled["source"] == "sacct"


def test_a_local_budget_refusal_submits_nothing(monkeypatch, tmp_path):
    from umrs.server.slurm_step_executor import SlurmStepExecutor
    from umrs.server.step_executor import StepExecutionError

    get_usage_ledger().tables["budgets"]["p1"] = 0.001
    executor = SlurmStepExecutor(image="img:1", poll_interval=0, workdir_root=str(tmp_path))
    monkeypatch.setenv("UMRS_STEP_GPU", "1")
    _accept_local_submit(monkeypatch, executor)
    with pytest.raises(StepExecutionError, match="hpcaccess.budget_exceeded"):
        executor.execute(entrypoint="m:S", config={"step_name": "s1"}, ctx=_local_ctx())
    assert _rows() == []


def test_a_definite_local_sbatch_refusal_is_not_submitted(monkeypatch, tmp_path):
    from umrs.server.slurm_step_executor import SlurmStepExecutor
    from umrs.server.step_executor import StepExecutionError

    executor = SlurmStepExecutor(image="img:1", workdir_root=str(tmp_path))
    result = SimpleNamespace(returncode=1, stdout="", stderr="refused")
    monkeypatch.setattr(executor, "_run_cmd", lambda *_a, **_k: result)
    with pytest.raises(StepExecutionError, match="sbatch submission failed"):
        executor.execute(entrypoint="m:S", config={"step_name": "s1"}, ctx=_local_ctx())
    assert [r["state"] for r in _rows()] == ["pending", "not_submitted"]


def test_an_unresolved_local_sbatch_answer_leaves_pending(monkeypatch, tmp_path):
    from umrs.server.slurm_step_executor import SlurmStepExecutor
    from umrs.server.step_executor import StepExecutionError

    executor = SlurmStepExecutor(image="img:1", workdir_root=str(tmp_path))
    result = SimpleNamespace(returncode=0, stdout="unparseable", stderr="")
    monkeypatch.setattr(executor, "_run_cmd", lambda *_a, **_k: result)
    with pytest.raises(StepExecutionError, match="Unexpected sbatch output"):
        executor.execute(entrypoint="m:S", config={"step_name": "s1"}, ctx=_local_ctx())
    assert [r["state"] for r in _rows()] == ["pending"]


# ── hpc_script ───────────────────────────────────────────────────────────────


def test_the_script_estimate_reads_its_own_sbatch_arguments():
    est = hpc_ledger.script_estimate(["--time=01:00:00", "-c", "8", "--gres=gpu:a100:2"])
    assert (est.cpus, est.gpus, est.time_s) == (8, 2, 3600)
    assert tuple(hpc_ledger.script_estimate([])) == (1, 0, 0)
    assert hpc_ledger.script_tag({"script_path": "/home/a/run.sh", "sbatch_args": []}) == "run.sh"
    assert hpc_ledger.script_tag({"script_path": "x", "sbatch_args": ["-J", "train"]}) == "train"


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (["--time=1-02:03:04", "--gpus=a100:2"], (1, 2, 93840)),
        (["--time=05:30", "--gpus=gpu:2"], (1, 2, 360)),
    ],
)
def test_script_estimate_reads_typed_gpu_forms_and_slurm_times(args, expected):
    assert tuple(hpc_ledger.script_estimate(args)) == expected


@pytest.mark.parametrize("args", [["--gpus=1"], ["--time=10", "--gpus=a100:wat"]])
def test_unknown_gpu_estimates_are_refused_under_a_budget_before_ssh(args):
    from umrs.jobs.hpc_script import HpcScriptBackend

    get_usage_ledger().tables["budgets"]["p1"] = 0.01
    job = _job()
    job.spec["sbatch_args"] = args
    called: list[str] = []

    async def factory() -> Any:
        called.append("ssh")
        return _Ssh("Submitted batch job 4242\n")

    with pytest.raises(LedgerRefused, match="hpcaccess.budget_exceeded"):
        asyncio.run(
            HpcScriptBackend(ssh=factory, identity=lambda _j: IDENTITY).submit_job(job, _attempt())
        )
    assert called == [] and _rows() == []


class _Ssh:
    def __init__(self, *outs: Any) -> None:
        self.outs, self.commands = list(outs), []

    async def run_remote(self, command: str, **_kw: Any) -> Any:
        self.commands.append(command)
        out = self.outs.pop(0)
        return SimpleNamespace(ok=True, returncode=0, stdout=out, stderr="")


def _job() -> Any:
    spec = {"script_path": "/home/a/run.sh", "sbatch_args": ["--time=10", "--gpus=1"]}
    return SimpleNamespace(job_id="job-1", project_id="p1", user_id="alice", spec=spec)


def _attempt() -> Any:
    return SimpleNamespace(attempt_id="att-1", metrics={}, backend_ref="")


def test_hpc_script_writes_pending_then_submitted_then_settles_at_the_terminal_state():
    from umrs.jobs.hpc_script import HpcScriptBackend, HpcScriptStatusChecker

    job, attempt = _job(), _attempt()
    ssh = _Ssh("Submitted batch job 4242\n")

    async def factory() -> Any:
        return ssh

    ref = asyncio.run(
        HpcScriptBackend(ssh=factory, identity=lambda _j: IDENTITY).submit_job(job, attempt)
    )
    assert ref.endswith("slurm=4242")
    assert [r["state"] for r in _rows()] == ["pending", "submitted"]
    assert (_rows()[0]["alloc_gpus"], _rows()[0]["elapsed_s"], _rows()[0]["lane"]) == (
        1,
        600,
        "hpc_script",
    )
    attempt.backend_ref = ref
    checker_ssh = _Ssh(
        "",
        "COMPLETED\n",
        "4242|COMPLETED|00:05:00|cpu=2,gres/gpu=1|2|2026-11-09T01:00:00|2026-11-09T01:05:00\n",
    )

    async def checker_factory() -> Any:
        return checker_ssh

    result = asyncio.run(
        HpcScriptStatusChecker(ssh_factory=checker_factory).check_status(attempt, job)
    )
    assert "succeeded" in str(result.status).lower()
    settled = _rows()[-1]
    assert (settled["state"], settled["source"], settled["elapsed_s"], settled["alloc_gpus"]) == (
        "COMPLETED",
        "sacct",
        300,
        1,
    )
    assert "Elapsed,AllocTRES,AllocCPUS" in checker_ssh.commands[-1]


def test_a_refused_hpc_script_pending_raises_before_any_ssh():
    from umrs.jobs.hpc_script import HpcScriptBackend

    get_usage_ledger().tables["budgets"]["p1"] = 0.01
    called: list[str] = []

    async def factory() -> Any:
        called.append("ssh")
        return _Ssh()

    with pytest.raises(LedgerRefused, match="hpcaccess.budget_exceeded"):
        asyncio.run(
            HpcScriptBackend(ssh=factory, identity=lambda _j: IDENTITY).submit_job(
                _job(), _attempt()
            )
        )
    assert called == [] and _rows() == []


def test_a_refused_sbatch_is_not_submitted():
    from umrs.jobs.hpc_script import HpcScriptBackend

    class _Refusing(_Ssh):
        async def run_remote(self, *_a: Any, **_kw: Any) -> Any:
            return SimpleNamespace(
                ok=False, returncode=1, stdout="", stderr="sbatch: error: invalid account"
            )

    async def factory() -> Any:
        return _Refusing()

    with pytest.raises(RuntimeError, match="sbatch failed"):
        asyncio.run(
            HpcScriptBackend(ssh=factory, identity=lambda _j: IDENTITY).submit_job(
                _job(), _attempt()
            )
        )
    assert [r["state"] for r in _rows()] == ["pending", "not_submitted"]


# ── cluster loops ────────────────────────────────────────────────────────────


def test_a_cluster_loop_job_is_ledgered_on_submit(monkeypatch):
    from umrs.jobs.cluster_venue import ClusterSlurmSubmitter
    from umrs.jobs.models import Job, JobAttempt, JobType
    from umrs.shared import handshake

    monkeypatch.setattr(handshake, "handshake_secret", lambda: "test-secret")

    async def submit(_wrapper: str, _env: dict) -> Any:
        return SimpleNamespace(job_id="777")

    job = Job(job_id="loop-job", project_id="p1", user_id="alice", job_type=JobType.RESEARCH_LOOP,
              spec={}, idempotency_key="k1")  # fmt: skip
    attempt = JobAttempt(attempt_id="att-9", job_id="loop-job", attempt_no=1, backend="slurm")
    sub = ClusterSlurmSubmitter(submit, identity=lambda _j: IDENTITY)
    ref = asyncio.run(sub.submit(job=job, attempt=attempt, spec={"loop_id": "L1"}))
    assert ref.endswith("slurm_id=777")
    assert [(r["state"], r["lane"], r["scheduler_job_id"]) for r in _rows()] == [
        ("pending", "cluster_loop", ""),
        ("submitted", "cluster_loop", "777"),
    ]
    tags = {r["tag"] for r in _rows()}
    assert len(tags) == 1 and tags.pop() != "umrs-loop-job"
    assert all(r["elapsed_s"] > 0 for r in _rows())
