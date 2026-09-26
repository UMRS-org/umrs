"""Remote Slurm step executor: the staged connected-HPC lane (ADR-074).

End to end through a fake SSH hop (local ``bash -c``) running the REAL
login-node helper against a tmp workdir root, with fake Slurm CLIs whose
``sbatch`` runs the real step worker against the staged directories (see
``step_lane_harness``). Staging in and out goes through the real
``step_staging`` stager and collector; the step reads and writes through the
real file shim. The transport of 2026-09-25 (``cat`` / ``test -f`` reads,
``output.pkl``, an S3 endpoint handed to the step) is gone by design (build
spec 2026-09-26 §3-§7), so its tests were replaced here.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import pytest
from step_lane_harness import HELPER, IMAGE, FakeS3, make_cluster, write_step_tree
from umrs.orchestrator.site_config import StepExecutorConfig, _step_executor_env
from umrs.orchestrator.ssh_control import SSHControlConfig, SSHControlManager
from umrs.pipeline_sdk.step import StepContext
from umrs.server.remote_slurm_step_executor import RemoteSlurmStepExecutor, safe_segment
from umrs.server.step_executor import (
    EXECUTOR_IMAGE_METADATA_KEY,
    EXECUTOR_SOURCE_DIR_METADATA_KEY,
    StepExecutionError,
    build_step_executor,
)
from umrs.server.step_reaper import step_jobs, unreaped_steps
from umrs.server.step_staging import STAGE_REQUEST_METADATA_KEY, register_bucket_registry

OWNER = "alice"
IN_BUCKET = "alice.proj.data"
OUT_BUCKET = "alice.proj.data-output"
RUN = "wf-0123456789ab"


@pytest.fixture(autouse=True)
def _no_bucket_registry():
    register_bucket_registry(None)
    yield
    register_bucket_registry(None)


@pytest.fixture
def cluster(tmp_path, monkeypatch):
    return make_cluster(tmp_path, monkeypatch)


@pytest.fixture
def tree(tmp_path) -> Path:
    return write_step_tree(tmp_path / "bundle")


def _ctx(s3, tree, *, step="s1", stage_inputs=None, params=None, metadata=None) -> StepContext:
    meta = {
        EXECUTOR_SOURCE_DIR_METADATA_KEY: str(tree),
        STAGE_REQUEST_METADATA_KEY: {
            "owner": OWNER,
            "step": step,
            "stage_inputs": stage_inputs or {},
            "params": params or {},
        },
        "user_key": "kept",
    }
    meta.update(metadata or {})
    return StepContext(
        s3=s3,
        input_bucket=IN_BUCKET,
        input_keys=[],
        output_bucket=OUT_BUCKET,
        config={"step_name": step, "api_key": "sk-platform"},
        metadata=meta,
        run_id=RUN,
    )


def _video_run(cluster, tree, **executor_kw):
    s3 = FakeS3({(IN_BUCKET, "in/video.txt"): b"frames"})
    ex = cluster.executor(**executor_kw)
    recorded: list[tuple[str, dict]] = []
    ex.bind_run(RUN, on_job=lambda step, job: recorded.append((step, dict(job))))
    config = {
        "step_name": "s1",
        "input_key": "in/video.txt",
        "api_key": "sk-platform",
        "headers": {"Authorization": "Bearer platform"},
        "_routing": {"endpoint_id": "e1"},
    }
    ctx = _ctx(
        s3,
        tree,
        stage_inputs={"params": ["video"]},
        params={"video": f"s3://{IN_BUCKET}/in/video.txt"},
    )
    result = ex.execute(entrypoint="steps.lane:Upper", config=config, ctx=ctx)
    return result, s3, recorded


def test_uploaded_step_runs_end_to_end_on_the_staged_lane(cluster, tree):
    result, s3, _recorded = _video_run(cluster, tree)
    physical = f"runs/{RUN}/s1/result/out.txt"
    assert result.output_keys == [physical]
    assert s3.objects[(OUT_BUCKET, physical)] == b"FRAMES!"
    assert result.extra["config_keys"] == ["input_key", "step_name"]  # no platform key
    assert result.extra["slurm"]["gpu_seconds"] == 0.0  # measured, the step's forgery gone


def test_the_job_is_recorded_and_its_workdir_removed_after_completion(cluster, tree):
    result, _s3, recorded = _video_run(cluster, tree)
    [job_id] = cluster.jobs()
    assert result.extra["slurm_job"]["job_id"] == job_id
    assert cluster.workdirs() == []  # removed once COMPLETED was confirmed
    states = _recorded(recorded, "state")
    # on record from mkwd (no job yet), then submitted, then ended
    assert (states[0], states[1], states[-1]) == ("", "SUBMITTED", "COMPLETED")
    assert set(_recorded(recorded, "job_id")[1:]) == {job_id}
    [workdir] = set(_recorded(recorded, "workdir"))
    assert workdir.startswith(f"{cluster.root}/{RUN}-s1-1-")


def _recorded(recorded: list, key: str) -> list:
    return [job[key] for _step, job in recorded]


def test_an_unmeasured_step_keeps_no_accounting_it_forged(cluster, tree):
    cluster.behave(acct_rc=1)  # sacct refused, as a forced-command key would
    result, _s3, _recorded = _video_run(cluster, tree)
    assert "slurm" not in result.extra  # the step wrote gpu_seconds=1e9 itself
    assert result.extra["slurm_job"]["job_id"] in cluster.jobs()


def test_the_lane_speaks_only_helper_verbs_to_the_cluster(cluster, tree):
    _video_run(cluster, tree)
    helper = f"{sys.executable} -I {HELPER} "
    other = [c for c in cluster.commands if not c.startswith(helper)]
    assert [c.split()[:2] for c in other] == [["sacct", "--job"]]  # accounting only
    verbs = cluster.helper_verbs()
    assert verbs[:5] == ["find-jobs", "mkwd", "stage-in", "put", "put"]
    assert verbs[5] == "submit" and "status" in verbs
    assert verbs[-3:] == ["read-reply", "collect-out", "rmwd"]


def test_the_payload_holds_no_platform_key_and_no_endpoint(cluster, tree):
    _result, _s3, recorded = _video_run(cluster, tree, keep_workdir=True)
    wd = Path(recorded[-1][1]["workdir"])
    payload = pickle.loads((wd / "input.pkl").read_bytes())
    keys = set(payload["config"]) | set(payload["ctx"]["config"])
    assert keys & {"api_key", "headers", "_routing"} == set()
    assert payload["s3_config"] == {}
    assert payload["ctx"]["metadata"] == {"user_key": "kept"}
    script = (wd / "job.sbatch").read_text()
    assert ("sk-platform" in script, "Bearer" in script) == (False, False)


def test_the_workdir_holds_the_staged_layout(cluster, tree):
    _result, _s3, recorded = _video_run(cluster, tree, keep_workdir=True)
    wd = Path(recorded[-1][1]["workdir"])
    assert (wd / "src" / "steps" / "lane.py").is_file()
    assert (wd / "in" / "manifest.json").is_file()
    assert (wd / "in" / IN_BUCKET / "in" / "video.txt").read_bytes() == b"frames"
    assert (wd / "log" / "slurm.err").exists()
    assert "--error=" + str(wd / "log" / "slurm.err") in (wd / "job.sbatch").read_text()


def test_a_failing_step_raises_its_error_and_leaves_nothing(cluster, tree):
    ex = cluster.executor()
    with pytest.raises(StepExecutionError, match="step exploded on purpose") as info:
        ex.execute(entrypoint="steps.lane:Boom", config={}, ctx=_ctx(FakeS3(), tree))
    assert info.value.phase == "run"
    assert "collect-out" not in cluster.helper_verbs()
    assert cluster.workdirs() == []


def test_a_symlink_left_in_out_fails_the_step_and_reaches_no_bucket(cluster, tree):
    s3 = FakeS3()
    ex = cluster.executor()
    with pytest.raises(StepExecutionError, match="not a regular"):
        ex.execute(entrypoint="steps.lane:Sneaky", config={}, ctx=_ctx(s3, tree))
    assert s3.objects == {}
    assert cluster.workdirs() == []


def test_a_pickle_reply_is_never_read_on_this_lane(cluster, tree):
    cluster.behave(
        run=False,
        states=["RUNNING", "COMPLETED"],
        files={"reply/output.pkl": "cos\nsystem\n(S'id'\ntR."},
    )
    with pytest.raises(StepExecutionError, match="wrote no JSON reply"):
        cluster.executor().execute(
            entrypoint="steps.lane:Upper", config={}, ctx=_ctx(FakeS3(), tree)
        )
    assert cluster.workdirs() == []


def test_a_reply_over_the_cap_or_through_a_link_is_refused(cluster, tree, tmp_path):
    cluster.behave(run=False, states=["COMPLETED"], files={"reply/output.json": "x" * 200})
    with pytest.raises(StepExecutionError, match="exceeds 100 bytes"):
        cluster.executor(reply_cap=100).execute(
            entrypoint="steps.lane:Upper", config={}, ctx=_ctx(FakeS3(), tree)
        )
    secret = tmp_path / "id_rsa"
    secret.write_text("-----BEGIN PRIVATE KEY-----")
    cluster.behave(run=False, states=["COMPLETED"], symlinks={"reply/output.json": str(secret)})
    with pytest.raises(StepExecutionError, match="symlink") as info:
        cluster.executor().execute(
            entrypoint="steps.lane:Upper", config={}, ctx=_ctx(FakeS3(), tree)
        )
    assert "PRIVATE" not in info.value.message


def test_a_closed_lane_refuses_before_any_ssh(cluster, tree):
    with pytest.raises(StepExecutionError, match="remote_enabled"):
        cluster.executor(enabled=False).execute(
            entrypoint="steps.lane:Upper", config={}, ctx=_ctx(FakeS3(), tree)
        )
    with pytest.raises(StepExecutionError, match="UMRS_STEP_HELPER"):
        cluster.executor(helper_path="").execute(
            entrypoint="steps.lane:Upper", config={}, ctx=_ctx(FakeS3(), tree)
        )
    with pytest.raises(StepExecutionError, match="UMRS_SLURM_WORKDIR_ROOT"):
        cluster.executor(workdir_root="relative/steps").execute(
            entrypoint="steps.lane:Upper", config={}, ctx=_ctx(FakeS3(), tree)
        )
    assert cluster.commands == []


def test_another_owners_input_is_refused_before_any_ssh(cluster, tree):
    s3 = FakeS3({("bob.proj.data", "secret.txt"): b"bob's"})
    ctx = _ctx(
        s3, tree, stage_inputs={"params": ["v"]}, params={"v": "s3://bob.proj.data/secret.txt"}
    )
    with pytest.raises(StepExecutionError, match="does not belong to the run owner"):
        cluster.executor().execute(entrypoint="steps.lane:Upper", config={}, ctx=ctx)
    assert cluster.commands == []


def test_an_image_off_the_allow_list_is_refused_before_any_ssh(cluster, tree, monkeypatch):
    monkeypatch.delenv("UMRS_STEP_ALLOWED_IMAGES", raising=False)
    ctx = _ctx(FakeS3(), tree, metadata={EXECUTOR_IMAGE_METADATA_KEY: "ghcr.io/evil/step:1"})
    with pytest.raises(StepExecutionError, match="not an allowed step image"):
        cluster.executor().execute(entrypoint="steps.lane:Upper", config={}, ctx=ctx)
    assert cluster.commands == []


def test_an_allow_listed_image_runs(cluster, tree, monkeypatch):
    monkeypatch.setenv("UMRS_STEP_ALLOWED_IMAGES", "localhost/gpu-step:2")
    ctx = _ctx(FakeS3(), tree, metadata={EXECUTOR_IMAGE_METADATA_KEY: "localhost/gpu-step:2"})
    ex = cluster.executor(keep_workdir=True)
    ex.execute(entrypoint="steps.lane:Upper", config={}, ctx=ctx)
    [wd] = cluster.workdirs()
    assert "_step_image=localhost/gpu-step:2" in (wd / "job.sbatch").read_text()


def test_two_step_names_never_share_a_job_name():
    assert safe_segment("a.b", "step") != safe_segment("a_b", "step")
    assert safe_segment("ok-name_1", "step") == "ok-name_1"
    long = "x" * 60
    assert len(safe_segment(long, "step")) == 40
    assert safe_segment("-lead", "step")[0].isalnum()
    assert safe_segment("...", "step").startswith("step-")


def test_cleanup_never_reaches_the_servers_own_filesystem(cluster, tmp_path):
    victim = tmp_path / "victim"
    victim.mkdir()
    ex = cluster.executor()
    ex._remove_workdir(str(victim))  # the base class would rmtree locally
    assert victim.exists()
    assert cluster.helper_verbs() == ["rmwd"]


# ── the runner, end to end ──────────────────────────────────────────────────


def test_a_two_step_uploaded_workflow_runs_through_the_runner(cluster, tree):
    """The runner hands the stager the owner, the child stages the parent's
    physical output key, and the step record carries the Slurm job."""
    from umrs.server import workflow_runner as wr
    from umrs.server.workflow_execution_registry import ExecutionRecord

    record = ExecutionRecord(execution_id=RUN, workflow_name="lane", user_id=OWNER)
    workflow = {
        "execution_mode": "slurm",
        "steps": [
            {"name": "parent", "entrypoint": "steps.lane:Upper",
             "config": {"out": "a.txt"}},
            {"name": "child", "entrypoint": "steps.lane:Upper", "depends_on": ["parent"],
             "config": {"out": "b.txt"}, "timeout_seconds": 600},
        ],
    }  # fmt: skip
    ex = cluster.executor()
    wr.run_workflow_record(record, source_path=str(tree), workflow_json=workflow, executor=ex)
    assert record.status == "succeeded", record.error
    blobs = record.results["blobs"]
    outputs = (
        blobs[f"wf-runner-output/runs/{RUN}/parent/a.txt"],
        blobs[f"wf-runner-output/runs/{RUN}/child/b.txt"],
    )
    assert outputs == (b"!", b"!!")
    child = record.steps["child"].extra
    assert child["output_keys"] == [f"runs/{RUN}/child/b.txt"]
    assert child["extra"]["slurm_job"]["job_id"] in cluster.jobs()
    names = sorted(job["name"].rsplit("-", 1)[-1] for job in cluster.jobs().values())
    assert names == ["child", "parent"]  # one job per step
    assert cluster.workdirs() == []
    _assert_nothing_left_to_reap(record, child["extra"]["slurm_job"]["job_id"])


def _assert_nothing_left_to_reap(record, child_job_id: str) -> None:
    """The result's set_step(extra=...) kept the job facts, all cleaned."""
    [facts] = step_jobs(record.steps["child"]).values()
    assert (facts["job_id"], facts["cleaned"]) == (child_job_id, True)
    assert unreaped_steps(record) == {}


def test_the_runner_records_the_job_while_it_runs(cluster, tree):
    from umrs.server import workflow_runner as wr
    from umrs.server.workflow_execution_registry import ExecutionRecord

    cluster.behave(run=False, states=["RUNNING", "FAILED"])
    record = ExecutionRecord(execution_id=RUN, workflow_name="lane", user_id=OWNER)
    workflow = {"steps": [{"name": "s1", "entrypoint": "steps.lane:Upper", "config": {}}]}
    wr.run_workflow_record(
        record, source_path=str(tree), workflow_json=workflow, executor=cluster.executor()
    )
    assert record.status == "failed"
    assert "status FAILED" in record.steps["s1"].error
    [job] = step_jobs(record.steps["s1"]).values()
    assert job["job_id"] in cluster.jobs() and job["state"] == "FAILED"
    assert job["workdir"].startswith(str(cluster.root))


# ── factory and site wiring (unchanged) ─────────────────────────────────────


def test_missing_ssh_target_fails_at_build():
    with pytest.raises(RuntimeError, match="ssh_target"):
        RemoteSlurmStepExecutor(ssh_config=SSHControlConfig(ssh_target=""))


def test_factory_selects_remote_executor(monkeypatch):
    import umrs.server.remote_slurm_step_executor as mod

    monkeypatch.setattr(mod, "_site_ssh_config", lambda: SSHControlConfig(ssh_target="login"))
    monkeypatch.setenv("UMRS_STEP_EXECUTOR", "slurm-remote")
    assert isinstance(build_step_executor(), RemoteSlurmStepExecutor)


def test_slurm_mode_on_a_remote_site_submits_remotely(monkeypatch):
    import umrs.server.remote_slurm_step_executor as mod

    monkeypatch.setattr(mod, "_site_ssh_config", lambda: SSHControlConfig(ssh_target="login"))
    monkeypatch.setenv("UMRS_SLURM_STEP_SUBMIT", "remote")
    assert isinstance(build_step_executor(name="slurm"), RemoteSlurmStepExecutor)


def test_executor_reads_the_lane_env(monkeypatch):
    monkeypatch.setenv("UMRS_STEP_REMOTE_ENABLED", "1")
    monkeypatch.setenv("UMRS_STEP_HELPER", "/c/umrs_step_helper.py")
    monkeypatch.setenv("UMRS_STEP_VENUE_SCRIPT", "/c/venue.sh")
    monkeypatch.setenv("UMRS_STEP_IMAGE_ARCHIVE_DIR", "/c/images")
    monkeypatch.setenv("UMRS_SLURM_QUEUE_TIMEOUT", "02:00:00")
    monkeypatch.setenv("UMRS_STEP_IMAGE", IMAGE)
    monkeypatch.delenv("UMRS_STEP_EXECUTOR_TIMEOUT", raising=False)
    ex = RemoteSlurmStepExecutor(ssh_config=SSHControlConfig(ssh_target="login"))
    assert (ex.enabled, ex.helper_path, ex.venue_script, ex.image_archive_dir) == (
        True, "/c/umrs_step_helper.py", "/c/venue.sh", "/c/images",
    )  # fmt: skip
    assert ex.queue_timeout == "02:00:00" and ex.image == IMAGE
    assert ex.timeout_seconds == 0  # the site's time_limit governs --time
    monkeypatch.setenv("UMRS_STEP_REMOTE_ENABLED", "")
    assert not RemoteSlurmStepExecutor(ssh_config=SSHControlConfig(ssh_target="login")).enabled


def test_slurm_remote_scheduler_marks_remote_submit():
    from umrs.orchestrator.site_config import SchedulerConfig, SiteConfig

    site = SiteConfig(
        scheduler=SchedulerConfig(kind="slurm-remote"),
        step_executor=StepExecutorConfig(kind="inproc", partition="p0"),
    )
    env = site.to_env()
    assert env["UMRS_SLURM_STEP_SUBMIT"] == "remote"
    assert env["UMRS_STEP_EXECUTOR"] == "inproc"  # default unchanged
    assert env["UMRS_SLURM_PARTITION"] == "p0"  # declared-slurm workflows need it


def test_site_config_exports_workdir_root_and_slurm_knobs():
    env = _step_executor_env(
        StepExecutorConfig(kind="slurm-remote", partition="p0", workdir_root="/lustre/x")
    )
    assert env["UMRS_STEP_EXECUTOR"] == "slurm-remote"
    assert env["UMRS_SLURM_WORKDIR_ROOT"] == "/lustre/x"
    assert env["UMRS_SLURM_PARTITION"] == "p0"


def test_command_argv_is_a_client_of_the_shared_master():
    mgr = SSHControlManager(
        SSHControlConfig(ssh_target="login", ssh_user="u", remote_env={"A": "1"})
    )
    argv = mgr.command_argv("squeue -h")
    assert argv[0] == "ssh"
    assert "ControlMaster=no" in argv
    assert argv[-2:] == ["u@login", "A=1 squeue -h"]
