"""Remote Slurm step executor (front-end host, no local Slurm CLI).

The SSH hop is replaced by a local ``bash -c`` whose PATH carries fake
``sbatch`` / ``squeue`` / ``sacct``; the "cluster filesystem" is a tmp dir.
Everything else — payload streaming, sbatch submission, polling, reading
``output.pkl`` back, source-tree staging, cleanup — runs for real.
"""

from __future__ import annotations

import os
import pickle
import subprocess
from pathlib import Path

import pytest
from umrs.orchestrator.site_config import StepExecutorConfig, _step_executor_env
from umrs.orchestrator.ssh_control import SSHControlConfig, SSHControlManager
from umrs.pipeline_sdk.step import StepContext, StepResult
from umrs.server.remote_slurm_step_executor import RemoteSlurmStepExecutor
from umrs.server.step_executor import (
    EXECUTOR_SOURCE_DIR_METADATA_KEY,
    StepExecutionError,
    build_step_executor,
)

_FAKE_SBATCH = """#!/bin/bash
d=$(dirname "$1")
cp "$FAKE_REPLY" "$d/output.pkl"
echo "$1" >> "$FAKE_LOG"
echo "Submitted batch job 777"
"""
_FAKE_SQUEUE = "#!/bin/bash\nexit 0\n"  # not queued any more
_FAKE_SACCT = "#!/bin/bash\necho COMPLETED\n"


def _ctx(metadata: dict | None = None) -> StepContext:
    return StepContext(
        s3=None,
        input_bucket="in",
        input_keys=[],
        output_bucket="out",
        metadata=metadata or {},
        run_id="run-remote",
    )


@pytest.fixture
def cluster(tmp_path: Path):
    """Fake login node: bin/ with Slurm CLIs, lustre/ as the shared FS."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in (
        ("sbatch", _FAKE_SBATCH),
        ("squeue", _FAKE_SQUEUE),
        ("sacct", _FAKE_SACCT),
    ):
        path = bin_dir / name
        path.write_text(body)
        path.chmod(0o755)
    reply = tmp_path / "reply.pkl"
    reply.write_bytes(pickle.dumps({"status": "ok", "result": StepResult(output_keys=["k1"])}))
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "FAKE_REPLY": str(reply),
        "FAKE_LOG": str(tmp_path / "sbatch.log"),
    }
    lustre = tmp_path / "lustre" / "steps"
    return {"env": env, "root": str(lustre), "log": tmp_path / "sbatch.log"}


def _executor(cluster, **kw) -> RemoteSlurmStepExecutor:
    ex = RemoteSlurmStepExecutor(
        image="img:latest",
        workdir_root=cluster["root"],
        poll_interval=0,
        ssh_config=SSHControlConfig(ssh_target="login"),
        **kw,
    )
    commands: list[str] = []

    def local_ssh(command, *, timeout, stdin=None):
        commands.append(command)
        return subprocess.run(
            ["bash", "-c", command],
            input=stdin,
            capture_output=True,
            timeout=timeout,
            env=cluster["env"],
            check=False,
        )

    ex._ssh_run = local_ssh  # type: ignore[method-assign]
    ex.commands = commands  # type: ignore[attr-defined]
    return ex


def test_step_runs_end_to_end_over_the_remote_hooks(cluster):
    ex = _executor(cluster)
    result = ex.execute(entrypoint="mod:Noop", config={"a": 1}, ctx=_ctx())
    assert result.output_keys == ["k1"]
    submitted = cluster["log"].read_text().strip()
    assert submitted.startswith(cluster["root"] + "/slurm-step-")
    assert any(c.startswith("sbatch ") for c in ex.commands)
    # Workdir removed on the cluster after the step.
    assert not Path(submitted).parent.exists()


def test_payload_and_script_land_on_the_cluster_fs(cluster):
    ex = _executor(cluster, keep_workdir=True)
    ex.execute(entrypoint="mod:Noop", config={"a": 1}, ctx=_ctx())
    workdir = Path(cluster["log"].read_text().strip()).parent
    payload = pickle.loads((workdir / "input.pkl").read_bytes())
    assert payload["entrypoint"] == "mod:Noop"
    assert payload["config"] == {"a": 1}
    script = (workdir / "run_step.sbatch").read_text()
    assert f"{workdir}:/work" in script


def test_uploaded_source_tree_is_staged_next_to_the_payload(cluster, tmp_path):
    src = tmp_path / "bundle"
    (src / "pkg").mkdir(parents=True)
    (src / "pkg" / "step.py").write_text("X = 1\n")
    ex = _executor(cluster, keep_workdir=True)
    ex.execute(
        entrypoint="pkg.step:Noop",
        config={},
        ctx=_ctx({EXECUTOR_SOURCE_DIR_METADATA_KEY: str(src)}),
    )
    workdir = Path(cluster["log"].read_text().strip()).parent
    assert (workdir / "src" / "pkg" / "step.py").read_text() == "X = 1\n"
    script = (workdir / "run_step.sbatch").read_text()
    assert f"UMRS_STEP_SOURCE_DIR={workdir}/src" in script
    assert str(src) not in script  # the front-end path never reaches the node


def test_s3_endpoint_rewritten_for_the_compute_node(cluster):
    class _Meta:
        endpoint_url = "http://127.0.0.1:8333"
        region_name = "us-east-1"

    class _S3:
        meta = _Meta()

    ex = _executor(cluster, keep_workdir=True, s3_endpoint="http://login01:9000")
    ctx = _ctx()
    ctx.s3 = _S3()
    ex.execute(entrypoint="mod:Noop", config={}, ctx=ctx)
    workdir = Path(cluster["log"].read_text().strip()).parent
    payload = pickle.loads((workdir / "input.pkl").read_bytes())
    assert payload["s3_config"] == {
        "endpoint_url": "http://login01:9000",
        "region_name": "us-east-1",
    }
    assert "S3_ENDPOINT_URL=http://login01:9000" in (workdir / "run_step.sbatch").read_text()


def test_workdir_root_is_required(cluster):
    ex = _executor(cluster)
    ex.workdir_root = ""
    with pytest.raises(StepExecutionError, match="UMRS_SLURM_WORKDIR_ROOT"):
        ex.execute(entrypoint="mod:Noop", config={}, ctx=_ctx())


def test_failed_job_surfaces_the_remote_error_log(cluster, tmp_path):
    sacct = Path(cluster["env"]["PATH"].split(":")[0]) / "sacct"
    sacct.write_text("#!/bin/bash\necho FAILED\n")
    sbatch = sacct.parent / "sbatch"
    sbatch.write_text(
        '#!/bin/bash\nd=$(dirname "$1")\necho "boom on node" > "$d/slurm-777.err"\n'
        'echo "Submitted batch job 777"\n'
    )
    ex = _executor(cluster)
    with pytest.raises(StepExecutionError, match="boom on node"):
        ex.execute(entrypoint="mod:Noop", config={}, ctx=_ctx())


def test_cleanup_refuses_paths_it_did_not_create(cluster):
    ex = _executor(cluster)
    ex._remove_workdir("/home/user")
    assert not any(c.startswith("rm ") for c in ex.commands)


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
