"""The staged step's sbatch script (build spec §7).

Golden pins on the rendered script (directives, the hardened podman argv,
exactly five mounts and four ``-e``, nothing about S3 or AWS), then the
script itself run by bash against a fake ``podman`` and venue library: the
image-ensure lines load the archive only when the node's image id differs
from the archive's ``.id``, refuse an image without the step-contract label,
and never pull; a probe in the same posture must read ``--pids-limit`` back;
the pod's output reaches the Slurm log capped; a watchdog stops a step whose
``out/`` or ``reply/`` pass their caps while it runs. The step's seccomp
profile is the node's default with ``fallocate`` failing (review round 3).
"""

from __future__ import annotations

import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import time

import pytest
from umrs.orchestrator.ssh_control import SSHControlConfig
from umrs.pipeline_sdk.step import StepContext
from umrs.server.remote_slurm_step_executor import (
    STEP_TIMEOUT_METADATA_KEY,
    RemoteSlurmStepExecutor,
    _StepJob,
)
from umrs.server.slurm_step_executor import (
    SECCOMP_FILTER_PY,
    WALL_GRACE_SEC,
    StagedJob,
    render_staged_job_script,
)
from umrs.server.step_executor import EXECUTOR_GPU_METADATA_KEY, StepExecutionError

WD = "/lvs0/rccs-dmp/u/umrs_steps/wf-0123456789-s1-1-abcdefgh"
IMAGE = "localhost/umrs-step:0123456789ab"


def _job(**overrides) -> StagedJob:
    values = {
        "workdir": WD,
        "job_name": "umrs-step-wf-0123456789-s1",
        "attempt": 1,
        "run_tag": "wf-0123456789ab",
        "partition": "ng-dgx-m0,ng-dgx-m1",
        "account": "cloud-users",
        "cpus": 8,
        "memory": "",
        "time_limit": "01:00:00",
        "gres_gpus": 0,
        "image": IMAGE,
        "image_archive_dir": "/lvs0/rccs-dmp/u/umrs_models/images",
        "venue_script": "/lvs0/rccs-dmp/u/ai4s-platform/umrs/src/umrs/mini-rivault-llm/venue.sh",
        "needs_gpu": False,
    }
    values.update(overrides)
    return StagedJob(**values)


def _podman_run_line(script: str) -> str:
    suffix = " 2>&1 | _step_log >&2"
    [line] = [x for x in script.splitlines() if x.startswith("podman run ") and x.endswith(suffix)]
    return line[: -len(suffix)]


def test_directives_are_the_spec_set_and_logs_live_outside_every_mount():
    script = render_staged_job_script(_job())
    directives = [x for x in script.splitlines() if x.startswith("#SBATCH")]
    assert directives == [
        "#SBATCH --job-name=umrs-step-wf-0123456789-s1",
        "#SBATCH --comment=umrs-step:wf-0123456789ab:1",
        "#SBATCH --no-requeue",
        "#SBATCH --time=01:00:00",
        "#SBATCH --partition=ng-dgx-m0,ng-dgx-m1",
        "#SBATCH --account=cloud-users",
        "#SBATCH --cpus-per-task=8",
        f"#SBATCH --output={WD}/log/slurm.out",
        f"#SBATCH --error={WD}/log/slurm.err",
    ]
    assert script.startswith("#!/bin/bash\n")
    assert "\nset -euo pipefail\numask 077\n" in script


def test_a_deadline_lets_slurm_drop_a_job_no_server_chases():
    script = render_staged_job_script(_job(deadline_seconds=25500))
    assert "#SBATCH --deadline=now+25500" in script.splitlines()
    ex = RemoteSlurmStepExecutor(
        ssh_config=SSHControlConfig(ssh_target="login"), workdir_root="/r", image=IMAGE,
        image_archive_dir="/a", venue_script="/v.sh", time_limit="01:00:00",
        queue_timeout="06:00:00",
    )  # fmt: skip
    job = _StepJob("wf-1", "s1", 1, "umrs-step-wf-1-s1", "wf-1-s1-1-", workdir=WD)
    ctx = StepContext(s3=None, input_bucket="b", input_keys=[], output_bucket="b-output",
                      metadata={STEP_TIMEOUT_METADATA_KEY: 600}, run_id="wf-1")  # fmt: skip
    deadline = 6 * 3600 + 600 + WALL_GRACE_SEC  # queue_timeout + --time + grace
    assert f"#SBATCH --deadline=now+{deadline}" in ex._job_script(job, ctx, IMAGE)


def _tokens(script: str) -> list[str]:
    line = _podman_run_line(script)
    return shlex.split(line.replace('"${_step_ctr}"', "CTR").replace('"${_step_image}"', "IMG"))


def _flag_values(tokens: list[str], flag: str) -> list[str]:
    return [tokens[i + 1] for i, token in enumerate(tokens) if token == flag]


def test_podman_argv_is_hardened():
    tokens = _tokens(render_staged_job_script(_job()))
    posture = [
        "--rm", "--init", "--name", "CTR", "--pull=never", "--network=none",
        "--http-proxy=false", "--read-only",
        "--tmpfs", "/tmp:rw,size=4g", "--cap-drop=all", "--security-opt", "no-new-privileges",
        "--userns=keep-id", "--user", "$(id -u):$(id -g)",
        "--security-opt", "seccomp=${_step_seccomp}", "--pids-limit=512",
        "--ulimit", f"fsize={8 * 1024**3}",
    ]  # fmt: skip
    assert tokens[2 : 2 + len(posture)] == posture
    assert tokens[-4:] == ["IMG", "python", "-m", "umrs.server.step_worker"]
    assert "--device" not in tokens


def test_podman_mounts_only_the_staged_dirs_and_sets_only_four_env():
    tokens = _tokens(render_staged_job_script(_job()))
    assert _flag_values(tokens, "-v") == [
        f"{WD}/in:/work/in:ro",
        f"{WD}/out:/work/out:rw",
        f"{WD}/reply:/work/reply:rw",
        f"{WD}/src:/src:ro",
        f"{WD}/input.pkl:/work/input.pkl:ro",
    ]
    assert _flag_values(tokens, "-e") == [
        "UMRS_STEP_INPUT=/work/input.pkl",
        "UMRS_STEP_OUTPUT_JSON=/work/reply/output.json",
        "UMRS_STEP_STAGED=1",
        "UMRS_STEP_SOURCE_DIR=/src",
    ]


def test_the_script_never_carries_object_store_or_credential_env(monkeypatch):
    for name in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "S3_ENDPOINT_URL", "MINIO_ENDPOINT"):
        monkeypatch.setenv(name, "leak-me")
    script = render_staged_job_script(_job())
    for needle in ("AWS_", "S3_", "MINIO", "leak-me", "--network=host", f"-v {WD}:"):
        assert needle not in script
    assert not re.search(r"-v \S*" + re.escape(WD) + r"(:|/log|/job\.sbatch)", script)
    assert "podman pull" not in script and "--pull=never" in script


def test_gpu_mem_and_gres_only_when_asked():
    plain = render_staged_job_script(_job(gres_gpus=2))
    assert "--gres" not in plain and "--device" not in plain and "--mem" not in plain
    gpu = render_staged_job_script(_job(needs_gpu=True, gres_gpus=0, memory="16G"))
    assert "--device nvidia.com/gpu=all" in _podman_run_line(gpu)
    assert "#SBATCH --gres" not in gpu  # ng-dgx-m* reject --gres
    assert "#SBATCH --mem=16G" in gpu
    assert "#SBATCH --gres=gpu:2" in render_staged_job_script(_job(needs_gpu=True, gres_gpus=2))


def test_the_gpu_form_picks_the_devices_of_a_step_that_needs_a_gpu():
    # step_executor.gpu_form: the NVIDIA CDI spec, or the ROCm device nodes.
    rocm = _podman_run_line(render_staged_job_script(_job(needs_gpu=True, gpu_form="rocm")))
    assert "--device /dev/kfd --device /dev/dri" in rocm and "nvidia.com" not in rocm
    cdi = _podman_run_line(render_staged_job_script(_job(needs_gpu=True, gpu_form="cdi")))
    assert "--device nvidia.com/gpu=all" in cdi and "/dev/kfd" not in cdi
    assert _tokens(render_staged_job_script(_job(needs_gpu=True, gpu_form="rocm")))[-4:] == [
        "IMG", "python", "-m", "umrs.server.step_worker",
    ]  # fmt: skip
    # No GPU asked: no device, whatever the form.
    assert "--device" not in render_staged_job_script(_job(gpu_form="rocm"))


def test_the_nvidia_devices_form_maps_what_the_node_has():
    # RCCS GB10 (R2 2026-10-01): podman 4.9.3 there cannot resolve the CDI
    # spec, so the step gets the node's device nodes + host driver libraries.
    script = render_staged_job_script(_job(needs_gpu=True, gpu_form="nvidia-devices"))
    run = _podman_run_line(script)
    assert '"${_step_gpu[@]}"' in run
    assert "nvidia.com/gpu=all" not in run
    for needle in ("/dev/nvidiactl", "/dev/nvidia-uvm", "libcuda.so.1", "no NVIDIA device nodes"):
        assert needle in script
    assert script.index("_step_gpu=()") < script.index(run)
    assert "_step_gpu" not in render_staged_job_script(_job(gpu_form="nvidia-devices"))


def test_the_nvidia_devices_script_is_valid_bash():
    script = render_staged_job_script(_job(needs_gpu=True, gpu_form="nvidia-devices"))
    proc = subprocess.run(["bash", "-n"], input=script.encode(), capture_output=True)
    assert proc.returncode == 0, proc.stderr


def test_a_gpu_form_the_script_cannot_render_is_refused_by_name():
    with pytest.raises(StepExecutionError, match="gpu_form 'opencl'"):
        render_staged_job_script(_job(needs_gpu=True, gpu_form="opencl"))


def test_the_executor_takes_the_gpu_form_from_the_site_env(monkeypatch):
    monkeypatch.setenv("UMRS_STEP_GPU", "1")
    monkeypatch.delenv("UMRS_STEP_GPU_FORM", raising=False)

    def executor() -> RemoteSlurmStepExecutor:
        return RemoteSlurmStepExecutor(ssh_config=SSHControlConfig(ssh_target="login"),
                                       workdir_root="/r", image=IMAGE, image_archive_dir="/a",
                                       venue_script="/v.sh")  # fmt: skip

    assert executor().gpu_form == "cdi"
    monkeypatch.setenv("UMRS_STEP_GPU_FORM", "rocm")
    ex = executor()
    assert ex.gpu_form == "rocm"
    job = _StepJob("wf-1", "s1", 1, "umrs-step-wf-1-s1", "wf-1-s1-1-", workdir=WD)
    ctx = StepContext(s3=None, input_bucket="b", input_keys=[], output_bucket="b-output",
                      metadata={EXECUTOR_GPU_METADATA_KEY: True}, run_id="wf-1")  # fmt: skip
    assert "--device /dev/kfd --device /dev/dri" in ex._job_script(job, ctx, IMAGE)


def test_every_python_value_is_quoted_in_the_script():
    hostile = "/lvs0/x $(touch /tmp/pwned)/wf-1-s-1-abcdefgh"
    script = render_staged_job_script(
        _job(workdir=hostile, venue_script="/v/$(id).sh", image_archive_dir="/a/`id`")
    )
    assert "'/v/$(id).sh'" in script
    assert f"'{hostile}/in:/work/in:ro'" in script
    assert "'/a/`id`/umrs-step-0123456789ab-'" in script
    assert subprocess.run(["bash", "-n"], input=script.encode(), check=False).returncode == 0


@pytest.mark.parametrize("image", ["img@sha256:" + "a" * 64, "img:$(x)", "a/b:c d"])
def test_an_image_without_an_archive_name_is_refused(image):
    with pytest.raises(StepExecutionError, match="archive name"):
        render_staged_job_script(_job(image=image))


def test_executor_time_is_the_least_of_site_step_and_executor_limits(monkeypatch):
    ex = RemoteSlurmStepExecutor(
        ssh_config=SSHControlConfig(ssh_target="login"),
        time_limit="02:00:00",
        timeout_seconds=0,
        workdir_root="/r",
        image=IMAGE,
        image_archive_dir="/a",
        venue_script="/v.sh",
    )
    job = _StepJob("wf-1", "s1", 1, "umrs-step-wf-1-s1", "wf-1-s1-1-", workdir=WD)

    def ctx(meta):
        return StepContext(s3=None, input_bucket="b", input_keys=[], output_bucket="b-output",
                           metadata=meta, run_id="wf-1")  # fmt: skip

    assert "#SBATCH --time=02:00:00" in ex._job_script(job, ctx({}), IMAGE)
    assert "#SBATCH --time=00:10:00" in ex._job_script(
        job, ctx({STEP_TIMEOUT_METADATA_KEY: 600}), IMAGE
    )
    ex.timeout_seconds = 300
    assert "#SBATCH --time=00:05:00" in ex._job_script(
        job, ctx({STEP_TIMEOUT_METADATA_KEY: 600}), IMAGE
    )
    # A GPU needs the step's hint AND a site that enables GPUs (site.yaml gpu).
    assert "--device" not in ex._job_script(job, ctx({EXECUTOR_GPU_METADATA_KEY: True}), IMAGE)
    monkeypatch.setenv("UMRS_STEP_GPU", "1")
    assert "--device" in ex._job_script(job, ctx({EXECUTOR_GPU_METADATA_KEY: True}), IMAGE)


# ── the script run against a fake podman ─────────────────────────────────────

#: The shape of containers-common's default profile: fallocate sits in the
#: big allow rule; the default action refuses everything unlisted.
DEFAULT_PROFILE = {
    "defaultAction": "SCMP_ACT_ERRNO",
    "defaultErrnoRet": 38,
    "syscalls": [
        {"names": ["read", "write", "fallocate", "openat"], "action": "SCMP_ACT_ALLOW"},
        {"names": ["fallocate"], "action": "SCMP_ACT_ALLOW", "includes": {"minKernel": "4.0"}},
        {"name": "fallocate", "action": "SCMP_ACT_ALLOW"},
        {"names": ["keyctl"], "action": "SCMP_ACT_ERRNO", "errnoRet": 1},
    ],
}

FAKE_PODMAN = r"""#!/usr/bin/env bash
state="${FAKE_STATE:?}"
all="$*"
printf '%s\n' "${all//$'\n'/ }" >> "$state/podman.log"
while [[ "${1:-}" == --* ]]; do shift; done  # global flags (--cgroup-manager=...)
case "$1" in
  info)
    if [[ -f "$state/seccomp_base.json" ]]; then echo "$state/seccomp_base.json"; fi ;;
  image)
    if [[ "$4" == *Labels* ]]; then cat "$state/label" 2>/dev/null; exit 0; fi
    [[ -f "$state/have_id" ]] || exit 125
    cat "$state/have_id" ;;
  rmi) rm -f "$state/have_id" ;;
  load) cp "$state/loaded_id" "$state/have_id" ;;
  run)
    if [[ "$*" == *umrs-seccomp-base.json* ]]; then
      args=("$@")
      for i in "${!args[@]}"; do
        if [[ "${args[$i]}" == "-c" ]]; then
          exec "$FAKE_PYTHON" -I -c "${args[$((i + 1))]}" "$state/seccomp_base.json"
        fi
      done
      exit 1
    fi
    if [[ "$*" == *pids.max* ]]; then
      cat "$state/probe_out" 2>/dev/null || echo 512
      exit "$(cat "$state/probe_rc" 2>/dev/null || echo 0)"
    fi
    if [[ -f "$state/run_script" ]]; then source "$state/run_script"; fi
    exit "$(cat "$state/run_rc" 2>/dev/null || echo 0)" ;;
  kill)
    if [[ -f "$state/run_pid" ]]; then kill "$(cat "$state/run_pid")"; rm "$state/run_pid"; exit 0; fi
    exit 1 ;;
  pull) echo "PULLED" >> "$state/podman.log"; exit 0 ;;
  ps) cat "$state/ps_out" 2>/dev/null ;;
esac
"""


@pytest.fixture
def node(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "podman").write_text(FAKE_PODMAN)
    (bin_dir / "podman").chmod(0o755)
    # squeue answers only when a test gives it live job ids.
    (bin_dir / "squeue").write_text('#!/bin/bash\ncat "$FAKE_STATE/squeue_out" 2>/dev/null\n')
    (bin_dir / "squeue").chmod(0o755)
    state = tmp_path / "state"
    state.mkdir()
    archive_dir = tmp_path / "images"
    archive_dir.mkdir()
    archive = archive_dir / f"umrs-step-0123456789ab-{platform.machine()}.tar"
    archive.write_bytes(b"tar")
    (archive_dir / (archive.name + ".id")).write_text("f" * 64 + "\n")
    (state / "loaded_id").write_text("f" * 64)
    (state / "label").write_text("1")
    (state / "seccomp_base.json").write_text(json.dumps(DEFAULT_PROFILE))
    venue = tmp_path / "venue.sh"
    venue.write_text('rivault_venue_prepare() { echo "prepared $1"; }\n')
    wd = tmp_path / "wd"
    for sub in ("out", "reply", "log"):
        (wd / sub).mkdir(parents=True)
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "FAKE_STATE": str(state),
        "FAKE_PYTHON": sys.executable,
        "SLURM_JOB_ID": "4242",
        "WD": str(wd),
    }

    def run_script(**overrides) -> subprocess.CompletedProcess:
        script = render_staged_job_script(
            _job(workdir=str(wd), image_archive_dir=str(archive_dir), venue_script=str(venue),
                 **overrides)
        )  # fmt: skip
        return subprocess.run(
            ["bash", "-c", script], env=env, capture_output=True, text=True, timeout=60
        )

    def log() -> list[str]:
        path = state / "podman.log"
        return path.read_text().splitlines() if path.exists() else []

    return {
        "run": run_script, "log": log, "state": state, "archive": archive, "wd": wd,
        "bin": bin_dir,
    }  # fmt: skip


def _step_runs(log: list[str]) -> list[str]:
    return [
        x for x in log
        if x.removeprefix("--cgroup-manager=cgroupfs ").startswith("run ")
        and "pids.max" not in x and "umrs-seccomp-base" not in x
    ]  # fmt: skip


def test_a_node_without_the_image_loads_the_archive_then_runs(node):
    proc = node["run"]()
    assert proc.returncode == 0, proc.stderr
    assert "prepared umrs-step-4242" in proc.stdout
    verbs = [line.split()[0] for line in node["log"]()]
    # never pull
    assert verbs == [
        "image", "rmi", "load", "image", "image", "info", "run", "run", "run", "kill", "rm"
    ]  # fmt: skip
    log = node["log"]()
    assert (log[1], log[2]) == ("rmi -f " + IMAGE, f"load -i {node['archive']}")
    assert log[8].startswith("run --rm --init --name umrs-step-4242 ")
    assert log[-2:] == ["kill umrs-step-4242", "rm -f umrs-step-4242"]  # the EXIT trap


def test_the_pids_probe_runs_first_in_the_steps_posture(node):
    assert node["run"]().returncode == 0
    probe = node["log"]()[7]
    seccomp = node["wd"] / "seccomp.json"
    assert probe.startswith("run --rm --init --name umrs-step-4242-pids --pull=never ")
    assert probe.endswith(
        f"--security-opt seccomp={seccomp} --pids-limit=512 --ulimit fsize={8 * 1024**3} "
        f"{IMAGE} cat /sys/fs/cgroup/pids.max"
    )


def test_a_node_whose_image_matches_the_archive_id_skips_the_load(node):
    (node["state"] / "have_id").write_text("f" * 64)
    assert node["run"]().returncode == 0
    assert not any(x.startswith(("load", "rmi")) for x in node["log"]())


def test_a_stale_image_is_replaced_and_a_bad_load_fails_by_name(node):
    (node["state"] / "have_id").write_text("0" * 64)
    (node["state"] / "loaded_id").write_text("e" * 64)
    proc = node["run"]()
    assert proc.returncode == 78
    assert "does not match" in proc.stderr
    assert not any(x.startswith("run ") for x in node["log"]())


def test_a_missing_archive_or_contract_label_fails_before_the_step(node):
    node["archive"].unlink()
    proc = node["run"]()
    assert proc.returncode == 78 and "no step image archive" in proc.stderr
    node["archive"].write_bytes(b"tar")
    (node["state"] / "label").write_text("")
    proc = node["run"]()
    assert proc.returncode == 78 and "org.umrs.step-contract=1" in proc.stderr
    assert not any(x.startswith("run ") for x in node["log"]())


def test_the_step_exit_code_is_the_job_exit_code(node):
    (node["state"] / "have_id").write_text("f" * 64)
    (node["state"] / "run_rc").write_text("3")
    assert node["run"]().returncode == 3


@pytest.mark.parametrize(
    "probe_out",
    [
        "max\n",
        "WARN[0000] Your kernel does not support pids limit capabilities or the cgroup is "
        "not mounted. PIDs limit discarded.\nmax\n",
    ],
)
def test_a_pids_limit_the_runtime_drops_fails_the_job_by_name(node, probe_out):
    (node["state"] / "have_id").write_text("f" * 64)
    (node["state"] / "probe_out").write_text(probe_out)
    proc = node["run"]()
    assert proc.returncode == 78
    assert "--pids-limit=512 is not applied on" in proc.stderr and "(pids.max max)" in proc.stderr
    assert _step_runs(node["log"]()) == []


def test_a_runtime_that_refuses_the_posture_fails_the_job_by_name(node):
    (node["state"] / "have_id").write_text("f" * 64)
    (node["state"] / "probe_out").write_text(
        "Error: OCI runtime error: runc: rootless needs no limits + no cgrouppath when no "
        "permission is granted for cgroups\n"
    )
    (node["state"] / "probe_rc").write_text("126")
    proc = node["run"]()
    assert proc.returncode == 78
    assert "the step posture was refused on" in proc.stderr
    assert "rootless needs no limits" in proc.stderr
    assert _step_runs(node["log"]()) == []


def test_pids_limit_zero_drops_the_flag_and_its_probe(node):
    (node["state"] / "have_id").write_text("f" * 64)
    (node["state"] / "probe_out").write_text("max\n")
    assert node["run"](pids_limit=0).returncode == 0
    [step] = _step_runs(node["log"]())
    assert "--pids-limit" not in step and "--ulimit fsize=" in step
    assert not any("pids.max" in x for x in node["log"]())


def test_a_venue_without_a_cgroup_runs_the_step_without_one_and_bounds_processes(node):
    # R2 on the rccs GB10 nodes (2026-09-26): Slurm's cgroup v2 grants the
    # job no cgroup to create the container's in, so runc refuses any
    # container that asks for one. The step then runs with --cgroups=disabled
    # inside the job's cgroup, RLIMIT_NPROC bounding its processes.
    (node["state"] / "have_id").write_text("f" * 64)
    assert node["run"](pids_limit=0, cgroups_disabled=True, nproc_limit=4096).returncode == 0
    [step] = _step_runs(node["log"]())
    for flag in ("--cgroups=disabled", "--ulimit nproc=4096", "--ulimit fsize="):
        assert flag in step
    assert "--pids-limit" not in step
    assert not [x for x in node["log"]() if "pids.max" in x]


def test_without_a_cgroup_every_container_pins_the_cgroupfs_manager(node):
    # The systemd manager would put conmon in a user@ scope outside the job's
    # Slurm cgroup: the container would outlive scancel (peer review).
    (node["state"] / "have_id").write_text("f" * 64)
    assert node["run"](pids_limit=0, cgroups_disabled=True).returncode == 0
    runs = [x for x in node["log"]() if " run " in f" {x} "]
    assert len(runs) == 2  # the seccomp probe and the step
    assert all(x.startswith("--cgroup-manager=cgroupfs run ") for x in runs)


def test_containers_a_scancel_left_are_removed_when_slurm_no_longer_lists_their_job(node):
    (node["state"] / "have_id").write_text("f" * 64)
    (node["state"] / "squeue_out").write_text("4242\n500\n")  # this job and a live one
    (node["state"] / "ps_out").write_text(
        "umrs-step-388530\numrs-step-388530-pids\numrs-step-500\numrs-step-4242-seccomp\n"
    )
    assert node["run"]().returncode == 0
    removed = [x for x in node["log"]() if x.startswith("rm -f ")]
    assert removed[:2] == ["rm -f umrs-step-388530", "rm -f umrs-step-388530-pids"]
    assert "rm -f umrs-step-500" not in removed  # its job is live
    assert removed[2:] == ["rm -f umrs-step-4242"]  # only the EXIT trap touches this job's


def test_nothing_is_removed_when_squeue_does_not_answer(node):
    (node["state"] / "have_id").write_text("f" * 64)
    (node["state"] / "ps_out").write_text("umrs-step-388530\n")
    (node["bin"] / "squeue").write_text("#!/bin/bash\nexit 1\n")
    assert node["run"]().returncode == 0
    assert [x for x in node["log"]() if x.startswith(("ps ", "rm -f umrs-step-388530"))] == []


def test_the_pods_output_reaches_the_log_capped_with_its_tail(node):
    (node["state"] / "have_id").write_text("f" * 64)
    (node["state"] / "run_script").write_text(
        "printf 'HEAD'; head -c 100000 /dev/zero | tr '\\0' m; printf 'TRACEBACK\\n' >&2\n"
    )
    proc = node["run"](log_head_bytes=1000, log_tail_bytes=100)
    assert proc.returncode == 0, proc.stderr
    step_log = proc.stderr.split("prepared", 1)[-1]
    assert "HEAD" in step_log and step_log.rstrip().endswith("TRACEBACK")
    assert len(step_log) <= 1100 + 200  # the head, the tail and the script's own lines


@pytest.mark.skipif(shutil.which("stdbuf") is None, reason="needs coreutils stdbuf")
def test_the_log_streams_while_the_pod_still_runs(tmp_path):
    # R2 2026-10-01: plain `head -c` held a still-running step's first line
    # in its stdio buffer, so the kill check never saw its sleeper's marker.
    script = render_staged_job_script(_job())
    start = script.index("if command -v stdbuf")
    # the function ends at its own closing-brace line (``${_step_head[@]}``
    # carries a brace too)
    definition = script[start : script.index("\n}\n", script.index("_step_log() {")) + 2]
    log = tmp_path / "slurm.err"
    log.write_text("")  # the redirect truncates it; never read a missing file
    proc = subprocess.Popen(
        ["bash", "-c", definition + f"\n( echo marker; sleep 5 ) | _step_log > {log}"]
    )
    try:
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline and "marker" not in log.read_text():
            time.sleep(0.1)
        assert "marker" in log.read_text()
    finally:
        proc.kill()
        proc.wait()


def test_the_watchdog_stops_a_step_whose_outputs_pass_their_caps(node):
    (node["state"] / "have_id").write_text("f" * 64)
    (node["state"] / "run_script").write_text(
        'head -c 5000 /dev/zero > "$WD/out/flood"\necho $$ > "$FAKE_STATE/run_pid"\nexec sleep 30\n'
    )
    proc = node["run"](watch_seconds=1, max_out_bytes=4096)
    assert proc.returncode != 0
    assert "out/ and reply/ passed their caps" in proc.stderr
    assert "kill umrs-step-4242" in node["log"]()


def test_the_watchdog_counts_entries_too(node):
    (node["state"] / "have_id").write_text("f" * 64)
    (node["state"] / "run_script").write_text(
        'for i in $(seq 1 40); do : > "$WD/out/f$i"; done\n'
        'echo $$ > "$FAKE_STATE/run_pid"\n'
        "exec sleep 30\n"
    )
    proc = node["run"](watch_seconds=1, max_out_entries=20)
    assert proc.returncode != 0
    assert "entries); the step was stopped" in proc.stderr


def test_the_watchdog_stops_a_step_that_hides_a_directory_from_it(node):
    # The watchdog runs as the job's uid without DAC override: du and find see
    # nothing inside a 0300 directory the step can still fill by name.
    (node["state"] / "have_id").write_text("f" * 64)
    (node["state"] / "run_script").write_text(
        'mkdir "$WD/out/h"; head -c 100000 /dev/zero > "$WD/out/h/big"; chmod 0300 "$WD/out/h"\n'
        'echo $$ > "$FAKE_STATE/run_pid"\nexec sleep 30\n'
    )
    try:
        proc = node["run"](watch_seconds=1, max_out_bytes=50_000)
    finally:
        (node["wd"] / "out" / "h").chmod(0o700)
    assert proc.returncode != 0
    assert "is hidden from the watchdog (not u+rx); the step was stopped" in proc.stderr
    assert "kill umrs-step-4242" in node["log"]()


def test_the_watchdog_stops_a_step_below_the_free_space_floor(node):
    (node["state"] / "have_id").write_text("f" * 64)
    (node["state"] / "run_script").write_text('echo $$ > "$FAKE_STATE/run_pid"\nexec sleep 30\n')
    proc = node["run"](watch_seconds=1, min_free_bytes=10**18)
    assert proc.returncode != 0
    assert f"bytes free (floor {10**18}); the step was stopped" in proc.stderr


def test_the_executor_sets_the_watchdog_floor_to_half_the_staging_floor():
    ex = RemoteSlurmStepExecutor(
        ssh_config=SSHControlConfig(ssh_target="login"), workdir_root="/r", image=IMAGE,
        image_archive_dir="/a", venue_script="/v.sh",
    )  # fmt: skip
    job = _StepJob("wf-1", "s1", 1, "umrs-step-wf-1-s1", "wf-1-s1-1-", workdir=WD)
    ctx = StepContext(s3=None, input_bucket="b", input_keys=[], output_bucket="b-output",
                      run_id="wf-1")  # fmt: skip
    floor = ex.caps.min_free_bytes // 2
    assert f"(( avail < {floor} ))" in ex._job_script(job, ctx, IMAGE)
    assert "(( avail <" not in render_staged_job_script(_job())  # 0: no floor


def test_a_step_within_its_caps_is_left_alone_by_the_watchdog(node):
    (node["state"] / "have_id").write_text("f" * 64)
    (node["state"] / "run_script").write_text('head -c 100 /dev/zero > "$WD/out/ok"; sleep 2\n')
    proc = node["run"](watch_seconds=1, max_out_bytes=1_000_000)
    assert proc.returncode == 0, proc.stderr
    assert "passed their caps" not in proc.stderr


# ── fallocate, and a watchdog no tree can delay (review round 3) ────────────


def _filtered(profile: dict) -> dict:
    proc = subprocess.run(
        [sys.executable, "-I", "-c", SECCOMP_FILTER_PY, "/dev/stdin"],
        input=json.dumps(profile), capture_output=True, text=True, check=True,
    )  # fmt: skip
    return json.loads(proc.stdout)


def test_the_step_profile_is_the_default_with_fallocate_failing_eopnotsupp():
    # glibc's posix_fallocate falls back to writing only on EOPNOTSUPP (95):
    # writes the file size limit and the watchdog both bound.
    out = _filtered(DEFAULT_PROFILE)
    assert out["defaultAction"] == "SCMP_ACT_ERRNO" and out["defaultErrnoRet"] == 38
    assert out["syscalls"] == [
        {"names": ["read", "write", "openat"], "action": "SCMP_ACT_ALLOW"},
        {"names": ["keyctl"], "action": "SCMP_ACT_ERRNO", "errnoRet": 1},
        {"names": ["fallocate"], "action": "SCMP_ACT_ERRNO", "errnoRet": 95},
    ]
    assert "'" not in SECCOMP_FILTER_PY  # it travels single-quoted in the script


def test_the_step_and_its_probe_run_under_the_written_profile(node):
    assert node["run"]().returncode == 0
    profile = json.loads((node["wd"] / "seccomp.json").read_text())
    assert profile["syscalls"][-1] == {
        "names": ["fallocate"], "action": "SCMP_ACT_ERRNO", "errnoRet": 95,
    }  # fmt: skip
    [step] = _step_runs(node["log"]())
    assert f"--security-opt seccomp={node['wd'] / 'seccomp.json'} " in step
    [transform] = [x for x in node["log"]() if "umrs-seccomp-base.json" in x]
    assert "seccomp=" not in transform.split(" python ")[0]  # the default profile


def test_a_node_without_a_default_seccomp_profile_runs_no_step(node):
    (node["state"] / "have_id").write_text("f" * 64)
    (node["state"] / "seccomp_base.json").unlink()
    proc = node["run"]()
    assert proc.returncode == 78
    assert "no default seccomp profile on" in proc.stderr
    assert not any(x.startswith("run ") for x in node["log"]())


def test_the_watchdog_counts_a_preallocated_file_at_its_allocated_size(node):
    # FALLOC_FL_KEEP_SIZE: 0 bytes apparent (du -sb), 200 KiB on the disk.
    (node["state"] / "have_id").write_text("f" * 64)
    (node["state"] / "run_script").write_text(
        ': > "$WD/out/held"; fallocate -n -l 200K "$WD/out/held"\n'
        'echo $$ > "$FAKE_STATE/run_pid"\nexec sleep 30\n'
    )
    proc = node["run"](watch_seconds=1, max_out_bytes=100_000)
    assert proc.returncode != 0
    assert "out/ and reply/ passed their caps" in proc.stderr


def test_the_watchdog_checks_the_floor_before_it_walks_the_tree(node):
    # A tree of millions of entries makes every walk slow: the floor comes first.
    (node["state"] / "have_id").write_text("f" * 64)
    (node["bin"] / "find").write_text("#!/bin/sh\nexec sleep 120\n")
    (node["bin"] / "find").chmod(0o755)
    (node["state"] / "run_script").write_text('echo $$ > "$FAKE_STATE/run_pid"\nexec sleep 30\n')
    proc = node["run"](watch_seconds=1, min_free_bytes=10**18)
    assert proc.returncode != 0
    assert f"bytes free (floor {10**18}); the step was stopped" in proc.stderr


def test_the_watchdog_stops_counting_past_the_entry_cap(node):
    (node["state"] / "have_id").write_text("f" * 64)
    (node["state"] / "run_script").write_text(
        'for i in $(seq 1 200); do : > "$WD/out/f$i"; done\n'
        'echo $$ > "$FAKE_STATE/run_pid"\nexec sleep 30\n'
    )
    proc = node["run"](watch_seconds=1, max_out_entries=20)
    assert proc.returncode != 0
    assert " 21 entries); the step was stopped" in proc.stderr  # cap + 1, not all 202
