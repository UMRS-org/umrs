"""Multi-node SGLang engine adapter for the serving wrapper (2026-11-12).

DeepSeek-V4.1-Flash-1017-NVFP4 runs under SGLang across the rccs
``ai-pro6000`` nodes (2x RTX PRO 6000 each, TP=4 over two nodes).
sglang_runner.sh starts one ``launch_server`` per node in rootless podman
under ``srun``. These tests source the library with stub srun/podman/scontrol
on PATH (the test_tensorfold_runner.py way) and pin the parts that break the
world when they drift:
- every rank gets the same --nnodes/--tp-size/--dist-init-addr, its own
  --node-rank, and rank 0's fabric IP as the rendezvous;
- the API key reaches rank 0 only as a read-only --config YAML file;
- workers follow the wrapper's step/detached rule;
- the GPU form (CDI or device nodes plus driver libraries) follows the probe;
- NCCL uses RDMA only when every node can;
- an unpinned image, a GPU shortfall or a bad layout is refused before launch.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

MINI_RIVAULT = Path(__file__).resolve().parents[1] / "src" / "umrs" / "mini-rivault-llm"
LIB = MINI_RIVAULT / "sglang_runner.sh"
SECRET = "sk-test-not-a-real-key-7b1e"
IMAGE = "lmsysorg/sglang@sha256:" + "b" * 64
LIBS = "libcuda.so.1,libnvidia-ml.so.1,libnvidia-ptxjitcompiler.so.1,libnvidia-gpucomp.so.580.1,"

_SRUN = r"""#!/bin/bash
echo "srun $*" >> "$STUB_DIR/calls.log"
while [ $# -gt 0 ]; do
  case "$1" in --*) shift ;; *) break ;; esac
done
exec "$@"
"""

_PODMAN = r"""#!/bin/bash
echo "podman $*" >> "$STUB_DIR/calls.log"
case "$1 $2" in
  "image exists") [ "${STUB_IMAGE_PRESENT:-1}" = 1 ]; exit $? ;;
esac
case "$1" in
  pull) exit 125 ;;
esac
exit 0
"""

_SCONTROL = """#!/bin/bash
printf 'node-a\\nnode-b\\n'
"""


def _write_exec(path: Path, body: str) -> None:
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


@pytest.fixture()
def env(tmp_path: Path) -> dict[str, str]:
    stub = tmp_path / "stub"
    stub.mkdir()
    _write_exec(stub / "srun", _SRUN)
    _write_exec(stub / "podman", _PODMAN)
    _write_exec(stub / "scontrol", _SCONTROL)
    model = tmp_path / "models" / "DeepSeek-V4.1-Flash-1017-NVFP4"
    model.mkdir(parents=True)
    (tmp_path / "wrapper" / "jobs").mkdir(parents=True)
    (tmp_path / "xdg").mkdir()
    return {
        "PATH": f"{stub}:{os.environ['PATH']}",
        "HOME": str(tmp_path),
        "XDG_RUNTIME_DIR": str(tmp_path / "xdg"),
        "STUB_DIR": str(stub),
        "SG_IMAGE": IMAGE,
        "SG_MODEL_DIR": str(model),
        "SG_WORKER_RELAUNCH_SLEEP": "0",
        "SG_WORKER_ATTEMPTS": "1",
        "RIVAULT_MN_WORKER_MODE": "detached",
        "VLLM_API_KEY": SECRET,
        "RIVAULT_WRAPPER_DIR": str(tmp_path / "wrapper"),
        "SLURM_JOB_ID": "4343",
        "SLURM_JOB_NUM_NODES": "2",
        "SLURM_JOB_NODELIST": "node-[a-b]",
        "SLURMD_NODENAME": "node-a",
        "PROBE_A": f"10.30.0.1 mlx5_0 3 2 1 unlimited enp1s0np0 - {LIBS}",
        "PROBE_B": f"10.30.0.2 mlx5_1 5 2 1 unlimited enp2s0np0 - {LIBS}",
    }


_PROBE_STUB = (
    'sg_probe_node() { if [ "$1" = node-a ]; then echo "$PROBE_A"; else echo "$PROBE_B"; fi; }\n'
)


def _run(env: dict[str, str], script: str) -> subprocess.CompletedProcess[str]:
    bash = shutil.which("bash") or "/bin/bash"
    full = f"set -u\nsource {LIB}\n{_PROBE_STUB}{script}\n"
    return subprocess.run([bash, "-c", full], env=env, capture_output=True, text=True, timeout=60)


def _calls(env: dict[str, str]) -> list[str]:
    log = Path(env["STUB_DIR"]) / "calls.log"
    return log.read_text().splitlines() if log.exists() else []


def _runs(env: dict[str, str]) -> dict[str, str]:
    """podman run lines by container name."""
    out = {}
    for call in _calls(env):
        if call.startswith("podman run "):
            words = call.split()
            out[words[words.index("--name") + 1]] = call
    return out


def _after(argv: list[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


def _argv(env: dict[str, str], rank: int, mode: str = "detached") -> list[str]:
    script = (
        "start_sglang_mn() { :; }\n"
        "sg_defaults\n"
        "declare -a RAY_NODES=(node-a node-b)\n"
        "sg_probe_all >/dev/null || exit 1\n"
        "SG_KEY_FILE=/tmp/sg-key.yaml\n"
        f"sg_rank_argv {rank} {mode}\n"
        'printf "%s\\n" "${SG_ARGV[@]}"\n'
    )
    res = _run(env, script)
    assert res.returncode == 0, res.stderr
    return res.stdout.splitlines()


def test_rank0_serves_http_with_key_file_only(env: dict[str, str]) -> None:
    argv = _argv(env, 0)
    cmd = argv[argv.index(IMAGE) + 1 :]
    facts = {
        "head": argv[:5],
        "entry": _after(argv, "--entrypoint"),
        "module": cmd[:2],
        "tp": _after(cmd, "--tp-size"),
        "nnodes": _after(cmd, "--nnodes"),
        "rank": _after(cmd, "--node-rank"),
        "dist": _after(cmd, "--dist-init-addr"),
        "config": _after(cmd, "--config"),
        "spec": (
            _after(cmd, "--speculative-algorithm"),
            _after(cmd, "--speculative-dspark-block-size"),
        ),
        "engram": _after(cmd, "--engram-config"),
        "no_attn": "--dsv4-attn-backend" not in cmd and "--pp-size" not in cmd,
        "missing": {"/tmp/sg-key.yaml:/run/secrets/sglang-key.yaml:ro", "SGLANG_HOST_IP=10.30.0.1"}
        - set(argv),
        "leaks": [a for a in argv if SECRET in a] + [a for a in argv if a == "--api-key"],
    }
    assert facts == {
        "head": ["podman", "run", "-d", "--name", "rivault-sg-r0"],
        "entry": "python3",
        "module": ["-m", "sglang.launch_server"],
        "tp": "4",
        "nnodes": "2",
        "rank": "0",
        "dist": "10.30.0.1:29640",
        "config": "/run/secrets/sglang-key.yaml",
        "spec": ("DSPARK", "5"),
        "engram": '{"placement":"cpu","replication":"none","lookup":"auto"}',
        "no_attn": True,
        "missing": set(),
        "leaks": [],
    }


def test_rank1_joins_rank0_without_key(env: dict[str, str]) -> None:
    argv = _argv(env, 1)
    cmd = argv[argv.index(IMAGE) + 1 :]
    facts = {
        "name": _after(argv, "--name"),
        "replace": "--replace" in argv,
        "rank": _after(cmd, "--node-rank"),
        "dist": _after(cmd, "--dist-init-addr"),
        "tp": _after(cmd, "--tp-size"),
        "keyless": "--config" not in cmd and not any("sglang-key" in a for a in argv),
        "nccl": {
            "NCCL_SOCKET_IFNAME=enp2s0np0",
            "GLOO_SOCKET_IFNAME=enp2s0np0",
            "SGLANG_HOST_IP=10.30.0.2",
        }
        - set(argv),
    }
    assert facts == {
        "name": "rivault-sg-r1",
        "replace": True,
        "rank": "1",
        "dist": "10.30.0.1:29640",
        "tp": "4",
        "keyless": True,
        "nccl": set(),
    }


def test_tp2_pp2_layout_and_attention_backend(env: dict[str, str]) -> None:
    env.update(SG_PP="2", SG_ATTN_BACKEND="flashmla", SG_SPEC="", SG_ENGRAM_CONFIG="")
    cmd = _argv(env, 0)
    facts = (
        _after(cmd, "--tp-size"),
        _after(cmd, "--pp-size"),
        _after(cmd, "--dsv4-attn-backend"),
        "--speculative-algorithm" in cmd or "--engram-config" in cmd,
    )
    assert facts == ("2", "2", "flashmla", False)


def test_cdi_gpu_form(env: dict[str, str]) -> None:
    argv = _argv(env, 0)
    devices = [argv[i + 1] for i, a in enumerate(argv) if a == "--device"]
    assert devices == ["/dev/infiniband", "nvidia.com/gpu=all"]
    assert not any(a.startswith("/dev/nvidia") or "libcuda" in a for a in argv)


def test_devices_gpu_form_binds_present_driver_libs(env: dict[str, str]) -> None:
    env["PROBE_B"] = (
        f"10.30.0.2 mlx5_1 5 2 0 unlimited enp2s0np0 /dev/nvidia-caps/nvidia-cap1, {LIBS}"
    )
    argv = _argv(env, 1)
    devices = [argv[i + 1] for i, a in enumerate(argv) if a == "--device"]
    lib = "/usr/lib/aarch64-linux-gnu"
    binds = {argv[i + 1] for i, a in enumerate(argv) if a == "-v"}
    facts = {
        "devices": devices,
        "missing_libs": {
            f"{lib}/{n}:{lib}/{n}:ro" for n in ("libcuda.so.1", "libnvidia-gpucomp.so.580.1")
        }
        - binds,
        "nvvm": any("libnvidia-nvvm" in b for b in binds),
    }
    assert facts == {
        "devices": [
            "/dev/infiniband",
            "/dev/nvidia0",
            "/dev/nvidia1",
            "/dev/nvidiactl",
            "/dev/nvidia-uvm",
            "/dev/nvidia-uvm-tools",
            "/dev/nvidia-caps/nvidia-cap1",
        ],
        "missing_libs": set(),
        "nvvm": False,
    }


def test_devices_form_without_libcuda_refused(env: dict[str, str]) -> None:
    env.update(SG_GPU_FORM="devices", PROBE_B="10.30.0.2 mlx5_1 5 2 0 unlimited enp2s0np0 - -")
    res = _run(env, "start_sglang_mn")
    assert res.returncode == 1
    assert "no libcuda.so.1 under /usr/lib/aarch64-linux-gnu on node-b" in res.stderr
    assert not _runs(env)


def test_rdma_on_sets_ib_env(env: dict[str, str]) -> None:
    argv = _argv(env, 1)
    assert {"NCCL_IB_HCA=mlx5_1", "NCCL_IB_GID_INDEX=5", "/dev/infiniband"} <= set(argv)
    assert "NCCL_IB_DISABLE=1" not in argv


@pytest.mark.parametrize(
    "probe_b",
    [
        f"10.30.0.2 - - 2 1 unlimited enp2s0np0 - {LIBS}",
        f"10.30.0.2 mlx5_1 5 2 1 65536 enp2s0np0 - {LIBS}",
    ],
)
def test_rdma_auto_falls_back_to_sockets(env: dict[str, str], probe_b: str) -> None:
    env["PROBE_B"] = probe_b
    argv = _argv(env, 0)
    assert "NCCL_IB_DISABLE=1" in argv
    assert "/dev/infiniband" not in argv
    assert not any(a.startswith("NCCL_IB_HCA=") for a in argv)


def test_rdma_forced_without_hca_refused(env: dict[str, str]) -> None:
    env.update(SG_RDMA="1", PROBE_B=f"10.30.0.2 - - 2 1 unlimited enp2s0np0 - {LIBS}")
    res = _run(env, "start_sglang_mn")
    assert res.returncode == 1
    assert "no InfiniBand HCA or RoCE v2 HCA/GID" in res.stderr and "node-b" in res.stderr
    assert not _runs(env)


@pytest.mark.parametrize(
    ("image", "ok"),
    [
        ("localhost/sglang:nightly-dev-cu13-20261111-314bce7c", True),
        (IMAGE, True),
        ("lmsysorg/sglang:nightly-dev-cu13-20261111-314bce7c", False),
        ("docker.io/lmsysorg/sglang:latest", False),
    ],
)
def test_image_must_be_staged_or_digest_pinned(env: dict[str, str], image: str, ok: bool) -> None:
    res = _run(env, f"sg_image_is_pinned '{image}'")
    assert (res.returncode == 0) is ok


def test_unpinned_image_refused_before_launch(env: dict[str, str]) -> None:
    env["SG_IMAGE"] = "lmsysorg/sglang:nightly-dev-cu13-20261111-314bce7c"
    res = _run(env, "start_sglang_mn")
    assert res.returncode == 1
    assert "must be localhost/* (staged) or pinned @sha256" in res.stderr
    assert not _runs(env)


def test_gpu_shortfall_refused(env: dict[str, str]) -> None:
    env["PROBE_B"] = f"10.30.0.2 mlx5_1 5 1 1 unlimited enp2s0np0 - {LIBS}"
    res = _run(env, "start_sglang_mn")
    assert res.returncode == 1
    assert "FATAL: node-b has 1 GPU(s), SG_GPUS_PER_NODE=2" in res.stderr
    assert not _runs(env)


def test_layout_mismatch_refused(env: dict[str, str]) -> None:
    env.update(SG_TP="4", SG_PP="2")
    res = _run(env, "start_sglang_mn")
    assert res.returncode == 1
    assert "SG_TP=4 x SG_PP=2 != 2 nodes x 2 GPUs" in res.stderr


def test_missing_setting_is_named(env: dict[str, str]) -> None:
    env.pop("SG_MODEL_DIR")
    res = _run(env, "sg_defaults\nsg_missing_setting")
    assert res.stdout.strip() == "SG_MODEL_DIR"


def test_detached_start_writes_key_yaml_and_repoints(env: dict[str, str], tmp_path: Path) -> None:
    cfg = tmp_path / "litellm.yaml"
    cfg.write_text("api_base: http://vllm:8000/v1\n")
    res = _run(env, f"start_sglang_mn && sg_repoint_litellm {cfg}")
    key = tmp_path / "wrapper" / "jobs" / "sg-api-key.4343.yaml"
    runs = _runs(env)
    facts = {
        "rc": res.returncode,
        "order": list(runs),
        "r1_via_srun": any(
            c.startswith("srun") and "podman run -d --name rivault-sg-r1" in c for c in _calls(env)
        ),
        "key": key.read_text() if key.exists() else None,
        "key_mode": oct(stat.S_IMODE(key.stat().st_mode)) if key.exists() else None,
        "litellm": cfg.read_text(),
        "leaks": [c for c in _calls(env) if SECRET in c],
    }
    assert facts == {
        "rc": 0,
        "order": ["rivault-sg-r1", "rivault-sg-r0"],
        "r1_via_srun": True,
        "key": f'api-key: "{SECRET}"\n',
        "key_mode": "0o600",
        "litellm": "api_base: http://10.30.0.1:30000/v1\n",
        "leaks": [],
    }, res.stderr


def test_step_mode_runs_worker_in_foreground_step(env: dict[str, str]) -> None:
    env.pop("RIVAULT_MN_WORKER_MODE")
    env["RIVAULT_PARTITION_OCI_RUNTIME"] = "runc"
    res = _run(env, "start_sglang_mn && wait")
    calls = _calls(env)
    step = [
        c
        for c in calls
        if c.startswith("srun") and "rivault_venue_prepare" in c and "exec podman run" in c
    ]
    runs = _runs(env)
    facts = {
        "rc": res.returncode,
        "step_srun": len(step),
        "r1_foreground": " -d " not in runs.get("rivault-sg-r1", " -d "),
        "r1_replace": "--replace" in runs.get("rivault-sg-r1", ""),
        "r0_detached": runs.get("rivault-sg-r0", "").startswith(
            "podman run -d --name rivault-sg-r0"
        ),
        "migrate": any(c == "podman system migrate" for c in calls),
    }
    assert facts == {
        "rc": 0,
        "step_srun": 1,
        "r1_foreground": True,
        "r1_replace": True,
        "r0_detached": True,
        "migrate": True,
    }, res.stderr


@pytest.mark.parametrize(
    ("override", "runtime", "mode"),
    [
        ("", "runc", "step"),
        ("", "crun", "detached"),
        ("detached", "runc", "detached"),
        ("step", "", "step"),
    ],
)
def test_worker_mode_rule(env: dict[str, str], override: str, runtime: str, mode: str) -> None:
    env.update(RIVAULT_MN_WORKER_MODE=override, RIVAULT_PARTITION_OCI_RUNTIME=runtime)
    assert _run(env, "sg_worker_mode").stdout.strip() == mode


def test_stop_removes_ranks_and_key(env: dict[str, str], tmp_path: Path) -> None:
    res = _run(env, "start_sglang_mn >/dev/null && sglang_stop")
    calls = _calls(env)
    facts = (
        res.returncode,
        any(c.startswith("srun") and "podman rm -f -t 30 rivault-sg-r1" in c for c in calls),
        "podman rm -f rivault-sg-r0" in calls,
        (tmp_path / "wrapper" / "jobs" / "sg-api-key.4343.yaml").exists(),
    )
    assert facts == (0, True, True, False), res.stderr


def test_probe_script_prints_nine_fields(env: dict[str, str]) -> None:
    res = _run(env, 'sg_defaults\nbash -c "$SG_PROBE_SCRIPT" _ lo /nonexistent')
    fields = res.stdout.split()
    assert (res.returncode, len(fields), fields[0], fields[6], fields[8]) == (
        0,
        9,
        "127.0.0.1",
        "lo",
        "-",
    ), res.stderr


def test_model_root_mounts_the_hf_repo_so_snapshot_links_resolve(env: dict[str, str]) -> None:
    """An HF cache snapshot's files are relative links into ../../blobs: the
    repo dir is mounted at its own path and the snapshot path is served."""
    snap = env["SG_MODEL_DIR"]
    root = str(Path(snap).parent)
    env["SG_MODEL_ROOT"] = root
    argv = _argv(env, 0)
    facts = (
        f"{root}:{root}:ro" in argv,
        _after(argv, "--model-path"),
        any(a.endswith(":/model:ro") for a in argv),
    )
    assert facts == (True, snap, False)


def test_wrapper_dispatches_and_supervises_sglang() -> None:
    text = (
        Path(__file__).resolve().parents[1] / "src/umrs/mini-rivault-llm/serve-model.slurm"
    ).read_text()
    dispatch = text.index('== "sglang-mn" ]]; then\n    # SGLang across nodes')
    block = text[dispatch : dispatch + 1200]
    assert "sglang_runner.sh" in block and "start_sglang_mn || exit 1" in block
    assert "_launch_engine_mn() { sg_launch_ranks; }" in block and "sg_repoint_litellm" in block
    assert "podman container exists rivault-sg-r0" in text
    assert '[ "$name" = "rivault-sg-r0" ]' in text
    assert "sglang_stop" in text[text.index("cleanup_stack() {") :]


def test_infiniband_hca_needs_no_gid_index(env: dict[str, str]) -> None:
    """ai-pro6000 (probe 2026-11-12): mlx5_0 is InfiniBand NDR 400 with IPoIB on
    ibP1p1s0; there is no RoCE v2 GID, and NCCL needs only the HCA name."""
    env["PROBE_A"] = f"10.129.76.0 mlx5_0 ib 2 0 unlimited ibP1p1s0 - {LIBS}"
    env["PROBE_B"] = f"10.129.76.1 mlx5_0 ib 2 0 unlimited ibP1p1s0 - {LIBS}"
    argv = _argv(env, 0)
    facts = (
        "NCCL_IB_HCA=mlx5_0" in argv,
        any(a.startswith("NCCL_IB_GID_INDEX") for a in argv),
        "NCCL_IB_DISABLE=1" in argv,
        "/dev/infiniband" in argv,
    )
    assert facts == (True, False, False, True)


_SINGULARITY = r"""#!/bin/bash
echo "singularity $*" >> "$STUB_DIR/calls.log"
exit 0
"""


def _sif(env: dict[str, str], tmp_path: Path) -> Path:
    sif = tmp_path / "sglang-test-arm64.sif"
    sif.write_text("x")
    (tmp_path / "sglang-test-arm64.sif.sha256").write_text("abc  sif\n")
    _write_exec(Path(env["STUB_DIR"]) / "singularity", _SINGULARITY)
    env.update(
        SG_RUNTIME="singularity",
        SG_IMAGE=str(sif),
        SG_WORKER_ATTEMPTS="1",
        SG_WORKER_RELAUNCH_SLEEP="0",
    )
    return sif


def test_singularity_runtime_runs_every_rank_without_podman(
    env: dict[str, str], tmp_path: Path
) -> None:
    """ai-pro6000 nodes have no podman (job 468124, 2026-11-12), only singularity-ce."""
    sif = _sif(env, tmp_path)
    res = _run(env, "start_sglang_mn && wait")
    calls = _calls(env)
    execs = [c for c in calls if c.startswith("singularity exec")]
    r0 = next((c for c in execs if "--node-rank 0" in c), "")
    r1 = next((c for c in execs if "--node-rank 1" in c), "")
    facts = {
        "rc": res.returncode,
        "podman_runs": [c for c in calls if c.startswith("podman")],
        "ranks": len(execs),
        "nv_and_sif": all("--nv" in c and str(sif) in c for c in execs),
        "key_only_on_rank0": ("--config" in r0, "--config" in r1),
        "leaks": [c for c in calls if SECRET in c],
    }
    assert facts == {
        "rc": 0,
        "podman_runs": [],
        "ranks": 2,
        "nv_and_sif": True,
        "key_only_on_rank0": (True, False),
        "leaks": [],
    }, res.stderr


@pytest.mark.parametrize("sidecar", [True, False])
def test_singularity_image_needs_a_sha256_sidecar(
    env: dict[str, str], tmp_path: Path, sidecar: bool
) -> None:
    sif = _sif(env, tmp_path)
    if not sidecar:
        (tmp_path / "sglang-test-arm64.sif.sha256").unlink()
    res = _run(env, f'sg_image_is_pinned "{sif}" && echo pinned')
    assert (res.stdout.strip() == "pinned") is sidecar


def test_singularity_ranks_run_in_steps_that_request_the_gpus(
    env: dict[str, str], tmp_path: Path
) -> None:
    """ai-pro6000 job 468620 (2026-11-12): the batch shell's device cgroup held one
    of the node's two GPUs, so TP rank 1 on node 0 failed "invalid device ordinal".
    Rank 0 now runs in its own srun step, and every rank's step asks for the GPUs."""
    _sif(env, tmp_path)
    env["SG_STEP_GRES"] = "gpu:2"
    res = _run(env, "start_sglang_mn && wait")
    steps = [c for c in _calls(env) if c.startswith("srun ") and "singularity exec" in c]
    facts = {
        "rc": res.returncode,
        "steps": len(steps),
        "all_request_gpus": all("--gres=gpu:2" in c for c in steps),
        "head_step": any("--nodelist=node-a" in c and "--node-rank 0" in c for c in steps),
        "worker_step": any("--nodelist=node-b" in c and "--node-rank 1" in c for c in steps),
    }
    assert facts == {
        "rc": 0,
        "steps": 2,
        "all_request_gpus": True,
        "head_step": True,
        "worker_step": True,
    }, res.stderr
