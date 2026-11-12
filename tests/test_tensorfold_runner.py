"""TensorFold engine adapter for the serving wrapper (ADR-106 D-8, 2026-11-01).

DeepSeek-V4.1-Flash-1017 runs on two GB10 through the MiaAI-Lab recipe's
TensorFold engine. Upstream's start.sh needs docker and ssh from the head to
the worker. tensorfold_runner.sh rebuilds its two run commands for rootless
podman under ``srun``. These tests source the library with stub
srun/podman/scontrol/systemd-run on PATH (the way test_stack_health.py does)
and pin the parts that break the pair when they drift:
- rank 1's pool and streams equal rank 0's context and parallel;
- the API key reaches the engine only as a read-only file;
- NCCL/RDMA follow the probed HCA and GID;
- an unpinned or unstaged image is refused before anything starts;
- rank 1 starts before rank 0;
- the wrapper dispatches to the adapter and supervises rank 0.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

MINI_RIVAULT = Path(__file__).resolve().parents[1] / "src" / "umrs" / "mini-rivault-llm"
LIB = MINI_RIVAULT / "tensorfold_runner.sh"
SERVE_MODEL_SLURM = MINI_RIVAULT / "serve-model.slurm"
SECRET = "sk-test-not-a-real-key-5f3a"

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
  "image exists") [ "${STUB_IMAGE_PRESENT:-1}" = 1 ] || [ -f "$STUB_DIR/loaded" ]; exit $? ;;
  "load -i") touch "$STUB_DIR/loaded"; echo "Loaded image: $IMAGE_REF"; exit 0 ;;
  "container exists") [ "${STUB_CTR_EXISTS:-1}" = 1 ]; exit $? ;;
esac
case "$1" in
  pull) exit 125 ;;
esac
exit 0
"""

_SCONTROL = """#!/bin/bash
printf 'node-a\\nnode-b\\n'
"""

_LOGGER = """#!/bin/bash
echo "$(basename "$0") $*" >> "$STUB_DIR/calls.log"
exit 0
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
    _write_exec(stub / "systemd-run", _LOGGER)
    hf = tmp_path / "hf"
    for rel in (
        "hub/models--Mia-AiLab--DS41/snapshots/45ee5d0f",
        "tensorfold-dsv41/45ee5d0f/model",
        "tensorfold-dsv41/45ee5d0f/rank-cache",
        "tensorfold-dsv41/kit-abe07c7d/kit",
    ):
        (hf / rel).mkdir(parents=True)
    (hf / "tensorfold-dsv41/45ee5d0f/dsv41_token_map.json").write_text("{}")
    (tmp_path / "engram").mkdir()
    (tmp_path / "r1cache").mkdir()
    (tmp_path / "wrapper" / "jobs").mkdir(parents=True)
    return {
        "PATH": f"{stub}:{os.environ['PATH']}",
        "HOME": str(tmp_path),
        "STUB_DIR": str(stub),
        "TF_IMAGE": "localhost/tensorfold-dsv41:exl3-tp2-7c9bf5d36350",
        "TF_HF_CACHE": str(hf),
        "TF_MODEL_SUB": "hub/models--Mia-AiLab--DS41/snapshots/45ee5d0f",
        "TF_DATA_SUB": "tensorfold-dsv41/45ee5d0f",
        "TF_KIT_SUB": "tensorfold-dsv41/kit-abe07c7d",
        "TF_ENGRAM_DIR": str(tmp_path / "engram"),
        "TF_WORKER_CACHE_DIR": str(tmp_path / "r1cache"),
        "TF_MIN_AVAIL_WAIT_S": "0",
        "VLLM_API_KEY": SECRET,
        "RIVAULT_WRAPPER_DIR": str(tmp_path / "wrapper"),
        "SLURM_JOB_ID": "4242",
        "SLURM_JOB_NUM_NODES": "2",
        "SLURM_JOB_NODELIST": "node-[a-b]",
        "SLURMD_NODENAME": "node-a",
        "PROBE_A": "10.200.0.1 rocep1s0f1 3 118 unlimited",
        "PROBE_B": "10.200.0.2 rocep1s0f1 3 117 unlimited",
    }


_PROBE_STUB = (
    'tf_probe_node() { if [ "$1" = node-a ]; then echo "$PROBE_A"; else echo "$PROBE_B"; fi; }\n'
)


def _run(env: dict[str, str], script: str) -> subprocess.CompletedProcess[str]:
    bash = shutil.which("bash") or "/bin/bash"
    full = f"set -u\nsource {LIB}\n{_PROBE_STUB}{script}\n"
    return subprocess.run([bash, "-c", full], env=env, capture_output=True, text=True, timeout=60)


def _argv(env: dict[str, str], fn: str, var: str) -> list[str]:
    script = (
        "tf_defaults\n"
        "declare -a TF_NODE_IP=(10.200.0.1 10.200.0.2) TF_NODE_HCA=(rocep1s0f1 rocep1s0f1)\n"
        "declare -a TF_NODE_GID=(3 5)\n"
        "TF_KEY_FILE=/tmp/keyfile\n"
        f"{fn}\n"
        f'printf "%s\\n" "${{{var}[@]}}"\n'
    )
    res = _run(env, script)
    assert res.returncode == 0, res.stderr
    return res.stdout.splitlines()


def _calls(env: dict[str, str]) -> list[str]:
    log = Path(env["STUB_DIR"]) / "calls.log"
    return log.read_text().splitlines() if log.exists() else []


def _after(argv: list[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


def test_rank0_serves_with_key_file_and_master_on_fabric(env: dict[str, str]) -> None:
    argv = _argv(env, "tf_rank0_argv", "TF_R0_ARGV")
    serve = argv[argv.index("tensorfold-native") :]
    facts = {
        "cmd": serve[:3],
        "name": _after(serve, "--name"),
        "context": _after(serve, "--context"),
        "key_flag": _after(serve, "--api-key-file"),
        "container": argv[argv.index("--name") + 1],
        "present": {
            "/tmp/keyfile:/run/secrets/tensorfold-api-key:ro",
            "TF_TP_MASTER=10.200.0.1",
            "NCCL_IB_GID_INDEX=3",
            "TF_DS_ENGRAM=/engram",
        }
        - set(argv),
        "leaks": [a for a in argv if SECRET in a],
    }
    assert facts == {
        "cmd": ["tensorfold-native", "serve", "/hf/tensorfold-dsv41/45ee5d0f/model"],
        "name": "DeepSeek-V4.1-Flash-1017",
        "context": "262144",
        "key_flag": "/run/secrets/tensorfold-api-key",
        "container": "rivault-tf-r0",
        "present": set(),
        "leaks": [],
    }


def test_rank1_follows_rank0_pool_streams_and_own_link(env: dict[str, str]) -> None:
    env["TF_CONTEXT"] = "524288"
    env["TF_PARALLEL"] = "6"
    argv = _argv(env, "tf_rank1_argv", "TF_R1_ARGV")
    follow = argv[argv.index("tf-dsv41-lanes") :]
    facts = {
        "positional": follow[1:8],
        "pool": _after(follow, "--pool"),
        "streams": _after(follow, "--streams"),
        "rdma": _after(follow, "--rdma"),
        "missing": {"NCCL_IB_GID_INDEX=5", "TF_DS_STREAMS=6", f"{env['TF_ENGRAM_DIR']}:/engram:ro"}
        - set(argv),
        "leaks": [a for a in argv if SECRET in a],
    }
    assert facts == {
        "positional": [
            "/hf/hub/models--Mia-AiLab--DS41/snapshots/45ee5d0f",
            "/rank-cache",
            "1",
            "2",
            "10.200.0.1",
            "29620",
            "/hf/tensorfold-dsv41/kit-abe07c7d/kit",
        ],
        "pool": "524288",
        "streams": "6",
        "rdma": "rocep1s0f1",
        "missing": set(),
        "leaks": [],
    }


def test_rdma_off_uses_sockets_only(env: dict[str, str]) -> None:
    env["TF_RDMA"] = "0"
    argv = _argv(env, "tf_rank1_argv", "TF_R1_ARGV")
    assert "NCCL_IB_DISABLE=1" in argv
    assert "--rdma" not in argv
    assert "/dev/infiniband" not in argv


def test_missing_setting_is_named(env: dict[str, str]) -> None:
    env.pop("TF_KIT_SUB")
    res = _run(env, "tf_defaults\ntf_missing_setting")
    assert res.stdout.strip() == "TF_KIT_SUB"


@pytest.mark.parametrize(
    ("image", "ok"),
    [
        ("localhost/tensorfold-dsv41:exl3-tp2-abc", True),
        ("ghcr.io/miaai-lab/tf@sha256:" + "a" * 64, True),
        ("ghcr.io/miaai-lab/tf:exl3-tp2-abc", False),
        ("docker.io/x/y:latest", False),
    ],
)
def test_image_must_be_staged_or_digest_pinned(env: dict[str, str], image: str, ok: bool) -> None:
    res = _run(env, f"tf_image_is_pinned '{image}'")
    assert (res.returncode == 0) is ok


def test_ipv4_gid_text(env: dict[str, str]) -> None:
    res = _run(env, "tf_ipv4_gid 10.200.0.1")
    assert res.stdout == "0000:0000:0000:0000:0000:ffff:0ac8:0001"


def test_probe_refuses_limited_memlock_with_rdma(env: dict[str, str]) -> None:
    env["PROBE_B"] = "10.200.0.2 rocep1s0f1 3 117 65536"
    res = _run(env, "start_tensorfold")
    assert res.returncode == 1
    assert "memlock hard limit is 65536 on node-b" in res.stderr
    assert not any("podman run" in c for c in _calls(env))


def test_memory_gate_refuses_short_node(env: dict[str, str]) -> None:
    env["PROBE_A"] = "10.200.0.1 rocep1s0f1 3 96 unlimited"
    res = _run(env, "start_tensorfold")
    assert res.returncode == 1
    assert "node-a=96GiB available, 112 GiB needed" in res.stderr
    assert not any("podman run" in c for c in _calls(env))


def test_unstaged_image_refused_before_launch(env: dict[str, str]) -> None:
    env["STUB_IMAGE_PRESENT"] = "0"
    res = _run(env, "start_tensorfold")
    assert res.returncode == 1
    assert "is not staged on node-a" in res.stderr
    assert not any("podman run" in c for c in _calls(env))


def test_start_launches_rank1_then_rank0_with_memguards(
    env: dict[str, str], tmp_path: Path
) -> None:
    cfg = tmp_path / "litellm.yaml"
    cfg.write_text("api_base: http://vllm:8000/v1\n")
    res = _run(env, f"start_tensorfold && tf_repoint_litellm {cfg}")
    calls = _calls(env)
    key = tmp_path / "wrapper" / "jobs" / "tf-api-key.4242"
    facts = {
        "rc": res.returncode,
        "run_order": [c.split()[4] for c in calls if c.startswith("podman run -d --name")],
        "memguards": sum(
            c.startswith("systemd-run --user --unit=rivault-tf-memguard-4242-") for c in calls
        ),
        "key": key.read_text().strip() if key.exists() else None,
        "key_mode": oct(stat.S_IMODE(key.stat().st_mode)) if key.exists() else None,
        "litellm": cfg.read_text(),
        "leaks": [c for c in calls if SECRET in c],
    }
    assert facts == {
        "rc": 0,
        "run_order": ["rivault-tf-r1", "rivault-tf-r0"],
        "memguards": 2,
        "key": SECRET,
        "key_mode": "0o600",
        "litellm": "api_base: http://10.200.0.1:8888/v1\n",
        "leaks": [],
    }, res.stderr


def test_stop_removes_both_ranks_and_key(env: dict[str, str], tmp_path: Path) -> None:
    res = _run(env, "start_tensorfold >/dev/null && tensorfold_stop")
    assert res.returncode == 0, res.stderr
    stops = [c for c in _calls(env) if "podman rm -f 'rivault-tf-r0' 'rivault-tf-r1'" in c]
    assert len(stops) == 2
    assert not (tmp_path / "wrapper" / "jobs" / "tf-api-key.4242").exists()


def test_wrapper_dispatches_and_supervises_tensorfold() -> None:
    text = SERVE_MODEL_SLURM.read_text()
    dispatch = text.index('== "tensorfold-dsv41" ]]; then')
    assert "tensorfold_runner.sh" in text[dispatch : dispatch + 800]
    assert "start_tensorfold || exit 1" in text[dispatch : dispatch + 800]
    assert "podman container exists rivault-tf-r0" in text
    assert '[ "$name" = "rivault-tf-r0" ]' in text
    assert "tensorfold_stop" in text[text.index("cleanup_stack() {") :]


def test_missing_image_loads_from_the_site_archive(env: dict[str, str], tmp_path: Path) -> None:
    """RCCS node image stores are node-local: a localhost/* image reaches a
    node only as <archive dir>/<name>-<tag>-<arch>.tar (venue.sh)."""
    archive_dir = tmp_path / "images"
    archive_dir.mkdir()
    arch = {"aarch64": "arm64", "x86_64": "amd64"}.get(os.uname().machine, os.uname().machine)
    (archive_dir / f"tensorfold-dsv41-exl3-tp2-7c9bf5d36350-{arch}.tar").write_text("x")
    env.update(
        STUB_IMAGE_PRESENT="0",
        RIVAULT_IMAGE_ARCHIVE_DIR=str(archive_dir),
        IMAGE_REF=env["TF_IMAGE"],
    )
    res = _run(env, "start_tensorfold")
    loads = [c for c in _calls(env) if c.startswith("podman load -i")]
    assert (res.returncode, len(loads) >= 1) == (0, True), res.stderr


def test_auto_fabric_uses_each_nodes_probed_link(env: dict[str, str]) -> None:
    env["TF_FABRIC_IFACE"] = "auto"
    env["PROBE_A"] = "10.10.0.1 roceA 3 118 unlimited enP2p1s0f1np1"
    env["PROBE_B"] = "10.10.0.2 roceB 4 117 unlimited enp1s0f1np1"
    res = _run(env, "start_tensorfold")
    runs = {c.split()[4]: c for c in _calls(env) if c.startswith("podman run -d --name")}
    facts = (
        res.returncode,
        "NCCL_SOCKET_IFNAME=enP2p1s0f1np1" in runs.get("rivault-tf-r0", ""),
        "NCCL_SOCKET_IFNAME=enp1s0f1np1" in runs.get("rivault-tf-r1", ""),
        "TF_TP_MASTER=10.10.0.1" in runs.get("rivault-tf-r0", ""),
    )
    assert facts == (0, True, True, True), res.stderr


def test_master_iface_carries_rendezvous_while_rdma_stays_on_fabric(
    env: dict[str, str], tmp_path: Path
) -> None:
    """rccs QSFP drops TCP (firewalld zone public) but passes RoCE: rank 1's
    connect, NCCL's socket bootstrap and LiteLLM go over the mgmt NIC, the
    RDMA ring and NCCL_IB stay on the QSFP HCA/GID (2026-11-12)."""
    env["TF_MASTER_IFACE"] = "enP7s7"
    env["PROBE_A"] = "192.168.1.12 roceP2p1s0f1 5 104 unlimited enP2p1s0f1np1 10.129.2.12"
    env["PROBE_B"] = "192.168.1.13 roceP2p1s0f1 5 104 unlimited enP2p1s0f1np1 10.129.2.13"
    env["TF_MIN_AVAIL_GIB"] = "100"
    cfg = tmp_path / "litellm.yaml"
    cfg.write_text("api_base: http://vllm:8000/v1\n")
    res = _run(env, f"start_tensorfold && tf_repoint_litellm {cfg}")
    runs = {c.split()[4]: c for c in _calls(env) if c.startswith("podman run -d --name")}
    r0, r1 = runs.get("rivault-tf-r0", ""), runs.get("rivault-tf-r1", "")
    facts = {
        "rc": res.returncode,
        "master": "TF_TP_MASTER=10.129.2.12" in r0,
        "r1_connects_to_mgmt": " 1 2 10.129.2.12 29620 " in r1,
        "nccl_socket": ("NCCL_SOCKET_IFNAME=enP7s7" in r0, "NCCL_SOCKET_IFNAME=enP7s7" in r1),
        "ib_on_fabric": ("NCCL_IB_HCA=roceP2p1s0f1" in r0, "NCCL_IB_GID_INDEX=5" in r1),
        "litellm": cfg.read_text(),
    }
    assert facts == {
        "rc": 0,
        "master": True,
        "r1_connects_to_mgmt": True,
        "nccl_socket": (True, True),
        "ib_on_fabric": (True, True),
        "litellm": "api_base: http://10.129.2.12:8888/v1\n",
    }, res.stderr


def test_master_iface_without_ipv4_is_refused(env: dict[str, str]) -> None:
    env["TF_MASTER_IFACE"] = "enP7s7"
    env["PROBE_A"] = "192.168.1.12 roceP2p1s0f1 5 117 unlimited enP2p1s0f1np1 10.129.2.12"
    env["PROBE_B"] = "192.168.1.13 roceP2p1s0f1 5 117 unlimited enP2p1s0f1np1 -"
    res = _run(env, "start_tensorfold")
    runs = [c for c in _calls(env) if c.startswith("podman run")]
    assert (res.returncode, runs) == (1, []), res.stderr
    assert "no IPv4 on TF_MASTER_IFACE enP7s7 on node-b" in res.stderr


def test_step_mode_runs_rank1_and_memguards_as_steps(env: dict[str, str], tmp_path: Path) -> None:
    """rccs (runc, no user systemd): a container detached from an srun step dies
    with the step, so rank 1 runs in the foreground of one backgrounded step and
    the memguards are background steps, not systemd-run units."""
    env.update(
        RIVAULT_PARTITION_OCI_RUNTIME="runc",
        STUB_CTR_EXISTS="0",
        TF_MEMGUARD_WAIT_S="0",
        XDG_RUNTIME_DIR=str(tmp_path / "xdg"),
    )
    res = _run(env, "start_tensorfold && wait")
    calls = _calls(env)
    facts = {
        "rc": res.returncode,
        "detached_r1": [c for c in calls if c.startswith("podman run -d --name rivault-tf-r1")],
        "foreground_r1": sum(
            c.startswith("podman run --rm --replace --name rivault-tf-r1") for c in calls
        ),
        "detached_r0": sum(c.startswith("podman run -d --name rivault-tf-r0") for c in calls),
        "systemd_run": [c for c in calls if c.startswith("systemd-run")],
        "memguard_checks": sum(c.startswith("podman container exists rivault-tf-r") for c in calls)
        >= 2,
        "r1_log": (tmp_path / "wrapper" / "jobs" / "tf-r1.4242.log").exists(),
    }
    assert facts == {
        "rc": 0,
        "detached_r1": [],
        "foreground_r1": 1,
        "detached_r0": 1,
        "systemd_run": [],
        "memguard_checks": True,
        "r1_log": True,
    }, res.stderr


def test_prestart_reclaim_runs_on_both_nodes_before_the_memory_gate(env: dict[str, str]) -> None:
    env["TF_PRESTART_RECLAIM_GIB"] = "100"
    res = _run(env, "start_tensorfold")
    calls = _calls(env)
    reclaims = [
        i for i, c in enumerate(calls) if c.startswith("podman run --rm") and "gb10_reclaim.py" in c
    ]
    first_rank = next(
        i for i, c in enumerate(calls) if c.startswith("podman run -d --name rivault-tf-r1")
    )
    facts = (
        res.returncode,
        len(reclaims),
        all(i < first_rank for i in reclaims),
        all("RIVAULT_MN_PRESTART_RECLAIM_GIB=100" in calls[i] for i in reclaims),
    )
    assert facts == (0, 2, True, True), res.stderr
