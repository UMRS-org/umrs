"""rccs DeepSeek-V4-Flash multi-node launch (2026-11-03, jobs 454912/454924/454939).

rccs is rootless runc (podman 4.9.3) without cgroup delegation: any container
memory limit fails ("rootless needs no limits + no cgrouppath"), which killed
the Ray worker on the second node; and the keep-id:uid userns default is
rejected there.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest
from umrs.orchestrator import provision_roster as pr

ROOT = Path(__file__).resolve().parents[1]
POOL = ROOT / "deploy/rccs/pool-rccs.yaml"
SLURM = ROOT / "src/umrs/mini-rivault-llm/serve-model.slurm"


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv(pr.ROSTER_ENV_VAR, str(POOL))
    pr._cache.clear()
    defaults = pr.resolve_model_provision_defaults("DeepSeek-V4-Flash")
    pr._cache.clear()
    assert defaults is not None
    return defaults["env_vars"]


def test_rccs_deepseek_sets_no_container_memory_cap(env):
    assert "RIVAULT_MN_MEMORY_CAP" not in env


def test_rccs_deepseek_spells_out_the_site_userns_mode(env):
    assert env["RIVAULT_USERNS_MODE"] == "host"


def test_launcher_skips_a_memory_cap_on_runc():
    text = SLURM.read_text(encoding="utf-8")
    guard = re.search(
        r'if \[\[ -n "\$\{RIVAULT_MN_MEMORY_CAP:-\}" && "\$\{RIVAULT_PARTITION_OCI_RUNTIME:-\}" == "runc" \]\]; then',
        text,
    )
    assert guard, "the runc guard must come before the --memory arguments"
    assert guard.start() < text.index('--memory="${RIVAULT_MN_MEMORY_CAP}"')


def test_runc_partitions_default_to_the_long_step_worker_mode():
    """rccs (runc, no user systemd) runs each worker in one long-lived step
    (probe job 455210); crun sites keep the detached launch."""
    text = SLURM.read_text(encoding="utf-8")
    assert '_mn_worker_mode="${RIVAULT_MN_WORKER_MODE:-}"' in text
    assert '[[ "${RIVAULT_PARTITION_OCI_RUNTIME:-}" == "runc" ]] && _mn_worker_mode=step' in text


def test_step_mode_owns_venue_pause_process_and_container_in_one_step():
    text = SLURM.read_text(encoding="utf-8")
    step = text[text.index('if [[ "${_mn_worker_mode}" == "step" ]]; then') :]
    step = step[: step.index("              else")]
    # One srun whose bash prepares the venue, resets the pause process and
    # execs the foreground worker: nothing podman-related in a short step.
    assert step.count("srun ") == 1
    assert "rivault_venue_prepare" in step and "podman system migrate" in step
    assert "exec ${_wcmd}" in step
    assert " -d " not in step


def test_rccs_deepseek_spells_out_the_step_worker_mode(env):
    """Job 456170 (2026-11-05): the runc → step default never fired on rccs
    (the wrapper's partition-profile loader fails there), so the entry names
    the mode itself."""
    assert env["RIVAULT_MN_WORKER_MODE"] == "step"


def _site_env_loader() -> str:
    text = SLURM.read_text(encoding="utf-8")
    start = text.index("load_site_env_defaults() {")
    return text[start : text.index("\n}\n", start) + 3]


def test_site_env_loader_failure_is_logged_not_silent(tmp_path):
    """rccs python3 cannot import umrs.orchestrator (no psycopg2); the loader
    used to swallow that and apply no site/partition default without a word."""
    fake = tmp_path / "bin"
    fake.mkdir()
    py = fake / "python3"
    py.write_text("#!/bin/sh\necho \"ModuleNotFoundError: No module named 'psycopg2'\" >&2\nexit 1\n")
    py.chmod(0o755)
    script = _site_env_loader() + 'load_site_env_defaults; echo "rc=$? MARK=${RIVAULT_PARTITION_OCI_RUNTIME:-unset}"\n'
    proc = subprocess.run(
        ["bash", "-c", script],
        capture_output=True,
        text=True,
        env={"PATH": f"{fake}:/usr/bin:/bin", "REPO_ROOT": str(tmp_path), "TMPDIR": str(tmp_path)},
        check=False,
    )
    assert proc.stdout.strip() == "rc=0 MARK=unset"
    assert "site.yaml/partition-profile env defaults NOT applied" in proc.stderr
    assert "No module named 'psycopg2'" in proc.stderr
    assert not list(tmp_path.glob("rivault-site-env.*"))


def test_rccs_deepseek_reclaims_cuda_memory_before_start(env):
    """Job 458139: CUDA free 59.4 of 121.6 GiB failed vLLM's 0.78 check; a
    100 GiB allocation reclaims the Lustre cache (probe jobs 458165/458166)."""
    assert env["RIVAULT_MN_PRESTART_RECLAIM_GIB"] == "100"


def test_both_ranks_reclaim_right_before_they_start():
    text = SLURM.read_text(encoding="utf-8")
    assert "'\"${_mn_reclaim}\"'exec vllm serve" in text
    assert text.count("&& ${_mn_reclaim}exec ray start") == 2  # step + detached worker


def _reclaim_block() -> str:
    text = SLURM.read_text(encoding="utf-8")
    start = text.index('    _mn_reclaim=""')
    return text[start : text.index("\n    fi\n", start) + 8]


@pytest.mark.parametrize(
    ("gib", "mounted", "cmd"),
    [("100", True, "{ python3 /opt/rivault/gb10_reclaim.py || true; } && "), ("", False, "")],
)
def test_reclaim_is_opt_in_and_never_blocks(gib, mounted, cmd):
    script = (
        "_mn_vllm_args=()\n"
        + _reclaim_block()
        + 'printf "%s\\n" "${_mn_vllm_args[@]}"; printf "CMD=[%s]\\n" "${_mn_reclaim}"\n'
    )
    proc = subprocess.run(
        ["bash", "-c", script],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", "SCRIPT_DIR": "/w", "RIVAULT_MN_PRESTART_RECLAIM_GIB": gib},
        check=True,
    )
    assert ("/w/gb10_reclaim.py:/opt/rivault/gb10_reclaim.py:ro" in proc.stdout) is mounted
    assert f"CMD=[{cmd}]" in proc.stdout
