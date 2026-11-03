"""rccs DeepSeek-V4-Flash multi-node launch (2026-11-03, jobs 454912/454924/454939).

rccs is rootless runc (podman 4.9.3) without cgroup delegation: any container
memory limit fails ("rootless needs no limits + no cgrouppath"), which killed
the Ray worker on the second node; and the keep-id:uid userns default is
rejected there.
"""

from __future__ import annotations

import re
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
