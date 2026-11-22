"""U3 S6d: the cluster-loop env carries neutral names at both ends.

The launcher exports every loop value under ``UMRS_<name>`` and ``R4_<name>``;
the readers take the neutral name first. A job queued before the deploy
(old script and new env, or the reverse) still runs.
"""

from __future__ import annotations

from pathlib import Path

import pytest

UMRS = Path(__file__).resolve().parents[1]


def test_both_names_carry_every_value():
    from umrs.jobs.cluster_venue import _both_names

    env = _both_names(LOOP_OWNER="alice", GATEWAY_BASE="https://gw")
    assert env == {
        "UMRS_LOOP_OWNER": "alice",
        "R4_LOOP_OWNER": "alice",
        "UMRS_GATEWAY_BASE": "https://gw",
        "R4_GATEWAY_BASE": "https://gw",
    }


@pytest.mark.parametrize(
    ("env", "base"),
    [
        ({"UMRS_GATEWAY_BASE": "https://new"}, "https://new"),
        ({"R4_GATEWAY_BASE": "https://old/"}, "https://old"),
        ({"UMRS_GATEWAY_BASE": "https://new", "R4_GATEWAY_BASE": "https://old"}, "https://new"),
    ],
)
def test_the_workflow_port_reads_the_neutral_base_first(monkeypatch, env, base):
    from umrs.knowledge import cluster_workflow_port as port

    for name in ("UMRS_GATEWAY_BASE", "R4_GATEWAY_BASE"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    source = Path(port.__file__).read_text(encoding="utf-8")
    assert 'env("GATEWAY_BASE")' in source
    from umrs.shared.compat_names import env as compat_env

    assert compat_env("GATEWAY_BASE").rstrip("/") == base


def test_the_job_script_and_wrapper_accept_both_names():
    script = (UMRS / "src/umrs/knowledge/_cluster_loop_job.sh").read_text(encoding="utf-8")
    for name in ("GATEWAY_BASE", "LOOP_JOB_ID", "LOOP_ATTEMPT_ID", "LOOP_OWNER", "LOOP_PROJECT"):
        assert f"${{UMRS_{name}:-${{R4_{name}" in script, name
        assert f'-e UMRS_{name}="' in script and f'-e R4_{name}="' in script, name
    assert "${UMRS_SRC_DIR:-${R4_DIR" in script
    wrapper = (UMRS / "deploy/rccs/loop-job.slurm").read_text(encoding="utf-8")
    assert "${UMRS_SRC_DIR:-${R4_DIR" in wrapper
    assert "export R4_DIR UMRS_SRC_DIR" in wrapper
