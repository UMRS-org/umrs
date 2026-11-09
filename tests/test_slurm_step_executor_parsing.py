"""Regression: squeue multi-line output parses identically to sacct.

Array/het Slurm jobs emit one state line per element. The squeue branch of
``_check_job_status`` used to compare the whole multi-line stdout against
known states, so such jobs never resolved and polls timed out. The sacct
branch already normalized with ``.split("\n")[0]`` — parity restored.
"""

from __future__ import annotations

import subprocess

import pytest
from umrs.orchestrator.ssh_control import SSHControlConfig
from umrs.server.remote_slurm_step_executor import RemoteSlurmStepExecutor
from umrs.server.slurm_step_executor import SlurmStepExecutor


class _Proc:
    def __init__(self, returncode: int, stdout: str, stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _executor() -> SlurmStepExecutor:
    return SlurmStepExecutor()


def test_multi_line_running_state_normalizes(monkeypatch):
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if cmd[0] == "squeue":
            return _Proc(0, "RUNNING\nPENDING\nRUNNING\n")
        if cmd[0] == "sacct":
            return _Proc(0, "")
        raise AssertionError(cmd)

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert _executor()._check_job_status("123") == "RUNNING"


def test_single_line_state_unchanged(monkeypatch):
    def fake_run(cmd, **kwargs):
        if cmd[0] == "squeue":
            return _Proc(0, "PENDING\n")
        return _Proc(0, "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert _executor()._check_job_status("123") == "PENDING"


def test_trailing_whitespace_state_parsed(monkeypatch):
    def fake_run(cmd, **kwargs):
        if cmd[0] == "squeue":
            return _Proc(0, "CONFIGURING   \nRUNNING\n")
        return _Proc(0, "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert _executor()._check_job_status("123") == "PENDING"


def test_sacct_branch_fallback_after_squeue_empty(monkeypatch):
    def fake_run(cmd, **kwargs):
        if cmd[0] == "squeue":
            return _Proc(0, "")  # job no longer in queue
        if cmd[0] == "sacct":
            return _Proc(0, "COMPLETED\nCOMPLETED\n")
        raise AssertionError(cmd)

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert _executor()._check_job_status("123") == "COMPLETED"


@pytest.mark.parametrize(
    "raw",
    ["not-json", "[]", '{"spring-8": 1}', '{"spring-8": "bad account"}', '{"": "rkp00101"}'],
)
def test_project_account_overrides_refuse_malformed_values(monkeypatch, raw):
    monkeypatch.setenv("UMRS_SLURM_PROJECT_ACCOUNTS", raw)
    with pytest.raises(ValueError, match="UMRS_SLURM_PROJECT_ACCOUNTS"):
        RemoteSlurmStepExecutor(ssh_config=SSHControlConfig(ssh_target="login"))


def test_project_account_field_preserves_positional_constructor_arguments(monkeypatch):
    monkeypatch.delenv("UMRS_SLURM_PROJECT_ACCOUNTS", raising=False)
    executor = RemoteSlurmStepExecutor(
        "gpu",
        1,
        8,
        "32G",
        "01:00:00",
        "acct",
        "img:1",
        ssh_config=SSHControlConfig(ssh_target="login"),
    )
    assert (executor.account, executor.image, executor.project_accounts) == (
        "acct",
        "img:1",
        {},
    )
