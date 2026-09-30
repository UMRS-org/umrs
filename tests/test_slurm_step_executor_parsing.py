"""Regression: squeue multi-line output parses identically to sacct.

Array/het Slurm jobs emit one state line per element. The squeue branch of
``_check_job_status`` used to compare the whole multi-line stdout against
known states, so such jobs never resolved and polls timed out. The sacct
branch already normalized with ``.split("\n")[0]`` — parity restored.
"""

from __future__ import annotations

import subprocess

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
