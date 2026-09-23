"""D-37 scheduler leg pins: the hpc_script durable job kind (ROADMAP 3.3)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from umrs.jobs.hpc_script import (
    HpcScriptBackend,
    HpcScriptStatusChecker,
    _sbatch_command,
    validate_hpc_script_spec,
)
from umrs.jobs.models import JobType


class _Result:
    def __init__(self, stdout="", stderr="", returncode=0):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode

    @property
    def ok(self):
        return self.returncode == 0


class _SSH:
    def __init__(self, results):
        self._results = list(results)
        self.commands = []

    async def run_remote(self, command, **_kw):
        self.commands.append(command)
        return self._results.pop(0)


def _job(spec):
    return SimpleNamespace(job_id="j-1", attempt_id=None, user_id="u", project_id="p", spec=spec)


def _attempt():
    return SimpleNamespace(attempt_id="a-1", metrics={}, backend_ref="")


def _run(coro):
    return asyncio.run(coro)


def test_spec_validation():
    good = validate_hpc_script_spec({"script_path": "/lvs0/x/run.sh"})
    assert good["script_path"] == "/lvs0/x/run.sh"
    assert good["sbatch_args"] == []
    with pytest.raises(ValueError, match="script_path"):
        validate_hpc_script_spec({})
    with pytest.raises(ValueError, match="sbatch_args"):
        validate_hpc_script_spec({"script_path": "x", "sbatch_args": "not-a-list"})


def test_sbatch_command_quotes_never_interpolates():
    cmd = _sbatch_command("/lvs0/x/my script.sh", ["-p", "short; rm -rf /"])
    assert "'/lvs0/x/my script.sh'" in cmd
    assert "'short; rm -rf /'" in cmd  # quoted — never executed by the shell


def test_backend_submits_and_parses_slurm_id():
    ssh = _SSH([_Result("Submitted batch job 385046\n")])

    async def factory():
        return ssh

    backend = HpcScriptBackend(ssh=factory)
    attempt = _attempt()
    ref = _run(backend.submit_job(_job({"script_path": "/lvs0/x/run.sh"}), attempt))
    assert ref == "hpc_script|script=/lvs0/x/run.sh|slurm=385046"
    assert attempt.metrics["slurm_job_id"] == "385046"
    assert ssh.commands[0].startswith("sbatch ")


def test_backend_sbatch_failure_raises_named():
    ssh = _SSH([_Result(stderr="Batch job submission failed", returncode=1)])

    async def factory():
        return ssh

    backend = HpcScriptBackend(ssh=factory)
    with pytest.raises(RuntimeError, match="sbatch failed"):
        _run(backend.submit_job(_job({"script_path": "/x"}), _attempt()))


def test_backend_degrades_without_ssh():
    backend = HpcScriptBackend(ssh=None)
    ref = _run(backend.submit_job(_job({"script_path": "/x"}), _attempt()))
    assert "degraded=no_ssh" in ref


def test_checker_maps_squeue_running_then_sacct_completed():
    ssh = _SSH(
        [_Result("RUNNING\n"), _Result("COMPLETED\n")]  # running cycle, then terminal
    )

    async def factory():
        return ssh

    checker = HpcScriptStatusChecker(ssh_factory=factory)
    attempt = _attempt()
    attempt.backend_ref = "hpc_script|script=/x|slurm=99"
    running = _run(checker.check_status(attempt, _job({})))
    assert running.status == "running"
    assert running.status_detail == "slurm state RUNNING"
    done = _run(checker.check_status(attempt, _job({})))
    assert done.status == "succeeded"
    assert done.metrics["slurm_state"] == "COMPLETED"


def _checker_for(state):
    ssh = _SSH([_Result(""), _Result(state + "\n")])  # squeue empty → sacct

    async def factory():
        return ssh

    return HpcScriptStatusChecker(ssh_factory=factory)


def test_checker_failed_terminal_states():
    for state in ("FAILED", "TIMEOUT", "CANCELLED", "OUT_OF_MEMORY"):
        checker = _checker_for(state)
        attempt = _attempt()
        attempt.backend_ref = "hpc_script|script=/x|slurm=1"
        result = _run(checker.check_status(attempt, _job({})))
        assert result.status == "failed", state


def test_checker_refuses_degraded_and_unknown():
    checker = HpcScriptStatusChecker(ssh_factory=None)
    attempt = _attempt()
    attempt.backend_ref = "hpc_script|script=/x|degraded=no_ssh"
    assert _run(checker.check_status(attempt, _job({}))).status == "failed"
    attempt.backend_ref = "hpc_script|script=/x|slurm="
    assert _run(checker.check_status(attempt, _job({}))).status == "failed"


def test_job_kind_registered_everywhere():
    from umrs.jobs.models import DEFAULT_BACKOFFS

    assert JobType.HPC_SCRIPT.value == "hpc_script"
    assert JobType.HPC_SCRIPT in DEFAULT_BACKOFFS
    assert JobType.HPC_SCRIPT not in ()  # dispatchable: adapters registered it
