"""Ops-watch #590 (bdrt, 2026-10-26): "Backend job 8318 failed before
publishing an endpoint: unknown failure", logged at ERROR with a traceback.

The remote-Slurm backend reported FAILED without a reason (the wrapper's EXIT
trap deletes the info file, and the squeue/sacct fallback kept only the mapped
status), and the orchestrator logged a classified job outcome as a server
defect. The job's failure is real; the lost reason and the traceback are not.
"""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace

from umrs.orchestrator.base import JobStatus
from umrs.orchestrator.endpoint_registry import EndpointRegistry
from umrs.orchestrator.readiness import ReadinessConfig, ReadinessPoller
from umrs.orchestrator.remote_slurm_backend import RemoteSlurmBackend
from umrs.orchestrator.service import OrchestrationService, ProvisionRequest


class _Ssh:
    def __init__(self, stdout: str):
        self.stdout = stdout
        self.commands: list[str] = []

    async def run_remote(self, command: str):
        self.commands.append(command)
        return SimpleNamespace(ok=True, stdout=self.stdout, stderr="")


def _remote_backend(monkeypatch, sacct_stdout: str):
    backend = RemoteSlurmBackend.__new__(RemoteSlurmBackend)
    ssh = _Ssh(sacct_stdout)

    async def no_info(_job_id):
        return None

    async def slurm_failed(_job_id):
        return JobStatus.FAILED, None

    async def get_ssh():
        return ssh

    monkeypatch.setattr(backend, "_read_remote_job_info", no_info)
    monkeypatch.setattr(backend, "_query_slurm_job_status", slurm_failed)
    monkeypatch.setattr(backend, "_get_ssh", get_ssh)
    return backend


def test_failed_job_carries_the_slurm_state(monkeypatch):
    backend = _remote_backend(monkeypatch, "OUT_OF_MEMORY|0:125\n")

    info = asyncio.run(backend.get_job_status_before_ready("8318"))

    assert info.status == JobStatus.FAILED
    assert "OUT_OF_MEMORY" in (info.error_message or "")
    assert "0:125" in (info.error_message or "")


class _FailedBackend:
    config = SimpleNamespace(readiness_poll_interval_sec=1)

    async def get_job_status_before_ready(self, job_id):
        return SimpleNamespace(
            job_id=job_id,
            status=JobStatus.FAILED,
            endpoint=None,
            error_message="Slurm state FAILED, exit code 1:0",
        )

    async def cleanup(self, _job_id, **_kwargs):  # the real one takes force=
        return None


class _PortManager:
    def release(self, port, host):
        pass


def test_job_ended_before_publishing_is_a_warning_not_a_traceback(caplog):
    registry = EndpointRegistry()
    registry.register(
        project_id="p", model="m", job_id="8318", host="pending", port=12001,
        backend="slurm-remote",
    )
    service = OrchestrationService(
        registry=registry,
        port_manager=_PortManager(),
        readiness_poller=ReadinessPoller(ReadinessConfig(timeout_sec=60)),
        backend=_FailedBackend(),
    )

    async def finalize():
        start = asyncio.get_running_loop().time()
        return await service._finalize_provision(
            request=ProvisionRequest("p", "m"),
            job_id="8318",
            target_host="pending",
            actual_host="pending",
            actual_port=12001,
            deadline=start + 60,
            progress=None,
            start_time=start,
        )

    with caplog.at_level(logging.INFO, logger="umrs.orchestrator.service"):
        result = asyncio.run(finalize())

    assert result.success is False
    assert "exit code 1:0" in result.error
    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert errors == []
    assert "failed before publishing an endpoint" in caplog.text
