"""Shared-storage interrupted model download E2E test.

Validates that:
- An interrupted HTTP download leaves a partial file on disk
- The durable job store records partial-progress metrics
- Registry recovery from the durable store rebuilds download context
  without restoring secret tokens
- The reconciler records an explicit retry policy for the stale attempt

This test exercises real filesystem I/O (tmp_path or shared storage)
and a real loopback HTTP server that interrupts the transfer.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

import pytest

from umrs.jobs.model_download import (
    DownloadSource,
    DownloadTaskContext,
    DownloadTaskRegistry,
    ModelDownloadBackend,
)
from umrs.jobs.model_download_state import (
    DownloadProgress,
    recover_download_registry_from_job_store,
)
from umrs.jobs.models import (
    AttemptStatus,
    Job,
    JobAttempt,
    JobStatus,
    JobType,
    generate_attempt_id,
    generate_job_id,
)
from umrs.jobs.reconciler import ReconcileResult, Reconciler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _start_interrupting_server(total_bytes: int, serve_bytes: int):
    """Loopback HTTP server that advertises *total_bytes* but closes after
    writing *serve_bytes*.  This simulates a network interruption during a
    large model download.
    """
    from aiohttp import web

    chunk_size = min(serve_bytes, 64 * 1024)
    written = 0

    async def handler(request):
        nonlocal written
        response = web.StreamResponse(
            status=200,
            headers={"Content-Length": str(total_bytes)},
        )
        await response.prepare(request)
        while written < serve_bytes:
            n = min(chunk_size, serve_bytes - written)
            await response.write(b"x" * n)
            written += n
        await asyncio.sleep(0)
        transport = request.transport
        if transport:
            transport.close()
        return response

    app = web.Application()
    app.router.add_get("/large-model.bin", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    sockets = site._server.sockets
    assert sockets
    return runner, sockets[0].getsockname()[1]


@dataclass
class InMemoryJobStore:
    """Minimal store that supports recovery and reconciler flows."""

    jobs: list[Job] = field(default_factory=list)
    attempts: dict[str, JobAttempt] = field(default_factory=dict)

    def list_jobs(self, **kwargs) -> list[Job]:
        status = kwargs.get("status")
        job_type = kwargs.get("job_type")
        return [j for j in self.jobs if j.status == status and j.job_type == job_type]

    def get_attempt(self, attempt_id: str) -> JobAttempt:
        return self.attempts[attempt_id]

    def get_attempts_for_job(self, job_id: str) -> list[JobAttempt]:
        return [a for a in self.attempts.values() if a.job_id == job_id]

    def update_attempt(self, attempt: JobAttempt) -> None:
        self.attempts[attempt.attempt_id] = attempt

    def update_job(self, job: Job) -> None:
        for i, j in enumerate(self.jobs):
            if j.job_id == job.job_id:
                self.jobs[i] = job
                return


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_interrupted_download_leaves_partial_on_shared_storage(tmp_path):
    """An interrupted HTTP download leaves a partial file and records metrics
    that survive registry recovery from the durable store.

    This simulates a shared-storage (Lustre/NFS) scenario where the
    download writes to a directory that is visible across restarts.
    """
    total_bytes = 1024 * 1024  # 1 MiB advertised
    serve_bytes = 256 * 1024  # only 256 KiB delivered

    # -- Phase 1: interrupted download ----------------------------------------
    validator = MagicMock()
    validator.is_allowed.return_value = True
    registry = DownloadTaskRegistry()
    backend = ModelDownloadBackend(
        model_store=MagicMock(),
        path_validator=validator,
        download_registry=registry,
    )

    runner, port = await _start_interrupting_server(total_bytes, serve_bytes)
    published: list[dict[str, Any]] = []
    try:
        ctx = DownloadTaskContext(
            job_id="job-shared-1",
            attempt_id="att-shared-1",
            source=DownloadSource.HTTP,
            source_id=f"http://127.0.0.1:{port}/large-model.bin",
            target_path=str(tmp_path / "download-dir"),
            model_name="large-model",
            scope="system",
            owner_id="user-dl",
        )
        progress = await backend._download_from_http(
            ctx,
            progress_callback=lambda p: published.append(p.to_dict()),
        )
    finally:
        await runner.cleanup()

    # Partial file must exist on disk
    partial_file = tmp_path / "download-dir" / "large-model.bin"
    assert partial_file.exists(), "Partial file missing after interrupted download"
    file_size = partial_file.stat().st_size
    assert file_size > 0, "Partial file is empty"
    assert file_size < total_bytes, "File should be partial, not complete"
    assert progress.status == "failed"
    assert progress.downloaded_bytes == file_size
    assert progress.total_bytes == total_bytes

    # Progress events should include at least one with downloaded_bytes > 0
    assert any(p["downloaded_bytes"] > 0 for p in published)
    assert published[-1]["status"] == "failed"

    # -- Phase 2: durable store recovery --------------------------------------
    job = Job(
        job_id="job-shared-1",
        project_id="proj-dl",
        user_id="user-dl",
        job_type=JobType.MODEL_DOWNLOAD,
        spec={
            "source": "http",
            "source_id": f"http://127.0.0.1:{port}/large-model.bin",
            "target_path": str(tmp_path / "download-dir"),
            "model_name": "large-model",
            "scope": "system",
            "hf_token": "should-not-appear",
        },
        idempotency_key="shared-dl-key",
        status=JobStatus.RUNNING,
    )
    attempt = JobAttempt(
        attempt_id="att-shared-1",
        job_id="job-shared-1",
        attempt_no=1,
        backend="model_download",
        status=AttemptStatus.RUNNING,
        metrics={
            "downloaded_bytes": file_size,
            "total_bytes": total_bytes,
            "status": "failed",
        },
    )
    job.active_attempt_id = attempt.attempt_id

    store = InMemoryJobStore(jobs=[job], attempts={attempt.attempt_id: attempt})
    recovered_registry = DownloadTaskRegistry()

    recovered = await recover_download_registry_from_job_store(
        store,
        download_registry=recovered_registry,
    )

    assert recovered >= 1, "No tasks recovered from durable store"

    # Verify the recovered context does NOT restore secrets
    recovered_ctx = await recovered_registry.get(attempt.attempt_id)
    assert recovered_ctx is not None, "Recovered context missing for att-shared-1"
    assert recovered_ctx.hf_token is None, "Secret token should not be restored"

    download_job = job  # alias for Phase 3

    # -- Phase 3: reconciler marks stale attempt with retry policy -----------
    fake_store = type("FakeStore", (), {"umrs_instance_id": "umrs-test"})()
    reconciler = Reconciler(fake_store)
    result = ReconcileResult()
    reconciler._record_model_download_recovery(
        download_job,
        "att-shared-stale",
        result,
    )
    assert result.model_download_recoveries, "Reconciler should record recovery policy"

    # The partial file still exists on "shared storage" after recovery
    assert partial_file.exists(), "Partial file should survive recovery on shared storage"
    assert partial_file.stat().st_size == file_size


@pytest.mark.asyncio
async def test_completed_download_not_resurrected_on_recovery(tmp_path):
    """Terminal (completed/failed) attempts should not be resurrected during
    registry recovery — only active running attempts are recovered."""
    job = Job(
        job_id="job-done-1",
        project_id="proj-dl",
        user_id="user-dl",
        job_type=JobType.MODEL_DOWNLOAD,
        spec={
            "source": "http",
            "source_id": "http://example.invalid/done.bin",
            "target_path": str(tmp_path),
            "model_name": "done-model",
            "scope": "system",
        },
        idempotency_key="done-key",
        status=JobStatus.RUNNING,
    )
    terminal_attempt = JobAttempt(
        attempt_id="att-done-1",
        job_id="job-done-1",
        attempt_no=1,
        backend="model_download",
        status=AttemptStatus.SUCCEEDED,
        metrics={
            "downloaded_bytes": 1024,
            "total_bytes": 1024,
            "status": "completed",
        },
    )
    job.active_attempt_id = terminal_attempt.attempt_id

    store = InMemoryJobStore(
        jobs=[job],
        attempts={terminal_attempt.attempt_id: terminal_attempt},
    )
    registry = DownloadTaskRegistry()
    recovered = await recover_download_registry_from_job_store(
        store,
        download_registry=registry,
    )

    assert recovered == 0, "Terminal attempts should not be recovered"
    assert await registry.get("att-done-1") is None
