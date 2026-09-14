"""Tests for durable model-download task recovery."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from umrs.jobs.model_download import (
    DownloadTaskRegistry,
    ModelDownloadStatusChecker,
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
from umrs.jobs.reconciler import Reconciler, ReconcileResult


@dataclass
class FakeDownloadJobStore:
    """Minimal durable job store facade for registry recovery tests."""

    jobs: list[Job]
    attempts: dict[str, JobAttempt]

    def list_jobs(self, **kwargs):
        status = kwargs.get("status")
        job_type = kwargs.get("job_type")
        return [job for job in self.jobs if job.status == status and job.job_type == job_type]

    def get_attempt(self, attempt_id: str) -> JobAttempt:
        return self.attempts[attempt_id]

    def get_attempts_for_job(self, job_id: str) -> list[JobAttempt]:
        return [attempt for attempt in self.attempts.values() if attempt.job_id == job_id]


@pytest.fixture
def download_job() -> Job:
    """Create a running durable model-download job."""
    return Job(
        job_id=generate_job_id(),
        project_id="proj-test",
        user_id="user-test",
        job_type=JobType.MODEL_DOWNLOAD,
        spec={
            "source": "huggingface",
            "source_id": "microsoft/DialoGPT-small",
            "target_path": "/shared/models/system/dialogpt-small",
            "model_name": "dialogpt-small",
            "scope": "system",
            "hf_token": "secret-token",
        },
        idempotency_key="test-key-123",
        status=JobStatus.RUNNING,
    )


@pytest.fixture
def running_attempt(download_job: Job) -> JobAttempt:
    """Create a running model-download attempt for the job."""
    attempt = JobAttempt(
        attempt_id=generate_attempt_id(),
        job_id=download_job.job_id,
        attempt_no=1,
        backend="model_download",
        status=AttemptStatus.RUNNING,
    )
    download_job.active_attempt_id = attempt.attempt_id
    return attempt


@pytest.mark.asyncio
async def test_recover_download_registry_from_active_attempt(
    download_job: Job,
    running_attempt: JobAttempt,
):
    """Active durable model download attempts rebuild registry context."""
    registry = DownloadTaskRegistry()
    store = FakeDownloadJobStore(
        jobs=[download_job],
        attempts={running_attempt.attempt_id: running_attempt},
    )

    recovered = await recover_download_registry_from_job_store(
        store,
        download_registry=registry,
    )

    task = await registry.get(running_attempt.attempt_id)
    assert recovered == 1
    assert task is not None
    assert task.job_id == download_job.job_id
    assert task.hf_token is None
    assert task.progress.status == "recovering"
    assert task.metadata["recovered_from_durable_job"] is True


@pytest.mark.asyncio
async def test_status_checker_treats_recovered_context_as_running(
    download_job: Job,
    running_attempt: JobAttempt,
):
    """Recovered contexts keep attempts visible until durable retry proceeds."""
    registry = DownloadTaskRegistry()
    store = FakeDownloadJobStore(
        jobs=[download_job],
        attempts={running_attempt.attempt_id: running_attempt},
    )
    await recover_download_registry_from_job_store(
        store,
        download_registry=registry,
    )

    checker = ModelDownloadStatusChecker(download_registry=registry)
    result = await checker.check_attempt(running_attempt, download_job)

    assert result.status == "running"
    assert result.metrics["status"] == "recovering"


@pytest.mark.asyncio
async def test_recovery_preserves_partial_progress_without_token(
    download_job: Job,
    running_attempt: JobAttempt,
):
    """Recovered interrupted downloads keep partial metrics but not secrets."""
    running_attempt.metrics = {
        "total_files": 4,
        "downloaded_files": 2,
        "total_bytes": 4096,
        "downloaded_bytes": 2048,
        "current_file": "model-00002.safetensors",
        "status": "downloading",
    }
    registry = DownloadTaskRegistry()
    store = FakeDownloadJobStore(
        jobs=[download_job],
        attempts={running_attempt.attempt_id: running_attempt},
    )

    recovered = await recover_download_registry_from_job_store(
        store,
        download_registry=registry,
    )

    task = await registry.get(running_attempt.attempt_id)
    assert recovered == 1
    assert task is not None
    assert task.hf_token is None
    assert task.progress.status == "recovering"
    assert task.progress.total_files == 4
    assert task.progress.downloaded_files == 2
    assert task.progress.total_bytes == 4096
    assert task.progress.downloaded_bytes == 2048
    assert task.progress.current_file == "model-00002.safetensors"
    assert task.progress.progress_percent == 50.0
    assert task.metadata["recovered_from_durable_job"] is True


@pytest.mark.asyncio
async def test_recovery_ignores_terminal_active_attempt(download_job: Job):
    """Terminal attempts must not be resurrected as interrupted downloads."""
    attempt = JobAttempt(
        attempt_id=generate_attempt_id(),
        job_id=download_job.job_id,
        attempt_no=1,
        backend="model_download",
        status=AttemptStatus.FAILED,
    )
    download_job.active_attempt_id = attempt.attempt_id
    registry = DownloadTaskRegistry()
    store = FakeDownloadJobStore(
        jobs=[download_job],
        attempts={attempt.attempt_id: attempt},
    )

    recovered = await recover_download_registry_from_job_store(
        store,
        download_registry=registry,
    )

    assert recovered == 0
    assert await registry.get(attempt.attempt_id) is None


def test_reconciler_records_model_download_restart_policy(download_job: Job):
    """Stale model download attempts get an explicit recovery policy marker."""
    fake_store = type("FakeStore", (), {"umrs_instance_id": "umrs-test"})()
    reconciler = Reconciler(fake_store)
    result = ReconcileResult()

    reconciler._record_model_download_recovery(
        download_job,
        "att-lost",
        result,
    )

    assert result.model_download_recoveries == ["att-lost:resume_on_retry"]
