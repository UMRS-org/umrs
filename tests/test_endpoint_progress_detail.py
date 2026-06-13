"""Focused tests for UMRS endpoint provisioning progress detail emission."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from umrs.jobs.adapters import EndpointProvisionStatusChecker
from umrs.jobs.models import AttemptStatus, Job, JobAttempt, JobStatus, JobType
from umrs.orchestrator.base import EndpointInfo, JobInfo, JobStatus as OrchestratorJobStatus
from umrs.orchestrator.endpoint_registry import EndpointRegistry


@pytest.mark.asyncio
async def test_checker_persists_scheduler_allocation_detail():
    registry = EndpointRegistry()
    registry.register(
        project_id="project-a",
        model="bonsai-8b",
        job_id="slurm-123",
        host="pending",
        port=11434,
        backend="slurm",
    )

    orchestrator = MagicMock()
    orchestrator.get_job_status = AsyncMock(
        return_value=JobInfo(
            job_id="slurm-123",
            status=OrchestratorJobStatus.PENDING,
            backend="slurm",
        )
    )
    readiness_poller = MagicMock()

    checker = EndpointProvisionStatusChecker(
        readiness_poller,
        registry,
        orchestrator=orchestrator,
    )
    job = Job(
        job_id="job-123",
        project_id="project-a",
        user_id="mnagaso",
        job_type=JobType.ENDPOINT_PROVISION,
        spec={"model": "bonsai-8b"},
        idempotency_key="idem-123",
        status=JobStatus.RUNNING,
        active_attempt_id="att-1",
    )
    attempt = JobAttempt(
        attempt_id="att-1",
        job_id="job-123",
        attempt_no=1,
        backend="slurm",
        backend_ref="slurm-123",
        status=AttemptStatus.SUBMITTED,
    )

    result = await checker.check_status(attempt, job)
    endpoint = registry.get_by_job_id("slurm-123")

    assert result.status == "running"
    assert result.status_detail == "waiting for scheduler allocation"
    assert endpoint is not None
    assert endpoint.metadata["detail"] == "waiting for scheduler allocation"
    assert endpoint.metadata["job_status"] == "pending"


@pytest.mark.asyncio
async def test_checker_persists_readiness_warmup_detail():
    registry = EndpointRegistry()
    registry.register(
        project_id="project-a",
        model="bonsai-8b",
        job_id="slurm-124",
        host="compute-01",
        port=11434,
        backend="slurm",
    )

    orchestrator = MagicMock()
    orchestrator.get_job_status = AsyncMock(
        return_value=JobInfo(
            job_id="slurm-124",
            status=OrchestratorJobStatus.STARTING,
            endpoint=EndpointInfo(node="compute-01", litellm_port=11434),
            backend="slurm",
        )
    )
    readiness_poller = MagicMock()
    readiness_poller.config.auth_token = None
    readiness_poller.check_ready_detail = AsyncMock(return_value=(False, False, [], "Chat probe timeout"))

    checker = EndpointProvisionStatusChecker(
        readiness_poller,
        registry,
        orchestrator=orchestrator,
    )
    job = Job(
        job_id="job-124",
        project_id="project-a",
        user_id="mnagaso",
        job_type=JobType.ENDPOINT_PROVISION,
        spec={"model": "bonsai-8b"},
        idempotency_key="idem-124",
        status=JobStatus.RUNNING,
        active_attempt_id="att-2",
    )
    attempt = JobAttempt(
        attempt_id="att-2",
        job_id="job-124",
        attempt_no=1,
        backend="slurm",
        backend_ref="slurm-124",
        status=AttemptStatus.RUNNING,
    )

    result = await checker.check_status(attempt, job)
    endpoint = registry.get_by_job_id("slurm-124")

    assert result.status == "running"
    assert result.status_detail == "loading model weights / waiting for inference warmup"
    assert endpoint is not None
    assert endpoint.metadata["detail"] == "loading model weights / waiting for inference warmup"
    assert endpoint.metadata["job_status"] == "starting"
