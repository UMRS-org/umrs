"""DurableJobRuntime.submit_job_sync — async shim used by ingest routes.

The knowledge ingest/export routes call ``runtime.submit_job_sync(...)``;
this is the delegation contract the production runtime must honor
(the route tests monkeypatch get_runtime with their own fakes, so only
this test covers the real method).
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

from umrs.jobs.service import JobType
from umrs.orchestrator.durable import DurableOrchestrationService
from umrs.server.durable_job_runtime import DurableJobRuntime


def test_submit_job_sync_delegates_to_service_off_loop():
    service = MagicMock()
    service.submit_job.return_value = "job-42"
    runtime = DurableJobRuntime(db_pool=object(), service=service)
    job_type = MagicMock(name="JobTypeStub")

    job_id = asyncio.run(
        runtime.submit_job_sync(
            job_type=job_type,
            project_id="proj-a",
            spec={"sources": [{"kind": "local_file", "uri": "/data/x.json"}]},
        )
    )

    assert job_id == "job-42"
    service.submit_job.assert_called_once_with(
        job_type=job_type,
        project_id="proj-a",
        user_id="",
        spec={"sources": [{"kind": "local_file", "uri": "/data/x.json"}]},
    )


def test_submit_job_sync_custom_user_id():
    service = MagicMock()
    service.submit_job.return_value = "job-7"
    runtime = DurableJobRuntime(db_pool=object(), service=service)

    asyncio.run(
        runtime.submit_job_sync(
            job_type=MagicMock(),
            project_id="proj-a",
            spec={},
            user_id="alice",
        )
    )

    assert service.submit_job.call_args.kwargs["user_id"] == "alice"


def test_orchestration_service_submit_job_alias():
    """Regression for live 500: shim delegates to
    ``DurableOrchestrationService.submit_job`` — the alias must exist and
    forward to ``JobService.submit_job`` with ``enqueue=True``.
    """
    service = DurableOrchestrationService.__new__(DurableOrchestrationService)
    job_service = MagicMock()
    job_service.submit_job.return_value = "job-99"
    service._job_service = job_service

    job_id = service.submit_job(
        job_type=JobType.KNOWLEDGE_INGEST,
        project_id="proj-a",
        spec={"sources": [{"kind": "local_file", "uri": "/data/x.json"}]},
        user_id="alice",
    )

    assert job_id == "job-99"
    job_service.submit_job.assert_called_once_with(
        project_id="proj-a",
        user_id="alice",
        job_type=JobType.KNOWLEDGE_INGEST,
        spec={"sources": [{"kind": "local_file", "uri": "/data/x.json"}]},
        enqueue=True,
    )
