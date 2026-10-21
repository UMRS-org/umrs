"""Cycle-2 umrs-api lane (2026-10-21): an identical re-ingest is a 409 that
names the job it duplicates, not a 500.

Found live: POST /umrs/k2/ingest twice with the same sources answered
``500 server.internal_error "Job already exists: job-…"``. The ingest job's
idempotency key is the spec hash (unlike export/verify, which carry a
request nonce), so the second submit raises ``DuplicateJobError`` and the
route's catch-all turned it into a server error. POST /umrs/jobs already
answers the same case with 409 + ``existing_job_id``.
"""

from __future__ import annotations

from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from umrs.api.files import get_project_id
from umrs.api.knowledge_v2 import router
from umrs.jobs.store import DuplicateJobError


class _DuplicateRuntime:
    async def submit_job_sync(self, **_kwargs):
        raise DuplicateJobError("job-first-ingest")


def test_identical_reingest_is_409_with_the_existing_job() -> None:
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_project_id] = lambda: "pa"
    with (
        patch("umrs.server.durable_job_runtime.get_runtime", return_value=_DuplicateRuntime()),
        patch("umrs.api.governance.guard_ingest", return_value=None),
    ):
        resp = TestClient(app).post(
            "/umrs/k2/ingest",
            json={"sources": [{"kind": "inline_text", "uri": "a.md", "content": "same"}]},
        )
    assert resp.status_code == 409, resp.text
    detail = resp.json()["detail"]
    assert detail["existing_job_id"] == "job-first-ingest"
    assert detail["error"] == "duplicate_ingest"
