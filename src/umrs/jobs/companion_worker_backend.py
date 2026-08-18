"""CompanionWorker durable backend (venue bookkeeping seam).

A companion worker venue can only exist as a real durable job — the
companion_turns FK demands it (live finding 2026-08-18: register_worker
alone used to land rows whose job_id no jobs row ever claimed, and the
first turn blew the FK). This backend is the bookkeeping attachment of
that submission: it validates the venue identity (registry key) and
returns a typed backend_ref; process supervision itself stays with
CompanionWorkerSupervisor explicitly (follow-up: spawn-on-submit).
"""

from __future__ import annotations

import logging

from umrs.jobs.models import Job, JobAttempt, JobType

from .monitor import AttemptCheckResult, BackendStatusChecker
from .reindex import _parse_ref
from .scheduler import OrchestratorBackend

logger = logging.getLogger(__name__)


def validate_companion_worker_spec(spec: dict) -> dict:
    if not isinstance(spec, dict):
        raise ValueError("companion_worker spec must be an object")
    registry_key = str(spec.get("registry_key") or "").strip()
    if not registry_key:
        raise ValueError("companion_worker spec requires registry_key")
    return {"registry_key": registry_key}


class CompanionWorkerBackend(OrchestratorBackend):
    """Durable marker backend for the per-(owner, project) venue worker."""

    @property
    def backend_name(self) -> str:
        return "companion_worker"

    async def submit_job(self, job: Job, attempt: JobAttempt) -> str:
        spec = validate_companion_worker_spec(job.spec or {})
        logger.info(
            "companion_worker venue job attached: job=%s registry_key=%s",
            job.job_id,
            spec["registry_key"],
        )
        return f"companion_worker|registry_key={spec['registry_key']}"


class CompanionWorkerStatusChecker(BackendStatusChecker):
    """Synchronous bookkeeping completion for venue worker jobs."""

    @property
    def job_type(self) -> JobType:
        return JobType.COMPANION_WORKER

    async def check_status(self, attempt: JobAttempt, job: Job) -> AttemptCheckResult:
        backend_ref = attempt.backend_ref
        if not backend_ref:
            return AttemptCheckResult.failed("No backend reference")
        if not backend_ref.startswith("companion_worker|"):
            return AttemptCheckResult.failed("Unexpected backend reference")
        parsed = _parse_ref(backend_ref)
        registry_key = parsed.get("registry_key")
        if not registry_key:
            return AttemptCheckResult.failed("Missing venue metadata")
        return AttemptCheckResult.succeeded(metrics={"registry_key": registry_key})
