"""``research_loop_job`` durable backend + status checker (ADR-036 D1/D2).

One job per server-hosted research loop; an attempt is one hosted run. The
backend does not execute anything itself: ``submit_job`` hands the loop to
``umrs.knowledge.loop_hosting.host`` (which spawns the engine/agent driver
against the loop store, resuming at the stored iteration) and returns a
typed backend_ref. The checker reads the loop record:

    running / paused              → running (iteration n/m)
    running + driver not alive    → failed  (the scheduler re-hosts = resume)
    stopped (completed/cancelled) → succeeded (metrics: iterations, findings)
    error                         → failed (the loop's own error text)
    record missing                → failed

LOST attempts (UMRS restart) never reach the checker: the reconciler marks
them LOST and requeues the job; the next attempt resumes the campaign.
"""

from __future__ import annotations

import logging

from umrs.jobs.models import Job, JobAttempt, JobType

from .monitor import AttemptCheckResult, BackendStatusChecker
from .reindex import _parse_ref
from .scheduler import OrchestratorBackend

logger = logging.getLogger(__name__)

REF_PREFIX = "research_loop_job|"
DRIVERS = frozenset({"engine", "agent"})


CLUSTER_VENUE_PATTERN = "cluster:"


def validate_research_loop_spec(spec: dict) -> dict:
    if not isinstance(spec, dict):
        raise TypeError("research_loop_job spec must be an object")
    loop_id = str(spec.get("loop_id") or "").strip()
    if not loop_id:
        raise ValueError("research_loop_job spec requires loop_id")
    driver = str(spec.get("driver") or "engine").strip()
    if driver not in DRIVERS:
        raise ValueError(f"research_loop_job driver must be one of {sorted(DRIVERS)}")
    venue = str(spec.get("venue") or "").strip()
    if venue and not (
        venue == "frontend" or venue.startswith(CLUSTER_VENUE_PATTERN)
    ):
        raise ValueError(
            f"research_loop_job venue must be 'frontend' or 'cluster:<name>' (got {venue!r})"
        )
    out = {"loop_id": loop_id, "driver": driver}
    if venue.startswith(CLUSTER_VENUE_PATTERN):
        if driver != "engine":
            raise ValueError(
                "research_loop_job on a cluster venue accepts driver 'engine' only "
                "(the companion agent driver needs cluster-side wiring — follow-on)"
            )
        out["cluster_site"] = split_cluster_venue_name(venue)
    elif venue and venue != "frontend":
        raise ValueError(
            f"research_loop_job venue must be 'frontend' or 'cluster:<site-name>' "
            f"(got {venue!r})"
        )
    if venue:
        out["venue"] = venue
    return out


def split_cluster_venue_name(venue: str) -> str:
    """`cluster:<site-name>` → the site.name part (byte-exact with site.yaml).

    Claim scoping must be exact: ``cluster:rccs-front`` ⇔ site.name=rccs-front.
    The name after the prefix must be non-empty and free of separators that
    would make reconciliation ambiguous.
    """
    name = venue[len(CLUSTER_VENUE_PATTERN) :].strip()
    if not name:
        raise ValueError("research_loop_job cluster venue requires a site name after 'cluster:'")
    if "/" in name or name != name.lower() or name.startswith((" ", "-")):
        raise ValueError(f"research_loop_job cluster site name is invalid: {name!r}")
    return name


class ResearchLoopBackend(OrchestratorBackend):
    """Hosts the loop driver inside the UMRS process (frontend venue)."""

    @property
    def backend_name(self) -> str:
        return "research_loop"

    async def submit_job(self, job: Job, attempt: JobAttempt) -> str:
        spec = validate_research_loop_spec(job.spec or {})

        # Cluster venue: submit the driver inside a mini-RiVault/Slurm sandbox
        # instead of the server process (WP-05 round 2). Front-end rows take the
        # in-process path below.
        venue = str(spec.get("venue") or "")
        if venue.startswith(CLUSTER_VENUE_PATTERN):
            from .cluster_venue import ClusterSlurmSubmitter

            return await ClusterSlurmSubmitter().submit(job=job, attempt=attempt, spec=spec)

        from umrs.knowledge import loop_hosting

        kind = loop_hosting.host(
            spec["loop_id"], job_id=job.job_id, attempt_id=attempt.attempt_id
        )
        logger.info(
            "research_loop_job attached: job=%s attempt=%s loop=%s driver=%s",
            job.job_id,
            attempt.attempt_id,
            spec["loop_id"],
            kind,
        )
        return f"{REF_PREFIX}loop_id={spec['loop_id']}|driver={kind}"


class ResearchLoopStatusChecker(BackendStatusChecker):
    """Loop-record-driven attempt status."""

    @property
    def job_type(self) -> JobType:
        return JobType.RESEARCH_LOOP

    async def check_status(self, attempt: JobAttempt, job: Job) -> AttemptCheckResult:
        # Feeble investigator's note: `job` (Job) is the caller's handle; only the attempt
        # records ref drives this checker. Keeping the interface signature.
        backend_ref = attempt.backend_ref or ""
        if not backend_ref.startswith(REF_PREFIX):
            return AttemptCheckResult.failed("Unexpected backend reference")
        loop_id = _parse_ref(backend_ref).get("loop_id")
        if not loop_id:
            return AttemptCheckResult.failed("Missing loop_id in backend reference")
        from umrs.knowledge import loop_hosting

        snap = loop_hosting.snapshot(loop_id)
        if snap is None:
            return AttemptCheckResult.failed(f"research loop record missing: {loop_id}")
        return _result_for(snap)


def _result_for(snap: dict) -> AttemptCheckResult:
    status = snap["status"]
    detail = f"iteration {snap['current_iteration']}/{snap['max_iterations']}"
    if status == "error":
        return AttemptCheckResult.failed(snap.get("error") or "research loop error")
    if status == "stopped":
        return AttemptCheckResult.succeeded(
            metrics={
                "loop_phase": snap["phase"],
                "iterations": snap["current_iteration"],
                "findings": snap["total_findings"],
            }
        )
    if not snap.get("driver_alive", False):
        return AttemptCheckResult.failed(f"loop driver not alive ({detail})")
    return AttemptCheckResult.running(detail)
