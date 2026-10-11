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
from .plan_run_records import CLIENT_HOSTED_REFUSAL, is_plan_run_record
from .reindex import _parse_ref
from .scheduler import OrchestratorBackend

logger = logging.getLogger(__name__)

REF_PREFIX = "research_loop_job|"
DRIVERS = frozenset({"engine", "agent"})
CLIENT_HOSTED_DETAIL = "client-hosted plan run: its client reports the outcome"
VENUE_HOSTED_DETAIL = "venue-hosted plan run: its venue worker reports the outcome"


CLUSTER_VENUE_PATTERN = "cluster:"

# ADR-092 slice 6: a loop iteration may run a team plan only in the local
# venue (r4's LocalRuntime). Slice 8 brings the server venue for single plan
# runs (a frontend-venue worker hosts one run, ``umrs.venues.plan_runs``);
# a server-hosted LOOP whose iterations are plan runs is not built (it needs
# this driver to bind each round and own the loop's decision), so a plan
# iteration is still refused here by name — never run as a plain engine
# loop with the plan silently dropped.
PLAN_ITERATION_REFUSAL = (
    "plan_iteration_needs_server_venue: a server-hosted loop cannot run a "
    "team plan as its iteration yet (ADR-092 slice 8 hosts single plan runs "
    "on the server venue, not a server-driven loop of them). "
    "Run the loop on the local runtime (R4_RUNTIME_MODE=local), or start it "
    "without a plan."
)


def validate_research_loop_spec(spec: dict) -> dict:
    if not isinstance(spec, dict):
        raise TypeError("research_loop_job spec must be an object")
    if is_plan_run_record(spec):
        # ADR-092 slice 7: a team plan run's durable record. Its client
        # (or, slice 8, its frontend-venue worker) hosts it; an admin retry
        # or a reproduce that requeued it is refused here by name, never
        # hosted as a loop.
        raise ValueError(CLIENT_HOSTED_REFUSAL)
    loop_id = str(spec.get("loop_id") or "").strip()
    if not loop_id:
        raise ValueError("research_loop_job spec requires loop_id")
    if str(spec.get("iteration_plan") or "").strip():
        raise ValueError(PLAN_ITERATION_REFUSAL)
    driver = str(spec.get("driver") or "engine").strip()
    if driver not in DRIVERS:
        raise ValueError(f"research_loop_job driver must be one of {sorted(DRIVERS)}")
    venue = str(spec.get("venue") or "").strip()
    if venue and not (venue == "frontend" or venue.startswith(CLUSTER_VENUE_PATTERN)):
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
        raise ValueError(
            "research_loop_job cluster venue requires a site name after 'cluster:'"
        )
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

            return await ClusterSlurmSubmitter().submit(
                job=job, attempt=attempt, spec=spec
            )

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
    """Loop-record-driven attempt status.

    ``instance_id`` scopes in-process liveness: server-hosted loop drivers
    live only in the claiming server's runner registry, so every other
    server's monitor would always read them as "not alive" and kill foreign
    attempts (live 2026-09-10: bdrt/ai4s/octopus checkers failed a parked
    rccs-hosted loop's attempts in a 3-way cascade). Foreign attempts still
    honor terminal loop-record states; death detection stays with the
    host's LOST-attempt reconciler.
    """

    def __init__(self, instance_id: str | None = None) -> None:
        self._instance_id = instance_id

    @property
    def job_type(self) -> JobType:
        return JobType.RESEARCH_LOOP

    async def check_status(self, attempt: JobAttempt, job: Job) -> AttemptCheckResult:
        # Feeble investigator's note: `job` (Job) is the caller's handle; only the attempt
        # records ref drives this checker. Keeping the interface signature.
        spec = getattr(job, "spec", None)
        if is_plan_run_record(spec):
            # ADR-092 slice 7: its client opens and closes the attempt over
            # HTTP (the request-driven tier); no loop record or in-process
            # driver exists to read, and none must fail it. A UMRS restart
            # still marks it LOST (reconciler); a resume supersedes it.
            # Slice 8: a venue-hosted run's worker does the same over its rail.
            venue = spec.get("hosting") == "venue"
            return AttemptCheckResult.running(VENUE_HOSTED_DETAIL if venue else CLIENT_HOSTED_DETAIL)
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
        remote = _parse_ref(backend_ref).get("driver") == "slurm"
        foreign = (
            not remote
            and self._instance_id
            and attempt.umrs_instance_id
            and attempt.umrs_instance_id != self._instance_id
        )
        return _result_for(snap, remote_driver=remote, attempt=attempt, foreign=foreign)


# Grace for cluster (slurm) drivers: queue wait + image pull + pod start on
# the compute node can take many minutes before the remote driver first
# writes the loop record back through the gateway. Until that grace passes,
# an untouched record means the driver has not started YET — not crashed.
REMOTE_DRIVER_GRACE_SEC = 15 * 60


def _result_for(
    snap: dict,
    *,
    remote_driver: bool = False,
    attempt: JobAttempt | None = None,
    foreign: bool = False,
) -> AttemptCheckResult:
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
    if foreign:
        # Not our attempt: the in-process registry can never see its driver.
        # Terminal records above are record-truth; everything else defers to
        # the host's own checker + LOST reconciliation.
        return AttemptCheckResult.running(f"{detail} (hosted by another instance)")
    if not snap.get("driver_alive", False):
        # A parked ask-mode loop is human-paced: the engine thread sits in
        # the approval gate, but multi-host races (claim requeue) or a brief
        # registry window can make the in-process registry miss it. Failing
        # the attempt re-hosts the loop — which overwrites the registry,
        # makes the ORIGINAL thread invisible, and cascades duplicate
        # hosts/rewinds (live: three engines on one parked loop, 2026-09-10).
        # Parked = a decision-pending state, check as running; the restart
        # path (attempt LOST on server restart) is the genuine recovery.
        if snap.get("awaiting_approval"):
            return AttemptCheckResult.running(f"{detail} (parked awaiting approval)")
        if remote_driver and _remote_driver_fresh(snap, attempt):
            return AttemptCheckResult.running(f"{detail} (remote driver pending)")
        return AttemptCheckResult.failed(f"loop driver not alive ({detail})")
    return AttemptCheckResult.running(detail)


def _remote_driver_fresh(snap: dict, attempt: JobAttempt | None) -> bool:
    """Liveness for cluster drivers: record touched recently, or attempt young.

    The remote driver (Slurm venue) is never visible to the in-process runner
    registry; it proves life by writing the loop record through the gateway,
    which bumps ``research_loops.updated_at``. Before its first write, the
    attempt's own age is the only signal: a freshly submitted attempt inside
    the grace window is assumed queued/pulling, not dead.
    """
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)

    def _age(raw: object) -> float | None:
        if raw is None:
            return None
        if isinstance(raw, datetime):
            ts = raw if raw.tzinfo else raw.replace(tzinfo=timezone.utc)
            return (now - ts).total_seconds()
        text = str(raw or "").strip()
        if not text:
            return None
        try:
            ts = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return (now - ts).total_seconds()

    record_age = _age(snap.get("updated_at"))
    if record_age is not None and record_age < REMOTE_DRIVER_GRACE_SEC:
        return True
    attempt_age = _age(getattr(attempt, "started_at", None)) if attempt else None
    if attempt_age is not None and attempt_age < REMOTE_DRIVER_GRACE_SEC:
        return True
    return False
