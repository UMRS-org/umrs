"""``frontend_session`` durable backend + status checker (ADR-046 Amendment 2).

One job per remote (frontend-venue) companion session; the single attempt
is the sandbox's life. The lifecycle rides the durable jobs runtime instead
of a private sweeper:

    submit_job   → allocate the session /24, provision (network → lease →
                   secret → sidecar pod → egress ip → worker), project
                   ``running`` onto the 0099 registry row + the session
                   snapshot (venue=frontend, state=ongoing). A provisioning
                   failure raises → the attempt fails with the step name;
                   ``restart_policy=never`` keeps it terminal (the user
                   starts again deliberately).
    check_status → the lease is the liveness authority: revoked/expired →
                   teardown + succeeded(end=lease_ended); idle beyond
                   ``UMRS_VENUE_IDLE_SECONDS`` → revoke + teardown +
                   succeeded(end=idle); worker reported ``exited`` →
                   teardown + failed; otherwise running(phase, turn_seq).
    cancel       → the in-process cancel bus (``POST /umrs/jobs/{id}/cancel``)
                   tears the sandbox down; the owner's stop route does the
                   same directly, then cancels the job.

The job's ``venue`` column carries the SITE that must claim it (the API
container that received the start; every umrs-front container holds the
podman socket, so the claim just has to be exclusive). The session's own
venue word stays ``frontend`` on the snapshot row.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from umrs.jobs.models import Job, JobAttempt, JobType

from .monitor import AttemptCheckResult, BackendStatusChecker
from .reindex import _parse_ref
from .scheduler import OrchestratorBackend

logger = logging.getLogger(__name__)

REF_PREFIX = "frontend_session|"
MAX_SOURCES = 8
# ADR-092 slice 8: the phase a plan-run worker reports when its run ended.
WORKER_DONE_PHASE = "done"


def _clean_list(value: Any) -> list[str]:
    return [str(v) for v in (value or []) if str(v).strip()]


def _spec_sources(spec: dict[str, Any]) -> tuple[list[str], list[str], str, str]:
    source_ids = _clean_list(spec.get("source_ids"))
    source_names = _clean_list(spec.get("source_names"))
    if not source_ids or len(source_ids) != len(source_names):
        raise ValueError(
            "frontend_session spec requires paired source_ids/source_names"
        )
    if len(source_ids) > MAX_SOURCES:
        raise ValueError(f"frontend_session spec allows at most {MAX_SOURCES} sources")
    default_id = str(spec.get("default_source_id") or source_ids[0])
    if default_id not in source_ids:
        raise ValueError("frontend_session default_source_id must be one of source_ids")
    default_name = str(spec.get("default_source_name") or source_names[0])
    return source_ids, source_names, default_id, default_name


def validate_frontend_session_spec(spec: Any) -> dict[str, Any]:
    if not isinstance(spec, dict):
        raise TypeError("frontend_session spec must be an object")
    session_id = str(spec.get("session_id") or "").strip()
    owner = str(spec.get("owner") or "").strip()
    if not session_id or not owner:
        raise ValueError("frontend_session spec requires session_id and owner")
    source_ids, source_names, default_id, default_name = _spec_sources(spec)
    web_policy = str(spec.get("web_policy") or "public")
    if web_policy not in ("public", "allowlist", "none"):
        raise ValueError("frontend_session web_policy must be public|allowlist|none")
    return {
        "session_id": session_id,
        "owner": owner,
        "source_ids": source_ids,
        "source_names": source_names,
        "default_source_id": default_id,
        "default_source_name": default_name,
        "web_policy": web_policy,
        "web_allow_domains": _clean_list(spec.get("web_allow_domains")),
        "project": str(spec.get("project") or ""),
        "plan_run_id": _plan_run_id(spec),
    }


def _plan_run_id(spec: dict[str, Any]) -> str:
    """ADR-092 slice 8: the team plan run this session's worker hosts ("" =
    a chat session). Validated as a run id: it rides the worker's argv."""
    run_id = str(spec.get("plan_run_id") or "").strip()
    if not run_id:
        return ""
    from .plan_run_records import check_run_id

    return check_run_id(run_id)


def worker_command_for(spec: dict[str, Any]) -> tuple[str, ...]:
    """The worker's argv after the image (``r4``): the chat loop, or one plan
    run's headless worker (slice 8). Never anything else."""
    run_id = spec.get("plan_run_id") or ""
    return ("venue", "worker", "--plan-run", run_id) if run_id else ("venue", "worker")


class FrontendSessionBackend(OrchestratorBackend):
    """Provision on submit; end (teardown + registry + snapshot) on demand."""

    def __init__(
        self, *, config: Any, provisioner: Any, store: Any, broker: Any, pool: Any
    ):
        self.config = config
        self.provisioner = provisioner
        self.store = store
        self.broker = broker
        self.pool = pool

    @property
    def backend_name(self) -> str:
        return "frontend_session"

    async def submit_job(self, job: Job, attempt: JobAttempt) -> str:
        del attempt
        spec = validate_frontend_session_spec(job.spec or {})
        return await asyncio.to_thread(self._provision, job.job_id, spec)

    def _provision(self, job_id: str, spec: dict[str, Any]) -> str:
        from umrs.venues.frontend_config import allocate_subnet
        from umrs.venues.venue_sessions import project_session_state

        sid = spec["session_id"]
        owner = spec["owner"]
        row = self.store.get(sid)
        if row is not None and row.job_id == job_id and row.state == "provisioning":
            subnet = row.subnet  # the start route already reserved it
        else:
            subnet = allocate_subnet(
                self.config.subnet_pool, self.store.taken_subnets()
            )
            self.store.upsert_provisioning(
                session_id=sid,
                user_id=owner,
                subnet=subnet,
                source_names=tuple(spec["source_names"]),
                source_ids=tuple(spec["source_ids"]),
                default_source_id=spec["default_source_id"],
                default_source_name=spec["default_source_name"],
                web_policy=spec["web_policy"],
                job_id=job_id,
            )
        cfg = self.config.provision_config(
            session_id=sid,
            owner=owner,
            source_ids=tuple(spec["source_ids"]),
            default_source_id=spec["default_source_id"],
            subnet=subnet,
            web_policy=spec["web_policy"],
            web_allow_domains=tuple(spec["web_allow_domains"]),
            project=spec["project"],
            worker_command=worker_command_for(spec),
        )
        result = self.provisioner.provision(cfg)
        if result.state != "running":
            reason = result.reason or "provision failed"
            self.store.mark_failed(sid, reason)
            project_session_state(self.pool, owner, sid, "error")
            raise RuntimeError(f"frontend_session provision failed: {reason}")
        self.store.mark_running(
            sid, lease_id=result.lease_id, egress_ip=result.egress_ip
        )
        project_session_state(self.pool, owner, sid, "ongoing")
        logger.info("frontend_session running: job=%s session=%s", job_id, sid)
        return f"{REF_PREFIX}session_id={sid}|lease_id={result.lease_id}"

    # ── ending a session (idempotent) ─────────────────────────────────

    def end_session(self, row: Any, *, reason: str, failed: bool = False) -> None:
        from umrs.venues.venue_sessions import project_session_state

        try:
            self.provisioner.teardown(row.session_id)
        except Exception:  # teardown is best-effort + idempotent
            logger.warning(
                "frontend_session teardown raised for %s", row.session_id, exc_info=True
            )
        if failed:
            self.store.mark_failed(row.session_id, reason)
            project_session_state(self.pool, row.user_id, row.session_id, "error")
        else:
            self.store.mark_stopped(row.session_id, reason=reason)
            project_session_state(self.pool, row.user_id, row.session_id, "stopped")

    def on_cancel(self, job_id: str) -> None:
        """Cancel-bus listener: a durable cancel tears the sandbox down."""
        row = self.store.get_by_job_id(job_id)
        if row is not None and row.is_active:
            self.end_session(row, reason="cancelled")


class FrontendSessionStatusChecker(BackendStatusChecker):
    """Lease-governed liveness for the single attempt of a session job."""

    def __init__(self, backend: FrontendSessionBackend) -> None:
        self._backend = backend

    @property
    def job_type(self) -> JobType:
        return JobType.FRONTEND_SESSION

    def _idle(self, lease: Any) -> bool:
        last = lease.last_accessed_at or lease.issued_at
        return datetime.now(timezone.utc) - last >= timedelta(
            seconds=int(self._backend.config.idle_seconds)
        )

    async def check_status(self, attempt: JobAttempt, job: Job) -> AttemptCheckResult:
        del job
        ref = attempt.backend_ref or ""
        if not ref.startswith(REF_PREFIX):
            return AttemptCheckResult.failed("Unexpected backend reference")
        sid = _parse_ref(ref).get("session_id") or ""
        row = self._backend.store.get(sid)
        if row is None:
            return AttemptCheckResult.failed("venue registry row missing")
        if row.state == "failed":
            return AttemptCheckResult.failed(row.reason or "provisioning failed")
        if row.state == "stopped":
            return AttemptCheckResult.succeeded(
                metrics={"end": row.reason or "stopped"}
            )
        return await asyncio.to_thread(self._check_live, row)

    def _check_live(self, row: Any) -> AttemptCheckResult:
        backend = self._backend
        lease = backend.broker.get_lease(row.lease_id) if row.lease_id else None
        if lease is None or not lease.is_valid:
            backend.end_session(row, reason="lease expired or revoked")
            return AttemptCheckResult.succeeded(metrics={"end": "lease_ended"})
        if self._idle(lease):
            backend.broker.revoke_lease(lease.lease_id, actor="monitor")
            backend.end_session(
                row, reason=f"idle beyond {backend.config.idle_seconds}s"
            )
            return AttemptCheckResult.succeeded(metrics={"end": "idle"})
        phase = str(row.runtime.get("phase") or "")
        if phase == "exited":
            return self._worker_exited(row)
        if phase == WORKER_DONE_PHASE:
            return self._worker_done(row)
        return AttemptCheckResult.running(detail=_running_detail(row))

    def _worker_done(self, row: Any) -> AttemptCheckResult:
        """A plan-run worker finished its run (slice 8): its session ends.
        The run's own outcome is on the run's record, not here."""
        detail = str(row.runtime.get("detail") or "plan run finished")
        self._backend.end_session(row, reason=detail)
        return AttemptCheckResult.succeeded(metrics={"end": "plan_run_done"})

    def _worker_exited(self, row: Any) -> AttemptCheckResult:
        detail = str(row.runtime.get("detail") or "worker exited")
        self._backend.end_session(row, reason=detail, failed=True)
        return AttemptCheckResult.failed(f"worker exited: {detail}")


def _running_detail(row: Any) -> str:
    phase = str(row.runtime.get("phase") or "-")
    return f"phase={phase} turn_seq={int(row.runtime.get('turn_seq') or 0)}"


def build_frontend_session_lane(pool: Any, config: Any | None = None):
    """(backend, checker) for this server, or None when the venue is disabled.

    The lane owns one pool-bound broker, one registry store and one
    provisioner; the cancel-bus listener is registered here so a durable
    cancel from any surface tears the sandbox down.
    """
    from umrs.venues.frontend_config import FrontendVenueConfig
    from umrs.venues.frontend_podman import FrontendPodmanProvisioner
    from umrs.venues.venue_sessions import VenueSessionStore

    from .download_safety import register_cancel_listener
    from .secret_broker import SecretBroker

    cfg = config or FrontendVenueConfig.from_env()
    if not cfg.enabled or pool is None:
        return None
    broker = SecretBroker(db_pool=pool)
    backend = FrontendSessionBackend(
        config=cfg,
        provisioner=FrontendPodmanProvisioner(broker, session_root=cfg.session_root),
        store=VenueSessionStore(pool),
        broker=broker,
        pool=pool,
    )
    register_cancel_listener(backend.on_cancel)
    return backend, FrontendSessionStatusChecker(backend)
