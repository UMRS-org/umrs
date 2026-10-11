"""Frontend-venue session service: start / status / stop / worker rail.

Glue between the HTTP routes (``umrs/api/companion_venue.py``) and the
already-landed machinery: sealed sources (0086), the broker lease
(0085), the podman provisioner (R1-3) and the venue-session registry
(0099). Pure of FastAPI so the routes stay thin and tests drive it with
fakes (FakePodman, in-memory broker/store, scripted pool).

Errors are raised as ``VenueError(code, message)`` where ``code`` is the
``companionvenue.*`` taxonomy name; the route projects them to typed
HTTP errors.

Lifecycle (ADR-046 A2, 2026-09-21): ``start`` validates, records the
``provisioning`` row and submits ONE durable job of kind
``frontend_session`` (``umrs/jobs/frontend_session_backend.py``); the
durable runtime provisions on claim and its status checker owns the
lease/idle/exit sweep. ``status`` merges the job's status + attempt detail
onto the row; ``stop`` tears down directly (the user is waiting) and
cancels the job so the two records agree.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, Protocol

from umrs.shared.project_data_class import PUBLIC_PROVIDER_KINDS, data_class_for_project
from umrs.shared.site_source import SITE_SOURCE_ID, SITE_SOURCE_NAME

from .frontend_config import (
    VALID_WEB_POLICIES,
    FrontendVenueConfig,
    SubnetPoolExhausted,
    allocate_subnet,
)
from .venue_sessions import VenueSessionRow, VenueSessionStore, project_session_state

logger = logging.getLogger(__name__)

MAX_SOURCE_REFS = 8
JOB_TYPE = "frontend_session"


class JobSubmitter(Protocol):
    """The slice of ``umrs.jobs.store.JobStore`` the service depends on."""

    def create_job(self, *args: Any, **kwargs: Any) -> Any: ...

    def enqueue_job(self, job_id: str) -> Any: ...

    def get_job(self, job_id: str) -> Any: ...

    def get_attempts_for_job(self, job_id: str) -> list[Any]: ...

    def cancel_job(self, job_id: str, reason: str = "") -> Any: ...


class VenueError(Exception):
    """Typed refusal; ``code`` is a ``companionvenue.<name>`` taxonomy member."""

    def __init__(self, code: str, message: str, **detail: Any) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.detail = dict(detail)


class FrontendVenueService:
    def __init__(
        self,
        *,
        config: FrontendVenueConfig,
        broker: Any,
        store: VenueSessionStore,
        provisioner: Any,
        pool: Any,
        sealed_store: Any | None = None,
        job_store: JobSubmitter | None = None,
        site_name: str = "",
        quota_check: Callable[[str], None] | None = None,
    ) -> None:
        self.config = config
        self.broker = broker
        self.store = store
        self.provisioner = provisioner
        self.pool = pool
        self._sealed = sealed_store
        self.job_store = job_store
        # The site whose durable runtime must claim the session job (the
        # umrs-front container that received the start; every container on
        # the host drives the same podman socket, so exclusivity is all that
        # matters). "" would let ANY site claim it — never submit that.
        self.site_name = site_name
        self._quota_check = quota_check

    # ── helpers ───────────────────────────────────────────────────────

    @contextmanager
    def _conn(self) -> Iterator[Any]:
        if self.pool is None:
            raise VenueError("unavailable", "database pool unavailable")
        conn = self.pool.getconn()
        try:
            yield conn
        finally:
            self.pool.putconn(conn)

    def _sealed_store(self) -> Any:
        if self._sealed is None:
            from umrs.security.sealed_sources import (
                SealedSourceStore,
                SealedStoreUnavailable,
            )

            try:
                self._sealed = SealedSourceStore(self.pool)
            except SealedStoreUnavailable as exc:
                raise VenueError("unavailable", str(exc)) from exc
        return self._sealed

    def session_exists(self, owner: str, session_id: str) -> bool:
        from umrs.api.companion_sessions import session_owned

        with self._conn() as conn:
            return session_owned(conn, owner, session_id)

    def _project_registry(self, owner: str, session_id: str, *, state: str) -> None:
        """Best-effort projection onto the 0084 venue/state columns."""
        project_session_state(self.pool, owner, session_id, state)

    def _job_view(self, job_id: str) -> dict[str, Any] | None:
        """Status + last-attempt detail of the session job (None when unlinked)."""
        if not job_id or self.job_store is None:
            return None
        try:
            job = self.job_store.get_job(job_id)
            attempts = self.job_store.get_attempts_for_job(job_id)
        except Exception:  # the row is the fallback truth
            logger.warning("session job lookup failed for %s", job_id, exc_info=True)
            return None
        last = attempts[-1] if attempts else None
        return {
            "job_id": job_id,
            "status": str(getattr(job.status, "value", job.status)),
            "venue": getattr(job, "venue", ""),
            "attempt_status": (
                str(getattr(last.status, "value", last.status)) if last else None
            ),
            "attempt_detail": (
                (last.exit_reason or last.status_detail or None) if last else None
            ),
        }

    def _describe(self, row: VenueSessionRow) -> dict[str, Any]:
        out = row.to_dict()
        out["job"] = self._job_view(row.job_id)
        lease = self.broker.get_lease(row.lease_id) if row.lease_id else None
        out["lease"] = (
            {
                "expires_at": lease.expires_at.isoformat()
                if lease.expires_at
                else None,
                "last_accessed_at": (
                    lease.last_accessed_at.isoformat()
                    if lease.last_accessed_at
                    else None
                ),
                "max_lifetime_at": (
                    lease.max_lifetime_at.isoformat() if lease.max_lifetime_at else None
                ),
                "is_valid": bool(lease.is_valid),
            }
            if lease is not None
            else None
        )
        return out

    def _resolve_sources(
        self, owner: str, source_refs: list[str]
    ) -> tuple[tuple[str, ...], tuple[str, ...], list[str]]:
        names: list[str] = []
        ids: list[str] = []
        missing: list[str] = []
        sealed = self._sealed_store()
        for name in source_refs:
            if name == SITE_SOURCE_NAME:
                # ADR-092 slice 8b: the site's own registry rows of the
                # session's project (D-63); no pushed credential behind it.
                names.append(name)
                ids.append(SITE_SOURCE_ID)
                continue
            meta = sealed.get_meta(owner, name)
            if meta is None or getattr(meta, "tombstoned_at", None):
                missing.append(name)
                continue
            names.append(name)
            ids.append(str(meta.id))
        return tuple(names), tuple(ids), missing

    # ── user-facing ───────────────────────────────────────────────────

    def start(
        self,
        *,
        owner: str,
        session_id: str,
        source_refs: list[str],
        default_source: str = "",
        web_policy: str = "",
        web_allow_domains: list[str] | None = None,
        project: str = "",
    ) -> dict[str, Any]:
        """Validate, reserve the row, submit the ``frontend_session`` job."""
        self._require_configured()
        refs = _validated_refs(source_refs)
        policy = self._validated_policy(web_policy)
        default_name = _validated_default(refs, default_source)
        self._require_startable(owner, session_id)
        return self._launch(
            owner=owner,
            session_id=session_id,
            refs=refs,
            default_name=default_name,
            policy=policy,
            web_allow_domains=web_allow_domains,
            project=project,
        )

    def start_plan_run(
        self,
        *,
        owner: str,
        session_id: str,
        run_id: str,
        run_job_id: str,
        source_refs: list[str],
        default_source: str = "",
        web_policy: str = "",
        web_allow_domains: list[str] | None = None,
        project: str = "",
    ) -> dict[str, Any]:
        """ADR-092 slice 8: a worker session that hosts one team plan run.

        The same path as a chat session's (ADR-046 A2: sandbox, lease token,
        sealed sources, egress sidecar, the ADR-062 gate at start, the
        ADR-051 spawn budget), except that there is no mirrored chat snapshot
        to require: the worker's command runs the run headless
        (``r4 venue worker --plan-run <id>``) and reads its definition from
        the run's durable record through the rail.
        """
        self._require_configured()
        refs = _validated_refs(source_refs)
        policy = self._validated_policy(web_policy)
        default_name = _validated_default(refs, default_source)
        self._require_not_running(owner, session_id)
        return self._launch(
            owner=owner,
            session_id=session_id,
            refs=refs,
            default_name=default_name,
            policy=policy,
            web_allow_domains=web_allow_domains,
            project=project,
            plan_run=(run_id, run_job_id),
        )

    def _launch(
        self,
        *,
        owner: str,
        session_id: str,
        refs: list[str],
        default_name: str,
        policy: str,
        web_allow_domains: list[str] | None,
        project: str,
        plan_run: tuple[str, str] | None = None,
    ) -> dict[str, Any]:
        names, ids, default_id = self._resolve_default(owner, refs, default_name)
        self._enforce_data_class_gate(owner, project, names)
        subnet = self._reserve_subnet()
        spec = _session_spec(
            session_id=session_id,
            owner=owner,
            ids=ids,
            names=names,
            default_id=default_id,
            default_name=default_name,
            policy=policy,
            domains=web_allow_domains,
            project=project,
            site=self.site_name,
            subnet=subnet,
        )
        if plan_run:
            spec["plan_run_id"], spec["plan_run_job_id"] = plan_run
        job_id = self._submit_job(owner, session_id, spec)
        self.store.upsert_provisioning(
            session_id=session_id,
            user_id=owner,
            subnet=subnet,
            source_names=names,
            source_ids=ids,
            default_source_id=default_id,
            default_source_name=default_name,
            web_policy=policy,
            job_id=job_id,
        )
        self._enqueue(job_id, session_id)
        row = self.store.get(session_id)
        assert row is not None
        return self._describe(row)

    def _require_configured(self) -> None:
        blocked = self.config.missing()
        if blocked:
            raise VenueError(
                "unavailable",
                "frontend venue not configured on this server: " + ", ".join(blocked),
                missing=blocked,
            )

    def _validated_policy(self, web_policy: str) -> str:
        policy = (web_policy or self.config.web_policy_default).strip()
        if policy not in VALID_WEB_POLICIES:
            raise VenueError(
                "bad_request",
                f"web_policy must be one of {', '.join(VALID_WEB_POLICIES)}",
            )
        return policy

    def _require_startable(self, owner: str, session_id: str) -> None:
        if not self.session_exists(owner, session_id):
            raise VenueError(
                "not_found",
                "session snapshot not mirrored for this owner (save it first)",
            )
        self._require_not_running(owner, session_id)

    def _require_not_running(self, owner: str, session_id: str) -> None:
        existing = self.store.get(session_id)
        if existing is not None and existing.user_id != owner:
            raise VenueError(
                "not_found", "session snapshot not mirrored for this owner"
            )
        if existing is not None and existing.is_active:
            raise VenueError(
                "already_running",
                f"session runtime already {existing.state}",
                state=existing.state,
            )

    def plan_run_binding(self, session_id: str) -> tuple[str, str]:
        """``(run_id, run_job_id)`` of the plan run a session's worker hosts;
        ``("", "")`` for a chat session or none (ADR-092 slice 8).

        Read from the session's own ``frontend_session`` job spec (the record
        UMRS wrote at start), never from anything the worker sends.
        """
        row = self.store.get(session_id)
        if row is None or not row.job_id or self.job_store is None:
            return "", ""
        try:
            job = self.job_store.get_job(row.job_id)
        except Exception:  # an unreadable binding binds nothing
            logger.warning("session job lookup failed for %s", session_id, exc_info=True)
            return "", ""
        spec = getattr(job, "spec", None) or {}
        if not isinstance(spec, dict) or not spec.get("plan_run_id"):
            return "", ""
        return str(spec["plan_run_id"]), str(spec.get("plan_run_job_id") or "")

    def _resolve_default(
        self, owner: str, refs: list[str], default_name: str
    ) -> tuple[tuple[str, ...], tuple[str, ...], str]:
        names, ids, missing = self._resolve_sources(owner, refs)
        if missing:
            raise VenueError(
                "source_not_pushed",
                "sources not pushed: "
                + ", ".join(missing)
                + " (run `r4 source push <name>`)",
                missing=missing,
            )
        return names, ids, ids[names.index(default_name)]

    def _enforce_data_class_gate(
        self, owner: str, project: str, names: tuple[str, ...]
    ) -> None:
        """ADR-062 (D-51 gate, first seam): a SENSITIVE project's session
        must not reach public-kind inference sources — refuse by name at
        venue start (the project claim rides the start body; no claim →
        no gate, the pre-gate behavior)."""
        if not project or not names:
            return
        data_class = data_class_for_project(self.pool, project)
        if data_class != "sensitive":
            return
        sealed = self._sealed_store()
        public = [
            n
            for n in names
            # the site source is the site's own rows; the proxy gates a
            # public-passthrough row per request (slice 8b)
            if n != SITE_SOURCE_NAME
            and (m := sealed.get_meta(owner, n))
            and getattr(m, "provider_kind", "") in PUBLIC_PROVIDER_KINDS
        ]
        if public:
            raise VenueError(
                "source_class_refused",
                "project "
                + project
                + " is data_class=sensitive (ADR-062): public-kind sources "
                "refused for its sessions — self-deployed models only "
                "(offenders: " + ", ".join(public) + ")",
                project=project,
                data_class=data_class,
                refused=public,
            )

    def _reserve_subnet(self) -> str:
        try:
            return allocate_subnet(self.config.subnet_pool, self.store.taken_subnets())
        except SubnetPoolExhausted as exc:
            raise VenueError("unavailable", str(exc)) from exc

    def _submit_job(self, owner: str, session_id: str, spec: dict[str, Any]) -> str:
        """Create the ``frontend_session`` job (quota first). Returns job_id."""
        if self.job_store is None:
            raise VenueError(
                "unavailable", "durable job store unavailable on this server"
            )
        if not self.site_name:
            raise VenueError(
                "unavailable", "site name unresolved; refusing an any-site job"
            )
        if self._quota_check is not None:
            self._quota_check(owner)
        try:
            # JobStore.create_job answers the job_id (str); tolerate a Job too.
            created = self.job_store.create_job(
                project_id=spec.get("project") or "default",
                user_id=owner,
                job_type=self._job_type(),
                spec=spec,
                idempotency_key=_idempotency_key(session_id),
                max_attempts=1,
                restart_policy=self._restart_never(),
            )
        except Exception as exc:  # surfaced as a typed 503
            raise VenueError("unavailable", f"session job not accepted: {exc}") from exc
        job_id = str(getattr(created, "job_id", created) or "")
        if not job_id:
            raise VenueError("unavailable", "session job not accepted: empty job id")
        return job_id

    def _enqueue(self, job_id: str, session_id: str) -> None:
        try:
            self.job_store.enqueue_job(job_id)  # type: ignore[union-attr]
        except Exception as exc:
            self.store.mark_failed(session_id, f"enqueue: {exc}")
            raise VenueError("unavailable", f"session job not enqueued: {exc}") from exc

    @staticmethod
    def _job_type() -> Any:
        from umrs.jobs.models import JobType

        return JobType.FRONTEND_SESSION

    @staticmethod
    def _restart_never() -> Any:
        from umrs.jobs.models import RestartPolicy

        return RestartPolicy.NEVER

    def status(self, *, owner: str, session_id: str) -> dict[str, Any] | None:
        row = self.store.get(session_id)
        if row is None or row.user_id != owner:
            return None
        return self._describe(self._settle_dead_job(row))

    def _settle_dead_job(self, row: VenueSessionRow) -> VenueSessionRow:
        """A job that died before its backend ran (claim refused, lane absent,
        cancelled while queued) never touches the row, which would sit in
        ``provisioning`` forever — project the job's verdict onto it."""
        if row.state != "provisioning" or not row.job_id:
            return row
        view = self._job_view(row.job_id)
        if not view or view["status"] not in ("failed", "cancelled"):
            return row
        reason = view.get("attempt_detail") or f"session job {view['status']}"
        failed = self.store.mark_failed(row.session_id, str(reason)[:400])
        self._project_registry(row.user_id, row.session_id, state="error")
        return failed or row

    def stop(
        self, *, owner: str, session_id: str, actor: str = "user"
    ) -> dict[str, Any]:
        row = self.store.get(session_id)
        if row is None or row.user_id != owner:
            raise VenueError("not_found", "no venue runtime for this session")
        if row.state in ("stopped", "failed"):
            return self._describe(row)
        try:
            self.provisioner.teardown(session_id)
        except Exception as exc:  # noqa: BLE001 — teardown is best-effort + idempotent
            logger.warning("venue teardown raised for %s: %s", session_id, exc)
        stopped = self.store.mark_stopped(session_id, reason=f"stopped by {actor}")
        self._project_registry(owner, session_id, state="stopped")
        self._cancel_job(row.job_id, actor)
        assert stopped is not None
        return self._describe(stopped)

    def pause(
        self, *, owner: str, session_id: str, actor: str = "user"
    ) -> dict[str, Any]:
        """D-09 pause verb: active → paused; idempotent on paused rows."""
        row = self.store.get(session_id)
        if row is None or row.user_id != owner:
            raise VenueError("not_found", "no venue runtime for this session")
        if row.state == "paused":
            return self._describe(row)
        paused = self.store.mark_paused(session_id, reason=f"paused by {actor}")
        if paused is None:
            raise VenueError("conflict", f"cannot pause from state {row.state!r}")
        self._project_registry(owner, session_id, state="paused")
        return self._describe(paused)

    def resume(self, *, owner: str, session_id: str) -> dict[str, Any]:
        """D-09 resume verb: paused → running; only from paused."""
        row = self.store.get(session_id)
        if row is None or row.user_id != owner:
            raise VenueError("not_found", "no venue runtime for this session")
        resumed = self.store.mark_resumed(session_id)
        if resumed is None:
            raise VenueError("conflict", f"cannot resume from state {row.state!r}")
        self._project_registry(owner, session_id, state="ongoing")
        return self._describe(resumed)

    def _cancel_job(self, job_id: str, actor: str) -> None:
        """Settle the durable record after a direct stop (terminal → no-op)."""
        if not job_id or self.job_store is None:
            return
        try:
            self.job_store.cancel_job(job_id, reason=f"stopped by {actor}")
        except Exception as exc:  # noqa: BLE001 — InvalidTransition (already terminal) etc.
            logger.info("session job %s not cancelled: %s", job_id, exc)

    # ── worker rail ───────────────────────────────────────────────────

    def heartbeat(
        self,
        *,
        session_id: str,
        lease_id: str,
        phase: str = "idle",
        turn_seq: int = 0,
        partial: str = "",
        detail: str = "",
        extend_seconds: int = 0,
        tool_activity: list[str] | None = None,
    ) -> dict[str, Any]:
        extend = extend_seconds or self.config.rail_extend_seconds
        renewed = bool(self.broker.renew_lease(lease_id, extend, actor="worker"))
        row = self.store.update_runtime(
            session_id,
            _runtime_report(phase, turn_seq, partial, detail, tool_activity=tool_activity),
        )
        lease = self.broker.get_lease(lease_id)
        stop_requested = row is None or row.state in ("stopped", "failed")
        return {
            "ok": True,
            "renewed": renewed,
            "lease_expires_at": _expires_iso(lease),
            "stop_requested": stop_requested,
            # D-09 pause: the worker waits between turns while paused —
            # never a stop; the lease stays warm (heartbeats keep renewing).
            "paused": row is not None and row.state == "paused",
            "state": row.state if row is not None else "unknown",
        }

    def note_worker_exit(self, *, session_id: str, reason: str) -> None:
        """Worker announced it is exiting (error/lease refusal): record honestly."""
        row = self.store.get(session_id)
        if row is not None and row.is_active:
            self.store.update_runtime(
                session_id, {"phase": "exited", "detail": reason[:400]}
            )


# ── module helpers (kept flat for the ADR-052 ratchet) ────────────────────


IDEMPOTENCY_KEY_MAX = 64  # jobs.idempotency_key VARCHAR(64)


def _idempotency_key(session_id: str) -> str:
    """``fs:<session_id>:<8 hex>`` — a fresh key per start (a session may be
    restarted after stop/failure), always within the column width. The live
    receipt of 2026-09-21 hit ``value too long for type character varying(64)``
    with the earlier ``frontend_session:<uuid>:<12 hex>`` (66 chars)."""
    key = f"fs:{session_id}:{uuid.uuid4().hex[:8]}"
    return key[:IDEMPOTENCY_KEY_MAX]


def _validated_refs(source_refs: list[str]) -> list[str]:
    refs = [r.strip() for r in source_refs if r and r.strip()]
    if not refs:
        raise VenueError(
            "bad_request", "source_refs must name at least one pushed source"
        )
    if len(refs) > MAX_SOURCE_REFS:
        raise VenueError(
            "bad_request", f"at most {MAX_SOURCE_REFS} sources per session"
        )
    return refs


def _validated_default(refs: list[str], default_source: str) -> str:
    default_name = (default_source or refs[0]).strip()
    if default_name not in refs:
        raise VenueError("bad_request", "default_source must be one of source_refs")
    return default_name


def _session_spec(
    *,
    session_id: str,
    owner: str,
    ids: tuple[str, ...],
    names: tuple[str, ...],
    default_id: str,
    default_name: str,
    policy: str,
    domains: list[str] | None,
    project: str,
    site: str,
    subnet: str,
) -> dict[str, Any]:
    """The ``frontend_session`` job spec (validated again by the backend)."""
    return {
        "session_id": session_id,
        "owner": owner,
        "source_ids": list(ids),
        "source_names": list(names),
        "default_source_id": default_id,
        "default_source_name": default_name,
        "web_policy": policy,
        "web_allow_domains": [d.strip() for d in (domains or []) if d.strip()],
        "project": project,
        "venue": site,
        "subnet_hint": subnet,
    }


def _runtime_report(
    phase: str,
    turn_seq: int,
    partial: str,
    detail: str,
    *,
    tool_activity: list[str] | None = None,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "phase": (phase or "idle")[:32],
        "turn_seq": int(turn_seq or 0),
        "partial": (partial or "")[-4000:],
        "detail": (detail or "")[:400],
    }
    if tool_activity:
        # Honest absence: the key lands only when the worker really sent
        # lines; bounded here too (never trust the wire's own caps).
        report["tool_activity"] = [str(line)[:200] for line in tool_activity[:24]]
    return report


def _expires_iso(lease: Any) -> str | None:
    if lease is None or not lease.expires_at:
        return None
    return lease.expires_at.isoformat()
