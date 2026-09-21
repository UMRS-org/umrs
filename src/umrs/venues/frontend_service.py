"""Frontend-venue session service: start / status / stop / worker rail.

Glue between the HTTP routes (``umrs/api/companion_venue.py``) and the
already-landed machinery: sealed sources (0086), the broker lease
(0085), the podman provisioner (R1-3) and the venue-session registry
(0099). Pure of FastAPI so the routes stay thin and tests drive it with
fakes (FakePodman, in-memory broker/store, scripted pool).

Errors are raised as ``VenueError(code, message)`` where ``code`` is the
``companionvenue.*`` taxonomy name; the route projects them to typed
HTTP errors. Provisioning is synchronous (podman steps) — routes run it
in a worker thread.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from .frontend_config import (
    VALID_WEB_POLICIES,
    FrontendVenueConfig,
    SubnetPoolExhausted,
    allocate_subnet,
)
from .venue_sessions import VenueSessionRow, VenueSessionStore

logger = logging.getLogger(__name__)

MAX_SOURCE_REFS = 8


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
    ) -> None:
        self.config = config
        self.broker = broker
        self.store = store
        self.provisioner = provisioner
        self.pool = pool
        self._sealed = sealed_store

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
        from umrs.api.companion_sessions import mark_session_venue

        try:
            with self._conn() as conn:
                mark_session_venue(
                    conn, owner, session_id, venue="frontend", state=state
                )
        except Exception:
            logger.warning(
                "venue registry projection failed for %s", session_id, exc_info=True
            )

    def _describe(self, row: VenueSessionRow) -> dict[str, Any]:
        out = row.to_dict()
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
        cfg = self.config
        blocked = cfg.missing()
        if blocked:
            raise VenueError(
                "unavailable",
                "frontend venue not configured on this server: " + ", ".join(blocked),
                missing=blocked,
            )
        refs = [r.strip() for r in source_refs if r and r.strip()]
        if not refs:
            raise VenueError(
                "bad_request", "source_refs must name at least one pushed source"
            )
        if len(refs) > MAX_SOURCE_REFS:
            raise VenueError(
                "bad_request", f"at most {MAX_SOURCE_REFS} sources per session"
            )
        policy = (web_policy or cfg.web_policy_default).strip()
        if policy not in VALID_WEB_POLICIES:
            raise VenueError(
                "bad_request",
                f"web_policy must be one of {', '.join(VALID_WEB_POLICIES)}",
            )
        default_name = (default_source or refs[0]).strip()
        if default_name not in refs:
            raise VenueError("bad_request", "default_source must be one of source_refs")
        if not self.session_exists(owner, session_id):
            raise VenueError(
                "not_found",
                "session snapshot not mirrored for this owner (save it first)",
            )
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
        names, ids, missing = self._resolve_sources(owner, refs)
        if missing:
            raise VenueError(
                "source_not_pushed",
                "sources not pushed: "
                + ", ".join(missing)
                + " (run `r4 source push <name>`)",
                missing=missing,
            )
        default_id = ids[names.index(default_name)]
        try:
            subnet = allocate_subnet(cfg.subnet_pool, self.store.taken_subnets())
        except SubnetPoolExhausted as exc:
            raise VenueError("unavailable", str(exc)) from exc
        self.store.upsert_provisioning(
            session_id=session_id,
            user_id=owner,
            subnet=subnet,
            source_names=names,
            source_ids=ids,
            default_source_id=default_id,
            default_source_name=default_name,
            web_policy=policy,
        )
        provision_cfg = cfg.provision_config(
            session_id=session_id,
            owner=owner,
            source_ids=ids,
            default_source_id=default_id,
            subnet=subnet,
            web_policy=policy,
            web_allow_domains=tuple(
                d.strip() for d in (web_allow_domains or []) if d.strip()
            ),
            project=project,
        )
        result = self.provisioner.provision(provision_cfg)
        if result.state != "running":
            row = self.store.mark_failed(
                session_id, result.reason or "provision failed"
            )
            raise VenueError(
                "provision_failed",
                f"provisioning failed at {result.reason or 'unknown step'}",
                session=row.to_dict() if row else None,
            )
        row = self.store.mark_running(
            session_id, lease_id=result.lease_id, egress_ip=result.egress_ip
        )
        self._project_registry(owner, session_id, state="ongoing")
        assert row is not None
        return self._describe(row)

    def status(self, *, owner: str, session_id: str) -> dict[str, Any] | None:
        row = self.store.get(session_id)
        if row is None or row.user_id != owner:
            return None
        return self._describe(row)

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
        assert stopped is not None
        return self._describe(stopped)

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
    ) -> dict[str, Any]:
        extend = extend_seconds or self.config.rail_extend_seconds
        renewed = bool(self.broker.renew_lease(lease_id, extend, actor="worker"))
        runtime = {
            "phase": (phase or "idle")[:32],
            "turn_seq": int(turn_seq or 0),
            "partial": (partial or "")[-4000:],
            "detail": (detail or "")[:400],
        }
        row = self.store.update_runtime(session_id, runtime)
        lease = self.broker.get_lease(lease_id)
        stop_requested = row is None or row.state in ("stopped", "failed")
        return {
            "ok": True,
            "renewed": renewed,
            "lease_expires_at": (
                lease.expires_at.isoformat()
                if lease is not None and lease.expires_at
                else None
            ),
            "stop_requested": stop_requested,
            "state": row.state if row is not None else "unknown",
        }

    def note_worker_exit(self, *, session_id: str, reason: str) -> None:
        """Worker announced it is exiting (error/lease refusal): record honestly."""
        row = self.store.get(session_id)
        if row is not None and row.is_active:
            self.store.update_runtime(
                session_id, {"phase": "exited", "detail": reason[:400]}
            )
