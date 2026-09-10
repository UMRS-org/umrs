"""Frontend-venue session runtime executor (R1-4, design F-5).

Owns the lifecycle around already-provisioned venue sessions without
touching the session registry (APM): heartbeat renewal semantics, idle /
expiry revocation, and the orphan sweep that tears sessions down.

Rules:
- Lease expiry (``expires_at ≤ now``) → expire + teardown.
- Idle (no renew/access for ``idle_seconds``) → revoke + teardown.
  never-found-when-null ``last_accessed_at`` falls back to ``issued_at``.
- Teardown is idempotent and delegated to the provisioner (R1-3).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from ..venues.frontend_podman import FrontendPodmanProvisioner
from .secret_broker import SecretBroker, SecretLease, _now

logger = logging.getLogger(__name__)

DEFAULT_IDLE_SECONDS = 30 * 60  # missed-heartbeat window (2× renew interval)


@dataclass
class SweepResult:
    expired: int = 0
    idled: int = 0
    torn_down: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class VenueRuntimeExecutor:
    """One-sweep-per-call lifecycle executor for frontend-venue sessions."""

    def __init__(
        self,
        broker: SecretBroker,
        provisioner: FrontendPodmanProvisioner,
        *,
        idle_seconds: int = DEFAULT_IDLE_SECONDS,
    ) -> None:
        self._broker = broker
        self._provisioner = provisioner
        self._idle_seconds = idle_seconds

    def heartbeat(self, lease_id: str, extend_seconds: int = 900, actor: str = "worker") -> bool:
        """Worker heartbeat → renew with hard-cap semantics (broker enforced)."""
        return self._broker.renew_lease(lease_id, extend_seconds, actor)

    def _is_idle(self, lease: SecretLease, now: datetime) -> bool:
        last = lease.last_accessed_at or lease.issued_at
        return now - last >= timedelta(seconds=self._idle_seconds)

    def sweep_once(self, *, actor: str = "sweeper") -> SweepResult:
        """Expire/revoke-then-teardown pass; idempotent and error-isolated.

        Semantics (design F-5): expiry and idle revoke+teardown; lease that
        was revoked by ANY actor (unpush cascade, session end) tears down
        when no still-valid lease for the same session remains.
        """
        result = SweepResult()
        now = _now()
        by_session: dict[str, list[SecretLease]] = {}
        for lease in self._broker.list_leases(scope="inference", limit=100_000):
            if not lease.session_id:
                continue
            by_session.setdefault(lease.session_id, []).append(lease)
            if lease.is_revoked:
                continue
            if lease.is_expired:
                result.expired += 1
                self._expire_session(lease, actor, result)
            elif self._is_idle(lease, now):
                result.idled += 1
                self._idle_session(lease, actor, result)
        self._teardown_revoked_sessions(by_session, actor, result)
        return result

    def _teardown_revoked_sessions(
        self, by_session: dict[str, list[SecretLease]], actor: str, result: SweepResult
    ) -> None:
        """Tear down sessions whose inference leases are all revoked/expired-left.

        Any still-valid lease keeps the session alive; revoked survivors without
        a live sibling get a teardown (idempotently on failure of prior sweeps).
        """
        for session_id, leases in by_session.items():
            if session_id in result.torn_down:
                continue  # torn earlier in this same sweep (expire/idle already fired)
            if any(err.startswith(f"{session_id}: ") for err in result.errors):
                continue  # a failed teardown already logged once this sweep
            if any(lease.is_valid for lease in leases):
                continue
            if not any(lease.is_revoked for lease in leases):
                continue
            try:
                self._provisioner.teardown(session_id)
                if session_id not in result.torn_down:
                    result.torn_down.append(session_id)
            except Exception as exc:  # noqa: BLE001
                logger.exception("venue revoked-session teardown failed for %s", session_id)
                result.errors.append(f"{session_id}: revoked teardown: {exc}")

    def _expire_session(self, lease: SecretLease, actor: str, result: SweepResult) -> None:
        self._revoke_and_teardown(lease, actor, "expires_at reached", result)

    def _idle_session(self, lease: SecretLease, actor: str, result: SweepResult) -> None:
        self._revoke_and_teardown(
            lease, actor, f"idle beyond {self._idle_seconds}s without heartbeat", result
        )

    def _revoke_and_teardown(
        self, lease: SecretLease, actor: str, detail: str, result: SweepResult
    ) -> None:
        try:
            if not self._broker.revoke_lease(lease.lease_id, actor=actor):
                return
            self._provisioner.teardown(lease.session_id)
            result.torn_down.append(lease.session_id)
        except Exception as exc:  # noqa: BLE001 — one bad session must not stall the sweep
            logger.exception("venue sweep failed for session %s", lease.session_id)
            result.errors.append(f"{lease.session_id}: {detail}: {exc}")
            # Best-effort state: lease stays revoked; the next sweep retries teardown.
