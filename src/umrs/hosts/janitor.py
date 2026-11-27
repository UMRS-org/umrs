"""Host heartbeat and the site-scoped hosts janitor (ADR-098 D-4, SH-H1a).

Every tick the UMRS process first heartbeats its own (inline) host, then
sweeps this site's registry:

1. a host whose last heartbeat is older than ``down_after_s`` turns ``down``
   (compare-and-set, so a racing heartbeat wins);
2. every session still live on a ``down`` host ends ``host_lost`` and its
   leases are revoked, so the inference proxy and the rail refuse them at
   once. The user restores from the journal (resume elsewhere is RS-5).

Step 2 runs for every down host on every tick, not only the one just marked,
so a crash between the two steps loses nothing. The janitor is site-scoped:
it reads only its venue's hosts (sites share one database).

Which sessions run on a host is the extension's knowledge (core never reads
an extension table): an extension registers a :class:`HostSessions` factory
with ``ExtensionContext.add_host_sessions``.

The heartbeat comes before the sweep in the same tick, so a process never
marks its own inline host down, even after a restart longer than the
timeout.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from umrs.hosts.placement import inline_host_profile
from umrs.hosts.registry import HostRecord, HostRegistry
from umrs.hosts.runtime import INLINE_HOST_ID
from umrs.venues.session_audit import audit_venue

logger = logging.getLogger(__name__)

HOST_LOST = "host_lost"
JANITOR_ACTOR = "hosts-janitor"

_HEARTBEAT_DEFAULT_SEC = 30.0
_HEARTBEAT_MIN_SEC = 5.0
_HEARTBEAT_MAX_SEC = 300.0


@dataclass(frozen=True)
class HostSession:
    """One live session on a host (references only)."""

    session_id: str
    owner: str = ""
    project: str = ""


class HostSessions(Protocol):
    """An extension's sessions on hosts (registered via ``add_host_sessions``)."""

    def sessions_on_host(self, venue: str, host_id: str) -> list[HostSession]: ...

    def mark_host_lost(self, session_id: str, reason: str) -> bool: ...


@dataclass
class SweepReport:
    down: list[str] = field(default_factory=list)
    lost: list[str] = field(default_factory=list)
    revoked: int = 0


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except ValueError:
        return default


def heartbeat_seconds() -> float:
    """``UMRS_HOST_HEARTBEAT_SEC`` (30 s), clamped to 5..300 s."""
    value = _float_env("UMRS_HOST_HEARTBEAT_SEC", _HEARTBEAT_DEFAULT_SEC)
    return min(_HEARTBEAT_MAX_SEC, max(_HEARTBEAT_MIN_SEC, value))


def down_after_seconds(heartbeat: float) -> float:
    """``UMRS_HOST_DOWN_AFTER_SEC`` (three missed beats), at least two beats."""
    value = _float_env("UMRS_HOST_DOWN_AFTER_SEC", 3 * heartbeat)
    return max(2 * heartbeat, value)


class HostJanitor:
    """Marks stale hosts down and ends the sessions of down hosts."""

    def __init__(
        self,
        registry: HostRegistry,
        *,
        broker: Any,
        sessions: Sequence[HostSessions] = (),
        down_after_s: float,
    ) -> None:
        self.registry = registry
        self.broker = broker
        self.sessions = list(sessions)
        self.down_after_s = down_after_s

    def sweep(self) -> SweepReport:
        report = SweepReport()
        reason = f"no heartbeat for {int(self.down_after_s)}s"
        for host in self.registry.stale(self.down_after_s):
            if self.registry.mark_down(host.host_id, self.down_after_s, reason):
                report.down.append(host.host_id)
                self._audit_down(host, reason)
        for host in self.registry.hosts(("down",)):
            self._end_sessions(host, report)
        return report

    def _end_sessions(self, host: HostRecord, report: SweepReport) -> None:
        reason = f"{HOST_LOST}: host {host.host_id} is down"
        for provider in self.sessions:
            for session in provider.sessions_on_host(host.venue, host.host_id):
                self._lose(provider, session, host, reason, report)

    def _lose(
        self,
        provider: HostSessions,
        session: HostSession,
        host: HostRecord,
        reason: str,
        report: SweepReport,
    ) -> None:
        revoked = int(
            self.broker.revoke_by_session(session.session_id, actor=JANITOR_ACTOR)
        )
        report.revoked += revoked
        if not provider.mark_host_lost(session.session_id, reason):
            return  # ended meanwhile; its leases are revoked either way
        report.lost.append(session.session_id)
        audit_venue(
            "lease.revoke",
            actor=JANITOR_ACTOR,
            session_id=session.session_id,
            owner=session.owner,
            project=session.project,
            revoked=revoked,
            via=HOST_LOST,
            host_id=host.host_id,
        )

    def _audit_down(self, host: HostRecord, reason: str) -> None:
        audit_venue(
            "host.down",
            actor=JANITOR_ACTOR,
            session_id="",
            outcome="error",
            scope="system",
            host_id=host.host_id,
            venue=host.venue,
            detail=reason,
        )


class HostMaintenance:
    """The process's host loop: heartbeat its own host, then sweep."""

    def __init__(
        self,
        registry: HostRegistry,
        janitor: HostJanitor,
        *,
        host_id: str = INLINE_HOST_ID,
        interval: float | None = None,
    ) -> None:
        self.registry = registry
        self.janitor = janitor
        self.host_id = host_id
        self.interval = interval if interval is not None else heartbeat_seconds()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._missing_logged = False

    def tick(self) -> SweepReport:
        if self.registry.heartbeat(self.host_id) is None:
            self._log_missing()
        return self.janitor.sweep()

    def _log_missing(self) -> None:
        if not self._missing_logged:
            logger.warning(
                "host %r is not registered (removed by an admin?): it gets no "
                "heartbeat until this process restarts",
                self.host_id,
            )
        self._missing_logged = True

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._loop, name="umrs-hosts", daemon=True
        )
        self._thread.start()
        logger.info("hosts heartbeat + janitor started (every %.0fs)", self.interval)

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)
            self._thread = None

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                self.tick()
            except Exception:
                logger.warning("hosts tick failed", exc_info=True)


def extension_host_sessions(db_pool: Any) -> list[HostSessions]:
    """Every enabled extension's :class:`HostSessions`, built on ``db_pool``."""
    from umrs.extensions.loader import get_extension_context

    built = []
    for factory in get_extension_context().host_session_factories:
        try:
            provider = factory(db_pool)
        except Exception:
            logger.warning("host sessions of %r not built", factory, exc_info=True)
            continue
        if provider is not None:
            built.append(provider)
    return built


def _site_extra() -> dict[str, Any]:
    """``site.yaml`` ``extra`` ({} when unreadable: the inline host then
    reports the default ``crun``/``squid-only`` isolation, never better)."""
    try:
        from umrs.orchestrator.site_config import load_site_config

        extra = load_site_config().extra
    except FileNotFoundError:  # no site file: a single node, default profile
        return {}
    except Exception:  # noqa: BLE001 — fail closed to the default profile
        logger.warning("site config unreadable; inline host profile defaulted")
        return {}
    return extra if isinstance(extra, dict) else {}


def site_inline_profile() -> dict[str, dict[str, Any]]:
    """This site's inline host profile (labels, capacity, isolation)."""
    return inline_host_profile(_site_extra())


def start_inline_host(
    db_pool: Any, *, broker: Any = None, venue: str | None = None
) -> HostMaintenance:
    """Register this process's own host (``inline``) and start its loop."""
    registry = HostRegistry(db_pool, venue=venue)
    registry.register(
        INLINE_HOST_ID,
        launch_mode="inline",
        roles=("session",),
        **site_inline_profile(),
    )
    if broker is None:
        from umrs.jobs.secret_broker import SecretBroker

        broker = SecretBroker(db_pool=db_pool, venue=registry.venue)
    interval = heartbeat_seconds()
    janitor = HostJanitor(
        registry,
        broker=broker,
        sessions=extension_host_sessions(db_pool),
        down_after_s=down_after_seconds(interval),
    )
    maintenance = HostMaintenance(registry, janitor, interval=interval)
    maintenance.start()
    return maintenance


__all__ = [
    "HOST_LOST",
    "HostJanitor",
    "HostMaintenance",
    "HostSession",
    "HostSessions",
    "SweepReport",
    "down_after_seconds",
    "extension_host_sessions",
    "heartbeat_seconds",
    "site_inline_profile",
    "start_inline_host",
]
