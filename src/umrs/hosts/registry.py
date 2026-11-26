"""The hosts registry (ADR-098 D-4, slice SH-H1a).

One row per host a site may place sandboxes on, keyed by (venue, host_id):
the sites share one database and every front's own host is ``local``. The
inline front registers itself at boot, so placement (SH-H2) reads one table
on a single node and on a site with agent hosts.

States: ``active`` (placeable), ``draining`` (keeps its sessions, gets no new
placement; admin), ``down`` (missed heartbeats; the janitor ended its
sessions ``host_lost`` and revoked their leases). A heartbeat brings a
``down`` host back to ``active``, or to ``draining`` when an admin had drained
it: the drain lives in ``drain_reason``, which only ``undrain`` clears, so a
blip never undoes it.

Staleness is judged on the database clock (``now()``, the clock that stamps
``last_heartbeat_at``), never this process's: a process clock ahead of the
database would otherwise down a fresh host, its own inline host included.
Admin transitions are compare-and-set on the state they checked, so a
racing janitor or admin is refused by name instead of overwritten.

Dual backend like ``SecretBroker``: in memory without a pool (tests, dev),
Postgres (migration ``<n>_hosts.sql``) with one. Every query is scoped to this
registry's venue, so one site never reads, drains or downs another's hosts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from typing import Any

HOST_STATES = ("active", "draining", "down")
#: States placement may choose (SH-H2 reads :meth:`HostRegistry.placeable`).
PLACEABLE_STATES = ("active",)
HOST_LAUNCH_MODES = ("inline", "agent")
_REMOVABLE_STATES = ("draining", "down")

_COLUMNS = (
    "venue",
    "host_id",
    "launch_mode",
    "roles",
    "state",
    "reason",
    "labels",
    "capacity",
    "isolation",
    "registered_at",
    "updated_at",
    "last_heartbeat_at",
    "drain_reason",
)
_SELECT = f"SELECT {', '.join(_COLUMNS)} FROM hosts"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: Any) -> str | None:
    return value.isoformat() if isinstance(value, datetime) else None


class HostRegistryError(ValueError):
    """A refused admin transition; ``code`` is the typed reason."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass
class HostRecord:
    host_id: str
    venue: str = ""
    launch_mode: str = "inline"
    roles: tuple[str, ...] = ("session",)
    state: str = "active"
    reason: str = ""
    labels: dict[str, Any] = field(default_factory=dict)
    capacity: dict[str, Any] = field(default_factory=dict)
    isolation: dict[str, Any] = field(default_factory=dict)
    registered_at: datetime | None = None
    updated_at: datetime | None = None
    last_heartbeat_at: datetime | None = None
    #: Non-empty while an admin drain stands (kept through ``down``).
    drain_reason: str = ""

    @property
    def placeable(self) -> bool:
        return self.state in PLACEABLE_STATES

    def to_dict(self) -> dict[str, Any]:
        return {
            "venue": self.venue,
            "host_id": self.host_id,
            "launch_mode": self.launch_mode,
            "roles": list(self.roles),
            "state": self.state,
            "reason": self.reason,
            "labels": dict(self.labels),
            "capacity": dict(self.capacity),
            "isolation": dict(self.isolation),
            "registered_at": _iso(self.registered_at),
            "updated_at": _iso(self.updated_at),
            "last_heartbeat_at": _iso(self.last_heartbeat_at),
            "drain_reason": self.drain_reason,
        }


def _json_obj(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        value = json.loads(value or "{}")
    return dict(value or {})


def _record(row: Any) -> HostRecord:
    data = dict(zip(_COLUMNS, row, strict=True))
    return HostRecord(
        host_id=str(data["host_id"]),
        venue=str(data["venue"] or ""),
        launch_mode=str(data["launch_mode"]),
        roles=tuple(data["roles"] or ()),
        state=str(data["state"]),
        reason=str(data["reason"] or ""),
        labels=_json_obj(data["labels"]),
        capacity=_json_obj(data["capacity"]),
        isolation=_json_obj(data["isolation"]),
        registered_at=data["registered_at"],
        updated_at=data["updated_at"],
        last_heartbeat_at=data["last_heartbeat_at"],
        drain_reason=str(data["drain_reason"] or ""),
    )


def _check_drain(record: HostRecord) -> None:
    if record.state == "down":
        raise HostRegistryError(
            "hosts.host_down",
            f"host {record.host_id!r} is down: remove it, or let it heartbeat first",
        )


def _check_undrain(record: HostRecord) -> None:
    if record.state != "draining":
        raise HostRegistryError(
            "hosts.not_draining", f"host {record.host_id!r} is {record.state}"
        )


def _check_remove(record: HostRecord) -> None:
    if record.state == "active":
        raise HostRegistryError(
            "hosts.host_active", f"host {record.host_id!r} is active: drain it first"
        )


# ── backends ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _Change:
    """An admin transition, applied only while the state is in ``from_states``."""

    from_states: tuple[str, ...]
    state: str
    reason: str
    drain_reason: str


class _MemoryHosts:
    def __init__(self) -> None:
        self._rows: dict[tuple[str, str], HostRecord] = {}

    def upsert(self, record: HostRecord) -> HostRecord:
        key = (record.venue, record.host_id)
        now = _now()
        current = self._rows.get(key)
        drained = current.drain_reason if current else ""
        self._rows[key] = replace(
            record,
            state="draining" if drained else "active",
            reason=drained,
            drain_reason=drained,
            registered_at=current.registered_at if current else now,
            updated_at=now,
            last_heartbeat_at=now,
        )
        return replace(self._rows[key])

    def heartbeat(self, venue: str, host_id: str) -> HostRecord | None:
        current = self._rows.get((venue, host_id))
        if current is None:
            return None
        current.last_heartbeat_at = current.updated_at = _now()
        if current.state == "down":
            current.state = "draining" if current.drain_reason else "active"
            current.reason = current.drain_reason
        return replace(current)

    def get(self, venue: str, host_id: str) -> HostRecord | None:
        current = self._rows.get((venue, host_id))
        return replace(current) if current else None

    def select(self, venue: str, states: tuple[str, ...]) -> list[HostRecord]:
        rows = [r for (v, _), r in sorted(self._rows.items()) if v == venue]
        return [replace(r) for r in rows if r.state in states]

    def stale(self, venue: str, down_after_s: float) -> list[HostRecord]:
        cutoff = _now() - timedelta(seconds=down_after_s)
        return [
            h
            for h in self.select(venue, ("active", "draining"))
            if h.last_heartbeat_at is None or h.last_heartbeat_at < cutoff
        ]

    def transition(self, venue: str, host_id: str, change: _Change) -> bool:
        current = self._rows.get((venue, host_id))
        if current is None or current.state not in change.from_states:
            return False
        current.state, current.reason = change.state, change.reason
        current.drain_reason, current.updated_at = change.drain_reason, _now()
        return True

    def mark_down(
        self, venue: str, host_id: str, down_after_s: float, reason: str
    ) -> bool:
        current = self._rows.get((venue, host_id))
        cutoff = _now() - timedelta(seconds=down_after_s)
        if current is None or current.state == "down":
            return False
        if (current.last_heartbeat_at or _now()) >= cutoff:
            return False
        current.state, current.reason, current.updated_at = "down", reason, _now()
        return True

    def delete(self, venue: str, host_id: str, from_states: tuple[str, ...]) -> bool:
        current = self._rows.get((venue, host_id))
        if current is None or current.state not in from_states:
            return False
        del self._rows[(venue, host_id)]
        return True


_UPSERT_SQL = (
    "INSERT INTO hosts (venue, host_id, launch_mode, roles, labels, capacity,"
    " isolation) VALUES (%s, %s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb)"
    " ON CONFLICT (venue, host_id) DO UPDATE SET"
    " launch_mode = EXCLUDED.launch_mode, roles = EXCLUDED.roles,"
    " labels = EXCLUDED.labels, capacity = EXCLUDED.capacity,"
    " isolation = EXCLUDED.isolation, updated_at = now(),"
    " last_heartbeat_at = now(),"
    " reason = hosts.drain_reason,"
    " state = CASE WHEN hosts.drain_reason <> '' THEN 'draining' ELSE 'active' END"
    f" RETURNING {', '.join(_COLUMNS)}"
)
_HEARTBEAT_SQL = (
    "UPDATE hosts SET last_heartbeat_at = now(), updated_at = now(),"
    " reason = CASE WHEN state = 'down' THEN drain_reason ELSE reason END,"
    " state = CASE WHEN state <> 'down' THEN state"
    " WHEN drain_reason <> '' THEN 'draining' ELSE 'active' END"
    f" WHERE venue = %s AND host_id = %s RETURNING {', '.join(_COLUMNS)}"
)
# Staleness on the database clock (the one that stamps last_heartbeat_at).
_STALE_SQL = (
    f"{_SELECT} WHERE venue = %s AND state <> 'down'"
    " AND last_heartbeat_at < now() - make_interval(secs => %s) ORDER BY host_id"
)
_MARK_DOWN_SQL = (
    "UPDATE hosts SET state = 'down', reason = %s, updated_at = now()"
    " WHERE venue = %s AND host_id = %s AND state <> 'down'"
    " AND last_heartbeat_at < now() - make_interval(secs => %s) RETURNING host_id"
)
_TRANSITION_SQL = (
    "UPDATE hosts SET state = %s, reason = %s, drain_reason = %s,"
    " updated_at = now() WHERE venue = %s AND host_id = %s"
    " AND state = ANY(%s) RETURNING host_id"
)
_DELETE_SQL = (
    "DELETE FROM hosts WHERE venue = %s AND host_id = %s AND state = ANY(%s)"
    " RETURNING host_id"
)


class _PgHosts:
    def __init__(self, pool: Any) -> None:
        self._pool = pool

    def upsert(self, record: HostRecord) -> HostRecord:
        params = (
            record.venue,
            record.host_id,
            record.launch_mode,
            list(record.roles),
            json.dumps(record.labels),
            json.dumps(record.capacity),
            json.dumps(record.isolation),
        )
        return _record(self._one(_UPSERT_SQL, params))

    def heartbeat(self, venue: str, host_id: str) -> HostRecord | None:
        row = self._one(_HEARTBEAT_SQL, (venue, host_id))
        return _record(row) if row else None

    def get(self, venue: str, host_id: str) -> HostRecord | None:
        row = self._one(
            f"{_SELECT} WHERE venue = %s AND host_id = %s", (venue, host_id)
        )
        return _record(row) if row else None

    def select(self, venue: str, states: tuple[str, ...]) -> list[HostRecord]:
        sql = f"{_SELECT} WHERE venue = %s AND state = ANY(%s) ORDER BY host_id"
        return [_record(r) for r in self._all(sql, (venue, list(states)))]

    def stale(self, venue: str, down_after_s: float) -> list[HostRecord]:
        rows = self._all(_STALE_SQL, (venue, float(down_after_s)))
        return [_record(r) for r in rows]

    def transition(self, venue: str, host_id: str, change: _Change) -> bool:
        params = (
            change.state,
            change.reason,
            change.drain_reason,
            venue,
            host_id,
            list(change.from_states),
        )
        return self._one(_TRANSITION_SQL, params) is not None

    def mark_down(
        self, venue: str, host_id: str, down_after_s: float, reason: str
    ) -> bool:
        params = (reason, venue, host_id, float(down_after_s))
        return self._one(_MARK_DOWN_SQL, params) is not None

    def delete(self, venue: str, host_id: str, from_states: tuple[str, ...]) -> bool:
        params = (venue, host_id, list(from_states))
        return self._one(_DELETE_SQL, params) is not None

    def _one(self, sql: str, params: tuple) -> Any:
        rows = self._all(sql, params)
        return rows[0] if rows else None

    def _all(self, sql: str, params: tuple) -> list[Any]:
        conn = self._pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall() if cur.description else []
            conn.commit()
            return list(rows)
        except Exception:
            conn.rollback()
            raise
        finally:
            self._pool.putconn(conn)


# ── the registry ──────────────────────────────────────────────────────────


class HostRegistry:
    """This site's hosts: register, heartbeat, list, drain, undrain, remove."""

    def __init__(self, pool: Any | None = None, *, venue: str | None = None) -> None:
        self._backend: Any = _PgHosts(pool) if pool is not None else _MemoryHosts()
        self._venue = venue

    @property
    def venue(self) -> str:
        """The site this registry reads and writes (``site.yaml`` ``name``)."""
        if self._venue is None:
            from umrs.jobs.store import configured_site_name

            self._venue = configured_site_name()
        return self._venue

    def register(
        self,
        host_id: str,
        *,
        launch_mode: str = "inline",
        roles: tuple[str, ...] = ("session",),
        labels: dict[str, Any] | None = None,
        capacity: dict[str, Any] | None = None,
        isolation: dict[str, Any] | None = None,
    ) -> HostRecord:
        """Insert or refresh a host (idempotent: one row per host id).

        A re-registration counts as a heartbeat; a ``draining`` host stays
        draining, a ``down`` one comes back ``active``."""
        if not host_id:
            raise HostRegistryError("hosts.invalid", "host id is empty")
        if launch_mode not in HOST_LAUNCH_MODES:
            raise HostRegistryError("hosts.invalid", f"launch mode {launch_mode!r}")
        record = HostRecord(
            host_id=host_id,
            venue=self.venue,
            launch_mode=launch_mode,
            roles=tuple(roles),
            labels=dict(labels or {}),
            capacity=dict(capacity or {}),
            isolation=dict(isolation or {}),
        )
        return self._backend.upsert(record)

    def heartbeat(self, host_id: str) -> HostRecord | None:
        """Stamp a heartbeat; None when the host is not registered (removed)."""
        return self._backend.heartbeat(self.venue, host_id)

    def get(self, host_id: str) -> HostRecord | None:
        return self._backend.get(self.venue, host_id)

    def hosts(self, states: tuple[str, ...] = HOST_STATES) -> list[HostRecord]:
        return self._backend.select(self.venue, states)

    def placeable(self) -> list[HostRecord]:
        """Hosts placement may choose: ``active`` only (never draining/down)."""
        return self.hosts(PLACEABLE_STATES)

    def stale(self, down_after_s: float) -> list[HostRecord]:
        """Hosts not yet down whose last heartbeat is older than
        ``down_after_s`` (database clock on Postgres)."""
        return self._backend.stale(self.venue, down_after_s)

    def mark_down(self, host_id: str, down_after_s: float, reason: str) -> bool:
        """Compare-and-set: down only if still stale (a racing heartbeat wins)."""
        return self._backend.mark_down(self.venue, host_id, down_after_s, reason)

    def drain(self, host_id: str, reason: str = "") -> HostRecord:
        _check_drain(self._require(host_id))
        why = reason or "drained"
        change = _Change(("active", "draining"), "draining", why, why)
        self._apply(host_id, change, _check_drain)
        return self._require(host_id)

    def undrain(self, host_id: str) -> HostRecord:
        _check_undrain(self._require(host_id))
        self._apply(host_id, _Change(("draining",), "active", "", ""), _check_undrain)
        return self._require(host_id)

    def remove(self, host_id: str) -> HostRecord:
        record = self._require(host_id)
        _check_remove(record)
        if not self._backend.delete(self.venue, host_id, _REMOVABLE_STATES):
            self._refuse_raced(host_id, _check_remove)
        return record

    def _apply(self, host_id: str, change: _Change, check: Any) -> None:
        if not self._backend.transition(self.venue, host_id, change):
            self._refuse_raced(host_id, check)

    def _refuse_raced(self, host_id: str, check: Any) -> None:
        """The checked state changed before the write: refuse by name."""
        record = self._require(host_id)
        check(record)
        raise HostRegistryError(
            "hosts.conflict",
            f"host {host_id!r} changed to {record.state} meanwhile: retry",
        )

    def _require(self, host_id: str) -> HostRecord:
        record = self.get(host_id)
        if record is None:
            raise HostRegistryError(
                "hosts.not_found", f"no host {host_id!r} at site {self.venue!r}"
            )
        return record


__all__ = [
    "HOST_LAUNCH_MODES",
    "HOST_STATES",
    "PLACEABLE_STATES",
    "HostRecord",
    "HostRegistry",
    "HostRegistryError",
]
