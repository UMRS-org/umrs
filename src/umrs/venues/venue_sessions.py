"""Durable frontend-venue session rows (migration 0099; wave 2026-09-21).

State machine per session: ``provisioning → running → stopped`` with
``failed`` as the provisioning dead-end. The store is in-memory when no
pool is given (tests, dev) and Postgres-backed otherwise — the same
dual-backend shape as ``SecretBroker``.

Rows carry references only (lease id, egress ip, source ids/names, the
worker-reported ``runtime`` projection). Never credential material.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

ACTIVE_STATES = ("provisioning", "running")
VALID_STATES = ("provisioning", "running", "failed", "stopped")

_COLUMNS = (
    "session_id",
    "user_id",
    "venue",
    "state",
    "reason",
    "lease_id",
    "egress_ip",
    "subnet",
    "source_names",
    "source_ids",
    "default_source_id",
    "default_source_name",
    "web_policy",
    "runtime",
    "started_at",
    "updated_at",
    "stopped_at",
)
_SELECT = f"SELECT {', '.join(_COLUMNS)} FROM companion_venue_sessions"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: Any) -> str | None:
    return (
        value.isoformat()
        if isinstance(value, datetime)
        else (str(value) if value else None)
    )


@dataclass
class VenueSessionRow:
    session_id: str
    user_id: str
    venue: str = "frontend"
    state: str = "provisioning"
    reason: str = ""
    lease_id: str = ""
    egress_ip: str = ""
    subnet: str = ""
    source_names: tuple[str, ...] = ()
    source_ids: tuple[str, ...] = ()
    default_source_id: str = ""
    default_source_name: str = ""
    web_policy: str = "public"
    runtime: dict[str, Any] = field(default_factory=dict)
    started_at: datetime | None = None
    updated_at: datetime | None = None
    stopped_at: datetime | None = None

    @property
    def is_active(self) -> bool:
        return self.state in ACTIVE_STATES

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "venue": self.venue,
            "state": self.state,
            "reason": self.reason,
            "lease_id": self.lease_id,
            "egress_ip": self.egress_ip,
            "subnet": self.subnet,
            "source_names": list(self.source_names),
            "default_source": self.default_source_name,
            "web_policy": self.web_policy,
            "runtime": dict(self.runtime),
            "started_at": _iso(self.started_at),
            "updated_at": _iso(self.updated_at),
            "stopped_at": _iso(self.stopped_at),
        }


def _row_from_tuple(row: Any) -> VenueSessionRow:
    data = dict(zip(_COLUMNS, row, strict=False))
    runtime = data.get("runtime") or {}
    if isinstance(runtime, str):
        try:
            runtime = json.loads(runtime)
        except json.JSONDecodeError:
            runtime = {}
    return VenueSessionRow(
        session_id=str(data["session_id"]),
        user_id=str(data["user_id"]),
        venue=str(data.get("venue") or "frontend"),
        state=str(data.get("state") or "provisioning"),
        reason=str(data.get("reason") or ""),
        lease_id=str(data.get("lease_id") or ""),
        egress_ip=str(data.get("egress_ip") or ""),
        subnet=str(data.get("subnet") or ""),
        source_names=tuple(str(s) for s in (data.get("source_names") or [])),
        source_ids=tuple(str(s) for s in (data.get("source_ids") or [])),
        default_source_id=str(data.get("default_source_id") or ""),
        default_source_name=str(data.get("default_source_name") or ""),
        web_policy=str(data.get("web_policy") or "public"),
        runtime=dict(runtime) if isinstance(runtime, dict) else {},
        started_at=data.get("started_at"),
        updated_at=data.get("updated_at"),
        stopped_at=data.get("stopped_at"),
    )


class VenueSessionStore:
    """In-memory (pool=None) or Postgres-backed venue-session registry."""

    def __init__(self, pool: Any = None) -> None:
        self._pool = pool
        self._rows: dict[str, VenueSessionRow] = {}

    # ── reads ────────────────────────────────────────────────────────

    def get(self, session_id: str) -> VenueSessionRow | None:
        if self._pool is None:
            return self._rows.get(session_id)
        row = self._fetchone(f"{_SELECT} WHERE session_id = %s", (session_id,))
        return _row_from_tuple(row) if row else None

    def list_active(self) -> list[VenueSessionRow]:
        if self._pool is None:
            return [r for r in self._rows.values() if r.is_active]
        rows = self._fetchall(
            f"{_SELECT} WHERE state = ANY(%s) ORDER BY started_at",
            (list(ACTIVE_STATES),),
        )
        return [_row_from_tuple(r) for r in rows]

    def taken_subnets(self) -> set[str]:
        return {r.subnet for r in self.list_active() if r.subnet}

    # ── writes ───────────────────────────────────────────────────────

    def upsert_provisioning(
        self,
        *,
        session_id: str,
        user_id: str,
        subnet: str,
        source_names: tuple[str, ...],
        source_ids: tuple[str, ...],
        default_source_id: str,
        default_source_name: str,
        web_policy: str,
    ) -> VenueSessionRow:
        now = _now()
        row = VenueSessionRow(
            session_id=session_id,
            user_id=user_id,
            state="provisioning",
            subnet=subnet,
            source_names=tuple(source_names),
            source_ids=tuple(source_ids),
            default_source_id=default_source_id,
            default_source_name=default_source_name,
            web_policy=web_policy,
            started_at=now,
            updated_at=now,
        )
        if self._pool is None:
            self._rows[session_id] = row
            return row
        self._execute(
            """
            INSERT INTO companion_venue_sessions
                (session_id, user_id, venue, state, reason, lease_id, egress_ip, subnet,
                 source_names, source_ids, default_source_id, default_source_name,
                 web_policy, runtime, started_at, updated_at, stopped_at)
            VALUES (%s, %s, 'frontend', 'provisioning', '', '', '', %s,
                    %s, %s::uuid[], %s::uuid, %s, %s, '{}'::jsonb, now(), now(), NULL)
            ON CONFLICT (session_id) DO UPDATE SET
                user_id = EXCLUDED.user_id,
                state = 'provisioning', reason = '', lease_id = '', egress_ip = '',
                subnet = EXCLUDED.subnet,
                source_names = EXCLUDED.source_names,
                source_ids = EXCLUDED.source_ids,
                default_source_id = EXCLUDED.default_source_id,
                default_source_name = EXCLUDED.default_source_name,
                web_policy = EXCLUDED.web_policy,
                runtime = '{}'::jsonb,
                started_at = now(), updated_at = now(), stopped_at = NULL
            """,
            (
                session_id,
                user_id,
                subnet,
                list(source_names),
                list(source_ids),
                default_source_id or None,
                default_source_name,
                web_policy,
            ),
        )
        return self.get(session_id) or row

    def mark_running(
        self, session_id: str, *, lease_id: str, egress_ip: str
    ) -> VenueSessionRow | None:
        return self._transition(
            session_id, state="running", lease_id=lease_id, egress_ip=egress_ip
        )

    def mark_failed(self, session_id: str, reason: str) -> VenueSessionRow | None:
        return self._transition(session_id, state="failed", reason=reason, stopped=True)

    def mark_stopped(self, session_id: str, reason: str = "") -> VenueSessionRow | None:
        return self._transition(
            session_id, state="stopped", reason=reason, stopped=True
        )

    def update_runtime(
        self, session_id: str, runtime: dict[str, Any]
    ) -> VenueSessionRow | None:
        """Merge worker-reported runtime fields (phase, turn_seq, partial…)."""
        current = self.get(session_id)
        if current is None:
            return None
        merged = {**current.runtime, **runtime, "reported_at": _now().isoformat()}
        if self._pool is None:
            current.runtime = merged
            current.updated_at = _now()
            return current
        self._execute(
            "UPDATE companion_venue_sessions SET runtime = %s::jsonb, updated_at = now() "
            "WHERE session_id = %s",
            (json.dumps(merged), session_id),
        )
        return self.get(session_id)

    def _transition(
        self,
        session_id: str,
        *,
        state: str,
        reason: str | None = None,
        lease_id: str | None = None,
        egress_ip: str | None = None,
        stopped: bool = False,
    ) -> VenueSessionRow | None:
        if state not in VALID_STATES:
            raise ValueError(f"invalid venue state {state!r}")
        current = self.get(session_id)
        if current is None:
            return None
        if self._pool is None:
            current.state = state
            if reason is not None:
                current.reason = reason
            if lease_id is not None:
                current.lease_id = lease_id
            if egress_ip is not None:
                current.egress_ip = egress_ip
            current.updated_at = _now()
            if stopped:
                current.stopped_at = _now()
            return current
        sets = ["state = %s", "updated_at = now()"]
        params: list[Any] = [state]
        if reason is not None:
            sets.append("reason = %s")
            params.append(reason)
        if lease_id is not None:
            sets.append("lease_id = %s")
            params.append(lease_id)
        if egress_ip is not None:
            sets.append("egress_ip = %s")
            params.append(egress_ip)
        if stopped:
            sets.append("stopped_at = now()")
        params.append(session_id)
        self._execute(
            f"UPDATE companion_venue_sessions SET {', '.join(sets)} WHERE session_id = %s",
            tuple(params),
        )
        return self.get(session_id)

    # ── pool plumbing ────────────────────────────────────────────────

    def _fetchone(self, sql: str, params: tuple) -> Any | None:
        conn = self._pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                return cur.fetchone()
        finally:
            self._pool.putconn(conn)

    def _fetchall(self, sql: str, params: tuple) -> list:
        conn = self._pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                return list(cur.fetchall())
        finally:
            self._pool.putconn(conn)

    def _execute(self, sql: str, params: tuple) -> None:
        conn = self._pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(sql, params)
            commit = getattr(conn, "commit", None)
            if callable(commit):
                commit()
        finally:
            self._pool.putconn(conn)
