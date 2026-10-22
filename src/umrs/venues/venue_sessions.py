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
VALID_STATES = ("provisioning", "running", "paused", "failed", "stopped")

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
    "job_id",
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
    # Durable job carrying the lifecycle (0100; "" for rows older than the link).
    job_id: str = ""

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
            "job_id": self.job_id,
        }


def _s(data: dict, key: str, default: str = "") -> str:
    return str(data.get(key) or default)


def _strs(data: dict, key: str) -> tuple[str, ...]:
    return tuple(str(v) for v in (data.get(key) or []))


def _runtime_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return {}
    return dict(value) if isinstance(value, dict) else {}


def _row_from_tuple(row: Any) -> VenueSessionRow:
    data = dict(zip(_COLUMNS, row, strict=False))
    return VenueSessionRow(
        session_id=str(data["session_id"]),
        user_id=str(data["user_id"]),
        venue=_s(data, "venue", "frontend"),
        state=_s(data, "state", "provisioning"),
        reason=_s(data, "reason"),
        lease_id=_s(data, "lease_id"),
        egress_ip=_s(data, "egress_ip"),
        subnet=_s(data, "subnet"),
        source_names=_strs(data, "source_names"),
        source_ids=_strs(data, "source_ids"),
        default_source_id=_s(data, "default_source_id"),
        default_source_name=_s(data, "default_source_name"),
        web_policy=_s(data, "web_policy", "public"),
        runtime=_runtime_dict(data.get("runtime")),
        started_at=data.get("started_at"),
        updated_at=data.get("updated_at"),
        stopped_at=data.get("stopped_at"),
        job_id=_s(data, "job_id"),
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

    def list_for_admin(
        self, *, user_id: str = "", state: str = "", project: str = "", limit: int = 100
    ) -> list[tuple[VenueSessionRow, str]]:
        """Every user's rows, newest first, with each row's project (the
        ``frontend_session`` job's ``project_id``; "" when unlinked). Empty
        filters match all (C-1 admin list, remote-session plan RS-0).

        In memory (tests, dev) no job table exists: the project is "" and a
        project filter matches nothing."""
        if self._pool is None:
            return _memory_admin_rows(
                self._rows.values(), user_id, state, project, limit
            )
        rows = self._fetchall(*_admin_list_sql(user_id, state, project, limit))
        return [(_row_from_tuple(r[:-1]), str(r[-1] or "")) for r in rows]

    def taken_subnets(self) -> set[str]:
        return {r.subnet for r in self.list_active() if r.subnet}

    def get_by_job_id(self, job_id: str) -> VenueSessionRow | None:
        if not job_id:
            return None
        if self._pool is None:
            return next((r for r in self._rows.values() if r.job_id == job_id), None)
        row = self._fetchone(f"{_SELECT} WHERE job_id = %s", (job_id,))
        return _row_from_tuple(row) if row else None

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
        job_id: str = "",
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
            job_id=job_id,
        )
        if self._pool is None:
            self._rows[session_id] = row
            return row
        self._execute(
            """
            INSERT INTO companion_venue_sessions
                (session_id, user_id, venue, state, reason, lease_id, egress_ip, subnet,
                 source_names, source_ids, default_source_id, default_source_name,
                 web_policy, runtime, started_at, updated_at, stopped_at, job_id)
            VALUES (%s, %s, 'frontend', 'provisioning', '', '', '', %s,
                    %s, %s::uuid[], %s::uuid, %s, %s, '{}'::jsonb, now(), now(), NULL, %s)
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
                started_at = now(), updated_at = now(), stopped_at = NULL,
                job_id = CASE WHEN EXCLUDED.job_id <> '' THEN EXCLUDED.job_id
                              ELSE companion_venue_sessions.job_id END
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
                job_id,
            ),
        )
        return self.get(session_id) or row

    def set_job_id(self, session_id: str, job_id: str) -> VenueSessionRow | None:
        current = self.get(session_id)
        if current is None:
            return None
        if self._pool is None:
            current.job_id = job_id
            return current
        self._execute(
            "UPDATE companion_venue_sessions SET job_id = %s, updated_at = now() "
            "WHERE session_id = %s",
            (job_id, session_id),
        )
        return self.get(session_id)

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

    def mark_paused(self, session_id: str, reason: str = "") -> VenueSessionRow | None:
        """D-09 pause verb: provisioning/running → paused (lease stays warm —
        the worker keeps heartbeating while paused; the sweep only reaps
        rows whose leases actually expire)."""
        current = self.get(session_id)
        if current is None or not current.is_active:
            return None
        return self._transition(session_id, state="paused", reason=reason)

    def mark_resumed(self, session_id: str) -> VenueSessionRow | None:
        """D-09 resume verb: paused → running (only from paused — never
        resurrects stopped/failed rows)."""
        current = self.get(session_id)
        if current is None or current.state != "paused":
            return None
        return self._transition(session_id, state="running", reason="resumed")

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
        fields = {"reason": reason, "lease_id": lease_id, "egress_ip": egress_ip}
        if self._pool is None:
            return _transition_memory(current, state, fields, stopped)
        self._execute(*_transition_sql(session_id, state, fields, stopped))
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


def project_session_state(pool: Any, owner: str, session_id: str, state: str) -> bool:
    """Project a runtime transition onto the session snapshot's 0084 columns.

    Plain SQL here (no api/ import — jobs/ and venues/ callers, ADR-053).
    ``state`` is one of the stored states (ongoing|stopped|error); venue is
    pinned to ``frontend``. Best-effort: False when the row is absent or the
    pool is missing.
    """
    # "paused" joins the vocabulary with the D-09 venue verbs (pause/resume).
    if pool is None or state not in ("ongoing", "paused", "stopped", "completed", "error", ""):
        return False
    conn = pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE companion_session_snapshots
                SET venue = %s, state = %s, updated_at = now()
                WHERE user_id = %s AND session_id = %s
                """,
                ("frontend", state, owner, session_id),
            )
            applied = int(getattr(cur, "rowcount", 0) or 0) > 0
        commit = getattr(conn, "commit", None)
        if callable(commit):
            commit()
        return applied
    except Exception:  # projection never masks the runtime result
        logger.warning(
            "session state projection failed for %s", session_id, exc_info=True
        )
        return False
    finally:
        pool.putconn(conn)


def _memory_admin_rows(
    rows: Any, user_id: str, state: str, project: str, limit: int
) -> list[tuple[VenueSessionRow, str]]:
    if project:
        return []
    picked = [
        r
        for r in rows
        if (not user_id or r.user_id == user_id) and (not state or r.state == state)
    ]
    picked.sort(key=lambda r: r.started_at or _now(), reverse=True)
    return [(r, "") for r in picked[: max(0, limit)]]


def _admin_list_sql(
    user_id: str, state: str, project: str, limit: int
) -> tuple[str, tuple]:
    """Rows joined to their session job for the project (one query)."""
    cols = ", ".join(f"s.{c}" for c in _COLUMNS)
    sql = (
        f"SELECT {cols}, COALESCE(j.project_id, '') "
        "FROM companion_venue_sessions s "
        "LEFT JOIN jobs j ON j.job_id = s.job_id "
        "WHERE (%s = '' OR s.user_id = %s) "
        "AND (%s = '' OR s.state = %s) "
        "AND (%s = '' OR j.project_id = %s) "
        "ORDER BY s.started_at DESC NULLS LAST LIMIT %s"
    )
    return sql, (user_id, user_id, state, state, project, project, max(0, limit))


def _transition_memory(
    current: VenueSessionRow, state: str, fields: dict[str, str | None], stopped: bool
) -> VenueSessionRow:
    current.state = state
    for key, value in fields.items():
        if value is not None:
            setattr(current, key, value)
    current.updated_at = _now()
    if stopped:
        current.stopped_at = _now()
    return current


def _transition_sql(
    session_id: str, state: str, fields: dict[str, str | None], stopped: bool
) -> tuple[str, tuple]:
    sets = ["state = %s", "updated_at = now()"]
    params: list[Any] = [state]
    for key, value in fields.items():
        if value is not None:
            sets.append(f"{key} = %s")
            params.append(value)
    if stopped:
        sets.append("stopped_at = now()")
    params.append(session_id)
    return (
        f"UPDATE companion_venue_sessions SET {', '.join(sets)} WHERE session_id = %s",
        tuple(params),
    )
