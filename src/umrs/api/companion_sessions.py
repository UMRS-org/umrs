"""Companion session snapshots (ADR-036 F2 first leg + venue registry).

Client companion sessions mirror their state here (write-through). A fresh
machine answers cross-client session presence/read by pulling these rows;
the companion SQLite store is the rebuildable read-model over them.

Venue/state registry columns (0084): the listing supports server-side
filters so a unified session view (local + remote) never has to emulate
remote filtering client-side. Plan:
docs/development/session-venue-selection-and-unified-listing-plan-2026-09-09.md
"""

from __future__ import annotations

import logging
import re
from typing import Any

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from .typed_errors import PlatformError, raise_typed

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/umrs/companion/sessions", tags=["companion-sessions"])

# Venue grammar (plan §4.1): local | frontend | cluster:<name>.
# '' means absent (client did not declare; never fabricated server-side).
_VENUE_RE = re.compile(r"^(local|frontend|cluster:[a-z0-9][a-z0-9-]{0,62})$")
# Recorded states. 'archived' is computed from tombstones, never stored.
_STATE_STORED = frozenset({"", "ongoing", "stopped", "completed", "error"})
# Filterable computed states include 'archived'.
_STATE_COMPUTED = _STATE_STORED | {"archived"}


def _validate_venue(value: str) -> str:
    if not _VENUE_RE.match(value or ""):
        raise_typed(
            PlatformError.companionsessions(
                "bad_request",
                f"invalid venue {value!r}: expected local | frontend | cluster:<name>",
            ),
        )
    return value


def _validate_stored_state(value: str) -> str:
    if value not in _STATE_STORED:
        raise_typed(
            PlatformError.companionsessions(
                "bad_request",
                (
                    f"invalid state {value!r}: expected one of "
                    f"{sorted(s for s in _STATE_STORED if s)} or '' "
                    "('archived' is computed, never stored)"
                ),
            ),
        )
    return value


def _csv_filter(param: str | None) -> list[str]:
    return [part.strip() for part in (param or "").split(",") if part.strip()]


class SessionSnapshotBody(BaseModel):
    """The mirrored companion session document."""

    name: str = Field(default="", max_length=256)
    updated_at: str = Field(default="")
    payload: str = Field(default="")
    venue: str = Field(default="")  # '' = absent; clients declare at PUT.
    state: str = Field(default="")


# ── Auth + pool helpers ────────────────────────────────────────────────────


def _uid(request: Request) -> str:
    """Owner identity, as asserted by the gateway-verified header."""
    uid = request.headers.get("x-user-id", "").strip()
    if not uid:
        raise_typed(
            PlatformError.companionsessions("unauthorized", "identity header missing")
        )
    return uid


def _pool(request: Request) -> Any:
    pool = getattr(request.app.state, "umrs_db_pool", None)
    if pool is None:
        raise_typed(
            PlatformError.companionsessions("unavailable", "database pool unavailable")
        )
    return pool


# ── Routes ─────────────────────────────────────────────────────────────────


@router.post("/{session_id}/snapshot", response_model=dict[str, Any])
async def put_session_snapshot(
    session_id: str, body: SessionSnapshotBody, request: Request
) -> dict[str, Any]:
    """Mirror one client session state. Newer updates supersede older ones.
    # auth: verified
    """
    uid = _uid(request)
    pool = _pool(request)
    venue = _validate_venue(body.venue) if body.venue else ""
    state = _validate_stored_state(body.state)
    # Venue: a PUT that omits venue must not stomp a recorded one — absent
    # means absent, not 'local'. Local clients always declare 'local'.
    conn = pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO companion_session_snapshots
                    (user_id, session_id, name, payload, update_marker, venue, state)
                VALUES (%s, %s, %s, %s, %s,
                        COALESCE(NULLIF(%s, ''), 'local'), %s)
                ON CONFLICT (user_id, session_id) DO UPDATE SET
                    name = EXCLUDED.name,
                    payload = EXCLUDED.payload,
                    update_marker = EXCLUDED.update_marker,
                    venue = CASE
                        WHEN EXCLUDED.venue <> '' THEN EXCLUDED.venue
                        ELSE companion_session_snapshots.venue
                    END,
                    state = CASE
                        WHEN EXCLUDED.state <> '' THEN EXCLUDED.state
                        ELSE companion_session_snapshots.state
                    END,
                    updated_at = now()
                WHERE companion_session_snapshots.update_marker <= EXCLUDED.update_marker
                """,
                (
                    uid,
                    session_id,
                    body.name,
                    body.payload,
                    body.updated_at,
                    venue,
                    state,
                ),
            )
            applied = cur.rowcount > 0
            conn.commit()
    finally:
        pool.putconn(conn)
    return {"status": "ok", "applied": applied}


@router.get("/{session_id}", response_model=dict[str, Any])
async def get_session_snapshot(session_id: str, request: Request) -> dict[str, Any]:
    """Answer a mirrored snapshot for the owner (404 for others/absent).
    # auth: verified
    """
    uid = _uid(request)
    pool = _pool(request)
    conn = pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT s.session_id, s.name, s.payload, s.update_marker,
                       s.venue, s.state, a.archived_at
                FROM companion_session_snapshots s
                LEFT JOIN companion_session_archives a
                    ON a.session_id = s.session_id
                WHERE s.user_id = %s AND s.session_id = %s
                """,
                (uid, session_id),
            )
            row = cur.fetchone()
    finally:
        pool.putconn(conn)
    if row is None:
        raise_typed(PlatformError.companionsessions("not_found", "snapshot not found"))
    sid, name, payload, updated, venue, state, archived_at = row
    return {
        "session_id": sid,
        "name": name,
        "payload": payload,
        "updated_at": updated,
        "venue": venue,
        "state": "archived" if archived_at else state,
        "archived_at": str(archived_at) if archived_at else "",
    }


@router.delete("/{session_id}", response_model=dict[str, Any])
async def delete_session_snapshot(session_id: str, request: Request) -> dict[str, Any]:
    """Delete one mirrored snapshot for the owner (WP-04 G-P6-1 round 2).
    # auth: verified

    Single txn drops the archive tombstone AND the snapshot row, so a
    client-side ``delete`` can never be undone by reconcile ``restored``
    semantics. 404 for others/absent — ownership is never revealed.
    """
    uid = _uid(request)
    pool = _pool(request)
    conn = pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM companion_session_archives WHERE session_id = %s",
                (session_id,),
            )
            cur.execute(
                """
                DELETE FROM companion_session_snapshots
                WHERE user_id = %s AND session_id = %s
                """,
                (uid, session_id),
            )
            deleted = cur.rowcount > 0
            if deleted:
                conn.commit()
            else:
                conn.rollback()
    finally:
        pool.putconn(conn)
    if not deleted:
        raise_typed(PlatformError.companionsessions("not_found", "snapshot not found"))
    return {"status": "ok", "session_id": session_id, "deleted": True}


def _list_where(
    uid: str,
    venues: list[str],
    states: list[str],
    include_archived: bool,
) -> tuple[str, list[Any]]:
    """Build the WHERE tail + params for the listing route."""
    clauses = ["user_id = %s"]
    params: list[Any] = [uid]
    if venues:
        clauses.append("venue = ANY(%s)")
        params.append(venues)
    if states:
        clauses.append("computed_state = ANY(%s)")
        params.append(states)
    if not include_archived:
        clauses.append("archived_at IS NULL")
    return " AND ".join(clauses), params


@router.get("", response_model=dict[str, Any])
async def list_session_snapshots(
    request: Request,
    limit: int = 50,
    venue: str | None = None,
    state: str | None = None,
    include_archived: bool = False,
    project: str | None = None,
) -> dict[str, Any]:
    """List my snapshots (summary rows only, no payloads).
    # auth: verified

    Filters compose server-side: venue csv (local/frontend/cluster:<name>),
    state csv (ongoing/stopped/completed/error/archived), project id, and
    archive inclusion. 'archived' is computed from the tombstone table.
    """
    uid = _uid(request)
    pool = _pool(request)
    limit = min(max(limit, 1), 200)
    venues = [_validate_venue(v) for v in _csv_filter(venue)]
    states = _csv_filter(state)
    invalid = [s for s in states if s not in _STATE_COMPUTED]
    if invalid:
        raise_typed(
            PlatformError.companionsessions(
                "bad_request", f"invalid state filter: {invalid}"
            )
        )
    where, params = _list_where(uid, venues, states, include_archived)
    project_clause = ""
    if project:
        project_clause = (
            " AND payload LIKE '{%'"
            " AND COALESCE(payload::jsonb -> 'metadata' -> 'config' ->> 'project_id',"
            "             payload::jsonb -> 'metadata' ->> 'project_id') = %s"
        )
        params.append(project)
    conn = pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT session_id, name, update_marker, venue, computed_state,
                       archived_at
                FROM (
                    SELECT s.session_id, s.name, s.update_marker, s.venue, s.state,
                           s.payload, a.archived_at,
                           CASE WHEN a.session_id IS NOT NULL THEN 'archived'
                                ELSE s.state END AS computed_state,
                           s.user_id
                    FROM companion_session_snapshots s
                    LEFT JOIN companion_session_archives a
                        ON a.session_id = s.session_id
                ) t
                WHERE {where}{project_clause}
                ORDER BY update_marker DESC
                LIMIT %s
                """,
                (*params, limit),
            )
            rows = cur.fetchall()
    finally:
        pool.putconn(conn)
    snapshots = [
        {
            "session_id": sid,
            "name": name,
            "updated_at": upd,
            "venue": venue_value,
            "state": computed_state,
            "archived_at": str(archived_at) if archived_at else "",
        }
        for sid, name, upd, venue_value, computed_state, archived_at in rows
    ]
    return {"snapshots": snapshots, "count": len(snapshots)}


__all__ = ["router"]
