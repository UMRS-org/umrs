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

import json
import logging
import re
from datetime import datetime, timezone
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
                -- F3 audit fix: comparing raw strings mixed naive-vs-tz-aware writer classes;
                -- timestamptz normalizes both (epoch fallback for empty/unparseable rows).
                WHERE COALESCE(
                    NULLIF(trim(companion_session_snapshots.update_marker), '')::timestamptz,
                    'epoch'::timestamptz
                ) <= COALESCE(
                    NULLIF(trim(EXCLUDED.update_marker), '')::timestamptz,
                    'epoch'::timestamptz
                )
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


# ── Turn accessor (ADR-057 D-2) ────────────────────────────────────────────


def _blob_messages(payload_blob: str) -> list[dict[str, Any]]:
    """JSON-parse blob → flat message dicts (no fabrication; malformed → [])."""
    import json

    try:
        blob = json.loads(payload_blob or "{}")
    except (TypeError, ValueError):
        return []
    messages = blob.get("messages")
    if isinstance(messages, str):
        try:
            messages = json.loads(messages)
        except (TypeError, ValueError):
            messages = []
    if not isinstance(messages, list):
        return []
    return [m for m in messages if isinstance(m, dict)]


def _turn_entries(payload_blob: str) -> list[dict[str, Any]]:
    """Project turn-index entries from a snapshot payload blob.

    Contract: only entries carrying ``metadata.turn_id`` are turns (ADR-057
    D-1 assignment stamps users/assistants at commit; older mirrored rows
    without the marker keep `turn_id=""` honestly — never synthesized).
    """
    entries: list[dict[str, Any]] = []
    for idx, m in enumerate(_blob_messages(payload_blob)):
        meta = m.get("metadata") or {}
        turn_id = str(meta.get("turn_id") or "")
        if not turn_id:
            continue
        entries.append(
            {
                "turn_id": turn_id,
                "index": idx,
                "role": m.get("role", ""),
                "committed_at": str(m.get("timestamp") or ""),
            }
        )
    return entries


def _turn_by_id(entries: list[dict[str, Any]], turn_id: str) -> dict[str, Any] | None:
    if not turn_id:
        return None
    return next((e for e in entries if e["turn_id"] == turn_id), None)


def _session_payload(conn: Any, session_id: str, owner: str) -> str | None:
    cur = conn.cursor()
    # F1 audit-fix (2026-09-21): the table's column is `user_id` — the legacy
    # `owner_id` read broke turns/detail under PG live (constraints never saw
    # it in pk-fake test rails). Never invent the wrong column name.
    cur.execute(
        "SELECT payload FROM companion_session_snapshots WHERE session_id = %s AND user_id = %s",
        (session_id, owner),
    )
    row = cur.fetchone()
    return row[0] if row else None


@router.get("/{session_id}/turns", response_model=dict[str, Any])
async def list_turns(session_id: str, request: Request) -> dict[str, Any]:
    """Ordered turn index for one session (ADR-057 D-2).
    # auth: verified
    """
    uid = _uid(request)
    pool = _pool(request)
    conn = pool.getconn()
    try:
        blob = _session_payload(conn, session_id, uid)
        if blob is None:
            raise_typed(
                PlatformError.companionsessions("not_found", "snapshot not found")
            )
        entries = _turn_entries(blob)
        return {
            "session_id": session_id,
            "entries": entries,
            "count": len(entries),
        }
    finally:
        pool.putconn(conn)


class QueueActionBody(BaseModel):
    """F9 design wave: server-side queue verb body (append | pop | clear)."""
    action: str
    content: str = ""
    client_id: str = ""
    updated_at: str = ""


@router.post("/{session_id}/queue", response_model=dict[str, Any])
async def queue_mutation(session_id: str, body: QueueActionBody, request: Request) -> dict[str, Any]:
    """Transactional alive-mutate for session.metadata.chat_queue (owner-scope +
     cross-client honor over the same "mirror's own journal"). Writers carry
    their client_id stamp so resumes always know which device wrote which row.
    Optimistic-concurrency coordination via FOR UPDATE (never fabricated side
    effects).
    """
    uid = _uid(request)
    pool = _pool(request)
    action = (body.action or "").strip().lower()
    if action not in {"append", "pop", "clear"}:
        raise_typed(
            PlatformError.companionsessions("invalid_action", f"action must be append|pop|clear, got {action!r}")
        )
    conn = pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT payload FROM companion_session_snapshots
                WHERE user_id = %s AND session_id = %s
                FOR UPDATE
                """,
                (uid, session_id),
            )
            row = cur.fetchone()
            if row is None:
                raise_typed(
                    PlatformError.companionsessions("not_found", "snapshot not found")
                )
            try:
                blob = json.loads(row[0]) if isinstance(row[0], str) else row[0]
            except json.JSONDecodeError:
                blob = {}
            metadata = blob.get("metadata") if isinstance(blob, dict) else {}
            if not isinstance(metadata, dict):
                metadata = {}
            entries = metadata.get("chat_queue") or []
            if not isinstance(entries, list):
                entries = []
            now = body.updated_at or datetime.now(timezone.utc).isoformat()
            if action == "append":
                content = (body.content or "").strip()
                if not content:
                    raise_typed(
                        PlatformError.companionsessions("invalid_value", "content is required for append")
                    )
                client_id = (body.client_id or "").strip() or uid or "unknown"
                entries.append({"id": f"q-{len(entries)+1}", "content": content, "client_id": client_id, "added_at": now})
                front = None
            elif action == "pop":
                front = entries.pop(0) if entries else None
            else:
                front = len(entries)
                entries = []
            blob["metadata"] = metadata
            metadata["chat_queue"] = entries
            cur.execute(
                """
                UPDATE companion_session_snapshots
                SET payload = %s, updated_at = now()
                WHERE user_id = %s AND session_id = %s
                """,
                (json.dumps(blob), uid, session_id),
            )
            applied = cur.rowcount > 0
            if applied:
                conn.commit()
            else:
                conn.rollback()
        return {
            "applied": applied,
            "queue": entries,
            "front": front,
        }
    finally:
        pool.putconn(conn)
async def get_turn(session_id: str, turn_id: str, request: Request) -> dict[str, Any]:
    """One turn's full payload (ADR-057 D-2).
    # auth: verified
    """
    uid = _uid(request)
    pool = _pool(request)
    conn = pool.getconn()
    try:
        blob = _session_payload(conn, session_id, uid)
        if blob is None:
            raise_typed(
                PlatformError.companionsessions("not_found", "snapshot not found")
            )
        entry = _turn_by_id(_turn_entries(blob), turn_id)
        if entry is None:
            raise_typed(
                PlatformError.companionsessions(
                    "not_found", f"turn {turn_id} not found in session"
                )
            )
        return {"session_id": session_id, "entry": entry}
    finally:
        pool.putconn(conn)


@router.get("/{session_id}", response_model=dict[str, Any])
async def get_session_snapshot(session_id: str, request: Request) -> dict[str, Any]:
    """Answer a mirrored snapshot for the owner (404 for others/absent).
    # auth: verified
    """
    # auth: verified
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
            # F5 audit-block (2026-09-21): tombstone-drop now rides the caller's
            # own grant column (archived_by matching), never an unscoped id —
            # a legit one's own session delete never strips another user's
            # same-id tombstone.
            cur.execute(
                "DELETE FROM companion_session_archives WHERE session_id = %s AND archived_by = %s",
                (session_id, uid),
            )
            cur.execute(
                """
                DELETE FROM companion_session_snapshots
                WHERE user_id = %s AND session_id = %s
                """,
                (uid, session_id),
            )
            deleted = cur.rowcount > 0
            # F15 audit-block (2026-09-21): approvals rows for this session
            # also die in the same transaction (dangling intents otherwise
            # live forever — cascade kill).
            cur.execute(
                "DELETE FROM chat_turn_approvals WHERE session_id = %s AND user_id = %s",
                (session_id, uid),
            )
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
