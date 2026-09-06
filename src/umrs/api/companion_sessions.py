"""Companion session snapshots (ADR-036 F2 first leg).

Client companion sessions mirror their state here (write-through). A fresh
machine answers cross-client session presence/read by pulling these rows;
the companion SQLite store is the rebuildable read-model over them.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/umrs/companion/sessions", tags=["companion-sessions"])


class SessionSnapshotBody(BaseModel):
    """The mirrored companion session document."""

    name: str = Field(default="", max_length=256)
    updated_at: str = Field(default="")
    payload: str = Field(default="")


# ── Auth + pool helpers ────────────────────────────────────────────────────


def _uid(request: Request) -> str:
    """Owner identity, as asserted by the gateway-verified header."""
    uid = request.headers.get("x-user-id", "").strip()
    if not uid:
        raise HTTPException(status_code=401, detail="identity header missing")
    return uid


def _pool(request: Request) -> Any:
    pool = getattr(request.app.state, "umrs_db_pool", None)
    if pool is None:
        raise HTTPException(status_code=503, detail="database pool unavailable")
    return pool


# ── Routes ─────────────────────────────────────────────────────────────────


@router.post("/{session_id}/snapshot", response_model=dict[str, Any])
async def put_session_snapshot(
    session_id: str, body: SessionSnapshotBody, request: Request
) -> dict[str, Any]:
    """Mirror one client session state. Newer updates supersede older ones."""
    uid = _uid(request)
    pool = _pool(request)
    conn = pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO companion_session_snapshots
                    (user_id, session_id, name, payload, update_marker)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (user_id, session_id) DO UPDATE SET
                    name = EXCLUDED.name,
                    payload = EXCLUDED.payload,
                    update_marker = EXCLUDED.update_marker,
                    updated_at = now()
                WHERE companion_session_snapshots.update_marker <= EXCLUDED.update_marker
                """,
                (uid, session_id, body.name, body.payload, body.updated_at),
            )
            applied = cur.rowcount > 0
            conn.commit()
    finally:
        pool.putconn(conn)
    return {"status": "ok", "applied": applied}


@router.get("/{session_id}", response_model=dict[str, Any])
async def get_session_snapshot(session_id: str, request: Request) -> dict[str, Any]:
    """Answer a mirrored snapshot for the owner (404 for others/absent)."""
    uid = _uid(request)
    pool = _pool(request)
    conn = pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT session_id, name, payload, update_marker
                FROM companion_session_snapshots
                WHERE user_id = %s AND session_id = %s
                """,
                (uid, session_id),
            )
            row = cur.fetchone()
    finally:
        pool.putconn(conn)
    if row is None:
        raise HTTPException(status_code=404, detail="snapshot not found")
    sid, name, payload, updated = row
    return {"session_id": sid, "name": name, "payload": payload, "updated_at": updated}


@router.get("", response_model=dict[str, Any])
async def list_session_snapshots(request: Request, limit: int = 50) -> dict[str, Any]:
    """List my snapshots (summary rows only, no payloads)."""
    uid = _uid(request)
    pool = _pool(request)
    limit = min(max(limit, 1), 200)
    conn = pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT session_id, name, update_marker
                FROM companion_session_snapshots
                WHERE user_id = %s
                ORDER BY updated_at DESC
                LIMIT %s
                """,
                (uid, limit),
            )
            rows = cur.fetchall()
    finally:
        pool.putconn(conn)
    snapshots = [
        {"session_id": sid, "name": name, "updated_at": upd} for sid, name, upd in rows
    ]
    return {"snapshots": snapshots, "count": len(snapshots)}


__all__ = ["router"]
