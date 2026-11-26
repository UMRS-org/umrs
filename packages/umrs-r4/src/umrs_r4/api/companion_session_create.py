"""Server-authored companion session create (UI plan §3.4, lane UX-V2, ADR-119).

    POST /umrs/companion/sessions
        body  {"name"?, "model"?, "project"?, "mode"?: chat|plan|autopilot,
               "venue"?: "frontend"}
        201   the detail row (``GET /umrs/companion/sessions/{sid}`` shape)

Until now only a client made a session: it saved its own row and mirrored
it here. A browser has no local store, so the server mints the id and
writes the first journal row itself, in the payload shape r4's
``CompanionSessionStore.save()`` writes (JSON-string ``messages``,
``active_skills``, ``memory`` and ``metadata``), so the venue worker's
bootstrap (``GET …/rail/snapshot`` → ``adopt_remote_detail`` → ``load``)
takes it like any mirrored session. The caller then starts it with
``POST …/{sid}/venue/start``. The row goes in with the revision compare-and-
swap at base 0 (ADR-057 D-7), so it never overwrites an existing row.

Only the front's venue is offered: ``local`` names the caller's own machine,
which a server cannot author, and the cluster and cpu venues have no
browser path yet.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field
from umrs.api.typed_errors import PlatformError, raise_typed

from umrs_r4.api.companion_sessions import (
    SessionSnapshotBody,
    _pool,
    _uid,
    snapshot_detail_row,
    write_snapshot_row,
)
from umrs_r4.api.companion_venue import _checked_start_project

router = APIRouter(prefix="/umrs/companion/sessions", tags=["companion-sessions"])


class SessionCreateBody(BaseModel):
    """What a client chooses when the server makes a session."""

    name: str = Field(default="", max_length=256)
    model: str = Field(default="", max_length=256)
    project: str = Field(default="", max_length=128)
    mode: Literal["chat", "plan", "autopilot"] = "chat"
    venue: Literal["frontend"] = "frontend"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def session_payload(session_id: str, body: SessionCreateBody, project: str, now: str) -> str:
    """The first journal payload, in ``CompanionSessionStore.save()``'s shape."""
    name = body.name.strip() or f"Session {now[:16].replace('T', ' ')}"
    metadata: dict[str, str] = {"venue": body.venue, "interaction_mode": body.mode}
    if project:
        metadata["project_id"] = project
    return json.dumps(
        {
            "id": session_id,
            "name": name,
            "created_at": now,
            "updated_at": now,
            "messages": "[]",
            "active_skills": "[]",
            "memory": "{}",
            "metadata": json.dumps(metadata),
            "model": body.model.strip(),
        }
    )


def create_session_row(
    conn: Any, uid: str, body: SessionCreateBody, project: str
) -> dict[str, Any]:
    """Write a new session's first row; the detail row it now reads as."""
    session_id = str(uuid.uuid4())
    now = _now_iso()
    payload = session_payload(session_id, body, project, now)
    write_snapshot_row(
        conn,
        uid,
        session_id,
        SessionSnapshotBody(
            name=json.loads(payload)["name"],
            payload=payload,
            updated_at=now,
            venue=body.venue,
            state="",
            base_revision=0,
        ),
    )
    detail = snapshot_detail_row(conn, uid, session_id)
    if detail is None:
        raise_typed(PlatformError.companionsessions("unavailable", "created row did not read back"))
    return detail


@router.post("", status_code=201, response_model=dict[str, Any])
async def create_session(body: SessionCreateBody, request: Request) -> dict[str, Any]:
    """Make a session on the server (see the module docstring).
    # auth: verified
    """
    uid = _uid(request)
    pool = _pool(request)
    # The verified caller's own project when none is named (as a venue start).
    project = _checked_start_project(request, body.project) or ""

    def _write() -> dict[str, Any]:
        conn = pool.getconn()
        try:
            return create_session_row(conn, uid, body, project)
        finally:
            pool.putconn(conn)

    return await asyncio.to_thread(_write)
