"""Circles — project-scoped context + audience cells (ADR-086) — store + routes.

Supersedes the platform-global groups authority (0097 / ADR-050 §3): a
circle lives INSIDE one project and is both the boundary that keeps the
contexts of works apart and the smallest addressable sharing audience
(grammar kind ``circle:<project_id>/<circle_id>``, ADR-080 one guard).

- ``circles`` rows are owner-attributed and tombstoned (retire never
  DELETEs); a retired circle stops resolving for access lookups immediately.
- ``circle_members`` rows are presence-only: membership is the current
  fact, not history. ``role`` is 'member' or 'admin' — circle admins manage
  the roster beside the owner.
- Personal circle: ``circle_id == owner_id`` is the caller's private cell
  in that project; it never gains other members and is the way a user
  "enters" a project (creation is casual, ADR-051 parity).
- Project affiliation derives from circle membership: a user's projects
  are the projects holding at least one of their circles — no
  users.projects list.

psycopg2-shaped sync pool (satellite pattern); store is import-free of
umrs.api internals beyond the shared typed-error helpers.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request, status
from pydantic import BaseModel, Field

from .jobs import CallerIdentity, get_caller_identity
from .typed_errors import PlatformError, raise_typed


class CircleNotFound(Exception):
    """Raised when a circle id does not exist (never silent)."""


class CirclesStore:
    """psycopg2-backed durable store over the 0119 registry."""

    def __init__(self, conn: Any) -> None:
        self._conn = conn

    # ── circles ──────────────────────────────────────────────────────────
    def _row(self, project_id: str, circle_id: str) -> dict[str, Any] | None:
        cur = self._conn.cursor()
        cur.execute(
            "SELECT project_id, circle_id, name, owner_id, status, retired_at,"
            " created_at, updated_at FROM circles"
            " WHERE project_id = %s AND circle_id = %s",
            (project_id, circle_id),
        )
        row = cur.fetchone()
        if not row:
            return None
        return _circle_row(row)

    def create(
        self, project_id: str, circle_id: str, *, name: str = "", owner_id: str
    ) -> dict[str, Any]:
        """Ensuring-shape: existing id returns as-is, never stomps."""
        pid, cid = project_id.strip(), circle_id.strip()
        if not pid or not cid:
            raise ValueError("project_id and circle_id must be non-empty")
        existing = self._row(pid, cid)
        if existing:
            return existing
        cur = self._conn.cursor()
        cur.execute(
            "INSERT INTO circles (project_id, circle_id, name, owner_id)"
            " VALUES (%s, %s, %s, %s)",
            (pid, cid, name, owner_id),
        )
        # the creator starts as the first member, with the admin role
        # (roster self-governance; the owner may delegate via member roles)
        cur.execute(
            "INSERT INTO circle_members (project_id, circle_id, user_id, role,"
            " added_by) VALUES (%s, %s, %s, 'admin', %s)",
            (pid, cid, owner_id, owner_id),
        )
        self._conn.commit()
        return self._row(pid, cid)

    def get(self, project_id: str, circle_id: str) -> dict[str, Any]:
        row = self._row(project_id, circle_id)
        if row is None:
            raise CircleNotFound(f"{project_id}/{circle_id}")
        return row

    def list_for_project(
        self, project_id: str, *, include_retired: bool = False
    ) -> list[dict[str, Any]]:
        cur = self._conn.cursor()
        sql = (
            "SELECT project_id, circle_id, name, owner_id, status, retired_at,"
            " created_at, updated_at FROM circles WHERE project_id = %s"
        )
        params: list[Any] = [project_id]
        if not include_retired:
            sql += " AND status <> 'retired'"
        sql += " ORDER BY created_at"
        cur.execute(sql, tuple(params))
        return [_circle_row(r) for r in cur.fetchall()]

    def list_mine(
        self, user_id: str, *, include_retired: bool = False
    ) -> list[dict[str, Any]]:
        """Circles the user OWNS or is a member of (across all projects)."""
        cur = self._conn.cursor()
        sql = (
            "SELECT DISTINCT c.project_id, c.circle_id, c.name, c.owner_id,"
            " c.status, c.retired_at, c.created_at, c.updated_at FROM circles c"
            " LEFT JOIN circle_members cm ON cm.project_id = c.project_id"
            " AND cm.circle_id = c.circle_id"
            " WHERE (c.owner_id = %s OR cm.user_id = %s)"
        )
        params: list[Any] = [user_id, user_id]
        if not include_retired:
            sql += " AND c.status <> 'retired'"
        sql += " ORDER BY c.project_id, c.created_at"
        cur.execute(sql, tuple(params))
        return [_circle_row(r) for r in cur.fetchall()]

    def retire(self, project_id: str, circle_id: str) -> dict[str, Any]:
        cur = self._conn.cursor()
        cur.execute(
            "UPDATE circles SET status = 'retired', retired_at = now(),"
            " updated_at = now()"
            " WHERE project_id = %s AND circle_id = %s AND status <> 'retired'",
            (project_id, circle_id),
        )
        self._conn.commit()
        return self.get(project_id, circle_id)

    # ── membership ───────────────────────────────────────────────────────
    def add_member(
        self,
        project_id: str,
        circle_id: str,
        user_id: str,
        *,
        role: str = "member",
        added_by: str,
    ) -> dict[str, Any]:
        """Presence insert, idempotent — re-adding is a no-op per PK."""
        uid = user_id.strip()
        if not uid:
            raise ValueError("user_id must be non-empty")
        if role not in ("member", "admin"):
            raise ValueError(f"role must be 'member' or 'admin', got {role!r}")
        cur = self._conn.cursor()
        cur.execute(
            "INSERT INTO circle_members (project_id, circle_id, user_id, role,"
            " added_by) VALUES (%s, %s, %s, %s, %s)"
            " ON CONFLICT (project_id, circle_id, user_id) DO NOTHING",
            (project_id, circle_id, uid, role, added_by),
        )
        self._conn.commit()
        return self.get(project_id, circle_id)

    def remove_member(self, project_id: str, circle_id: str, user_id: str) -> None:
        cur = self._conn.cursor()
        cur.execute(
            "DELETE FROM circle_members"
            " WHERE project_id = %s AND circle_id = %s AND user_id = %s",
            (project_id, circle_id, user_id),
        )
        self._conn.commit()

    def members(self, project_id: str, circle_id: str) -> list[dict[str, Any]]:
        cur = self._conn.cursor()
        cur.execute(
            "SELECT user_id, role, added_by, added_at FROM circle_members"
            " WHERE project_id = %s AND circle_id = %s ORDER BY added_at",
            (project_id, circle_id),
        )
        return [
            {
                "user_id": r[0],
                "role": r[1],
                "added_by": r[2],
                "added_at": r[3].isoformat() if r[3] else None,
            }
            for r in cur.fetchall()
        ]

    def member_role(self, project_id: str, circle_id: str, user_id: str) -> str | None:
        """The user's role in the circle, or None when not a member."""
        cur = self._conn.cursor()
        cur.execute(
            "SELECT role FROM circle_members"
            " WHERE project_id = %s AND circle_id = %s AND user_id = %s",
            (project_id, circle_id, user_id),
        )
        row = cur.fetchone()
        return str(row[0]) if row else None

    def affiliated(self, project_id: str, user_id: str) -> bool:
        """True when the user holds any active-circle membership in the project."""
        cur = self._conn.cursor()
        cur.execute(
            "SELECT 1 FROM circle_members cm"
            " JOIN circles c ON c.project_id = cm.project_id"
            " AND c.circle_id = cm.circle_id"
            " WHERE cm.project_id = %s AND cm.user_id = %s"
            " AND c.status = 'active' LIMIT 1",
            (project_id, user_id),
        )
        return cur.fetchone() is not None

    # ── containing project ───────────────────────────────────────────────
    def project_row(self, project_id: str) -> dict[str, Any] | None:
        """The containing project's row (0095 registry) or None."""
        cur = self._conn.cursor()
        cur.execute(
            "SELECT project_id, owner_id, status FROM projects WHERE project_id = %s",
            (project_id,),
        )
        row = cur.fetchone()
        if not row:
            return None
        return {"project_id": row[0], "owner_id": row[1], "status": row[2]}


def _circle_row(row: Any) -> dict[str, Any]:
    return {
        "project_id": row[0],
        "circle_id": row[1],
        "name": row[2],
        "owner_id": row[3],
        "status": row[4],
        "retired_at": row[5].isoformat() if row[5] else None,
        "created_at": row[6].isoformat() if row[6] else None,
        "updated_at": row[7].isoformat() if row[7] else None,
    }


def circles_for_user(conn: Any, user_id: str) -> list[str]:
    """Active circle refs (the access-guard resolution, ADR-080).

    Retired circles never resolve — their access entries stop counting
    immediately. The SQL authority is
    ``umrs.shared.circle_membership.circle_refs_for_user``; this wrapper
    keeps the established api import surface.
    """
    from umrs.shared.circle_membership import circle_refs_for_user

    return circle_refs_for_user(conn, user_id)


# ── routes ───────────────────────────────────────────────────────────────

router = APIRouter(prefix="/umrs/circles", tags=["circles"])


class CircleCreateBody(BaseModel):
    project_id: str = Field(..., min_length=1, max_length=200)
    circle_id: str = Field(..., min_length=1, max_length=200)
    name: str = Field(default="", max_length=500)
    for_user: str | None = None


class MemberBody(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=200)
    role: str = Field(default="member", max_length=20)


def _store_dep(request: Request):
    """FastAPI generator dependency — pool conn is always returned on every
    path (the 2026-09-09 exhaustion class; plain factories leaked)."""
    pool = getattr(request.app.state, "umrs_db_pool", None)
    if pool is None:
        raise_typed(
            PlatformError.circles(
                "unavailable", "circles store unavailable — shared umrs_db pool absent"
            )
        )
    conn = pool.getconn()
    try:
        yield CirclesStore(conn)
    finally:
        pool.putconn(conn)


def _row_response(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "project_id": row["project_id"],
        "circle_id": row["circle_id"],
        "name": row.get("name", ""),
        "owner_id": row["owner_id"],
        "status": row["status"],
        "retired_at": row.get("retired_at"),
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
    }


def _is_personal(row: dict[str, Any]) -> bool:
    """A circle named after its owner is the owner's private cell."""
    return row["circle_id"] == row["owner_id"]


def _manager_or_admin(
    identity: CallerIdentity, row: dict[str, Any], store: CirclesStore
) -> None:
    """Roster mutation gate: platform admin, circle owner, or circle admin."""
    actor = (identity.user_id or "").strip()
    if identity.is_admin or actor == row["owner_id"]:
        return
    if store.member_role(row["project_id"], row["circle_id"], actor) == "admin":
        return
    raise_typed(
        PlatformError.circles(
            "forbidden",
            "only the circle owner, a circle admin, or a platform admin"
            " may manage this roster",
        )
    )


def _owner_or_admin(identity: CallerIdentity, row: dict[str, Any]) -> None:
    actor = (identity.user_id or "").strip()
    if identity.is_admin or actor == row["owner_id"]:
        return
    raise_typed(
        PlatformError.circles(
            "forbidden", "only the circle owner or an admin may perform this"
        )
    )


def _project_or_404(store: CirclesStore, project_id: str) -> dict[str, Any]:
    row = store.project_row(project_id)
    if row is None:
        raise_typed(
            PlatformError.circles(
                "not_found", f"project {project_id} not found — circles need a project"
            )
        )
    return row


@router.get("/mine")
def list_my_circles(
    include_retired: bool = False,
    store: CirclesStore = Depends(_store_dep),
    identity: CallerIdentity = Depends(get_caller_identity),
) -> dict[str, Any]:
    """Every circle the caller owns or belongs to, plus the derived
    project list (projects = projects holding at least one circle of mine).
    # auth: verified
    """
    rows = store.list_mine(
        (identity.user_id or "").strip(), include_retired=include_retired
    )
    projects = sorted({r["project_id"] for r in rows})
    return {
        "status": "ok",
        "circles": [_row_response(r) for r in rows],
        "count": len(rows),
        "projects": projects,
    }


@router.post("", status_code=status.HTTP_201_CREATED)
def create_circle(
    body: CircleCreateBody,
    store: CirclesStore = Depends(_store_dep),
    identity: CallerIdentity = Depends(get_caller_identity),
) -> dict[str, Any]:
    """Ensuring-shape creation (idempotent — existing row returned as-is).

    Casual parity with projects: a caller creates their personal circle
    (``circle_id == caller``) in any project — that IS entering the
    project. A named circle requires the caller to be the project owner,
    an admin, or already affiliated with the project (member of any of
    its circles). Minting for another user is admin-only.
    # auth: verified
    """
    actor = (identity.user_id or "").strip()
    owner = _circle_owner(body, actor, identity.is_admin)
    project = _live_project_or_raise(store, body.project_id)
    _require_create_right(
        store, body, project, actor=actor, owner=owner, is_admin=identity.is_admin
    )
    row = store.create(body.project_id, body.circle_id, name=body.name, owner_id=owner)
    return {"status": "ok", "circle": _row_response(row), "created": True}


def _circle_owner(body: CircleCreateBody, actor: str, is_admin: bool) -> str:
    """The circle's owner: the caller, or ``for_user`` when an admin mints it."""
    if body.for_user and body.for_user != actor and not is_admin:
        raise_typed(
            PlatformError.circles(
                "forbidden", "minting circles for another user requires the admin role"
            )
        )
    owner = body.for_user or actor
    if not owner:
        raise_typed(PlatformError.circles("bad_request", "caller identity required"))
    return owner


def _live_project_or_raise(store: CirclesStore, project_id: str) -> dict[str, Any]:
    project = _project_or_404(store, project_id)
    if project["status"] == "retired":
        raise_typed(
            PlatformError.circles("conflict", f"project {project_id} is retired")
        )
    return project


def _require_create_right(
    store: CirclesStore,
    body: CircleCreateBody,
    project: dict[str, Any],
    *,
    actor: str,
    owner: str,
    is_admin: bool,
) -> None:
    """Personal circle, admin, project owner, or an affiliated caller."""
    allowed = (
        body.circle_id == owner
        or is_admin
        or actor == project["owner_id"]
        or store.affiliated(body.project_id, actor)
    )
    if not allowed:
        raise_typed(
            PlatformError.circles(
                "forbidden",
                "a named circle requires the project owner, an admin, or an"
                " affiliated caller — create your personal circle first",
            )
        )


@router.get("/{project_id}")
def list_project_circles(
    project_id: str,
    include_retired: bool = False,
    store: CirclesStore = Depends(_store_dep),
    identity: CallerIdentity = Depends(get_caller_identity),
) -> dict[str, Any]:
    """Circles of the project. Affiliated callers, the project owner and
    admins see the roster (discoverability for sharing); unaffiliated
    callers see only their own rows in the project.
    # auth: verified
    """
    project = _project_or_404(store, project_id)
    actor = (identity.user_id or "").strip()
    privileged = (
        identity.is_admin
        or actor == project["owner_id"]
        or store.affiliated(project_id, actor)
    )
    if privileged:
        rows = store.list_for_project(project_id, include_retired=include_retired)
    else:
        rows = [
            r
            for r in store.list_mine(actor, include_retired=include_retired)
            if r["project_id"] == project_id
        ]
    return {
        "status": "ok",
        "project_id": project_id,
        "circles": [_row_response(r) for r in rows],
        "count": len(rows),
    }


def _may_see_circle(
    store: CirclesStore, row: dict[str, Any], identity: CallerIdentity
) -> bool:
    """The project listing's rule (admin, project owner, affiliated caller)
    plus the circle's own owner and members (a retired circle included)."""
    actor = (identity.user_id or "").strip()
    if identity.is_admin or actor == row["owner_id"]:
        return True
    project = store.project_row(row["project_id"]) or {}
    if actor == project.get("owner_id") or store.affiliated(row["project_id"], actor):
        return True
    return store.member_role(row["project_id"], row["circle_id"], actor) is not None


def _visible_circle(
    store: CirclesStore, project_id: str, circle_id: str, identity: CallerIdentity
) -> dict[str, Any]:
    """The circle when the caller may see it; otherwise "not found" (H-1,
    2026-10-17: the circle and roster reads took no identity at all)."""
    try:
        row = store.get(project_id, circle_id)
    except CircleNotFound:
        row = None
    if row is None or not _may_see_circle(store, row, identity):
        raise_typed(
            PlatformError.circles(
                "not_found", f"circle {project_id}/{circle_id} not found"
            )
        )
    return row


@router.get("/{project_id}/{circle_id}")
def get_circle(
    project_id: str,
    circle_id: str,
    store: CirclesStore = Depends(_store_dep),
    identity: CallerIdentity = Depends(get_caller_identity),
) -> dict[str, Any]:
    """One circle — admin, project owner, affiliated caller, circle member.
    # auth: verified
    """
    row = _visible_circle(store, project_id, circle_id, identity)
    return {"status": "ok", "circle": _row_response(row)}


@router.patch("/{project_id}/{circle_id}")
def patch_circle(
    project_id: str,
    circle_id: str,
    body: dict[str, str],
    store: CirclesStore = Depends(_store_dep),
    identity: CallerIdentity = Depends(get_caller_identity),
) -> dict[str, Any]:
    """Only 'retire' this round — owner-or-admin; tombstone, never DELETE.
    # auth: verified
    """
    try:
        row = store.get(project_id, circle_id)
    except CircleNotFound:
        raise_typed(
            PlatformError.circles(
                "not_found", f"circle {project_id}/{circle_id} not found"
            )
        )
    if body.get("action") != "retire":
        raise_typed(
            PlatformError.circles(
                "bad_request", "action must be 'retire' for the current round"
            )
        )
    _owner_or_admin(identity, row)
    row = store.retire(project_id, circle_id)
    return {"status": "ok", "circle": _row_response(row)}


@router.get("/{project_id}/{circle_id}/members")
def list_members(
    project_id: str,
    circle_id: str,
    store: CirclesStore = Depends(_store_dep),
    identity: CallerIdentity = Depends(get_caller_identity),
) -> dict[str, Any]:
    """The roster — same visibility as the circle itself.
    # auth: verified
    """
    _visible_circle(store, project_id, circle_id, identity)
    members = store.members(project_id, circle_id)
    return {
        "status": "ok",
        "project_id": project_id,
        "circle_id": circle_id,
        "members": members,
        "count": len(members),
    }


@router.post("/{project_id}/{circle_id}/members", status_code=status.HTTP_201_CREATED)
def add_member(
    project_id: str,
    circle_id: str,
    body: MemberBody,
    store: CirclesStore = Depends(_store_dep),
    identity: CallerIdentity = Depends(get_caller_identity),
) -> dict[str, Any]:
    """Owner / circle-admin / platform-admin adds a member (idempotent,
    presence-only). Personal circles never gain other members.
    # auth: verified
    """
    try:
        row = store.get(project_id, circle_id)
    except CircleNotFound:
        raise_typed(
            PlatformError.circles(
                "not_found", f"circle {project_id}/{circle_id} not found"
            )
        )
    if row["status"] == "retired":
        raise_typed(
            PlatformError.circles(
                "conflict", "cannot grow membership of a retired circle"
            )
        )
    if _is_personal(row):
        raise_typed(
            PlatformError.circles(
                "conflict",
                "a personal circle (circle_id = owner) never gains other"
                " members — create a named circle to collaborate",
            )
        )
    _manager_or_admin(identity, row, store)
    try:
        out = store.add_member(
            project_id,
            circle_id,
            body.user_id,
            role=body.role,
            added_by=(identity.user_id or "").strip() or "unknown",
        )
    except ValueError as exc:
        raise_typed(PlatformError.circles("bad_request", str(exc)), cause=exc)
    return {
        "status": "ok",
        "circle": _row_response(out),
        "members": store.members(project_id, circle_id),
    }


@router.delete("/{project_id}/{circle_id}/members/{user_id}")
def remove_member(
    project_id: str,
    circle_id: str,
    user_id: str,
    store: CirclesStore = Depends(_store_dep),
    identity: CallerIdentity = Depends(get_caller_identity),
) -> dict[str, Any]:
    """Owner / circle-admin / platform-admin removes a member (presence
    removal; no tombstone). The circle owner is never removed.
    # auth: verified
    """
    try:
        row = store.get(project_id, circle_id)
    except CircleNotFound:
        raise_typed(
            PlatformError.circles(
                "not_found", f"circle {project_id}/{circle_id} not found"
            )
        )
    if user_id.strip() == row["owner_id"]:
        raise_typed(
            PlatformError.circles(
                "conflict", "the circle owner is never removed — retire instead"
            )
        )
    _manager_or_admin(identity, row, store)
    store.remove_member(project_id, circle_id, user_id)
    return {
        "status": "ok",
        "project_id": project_id,
        "circle_id": circle_id,
        "members": store.members(project_id, circle_id),
    }


__all__ = ["CircleNotFound", "CirclesStore", "circles_for_user", "router"]
