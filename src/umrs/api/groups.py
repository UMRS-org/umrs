"""Groups + membership authority (ADR-050 §3 groundwork) — store + routes.

The scope grammar always spoke ``group:<id>`` and the asset-grant ladder
named it as a grantee, but no authority existed to resolve it. This module
is the authority:

- ``groups`` rows are owner-attributed and tombstoned (retire never
  DELETEs); a retired group stops resolving for grant lookups immediately.
- ``group_members`` rows are presence-only (no tombstones): membership is
  the current fact, not history.
- ``groups_for_user`` is the single read resolution used by every grant
  listing endpoint — callers get the grants of the groups they are in.

psycopg2-shaped sync pool (satellite pattern); store is import-free of
umrs.api internals beyond the shared typed-error helpers.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request, status
from pydantic import BaseModel, Field

from .jobs import CallerIdentity, get_caller_identity
from .typed_errors import PlatformError, raise_typed


class GroupNotFound(Exception):
    """Raised when a group id does not exist (never silent)."""


class GroupsStore:
    """psycopg2-backed durable store over the 0097 registry."""

    def __init__(self, conn: Any) -> None:
        self._conn = conn

    # ── groups ───────────────────────────────────────────────────────────
    def _row(self, group_id: str) -> dict[str, Any] | None:
        cur = self._conn.cursor()
        cur.execute(
            "SELECT group_id, name, owner_id, status, retired_at, created_at, updated_at"
            " FROM groups WHERE group_id = %s",
            (group_id,),
        )
        row = cur.fetchone()
        if not row:
            return None
        return {
            "group_id": row[0],
            "name": row[1],
            "owner_id": row[2],
            "status": row[3],
            "retired_at": row[4].isoformat() if row[4] else None,
            "created_at": row[5].isoformat() if row[5] else None,
            "updated_at": row[6].isoformat() if row[6] else None,
        }

    def create(self, group_id: str, *, name: str = "", owner_id: str) -> dict[str, Any]:
        """Ensuring-shape: existing id returns as-is, never stomps."""
        gid = group_id.strip()
        if not gid:
            raise ValueError("group_id must be non-empty")
        existing = self._row(gid)
        if existing:
            return existing
        cur = self._conn.cursor()
        cur.execute(
            "INSERT INTO groups (group_id, name, owner_id) VALUES (%s, %s, %s)",
            (gid, name, owner_id),
        )
        # the creator starts as the first member (owner-attributed presence)
        cur.execute(
            "INSERT INTO group_members (group_id, user_id, added_by) VALUES (%s, %s, %s)",
            (gid, owner_id, owner_id),
        )
        self._conn.commit()
        return self._row(gid)

    def get(self, group_id: str) -> dict[str, Any]:
        row = self._row(group_id)
        if row is None:
            raise GroupNotFound(group_id)
        return row

    def list(self, *, owner: str | None = None, include_retired: bool = False) -> list[dict[str, Any]]:
        cur = self._conn.cursor()
        sql = "SELECT group_id, name, owner_id, status, retired_at, created_at, updated_at FROM groups"
        tail = " ORDER BY created_at"
        params: list[Any] = []
        where: list[str] = []
        if owner is not None:
            where.append("owner_id = %s")
            params.append(owner)
        if not include_retired:
            where.append("status <> 'retired'")
        if where:
            sql += " WHERE " + " AND ".join(where)
        cur.execute(sql + tail, tuple(params))
        return [
            {
                "group_id": r[0],
                "name": r[1],
                "owner_id": r[2],
                "status": r[3],
                "retired_at": r[4].isoformat() if r[4] else None,
                "created_at": r[5].isoformat() if r[5] else None,
                "updated_at": r[6].isoformat() if r[6] else None,
            }
            for r in cur.fetchall()
        ]

    def list_mine(self, user_id: str, *, include_retired: bool = False) -> list[dict[str, Any]]:
        """Groups the user OWNS or is a member of."""
        cur = self._conn.cursor()
        sql = (
            "SELECT DISTINCT g.group_id, g.name, g.owner_id, g.status, g.retired_at,"
            " g.created_at, g.updated_at FROM groups g"
            " LEFT JOIN group_members gm ON gm.group_id = g.group_id"
            " WHERE (g.owner_id = %s OR gm.user_id = %s)"
        )
        params: list[Any] = [user_id, user_id]
        if not include_retired:
            sql += " AND g.status <> 'retired'"
        sql += " ORDER BY g.created_at"
        cur.execute(sql, tuple(params))
        return [
            {
                "group_id": r[0],
                "name": r[1],
                "owner_id": r[2],
                "status": r[3],
                "retired_at": r[4].isoformat() if r[4] else None,
                "created_at": r[5].isoformat() if r[5] else None,
                "updated_at": r[6].isoformat() if r[6] else None,
            }
            for r in cur.fetchall()
        ]

    def retire(self, group_id: str) -> dict[str, Any]:
        cur = self._conn.cursor()
        cur.execute(
            "UPDATE groups SET status = 'retired', retired_at = now(), updated_at = now()"
            " WHERE group_id = %s AND status <> 'retired'",
            (group_id,),
        )
        self._conn.commit()
        return self.get(group_id)

    # ── membership ───────────────────────────────────────────────────────
    def add_member(self, group_id: str, user_id: str, *, added_by: str) -> dict[str, Any]:
        """Presence insert, idempotent — re-adding is a no-op per PK."""
        uid = user_id.strip()
        if not uid:
            raise ValueError("user_id must be non-empty")
        cur = self._conn.cursor()
        cur.execute(
            "INSERT INTO group_members (group_id, user_id, added_by) VALUES (%s, %s, %s)"
            " ON CONFLICT (group_id, user_id) DO NOTHING",
            (group_id, uid, added_by),
        )
        self._conn.commit()
        return self.get(group_id)

    def remove_member(self, group_id: str, user_id: str) -> None:
        cur = self._conn.cursor()
        cur.execute(
            "DELETE FROM group_members WHERE group_id = %s AND user_id = %s",
            (group_id, user_id),
        )
        self._conn.commit()

    def members(self, group_id: str) -> list[dict[str, Any]]:
        cur = self._conn.cursor()
        cur.execute(
            "SELECT user_id, role, added_by, added_at FROM group_members"
            " WHERE group_id = %s ORDER BY added_at",
            (group_id,),
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


def groups_for_user(conn: Any, user_id: str) -> list[str]:
    """Active group ids the user is a member of (the grant-ladder resolution).

    Retired groups never resolve — their grants stop counting immediately
    (tombstone semantics from the migration comment). The SQL authority is
    ``umrs.shared.group_membership.group_ids_for_user`` (ADR-080); this
    wrapper keeps the established api import surface.
    """
    from umrs.shared.group_membership import group_ids_for_user

    return group_ids_for_user(conn, user_id)


# ── routes ───────────────────────────────────────────────────────────────

router = APIRouter(prefix="/umrs/groups", tags=["groups"])


class GroupCreateBody(BaseModel):
    group_id: str = Field(..., min_length=1, max_length=200)
    name: str = Field(default="", max_length=500)
    for_user: str | None = None


class MemberBody(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=200)


def _store_dep(request: Request):
    """FastAPI generator dependency — pool conn is always returned on every
    path (the 2026-09-09 exhaustion class; plain factories leaked)."""
    pool = getattr(request.app.state, "umrs_db_pool", None)
    if pool is None:
        raise_typed(
            PlatformError.groups(
                "unavailable", "groups store unavailable — shared umrs_db pool absent"
            )
        )
    conn = pool.getconn()
    try:
        yield GroupsStore(conn)
    finally:
        pool.putconn(conn)


def _row_response(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "group_id": row["group_id"],
        "name": row.get("name", ""),
        "owner_id": row["owner_id"],
        "status": row["status"],
        "retired_at": row.get("retired_at"),
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
    }


def _owner_or_admin(identity: CallerIdentity, row: dict[str, Any] | None = None) -> None:
    actor_owner = (identity.user_id or "").strip()
    if identity.is_admin:
        return
    if row is not None and actor_owner == row["owner_id"]:
        return
    raise_typed(
        PlatformError.groups(
            "forbidden", "only the group owner or an admin may perform this"
        )
    )


@router.post("", status_code=status.HTTP_201_CREATED)
def create_group(
    body: GroupCreateBody,
    store: GroupsStore = Depends(_store_dep),
    identity: CallerIdentity = Depends(get_caller_identity),
) -> dict[str, Any]:
    """Ensuring-shape creation (idempotent — existing row returned as-is).
    # auth: verified
    """
    actor = (identity.user_id or "").strip()
    if body.for_user and body.for_user != actor and not identity.is_admin:
        raise_typed(
            PlatformError.groups(
                "forbidden", "minting groups for another user requires the admin role"
            )
        )
    owner = body.for_user if body.for_user else actor
    if not owner:
        raise_typed(PlatformError.groups("bad_request", "caller identity required"))
    row = store.create(body.group_id, name=body.name, owner_id=owner)
    return {"status": "ok", "group": _row_response(row), "created": True}


@router.get("")
def list_groups(
    include_retired: bool = False,
    store: GroupsStore = Depends(_store_dep),
    identity: CallerIdentity = Depends(get_caller_identity),
) -> dict[str, Any]:
    """Mine (owned or member-of); admin sees every row.
    # auth: verified
    """
    if identity.is_admin:
        rows = store.list(include_retired=include_retired)
    else:
        rows = store.list_mine(
            (identity.user_id or "").strip(), include_retired=include_retired
        )
    return {"status": "ok", "groups": [_row_response(r) for r in rows], "count": len(rows)}


@router.get("/{group_id}")
def get_group(group_id: str, store: GroupsStore = Depends(_store_dep)) -> dict[str, Any]:
    """# auth: verified
    """
    # auth: verified
    try:
        row = store.get(group_id)
    except GroupNotFound:
        raise_typed(PlatformError.groups("not_found", f"group {group_id} not found"))
    return {"status": "ok", "group": _row_response(row)}


@router.patch("/{group_id}")
def patch_group(
    group_id: str,
    body: dict[str, str],
    store: GroupsStore = Depends(_store_dep),
    identity: CallerIdentity = Depends(get_caller_identity),
) -> dict[str, Any]:
    """Only 'retire' this round — owner-or-admin; tombstone, never DELETE.
    # auth: verified
    """
    try:
        row = store.get(group_id)
    except GroupNotFound:
        raise_typed(
            PlatformError.groups("not_found", f"group {group_id} not found")
        )
    if body.get("action") != "retire":
        raise_typed(
            PlatformError.groups(
                "bad_request", "action must be 'retire' for the current round"
            )
        )
    _owner_or_admin(identity, row)
    row = store.retire(group_id)
    return {"status": "ok", "group": _row_response(row)}


@router.get("/{group_id}/members")
def list_members(group_id: str, store: GroupsStore = Depends(_store_dep)) -> dict[str, Any]:
    """# auth: verified
    """
    # auth: verified
    try:
        store.get(group_id)
    except GroupNotFound:
        raise_typed(
            PlatformError.groups("not_found", f"group {group_id} not found")
        )
    members = store.members(group_id)
    return {"status": "ok", "group_id": group_id, "members": members, "count": len(members)}


@router.post("/{group_id}/members", status_code=status.HTTP_201_CREATED)
def add_member(
    group_id: str,
    body: MemberBody,
    store: GroupsStore = Depends(_store_dep),
    identity: CallerIdentity = Depends(get_caller_identity),
) -> dict[str, Any]:
    """Owner-or-admin adds a member (idempotent, presence-only semantics).
    # auth: verified
    """
    try:
        row = store.get(group_id)
    except GroupNotFound:
        raise_typed(
            PlatformError.groups("not_found", f"group {group_id} not found")
        )
    if row["status"] == "retired":
        raise_typed(
            PlatformError.groups(
                "conflict", "cannot grow membership of a retired group"
            )
        )
    _owner_or_admin(identity, row)
    out = store.add_member(
        group_id, body.user_id, added_by=(identity.user_id or "").strip() or "unknown"
    )
    return {"status": "ok", "group": _row_response(out), "members": store.members(group_id)}


@router.delete("/{group_id}/members/{user_id}")
def remove_member(
    group_id: str,
    user_id: str,
    store: GroupsStore = Depends(_store_dep),
    identity: CallerIdentity = Depends(get_caller_identity),
) -> dict[str, Any]:
    """Owner-or-admin removes a member (presence removal; no tombstone).
    # auth: verified
    """
    try:
        row = store.get(group_id)
    except GroupNotFound:
        raise_typed(
            PlatformError.groups("not_found", f"group {group_id} not found")
        )
    _owner_or_admin(identity, row)
    store.remove_member(group_id, user_id)
    return {"status": "ok", "group_id": group_id, "members": store.members(group_id)}


__all__ = ["GroupNotFound", "GroupsStore", "groups_for_user", "router"]
