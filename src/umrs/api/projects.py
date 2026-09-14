"""Durable projects registry (ADR-051 D-1) — store + routes.

Invariants pinned below:

- creation is casual (``POST /umrs/projects`` is ensure-shape, idempotent,
  an existing row is returned as-is); issued accounts are never a ceiling.
- every row carries the owner (creator); admin may mint for another user.
- retire is a tombstone, never a DELETE.
- unknown project ids 404 — never a smudged empty payload.

psycopg2-shaped sync pool (satellite pattern); store is a foundation
module (no imports from umrs.api).
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request, status
from pydantic import BaseModel, Field

from .jobs import CallerIdentity, get_caller_identity
from .typed_errors import PlatformError, raise_typed


class ProjectNotFound(Exception):
    """Raised when a project id does not exist (never silent)."""


class ProjectStore:
    """psycopg2-backed durable store over the 0095 registry."""

    def __init__(self, conn: Any) -> None:
        self._conn = conn

    def _row(self, project_id: str) -> dict[str, Any] | None:
        cur = self._conn.cursor()
        cur.execute(
            "SELECT project_id, name, owner_id, status, retired_at, created_at, updated_at"
            " FROM projects WHERE project_id = %s",
            (project_id,),
        )
        row = cur.fetchone()
        if not row:
            return None
        return {
            "project_id": row[0],
            "name": row[1],
            "owner_id": row[2],
            "status": row[3],
            "retired_at": row[4].isoformat() if row[4] else None,
            "created_at": row[5].isoformat() if row[5] else None,
            "updated_at": row[6].isoformat() if row[6] else None,
        }

    def create(
        self,
        project_id: str,
        *,
        name: str = "",
        owner_id: str,
    ) -> dict[str, Any]:
        """Ensuring-shape: returns the existing row if the id is taken (and
        never writes an invented mutation on a known id)."""
        pid = project_id.strip()
        if not pid:
            raise ValueError("project_id must be non-empty")
        existing = self._row(pid)
        if existing:
            return existing
        cur = self._conn.cursor()
        cur.execute(
            """INSERT INTO projects (project_id, name, owner_id)
               VALUES (%s, %s, %s)
               ON CONFLICT (project_id) DO NOTHING""",
            (pid, name, owner_id),
        )
        self._conn.commit()
        row = self._row(pid)
        if row is None:
            raise RuntimeError(f"project insert lost (project_id={pid})")
        return row

    def get(self, project_id: str) -> dict[str, Any]:
        row = self._row(project_id)
        if row is None:
            raise ProjectNotFound(project_id)
        return row

    def list(
        self, *, owner: str | None = None, include_retired: bool = False
    ) -> list[dict[str, Any]]:
        cur = self._conn.cursor()
        clauses: list[str] = []
        params: list[Any] = []
        if not include_retired:
            clauses.append("status != 'retired'")
        if owner:
            clauses.append("owner_id = %s")
            params.append(owner)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        cur.execute(
            "SELECT project_id, name, owner_id, status, retired_at, created_at,"
            " updated_at"
            " FROM projects" + where + " ORDER BY created_at",
            tuple(params),
        )
        out: list[dict[str, Any]] = []
        for row in cur.fetchall():
            out.append(
                {
                    "project_id": row[0],
                    "name": row[1],
                    "owner_id": row[2],
                    "status": row[3],
                    "retired_at": row[4].isoformat() if row[4] else None,
                    "created_at": row[5].isoformat() if row[5] else None,
                    "updated_at": row[6].isoformat() if row[6] else None,
                }
            )
        return out

    def retire(self, project_id: str) -> dict[str, Any]:
        row = self.get(project_id)  # raises ProjectNotFound
        if row["status"] == "retired":
            return row
        cur = self._conn.cursor()
        cur.execute(
            """UPDATE projects
               SET status = 'retired', retired_at = now(), updated_at = now()
               WHERE project_id = %s""",
            (project_id,),
        )
        self._conn.commit()
        return self._row(project_id)


# -- HTTP routes ---------------------------------------------------------------

router = APIRouter(prefix="/umrs/projects", tags=["projects"])


class ProjectCreateBody(BaseModel):
    project_id: str = Field(..., min_length=1)
    name: str = Field(default="")
    for_user: str | None = Field(
        default=None,
        description="Admin-only: owner differs from the caller (minting-on-behalf)",
    )


class ProjectPatchBody(BaseModel):
    action: str = Field(default="")


def _store_for(request: Request) -> ProjectStore:
    app = request.app
    pool_getter = getattr(app.state, "umrs_db_pool", None)
    if pool_getter is None:
        raise_typed(
            PlatformError.projects(
                "unavailable",
                "projects store unavailable — shared umrs_db pool absent",
            )
        )
    return ProjectStore(pool_getter.getconn())


def _row_response(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "project_id": row["project_id"],
        "name": row.get("name", ""),
        "owner_id": row["owner_id"],
        "status": row["status"],
        "retired_at": row.get("retired_at"),
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
    }


@router.post("", status_code=status.HTTP_201_CREATED)
def create_project(
    body: ProjectCreateBody,
    request: Request,
    identity: CallerIdentity = Depends(get_caller_identity),
) -> dict[str, Any]:
    """Ensuring-shape creation (idempotent — existing row returned as-is)."""
    store = _store_for(request)
    actor = (identity.user_id or "").strip()
    if body.for_user and body.for_user != actor:
        if not identity.is_admin:
            raise_typed(
                PlatformError.projects(
                    "forbidden",
                    "minting projects for another user requires the admin role",
                )
            )
    owner = body.for_user if body.for_user else actor
    if not owner:
        raise_typed(PlatformError.projects("bad_request", "caller identity required"))
    row = store.create(
        body.project_id,
        name=body.name,
        owner_id=owner,
    )
    return {"status": "ok", "project": _row_response(row), "created": True}


@router.get("")
def list_my_projects(
    request: Request,
    include_retired: bool = False,
    identity: CallerIdentity = Depends(get_caller_identity),
) -> dict[str, Any]:
    """Mine only; admin sees every row."""
    store = _store_for(request)
    owner = None if identity.is_admin else (identity.user_id or "").strip()
    rows = store.list(owner=owner, include_retired=include_retired)
    return {
        "status": "ok",
        "projects": [_row_response(r) for r in rows],
        "count": len(rows),
    }


@router.get("/{project_id}")
def get_project(project_id: str, request: Request) -> dict[str, Any]:
    store = _store_for(request)
    try:
        row = store.get(project_id)
    except ProjectNotFound:
        raise_typed(
            PlatformError.projects("not_found", f"project {project_id} not found")
        )
    return {"status": "ok", "project": _row_response(row)}


@router.patch("/{project_id}")
def patch_project(
    project_id: str,
    body: ProjectPatchBody,
    request: Request,
    identity: CallerIdentity = Depends(get_caller_identity),
) -> dict[str, Any]:
    """Only 'retire' this round — owner-or-admin, tombstone, never DELETE."""
    store = _store_for(request)
    try:
        row = store.get(project_id)
    except ProjectNotFound:
        raise_typed(
            PlatformError.projects("not_found", f"project {project_id} not found")
        )
    if body.action != "retire":
        raise_typed(
            PlatformError.projects(
                "bad_request", "action must be 'retire' for the current round"
            )
        )
    if not identity.is_admin and (identity.user_id or "") != row["owner_id"]:
        raise_typed(
            PlatformError.projects(
                "forbidden", "only the owner or an admin may retire this project"
            )
        )
    row = store.retire(project_id)
    return {"status": "ok", "project": _row_response(row)}


__all__ = ["ProjectNotFound", "ProjectStore", "router"]
