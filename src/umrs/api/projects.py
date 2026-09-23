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

import json
from typing import Any

from fastapi import APIRouter, Depends, Request, status
from pydantic import BaseModel, Field

from .jobs import CallerIdentity, get_caller_identity
from .typed_errors import PlatformError, raise_typed


class ProjectNotFound(Exception):
    """Raised when a project id does not exist (never silent)."""


def _as_triggers(raw: Any) -> list[str]:
    """0111 JSONB column: list through psycopg, JSON string in fakes."""
    if isinstance(raw, list):
        return [str(t) for t in raw]
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            return [str(t) for t in parsed] if isinstance(parsed, list) else []
        except ValueError:
            return []
    return []


class ProjectStore:
    """psycopg2-backed durable store over the 0095 registry."""

    def __init__(self, conn: Any) -> None:
        self._conn = conn

    def _row(self, project_id: str) -> dict[str, Any] | None:
        cur = self._conn.cursor()
        cur.execute(
            "SELECT project_id, name, owner_id, status, retired_at, created_at,"
            " updated_at, data_class, ask_mode, pause_triggers"
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
            "data_class": row[7] if len(row) > 7 else "sensitive",
            "ask_mode": row[8] if len(row) > 8 else "",
            "pause_triggers": _as_triggers(row[9] if len(row) > 9 else None),
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

    # -- budget caps (ADR-067, gap row D-36) --------------------------------

    def set_ask_policy(
        self, project_id: str, ask_mode: str, pause_triggers: list[str]
    ) -> dict[str, Any]:
        """D-33 tail: persist the project's ask-policy default (owner-or-admin
        enforced at the ROUTE; explicit loop requests always win)."""
        if ask_mode not in ("", "auto", "ask", "profile"):
            raise ValueError(f"invalid ask_mode {ask_mode!r}")
        if not isinstance(pause_triggers, list) or not all(
            isinstance(t, str) and t.strip() for t in pause_triggers
        ):
            raise ValueError("pause_triggers must be a list of non-empty strings")
        if self._row(project_id) is None:
            raise ProjectNotFound(project_id)
        cur = self._conn.cursor()
        cur.execute(
            "UPDATE projects SET ask_mode = %s, pause_triggers = %s,"
            " updated_at = now() WHERE project_id = %s",
            (ask_mode, json.dumps(pause_triggers), project_id),
        )
        self._conn.commit()
        row = self._row(project_id)
        if row is None:
            raise RuntimeError(f"project update lost (project_id={project_id})")
        return row

    def set_data_class(self, project_id: str, data_class: str) -> dict[str, Any]:
        """ADR-062: set the project's data class (widening/narrowing both
        ride this verb; the ROUTE enforces owner-or-admin)."""
        if data_class not in ("sensitive", "non_sensitive"):
            raise ValueError(f"invalid data_class {data_class!r}")
        if self._row(project_id) is None:
            raise ProjectNotFound(project_id)
        cur = self._conn.cursor()
        cur.execute(
            "UPDATE projects SET data_class = %s, updated_at = now()"
            " WHERE project_id = %s",
            (data_class, project_id),
        )
        self._conn.commit()
        row = self._row(project_id)
        if row is None:
            raise RuntimeError(f"project update lost (project_id={project_id})")
        return row

    def get_budget(self, project_id: str) -> dict[str, Any]:
        """Budget row for the project (zeros = uncapped when absent)."""
        cur = self._conn.cursor()
        cur.execute(
            "SELECT project_id, max_gpu_hours, max_wall_clock_hours, updated_at"
            " FROM project_budgets WHERE project_id = %s",
            (project_id,),
        )
        row = cur.fetchone()
        if not row:
            return {
                "project_id": project_id,
                "max_gpu_hours": 0.0,
                "max_wall_clock_hours": 0.0,
                "updated_at": None,
            }
        return {
            "project_id": row[0],
            "max_gpu_hours": float(row[1]),
            "max_wall_clock_hours": float(row[2]),
            "updated_at": row[3].isoformat() if row[3] else None,
        }

    def set_budget(
        self,
        project_id: str,
        *,
        max_gpu_hours: float,
        max_wall_clock_hours: float,
    ) -> dict[str, Any]:
        """Upsert budget caps (0 = uncapped; the row says what it enforces)."""
        self.get(project_id)  # raises ProjectNotFound — no orphan budgets
        cur = self._conn.cursor()
        cur.execute(
            """INSERT INTO project_budgets (project_id, max_gpu_hours,
                                          max_wall_clock_hours, updated_at)
               VALUES (%s, %s, %s, NOW())
               ON CONFLICT (project_id) DO UPDATE SET
                   max_gpu_hours = EXCLUDED.max_gpu_hours,
                   max_wall_clock_hours = EXCLUDED.max_wall_clock_hours,
                   updated_at = NOW()""",
            (project_id, max_gpu_hours, max_wall_clock_hours),
        )
        self._conn.commit()
        return self.get_budget(project_id)


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
    data_class: str | None = Field(
        default=None,
        description="ADR-062: with action='set_data_class' — 'sensitive' | 'non_sensitive'",
    )
    ask_mode: str | None = Field(
        default=None,
        description=(
            "D-33: with action='set_ask_policy' — '' | 'auto' | 'ask' | 'profile'"
            " (the project's persisted loop-start default; explicit requests win)"
        ),
    )
    pause_triggers: list[str] | None = Field(
        default=None,
        description="D-33: trigger classes persisted for ask_mode='profile'",
    )


def _store_dep(request: Request):
    """FastAPI generator dependency — pool conn is always returned on every
    path (the 2026-09-09 exhaustion class; the prior plain factory leaked
    one conn per route call)."""
    app = request.app
    pool = getattr(app.state, "umrs_db_pool", None)
    if pool is None:
        raise_typed(
            PlatformError.projects(
                "unavailable",
                "projects store unavailable — shared umrs_db pool absent",
            )
        )
    conn = pool.getconn()
    try:
        yield ProjectStore(conn)
    finally:
        pool.putconn(conn)


def _row_response(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "project_id": row["project_id"],
        "name": row.get("name", ""),
        "owner_id": row["owner_id"],
        "status": row["status"],
        "data_class": row.get("data_class", "sensitive"),
        "ask_mode": row.get("ask_mode", ""),
        "pause_triggers": row.get("pause_triggers", []),
        "retired_at": row.get("retired_at"),
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
    }


@router.post("", status_code=status.HTTP_201_CREATED)
def create_project(
    body: ProjectCreateBody,
    store: ProjectStore = Depends(_store_dep),
    identity: CallerIdentity = Depends(get_caller_identity),
) -> dict[str, Any]:
    """Ensuring-shape creation (idempotent — existing row returned as-is).
    # auth: verified
    """
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
    include_retired: bool = False,
    store: ProjectStore = Depends(_store_dep),
    identity: CallerIdentity = Depends(get_caller_identity),
) -> dict[str, Any]:
    """Mine only; admin sees every row.
    # auth: verified
    """
    owner = None if identity.is_admin else (identity.user_id or "").strip()
    rows = store.list(owner=owner, include_retired=include_retired)
    return {
        "status": "ok",
        "projects": [_row_response(r) for r in rows],
        "count": len(rows),
    }


@router.get("/{project_id}")
def get_project(
    project_id: str, store: ProjectStore = Depends(_store_dep)
) -> dict[str, Any]:
    """# auth: verified"""
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
    store: ProjectStore = Depends(_store_dep),
    identity: CallerIdentity = Depends(get_caller_identity),
) -> dict[str, Any]:
    """Only 'retire' this round — owner-or-admin, tombstone, never DELETE.
    # auth: verified
    """
    try:
        row = store.get(project_id)
    except ProjectNotFound:
        raise_typed(
            PlatformError.projects("not_found", f"project {project_id} not found")
        )
    if body.action not in ("retire", "set_data_class", "set_ask_policy"):
        raise_typed(
            PlatformError.projects(
                "bad_request",
                "action must be 'retire', 'set_data_class' or 'set_ask_policy'",
            )
        )
    if not identity.is_admin and (identity.user_id or "") != row["owner_id"]:
        raise_typed(
            PlatformError.projects(
                "forbidden", "only the owner or an admin may modify this project"
            )
        )
    if body.action == "set_ask_policy":
        ask_mode = str(body.ask_mode or "").strip()
        triggers = [
            str(t).strip() for t in (body.pause_triggers or []) if str(t).strip()
        ]
        row = store.set_ask_policy(project_id, ask_mode, triggers)
        return {"status": "ok", "project": _row_response(row)}
    if body.action == "set_data_class":
        data_class = str(body.data_class or "").strip()
        if data_class not in ("sensitive", "non_sensitive"):
            raise_typed(
                PlatformError.projects(
                    "bad_request",
                    "set_data_class requires 'sensitive' or 'non_sensitive'",
                )
            )
        row = store.set_data_class(project_id, data_class)
        return {"status": "ok", "project": _row_response(row)}
    row = store.retire(project_id)
    return {"status": "ok", "project": _row_response(row)}


class BudgetBody(BaseModel):
    """Loop budget caps (0 = uncapped; ADR-067)."""

    max_gpu_hours: float = Field(default=0.0, ge=0.0)
    max_wall_clock_hours: float = Field(default=0.0, ge=0.0)


@router.get("/{project_id}/budget")
def get_project_budget(
    project_id: str, store: ProjectStore = Depends(_store_dep)
) -> dict[str, Any]:
    """Loop budget caps for a project (zeros = uncapped when absent).
    # auth: verified
    """
    try:
        budget = store.get_budget(project_id)
    except ProjectNotFound:
        raise_typed(
            PlatformError.projects("not_found", f"project {project_id} not found")
        )
    return {"status": "ok", "budget": budget}


@router.put("/{project_id}/budget")
def put_project_budget(
    project_id: str,
    body: BudgetBody,
    store: ProjectStore = Depends(_store_dep),
    identity: CallerIdentity = Depends(get_caller_identity),
) -> dict[str, Any]:
    """Set loop budget caps — owner-or-admin; 0 clears the cap.
    # auth: verified
    """
    try:
        row = store.get(project_id)
    except ProjectNotFound:
        raise_typed(
            PlatformError.projects("not_found", f"project {project_id} not found")
        )
    if not identity.is_admin and (identity.user_id or "") != row["owner_id"]:
        raise_typed(
            PlatformError.projects(
                "forbidden", "only the owner or an admin may set budgets"
            )
        )
    budget = store.set_budget(
        project_id,
        max_gpu_hours=body.max_gpu_hours,
        max_wall_clock_hours=body.max_wall_clock_hours,
    )
    return {"status": "ok", "budget": budget}


__all__ = ["ProjectNotFound", "ProjectStore", "router"]
