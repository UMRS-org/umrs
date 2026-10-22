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
import logging
from typing import Any, NoReturn

from fastapi import APIRouter, Depends, Request, status
from pydantic import BaseModel, Field

from .jobs import CallerIdentity, get_caller_identity
from .typed_errors import PlatformError, raise_typed

logger = logging.getLogger(__name__)


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


def _listed_row(row: Any) -> dict[str, Any]:
    """Map one list() SELECT row (the _row() columns)."""
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
        from umrs.shared.project_data_class import default_project_data_class

        cur = self._conn.cursor()
        cur.execute(
            """INSERT INTO projects (project_id, name, owner_id, data_class)
               VALUES (%s, %s, %s, %s)
               ON CONFLICT (project_id) DO NOTHING""",
            # ADR-062 amendment D-4: the deployment's default for NEW rows
            # (stand-alone: non_sensitive). DO NOTHING + the ensuring-shape
            # return mean an owner's later choice is never rewritten here.
            (pid, name, owner_id, default_project_data_class()),
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

    def circle_affiliated(self, project_id: str, user_id: str) -> bool:
        """True when the user holds an active circle in the project (ADR-086:
        project affiliation derives from circle membership). Fails closed."""
        if not user_id:
            return False
        from .circles import CirclesStore

        try:
            return CirclesStore(self._conn).affiliated(project_id, user_id)
        except Exception:  # noqa: BLE001 - an unreadable roster grants nothing
            logger.warning("circle affiliation unreadable for %s", project_id)
            self._conn.rollback()
            return False

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
            " updated_at, data_class, ask_mode, pause_triggers"
            " FROM projects" + where + " ORDER BY created_at",
            tuple(params),
        )
        return [_listed_row(row) for row in cur.fetchall()]

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
    maturity_stage: str | None = Field(
        default=None,
        description=(
            "ADR-066 amendment 3: with action='set_ask_policy' — a named preset "
            "('bootstrapped' | 'learning' | 'competent' | 'autonomous') that "
            "expands to ask_mode + pause_triggers; not combinable with them"
        ),
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


def _requested_ask_policy(body: ProjectPatchBody) -> tuple[str, list[str]]:
    """(ask_mode, pause_triggers) from explicit fields or a maturity stage."""
    from umrs.knowledge.ask_policy import stage_preset

    stage = str(body.maturity_stage or "").strip()
    if not stage:
        return _explicit_ask_policy(body)
    if body.ask_mode or body.pause_triggers:
        raise_typed(
            PlatformError.projects(
                "bad_request",
                "give either maturity_stage or ask_mode/pause_triggers, not both",
            )
        )
    try:
        return stage_preset(stage)
    except KeyError as exc:
        raise_typed(PlatformError.projects("bad_request", str(exc.args[0])))


def _explicit_ask_policy(body: ProjectPatchBody) -> tuple[str, list[str]]:
    triggers = [str(t).strip() for t in (body.pause_triggers or []) if str(t).strip()]
    return str(body.ask_mode or "").strip(), triggers


def _authorize_patch(row: dict[str, Any], action: str, identity: Any) -> None:
    """The PATCH action is known and the caller is the owner or an admin."""
    if action not in ("retire", "set_data_class", "set_ask_policy"):
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


def _set_data_class_audited(
    store: ProjectStore, row: dict[str, Any], body: ProjectPatchBody, identity: Any
) -> dict[str, Any]:
    """'set_data_class' (ADR-062): write the class and audit the act.

    Widening must be attributable: the audit trail records who changed the
    class (owner or admin), on whose project, from which class to which.
    """
    data_class = str(body.data_class or "").strip()
    if data_class not in ("sensitive", "non_sensitive"):
        raise_typed(
            PlatformError.projects(
                "bad_request",
                "set_data_class requires 'sensitive' or 'non_sensitive'",
            )
        )
    before = str(row.get("data_class") or "sensitive")
    updated = store.set_data_class(row["project_id"], data_class)
    from .audit_events import emit_audit

    actor = (identity.user_id or "").strip()
    emit_audit(
        "project.data_class_changed",
        actor=actor,
        project_id=row["project_id"],
        outcome="success",
        scope="project",
        details={
            "from_class": before,
            "to_class": data_class,
            "widening": before == "sensitive" and data_class == "non_sensitive",
            "owner_id": row["owner_id"],
            "by_admin": bool(identity.is_admin) and actor != row["owner_id"],
        },
    )
    return updated


def _row_response(row: dict[str, Any]) -> dict[str, Any]:
    from umrs.knowledge.ask_policy import stage_for

    return {
        "project_id": row["project_id"],
        "name": row.get("name", ""),
        "owner_id": row["owner_id"],
        "status": row["status"],
        "data_class": row.get("data_class", "sensitive"),
        "ask_mode": row.get("ask_mode", ""),
        "pause_triggers": row.get("pause_triggers", []),
        # ADR-066 amendment 3: the stage the stored policy corresponds to
        "maturity_stage": stage_for(
            row.get("ask_mode", ""), list(row.get("pause_triggers") or [])
        ),
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
    # H-1 (2026-10-18): an id already taken returned that project's row to
    # anyone, while GET hides it as 404; a caller who may not read it learns
    # only that the id is taken.
    row = _readable_project(store, row["project_id"], identity, refusal=_ID_TAKEN)
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
        "memberships": live_memberships(store, (identity.user_id or "").strip()),
    }


def _record_projects(user: str) -> list[str]:
    """The project list of the venue's user record (what ``umrs admin user
    add-to-project`` edits); [] for an unknown or disabled user."""
    from ..admin.user_store import UserStore, UserStoreError

    try:
        record = UserStore().get(user)
    except UserStoreError:
        return []
    if record.disabled:
        return []
    return [p.strip() for p in record.projects if isinstance(p, str) and p.strip()]


def live_memberships(store: ProjectStore, user: str) -> list[str] | None:
    """ADR-097 D-1: the caller's live project ids — the user record's
    projects plus the active projects they own, a retired row dropped.
    None when either source is unreadable: the gateway then fails closed
    instead of reading "no memberships"."""
    if not user:
        return []
    try:
        owned = [row["project_id"] for row in store.list(owner=user)]
        listed = [pid for pid in _record_projects(user) if not _retired(store, pid)]
    except Exception:  # unreadable is not "none"
        logger.warning("live memberships unreadable for %s", user, exc_info=True)
        return None
    return sorted(set(owned) | set(listed))


def _retired(store: ProjectStore, project_id: str) -> bool:
    """A listed project whose row is retired grants nothing; an id with no
    row (``default``, pre-registry projects) stays a membership."""
    row = store._row(project_id)
    return bool(row) and row["status"] == "retired"


def _may_read_project(
    store: ProjectStore, row: dict[str, Any], identity: CallerIdentity
) -> bool:
    """Owner, admin, a caller verified into the project, or a circle member."""
    user = (identity.user_id or "").strip()
    if identity.is_admin or (user and user == row["owner_id"]):
        return True
    if (identity.project_id or "").strip() == row["project_id"]:
        return True
    return store.circle_affiliated(row["project_id"], user)


_ID_TAKEN = PlatformError.projects("conflict", "project id is taken")


def _readable_project(
    store: ProjectStore,
    project_id: str,
    identity: CallerIdentity,
    *,
    refusal: PlatformError | None = None,
) -> dict[str, Any]:
    """The project row for its owner, members and admins; anyone else reads
    "not found" (H-1, 2026-10-17: these reads took no identity at all), or
    ``refusal`` when the caller already knows the id exists (create)."""
    try:
        row = store.get(project_id)
    except ProjectNotFound:
        row = None
    if row is None or not _may_read_project(store, row, identity):
        raise_typed(
            refusal
            or PlatformError.projects("not_found", f"project {project_id} not found")
        )
    return row


@router.get("/{project_id}")
def get_project(
    project_id: str,
    store: ProjectStore = Depends(_store_dep),
    identity: CallerIdentity = Depends(get_caller_identity),
) -> dict[str, Any]:
    """Owner, member or admin (another project reads as not found).
    # auth: verified
    """
    row = _readable_project(store, project_id, identity)
    return {"status": "ok", "project": _row_response(row)}


@router.patch("/{project_id}")
def patch_project(
    project_id: str,
    body: ProjectPatchBody,
    store: ProjectStore = Depends(_store_dep),
    identity: CallerIdentity = Depends(get_caller_identity),
) -> dict[str, Any]:
    """Owner-or-admin actions: 'retire' (tombstone, never DELETE),
    'set_data_class' (ADR-062) and 'set_ask_policy' (D-33 default; explicit
    fields or an ADR-066 amendment 3 maturity stage).
    # auth: verified
    """
    try:
        row = store.get(project_id)
    except ProjectNotFound:
        raise_typed(
            PlatformError.projects("not_found", f"project {project_id} not found")
        )
    _authorize_patch(row, body.action, identity)
    if body.action == "set_ask_policy":
        ask_mode, triggers = _requested_ask_policy(body)
        row = store.set_ask_policy(project_id, ask_mode, triggers)
        return {"status": "ok", "project": _row_response(row)}
    if body.action == "set_data_class":
        row = _set_data_class_audited(store, row, body, identity)
        return {"status": "ok", "project": _row_response(row)}
    row = store.retire(project_id)
    return {"status": "ok", "project": _row_response(row)}


class BudgetBody(BaseModel):
    """Loop budget caps (0 = uncapped; ADR-067)."""

    max_gpu_hours: float = Field(default=0.0, ge=0.0)
    max_wall_clock_hours: float = Field(default=0.0, ge=0.0)


@router.get("/{project_id}/budget")
def get_project_budget(
    project_id: str,
    store: ProjectStore = Depends(_store_dep),
    identity: CallerIdentity = Depends(get_caller_identity),
) -> dict[str, Any]:
    """Loop budget caps for a project (zeros = uncapped when absent);
    owner, member or admin (another or unknown project reads as not found).
    # auth: verified
    """
    _readable_project(store, project_id, identity)
    return {"status": "ok", "budget": store.get_budget(project_id)}


@router.put("/{project_id}/budget")
def put_project_budget(
    project_id: str,
    body: BudgetBody,
    store: ProjectStore = Depends(_store_dep),
    identity: CallerIdentity = Depends(get_caller_identity),
) -> dict[str, Any]:
    """Set loop budget caps — owner-or-admin; 0 clears the cap. A caller who
    cannot read the project gets "not found", as on GET (H-1, 2026-10-18).
    # auth: verified
    """
    row = _readable_project(store, project_id, identity)
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


# -- external storage binding (ADR-077, gap row D-17) -------------------------


class StorageBindingBody(BaseModel):
    """Bind a project's stores by locator; ``None`` keeps, ``""`` unbinds."""

    s3_url: str | None = Field(
        default=None,
        description=(
            "http(s)://ACCESS_KEY:SECRET_KEY@host:port[/bucket][?region=&verify=&backend=]"
            " — '' unbinds the object store"
        ),
    )
    postgres_uri: str | None = Field(
        default=None,
        description="postgresql://user:password@host:port/db[?sslmode=] — '' unbinds Postgres",
    )
    check: bool = Field(
        default=True,
        description="probe the resulting binding from this server first; a failing probe refuses",
    )


class _BindingStores:
    """The project row store and the binding store over one pooled conn."""

    def __init__(self, conn: Any) -> None:
        from umrs.shared.storage.project_storage import ProjectStorageStore

        self.projects = ProjectStore(conn)
        self.bindings = ProjectStorageStore(conn)


def _binding_stores_dep(request: Request):
    """Borrow one conn for the request; always returned (2026-09-09 class)."""
    pool = getattr(request.app.state, "umrs_db_pool", None)
    if pool is None:
        raise_typed(
            PlatformError.projects(
                "unavailable", "projects store unavailable — shared umrs_db pool absent"
            )
        )
    conn = pool.getconn()
    try:
        yield _BindingStores(conn)
    finally:
        try:
            conn.rollback()  # reads must not leave the conn idle in transaction
        finally:
            pool.putconn(conn)


def _project_or_404(stores: _BindingStores, project_id: str) -> dict[str, Any]:
    try:
        return stores.projects.get(project_id)
    except ProjectNotFound:
        raise_typed(
            PlatformError.projects("not_found", f"project {project_id} not found")
        )


def _require_owner_or_admin(row: dict[str, Any], identity: Any) -> None:
    if not identity.is_admin and (identity.user_id or "") != row["owner_id"]:
        raise_typed(
            PlatformError.projects(
                "forbidden",
                "only the owner or an admin may manage this project's storage",
            )
        )


def _binding_error(exc: Exception) -> NoReturn:
    """Typed refusal for a locator/sealing failure (never echoes credentials)."""
    reason = getattr(exc, "reason", "invalid")
    if reason in ("sealing_key_unavailable", "sealing_key_changed"):
        raise_typed(
            PlatformError.projects("unavailable", f"storage binding: {exc}"), cause=exc
        )
    raise_typed(
        PlatformError.projects("bad_request", f"storage binding: {exc}"), cause=exc
    )


def _empty_binding(project_id: str) -> dict[str, Any]:
    from umrs.shared.storage.project_storage import StorageBinding

    return StorageBinding(project_id).redacted()


def _current_binding(stores: _BindingStores, project_id: str):
    from umrs.shared.storage.project_storage import StorageBindingError

    try:
        return stores.bindings.get(project_id)
    except StorageBindingError as exc:
        _binding_error(exc)


def _parsed_locators(body: StorageBindingBody) -> tuple[Any, Any]:
    """Parse + host-check the locators the body sets (None/'' carry no locator)."""
    from umrs.shared.storage.project_storage import (
        check_target_host,
        parse_postgres_uri,
        parse_s3_url,
    )

    s3 = parse_s3_url(body.s3_url) if (body.s3_url or "").strip() else None
    pg = (
        parse_postgres_uri(body.postgres_uri)
        if (body.postgres_uri or "").strip()
        else None
    )
    for locator in (s3, pg):
        if locator is not None:
            check_target_host(locator.host)
    return s3, pg


def _candidate(
    project_id: str, body: StorageBindingBody, current: Any, s3: Any, pg: Any
):
    """The binding the PUT would leave behind (for the pre-save probe)."""
    from umrs.shared.storage.project_storage import StorageBinding

    def pick(value: str | None, parsed: Any, kept: Any) -> Any:
        if value is None:
            return kept
        return parsed  # None when value is blank (unbind)

    return StorageBinding(
        project_id,
        s3=pick(body.s3_url, s3, current.s3 if current else None),
        postgres=pick(body.postgres_uri, pg, current.postgres if current else None),
    )


def _default_bucket() -> str:
    from umrs.shared.storage.s3_config import get_s3_config

    return get_s3_config().bucket


def _probe_or_refuse(candidate: Any) -> dict[str, Any] | None:
    from umrs.shared.storage.project_storage import probe_binding

    if not candidate.bound:
        return None
    report = probe_binding(candidate, default_bucket=_default_bucket())
    if not report["ok"]:
        failed = {
            name: (report[name] or {}).get("detail")
            or (report[name] or {}).get("error")
            for name in ("s3", "postgres")
            if report[name] is not None and not report[name].get("ok")
        }
        raise_typed(
            PlatformError.projects(
                "conflict", f"storage binding probe failed — nothing saved: {failed}"
            )
        )
    return report


def _invalidate_binding_cache(project_id: str) -> None:
    from umrs.shared.storage.project_storage import get_project_storage_resolver

    get_project_storage_resolver().invalidate(project_id)


@router.get("/{project_id}/storage")
def get_project_storage(
    project_id: str,
    stores: _BindingStores = Depends(_binding_stores_dep),
    identity: CallerIdentity = Depends(get_caller_identity),
) -> dict[str, Any]:
    """The project's storage binding, redacted (owner-or-admin).
    # auth: verified
    """
    row = _project_or_404(stores, project_id)
    _require_owner_or_admin(row, identity)
    binding = _current_binding(stores, project_id)
    return {
        "status": "ok",
        "storage": binding.redacted() if binding else _empty_binding(project_id),
    }


@router.put("/{project_id}/storage")
def put_project_storage(
    project_id: str,
    body: StorageBindingBody,
    stores: _BindingStores = Depends(_binding_stores_dep),
    identity: CallerIdentity = Depends(get_caller_identity),
) -> dict[str, Any]:
    """Bind the project's object store and/or Postgres by locator.

    Owner-or-admin; an owner needs the operator allowlist
    (``UMRS_PROJECT_STORAGE_ALLOWED``). With ``check`` (default) the
    resulting binding is probed from this server first. Credentials are
    sealed at rest and never returned.
    # auth: verified
    """
    from umrs.shared.storage.project_storage import (
        StorageBindingError,
        allowed_targets,
        binding_refusal,
    )

    row = _project_or_404(stores, project_id)
    refusal = binding_refusal(
        is_admin=bool(identity.is_admin),
        is_owner=(identity.user_id or "") == row["owner_id"],
        allowed=allowed_targets(),
    )
    if refusal:
        raise_typed(PlatformError.projects("forbidden", _REFUSAL_TEXT[refusal]))
    try:
        binding, probe = _save_binding(stores, project_id, body, actor=identity.user_id)
    except StorageBindingError as exc:
        _binding_error(exc)
    _invalidate_binding_cache(project_id)
    return {
        "status": "ok",
        "storage": binding.redacted() if binding else _empty_binding(project_id),
        "probe": probe,
    }


def _save_binding(
    stores: _BindingStores,
    project_id: str,
    body: StorageBindingBody,
    *,
    actor: str | None,
) -> tuple[Any, dict[str, Any] | None]:
    """Validate, optionally probe, then persist; returns (binding, probe report)."""
    s3, pg = _parsed_locators(body)
    probe = None
    if body.check:
        # The stored side is read only when the body keeps one of the stores.
        keeps_one = body.s3_url is None or body.postgres_uri is None
        current = stores.bindings.get(project_id) if keeps_one else None
        probe = _probe_or_refuse(_candidate(project_id, body, current, s3, pg))
    binding = stores.bindings.put(
        project_id,
        s3_url=body.s3_url,
        postgres_uri=body.postgres_uri,
        updated_by=(actor or "").strip(),
    )
    return binding, probe


_REFUSAL_TEXT = {
    "forbidden": "only the owner or an admin may bind this project's storage",
    "admin_required": (
        "binding external storage requires the admin role unless the operator "
        "allowlists targets (UMRS_PROJECT_STORAGE_ALLOWED)"
    ),
}


@router.delete("/{project_id}/storage")
def delete_project_storage(
    project_id: str,
    stores: _BindingStores = Depends(_binding_stores_dep),
    identity: CallerIdentity = Depends(get_caller_identity),
) -> dict[str, Any]:
    """Unbind both stores — the project returns to the server's own stores.

    Objects already written to the external stores stay there.
    # auth: verified
    """
    row = _project_or_404(stores, project_id)
    _require_owner_or_admin(row, identity)
    existed = stores.bindings.clear(project_id)
    _invalidate_binding_cache(project_id)
    return {"status": "ok", "cleared": existed, "storage": _empty_binding(project_id)}


@router.post("/{project_id}/storage/check")
def check_project_storage(
    project_id: str,
    stores: _BindingStores = Depends(_binding_stores_dep),
    identity: CallerIdentity = Depends(get_caller_identity),
) -> dict[str, Any]:
    """Probe the saved binding from this server (owner-or-admin).

    S3: project bucket + ``-output`` ensured, then a put/get/delete
    round-trip; Postgres: connect, server facts, TLS, pgvector round-trip.
    # auth: verified
    """
    from umrs.shared.storage.project_storage import (
        StorageBindingError,
        check_target_host,
        probe_binding,
    )

    row = _project_or_404(stores, project_id)
    _require_owner_or_admin(row, identity)
    binding = _current_binding(stores, project_id)
    if binding is None:
        raise_typed(
            PlatformError.projects(
                "not_found", f"project {project_id} has no storage binding"
            )
        )
    try:
        for locator in (binding.s3, binding.postgres):
            if locator is not None:
                check_target_host(locator.host)
    except StorageBindingError as exc:
        _binding_error(exc)
    report = probe_binding(binding, default_bucket=_default_bucket())
    return {"status": "ok", "storage": binding.redacted(), "probe": report}


__all__ = ["ProjectNotFound", "ProjectStore", "router"]
