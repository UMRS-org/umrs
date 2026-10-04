"""Scientific Catalog registry (gap row D-39, Phase 1) — store + routes.

The catalog is a named, versioned object: one row bundles the ids of its
members (knowledge graphs, skills, MCP servers, embedding models) plus the
metadata that makes it shareable (contributors, licence, provenance).

Invariants pinned:

- one row per (project_id, name, version); creation of the same triple is
  idempotent (existing row returned as-is).
- version rows are immutable in their membership; PATCH touches description
  and metadata fields only — a new version is a new row.
- retire is a tombstone, never a DELETE.
- unknown catalog ids 404 — never a smudged empty payload.

psycopg2-shaped sync store over the 0101 table (satellite pattern);
no imports from umrs.jobs (ADR-053).
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request, status
from pydantic import BaseModel, Field

from umrs.scientific_catalogs.store import (
    CatalogNotFound,
    CatalogPatch,
    CatalogSpec,
    ScientificCatalogStore,
)
from umrs.shared.access_scopes import access_admits

from ._access import checked_access, resource_caller
from .jobs import CallerIdentity, get_caller_identity
from .typed_errors import PlatformError, raise_typed

# -- HTTP routes ---------------------------------------------------------------

router = APIRouter(prefix="/umrs/scientific-catalogs", tags=["scientific-catalogs"])


def _store_dep(request: Request):
    """FastAPI generator dependency — pool conn always returned (the
    2026-09-09 exhaustion class; no plain factory)."""
    app = request.app
    pool = getattr(app.state, "umrs_db_pool", None)
    if pool is None:
        raise_typed(
            PlatformError.scientificcatalogs(
                "unavailable",
                "scientific-catalogs store unavailable — shared umrs_db pool absent",
            )
        )
    conn = pool.getconn()
    try:
        yield ScientificCatalogStore(conn)
    finally:
        pool.putconn(conn)


def _require_project_scope(body_project: str, identity: CallerIdentity) -> None:
    """Project-scoped write: caller's project must match (or admin)."""
    if identity.is_admin:
        return
    caller_project = (identity.project_id or "").strip()
    if caller_project and caller_project == body_project:
        return
    raise_typed(
        PlatformError.scientificcatalogs(
            "forbidden",
            f"project {body_project!r} outside the caller's project scope",
        )
    )


def _catalog_read_allowed(row: dict[str, Any], request: Request, identity: CallerIdentity) -> bool:
    """Owning project, admin, or the row's ADR-080 access list (lazy groups)."""
    if identity.is_admin:
        return True
    if (identity.project_id or "").strip() and identity.project_id == row.get("project_id"):
        return True
    caller = resource_caller(
        request,
        user_id=identity.user_id,
        project_id=identity.project_id or "",
        is_admin=identity.is_admin,
    )
    return access_admits(row.get("access"), caller)


class AccessUpdate(BaseModel):
    """ADR-080 access list update (validated against the canonical grammar)."""

    access: list[str] = Field(
        default_factory=list,
        description=(
            "user:<id> | project:<id> | circle:<proj>/<id> | system; [] = owning project only"
        ),
    )


def _resolve_mcp_trust_classes(
    request: Request, server_ids: list[str]
) -> dict[str, str]:
    """D-24 declaration: resolve each member MCP server's trust class from
    the live registry; unresolvable members carry "" — the registry stays
    the truth, never an invented class."""
    if not server_ids:
        return {}
    classes: dict[str, str] = {}
    try:
        from .mcp import _get_store

        store = _get_store(request)
        for sid in server_ids:
            try:
                server = store.get_server(sid)
                classes[sid] = str(getattr(server, "trust_class", "") or "")
            except Exception:  # noqa: BLE001 — declaration is fail-open
                classes[sid] = ""
    except Exception:  # noqa: BLE001 — the MCP registry may be unwired; declaration fails open
        return {sid: "" for sid in server_ids}
    return classes


@router.post("", status_code=status.HTTP_201_CREATED)
def create_catalog(
    body: CatalogSpec,
    request: Request,
    store: ScientificCatalogStore = Depends(_store_dep),
    identity: CallerIdentity = Depends(get_caller_identity),
) -> dict[str, Any]:
    """Create a catalog version (idempotent on the project/name/version triple).
    # auth: verified
    """
    _require_project_scope(body.project_id, identity)
    body.mcp_trust_classes = _resolve_mcp_trust_classes(request, body.mcp_server_ids)
    owner = (identity.user_id or "").strip() or "unknown"
    return store.create(spec=body, owner_id=owner)


@router.get("")
def list_catalogs(
    request: Request,
    project_id: str | None = None,
    store: ScientificCatalogStore = Depends(_store_dep),
    identity: CallerIdentity = Depends(get_caller_identity),
) -> dict[str, Any]:
    """List active catalogs, optionally scoped to one project.

    Non-admin callers see their own project's rows plus rows whose ADR-080
    access list admits them (shared catalogs are marked by their access).
    # auth: verified
    """
    scope = project_id
    if not identity.is_admin:
        scope = identity.project_id or project_id
    catalogs = store.list(project_id=scope)
    if not identity.is_admin:
        seen = {row["catalog_id"] for row in catalogs}
        for shared in store.list_shared():
            if shared["catalog_id"] in seen:
                continue
            if _catalog_read_allowed(shared, request, identity):
                catalogs.append(shared)
                seen.add(shared["catalog_id"])
    return {"catalogs": catalogs}


@router.get("/{catalog_id}")
def get_catalog(
    catalog_id: str,
    request: Request,
    store: ScientificCatalogStore = Depends(_store_dep),
    identity: CallerIdentity = Depends(get_caller_identity),
) -> dict[str, Any]:
    """Fetch one catalog row by id (404 for unknown ids, never empty-smudge).

    Readable by the owning project, admins, and callers the row's ADR-080
    access list admits — a row outside all three is a 404 (no disclosure).
    # auth: verified
    """
    try:
        row = store.get(catalog_id)
    except CatalogNotFound:
        raise_typed(
            PlatformError.scientificcatalogs(
                "not_found", f"scientific catalog {catalog_id!r} not found"
            )
        )
    if not _catalog_read_allowed(row, request, identity):
        raise_typed(
            PlatformError.scientificcatalogs(
                "not_found", f"scientific catalog {catalog_id!r} not found"
            )
        )
    return row


@router.patch("/{catalog_id}")
def patch_catalog(
    catalog_id: str,
    body: CatalogPatch,
    store: ScientificCatalogStore = Depends(_store_dep),
    identity: CallerIdentity = Depends(get_caller_identity),
) -> dict[str, Any]:
    """Metadata-only update of a catalog version row.
    # auth: verified
    """
    try:
        row = store.get(catalog_id)
    except CatalogNotFound:
        raise_typed(
            PlatformError.scientificcatalogs(
                "not_found", f"scientific catalog {catalog_id!r} not found"
            )
        )
    _require_project_scope(row["project_id"], identity)
    return store.update_metadata(catalog_id, patch=body)


class PinBody(BaseModel):
    """One external pin onto a catalog version (D-43)."""

    kind: str = Field(..., min_length=1, description="job | result | finding | paper")
    ref: str = Field(..., min_length=1, description="job id / artifact id / doi / url")
    note: str = Field(default="")


@router.post("/{catalog_id}/pins", status_code=status.HTTP_201_CREATED)
def add_catalog_pin(
    catalog_id: str,
    body: PinBody,
    store: ScientificCatalogStore = Depends(_store_dep),
    identity: CallerIdentity = Depends(get_caller_identity),
) -> dict[str, Any]:
    """Append one pin (job/result/finding ref) to the catalog version row.
    # auth: verified
    """
    try:
        row = store.get(catalog_id)
    except CatalogNotFound:
        raise_typed(
            PlatformError.scientificcatalogs(
                "not_found", f"scientific catalog {catalog_id!r} not found"
            )
        )
    _require_project_scope(row["project_id"], identity)
    return store.add_pin(catalog_id, kind=body.kind, ref=body.ref, note=body.note)


@router.get("/pins/by-ref/{ref}")
def pins_by_ref(
    ref: str,
    project_id: str | None = None,
    store: ScientificCatalogStore = Depends(_store_dep),
    identity: CallerIdentity = Depends(get_caller_identity),
) -> dict[str, Any]:
    """Inverse pin index (D-45): which catalog versions pinned this ref.

    The join a draft needs to stamp catalog ids into its provenance graph.
    Project-scoped for non-admin callers (retired versions included —
    a retired catalog's pin is still provenance).
    # auth: verified
    """
    scope = project_id
    if not identity.is_admin:
        scope = identity.project_id or project_id
    return {"ref": ref, "pins": store.pins_by_ref(ref, project_id=scope)}


@router.get("/versions/list")
def list_versions(
    project_id: str,
    name: str,
    request: Request,
    store: ScientificCatalogStore = Depends(_store_dep),
    identity: CallerIdentity = Depends(get_caller_identity),
) -> dict[str, Any]:
    """One catalog's version chain (changelog view, oldest first).

    Rows the caller may not read (outside their project and not shared via
    ADR-080) are filtered out — the chain never leaks across projects.
    # auth: verified
    """
    rows = store.versions(project_id=project_id, name=name)
    visible = [row for row in rows if _catalog_read_allowed(row, request, identity)]
    return {"versions": visible}


@router.get("/{catalog_id}/access")
def get_catalog_access(
    catalog_id: str,
    request: Request,
    store: ScientificCatalogStore = Depends(_store_dep),
    identity: CallerIdentity = Depends(get_caller_identity),
) -> dict[str, Any]:
    """The catalog's ADR-080 access list (any caller who may read the row).
    # auth: verified
    """
    try:
        row = store.get(catalog_id)
    except CatalogNotFound:
        raise_typed(
            PlatformError.scientificcatalogs(
                "not_found", f"scientific catalog {catalog_id!r} not found"
            )
        )
    if not _catalog_read_allowed(row, request, identity):
        raise_typed(
            PlatformError.scientificcatalogs(
                "not_found", f"scientific catalog {catalog_id!r} not found"
            )
        )
    return {"catalog_id": row["catalog_id"], "access": row.get("access", [])}


@router.put("/{catalog_id}/access")
def set_catalog_access(
    catalog_id: str,
    body: AccessUpdate,
    request: Request,
    store: ScientificCatalogStore = Depends(_store_dep),
    identity: CallerIdentity = Depends(get_caller_identity),
) -> dict[str, Any]:
    """Replace the catalog's access list (owning project or admin; audited).

    Governs READS beyond the owning project; ``system`` is admin-only.
    Lifecycle writes stay project-scoped.
    # auth: verified
    """
    try:
        row = store.get(catalog_id)
    except CatalogNotFound:
        raise_typed(
            PlatformError.scientificcatalogs(
                "not_found", f"scientific catalog {catalog_id!r} not found"
            )
        )
    _require_project_scope(row["project_id"], identity)
    caller = resource_caller(
        request,
        user_id=identity.user_id,
        project_id=identity.project_id or "",
        is_admin=identity.is_admin,
    )
    access = checked_access(body.access, caller, error=PlatformError.scientificcatalogs)
    updated = store.set_access(catalog_id, access)
    from .audit_events import emit_audit

    emit_audit(
        "scientificcatalog.access_changed",
        project_id=row["project_id"],
        details={"catalog_id": catalog_id, "access": access, "by": identity.user_id or ""},
    )
    return updated


@router.delete("/{catalog_id}")
def retire_catalog(
    catalog_id: str,
    store: ScientificCatalogStore = Depends(_store_dep),
    identity: CallerIdentity = Depends(get_caller_identity),
) -> dict[str, Any]:
    """Tombstone a catalog version (retire — never a DELETE).
    # auth: verified
    """
    try:
        row = store.get(catalog_id)
    except CatalogNotFound:
        raise_typed(
            PlatformError.scientificcatalogs(
                "not_found", f"scientific catalog {catalog_id!r} not found"
            )
        )
    _require_project_scope(row["project_id"], identity)
    return store.retire(catalog_id)
