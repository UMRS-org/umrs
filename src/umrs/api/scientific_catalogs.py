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


@router.post("", status_code=status.HTTP_201_CREATED)
def create_catalog(
    body: CatalogSpec,
    store: ScientificCatalogStore = Depends(_store_dep),
    identity: CallerIdentity = Depends(get_caller_identity),
) -> dict[str, Any]:
    """Create a catalog version (idempotent on the project/name/version triple).
    # auth: verified
    """
    _require_project_scope(body.project_id, identity)
    owner = (identity.user_id or "").strip() or "unknown"
    return store.create(spec=body, owner_id=owner)


@router.get("")
def list_catalogs(
    project_id: str | None = None,
    store: ScientificCatalogStore = Depends(_store_dep),
    identity: CallerIdentity = Depends(get_caller_identity),
) -> dict[str, Any]:
    """List active catalogs, optionally scoped to one project.
    # auth: verified
    """
    scope = project_id
    if not identity.is_admin:
        scope = identity.project_id or project_id
    return {"catalogs": store.list(project_id=scope)}


@router.get("/{catalog_id}")
def get_catalog(
    catalog_id: str,
    store: ScientificCatalogStore = Depends(_store_dep),
    identity: CallerIdentity = Depends(get_caller_identity),
) -> dict[str, Any]:
    """Fetch one catalog row by id (404 for unknown ids, never empty-smudge).
    # auth: verified
    """
    try:
        return store.get(catalog_id)
    except CatalogNotFound:
        raise_typed(
            PlatformError.scientificcatalogs(
                "not_found", f"scientific catalog {catalog_id!r} not found"
            )
        )
    raise AssertionError("unreachable")  # raise_typed is NoReturn


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
    store: ScientificCatalogStore = Depends(_store_dep),
    identity: CallerIdentity = Depends(get_caller_identity),
) -> dict[str, Any]:
    """One catalog's version chain (changelog view, oldest first).
    # auth: verified
    """
    return {"versions": store.versions(project_id=project_id, name=name)}


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
