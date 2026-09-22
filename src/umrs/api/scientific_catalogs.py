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

import hashlib
import json
from typing import Any

from fastapi import APIRouter, Depends, Request, status
from pydantic import BaseModel, Field

from .jobs import CallerIdentity, get_caller_identity
from .typed_errors import PlatformError, raise_typed

_COLUMNS = (
    "catalog_id, project_id, name, version, description, ontology,"
    " member_graph_ids, skill_ids, mcp_server_ids, embedding_model_ids,"
    " contributors, licence, provenance, change_note, owner_id, status,"
    " created_at, updated_at"
)


class CatalogNotFound(Exception):
    """Raised when a catalog id does not exist (never silent)."""


class ScientificCatalogStore:
    """psycopg2-backed durable store over the 0101 registry."""

    def __init__(self, conn: Any) -> None:
        self._conn = conn

    # -- row helpers ------------------------------------------------------

    @staticmethod
    def _row_dict(row: Any) -> dict[str, Any]:
        return {
            "catalog_id": row[0],
            "project_id": row[1],
            "name": row[2],
            "version": row[3],
            "description": row[4],
            "ontology": row[5],
            "member_graph_ids": _as_list(row[6]),
            "skill_ids": _as_list(row[7]),
            "mcp_server_ids": _as_list(row[8]),
            "embedding_model_ids": _as_list(row[9]),
            "contributors": _as_list(row[10]),
            "licence": row[11],
            "provenance": _as_dict(row[12]),
            "change_note": row[13],
            "owner_id": row[14],
            "status": row[15],
            "created_at": row[16].isoformat() if row[16] else None,
            "updated_at": row[17].isoformat() if row[17] else None,
        }

    def _row(self, catalog_id: str) -> dict[str, Any] | None:
        cur = self._conn.cursor()
        cur.execute(
            f"SELECT {_COLUMNS} FROM scientific_catalogs WHERE catalog_id = %s",
            (catalog_id,),
        )
        row = cur.fetchone()
        return self._row_dict(row) if row else None

    def _row_by_triple(
        self, project_id: str, name: str, version: str
    ) -> dict[str, Any] | None:
        cur = self._conn.cursor()
        cur.execute(
            f"SELECT {_COLUMNS} FROM scientific_catalogs"
            " WHERE project_id = %s AND name = %s AND version = %s",
            (project_id, name, version),
        )
        row = cur.fetchone()
        return self._row_dict(row) if row else None

    # -- CRUD -------------------------------------------------------------

    def create(self, *, spec: "CatalogSpec", owner_id: str) -> dict[str, Any]:
        """Ensuring-shape: identical (project, name, version) returns the
        existing row; a mismatched re-create is a typed conflict."""
        existing = self._row_by_triple(spec.project_id, spec.name, spec.version)
        if existing is not None:
            return existing
        catalog_id = spec.catalog_id or _mint_catalog_id(spec, owner_id)
        cur = self._conn.cursor()
        cur.execute(
            """INSERT INTO scientific_catalogs (
                   catalog_id, project_id, name, version, description, ontology,
                   member_graph_ids, skill_ids, mcp_server_ids,
                   embedding_model_ids, contributors, licence, provenance,
                   change_note, owner_id
               ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                         %s, %s)
               ON CONFLICT (project_id, name, version) DO NOTHING""",
            (
                catalog_id,
                spec.project_id,
                spec.name,
                spec.version,
                spec.description,
                spec.ontology,
                json.dumps(spec.member_graph_ids),
                json.dumps(spec.skill_ids),
                json.dumps(spec.mcp_server_ids),
                json.dumps(spec.embedding_model_ids),
                json.dumps(spec.contributors),
                spec.licence,
                json.dumps(spec.provenance),
                spec.change_note,
                owner_id,
            ),
        )
        self._conn.commit()
        row = self._row_by_triple(spec.project_id, spec.name, spec.version)
        if row is None:
            raise RuntimeError(
                f"catalog insert lost (project={spec.project_id} name={spec.name})"
            )
        return row

    def get(self, catalog_id: str) -> dict[str, Any]:
        row = self._row(catalog_id)
        if row is None:
            raise CatalogNotFound(catalog_id)
        return row

    def list(self, *, project_id: str | None = None) -> list[dict[str, Any]]:
        cur = self._conn.cursor()
        where = " WHERE status != 'retired'"
        params: tuple = ()
        if project_id:
            where += " AND project_id = %s"
            params = (project_id,)
        cur.execute(
            f"SELECT {_COLUMNS} FROM scientific_catalogs{where}"
            " ORDER BY project_id, name, created_at",
            params,
        )
        return [self._row_dict(row) for row in cur.fetchall()]

    def update_metadata(
        self, catalog_id: str, *, patch: "CatalogPatch"
    ) -> dict[str, Any]:
        row = self.get(catalog_id)  # raises CatalogNotFound
        cur = self._conn.cursor()
        cur.execute(
            """UPDATE scientific_catalogs SET
                   description = COALESCE(%s, description),
                   ontology = COALESCE(%s, ontology),
                   contributors = COALESCE(%s, contributors),
                   licence = COALESCE(%s, licence),
                   provenance = COALESCE(%s, provenance),
                   updated_at = now()
               WHERE catalog_id = %s""",
            (
                patch.description,
                patch.ontology,
                json.dumps(patch.contributors)
                if patch.contributors is not None
                else None,
                patch.licence,
                json.dumps(patch.provenance) if patch.provenance is not None else None,
                catalog_id,
            ),
        )
        self._conn.commit()
        updated = self._row(catalog_id)
        return updated if updated is not None else row

    def retire(self, catalog_id: str) -> dict[str, Any]:
        row = self.get(catalog_id)  # raises CatalogNotFound
        if row["status"] == "retired":
            return row
        cur = self._conn.cursor()
        cur.execute(
            """UPDATE scientific_catalogs
               SET status = 'retired', updated_at = now()
               WHERE catalog_id = %s""",
            (catalog_id,),
        )
        self._conn.commit()
        return self._row(catalog_id)

    def versions(self, *, project_id: str, name: str) -> list[dict[str, Any]]:
        """One catalog's version chain (changelog view, oldest first)."""
        cur = self._conn.cursor()
        cur.execute(
            f"SELECT {_COLUMNS} FROM scientific_catalogs"
            " WHERE project_id = %s AND name = %s"
            " ORDER BY created_at, catalog_id",
            (project_id, name),
        )
        return [self._row_dict(row) for row in cur.fetchall()]

    def add_pin(
        self, catalog_id: str, *, kind: str, ref: str, note: str = ""
    ) -> dict[str, Any]:
        """Append one external pin (job/result/finding/paper ref) to the
        row's provenance.pins array — append-only, the row otherwise stays
        immutable (D-43)."""
        self.get(catalog_id)  # raises CatalogNotFound
        pin = json.dumps({"kind": kind, "ref": ref, "note": note})
        cur = self._conn.cursor()
        cur.execute(
            """UPDATE scientific_catalogs
               SET provenance = jsonb_set(
                       provenance, '{pins}',
                       COALESCE(provenance->'pins', '[]'::jsonb) || %s::jsonb,
                       true
                   ),
                   updated_at = now()
               WHERE catalog_id = %s""",
            (f"[{pin}]", catalog_id),
        )
        self._conn.commit()
        return self._row(catalog_id)


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except ValueError:
            return []
        return parsed if isinstance(parsed, list) else []
    return []


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except ValueError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _mint_catalog_id(spec: "CatalogSpec", owner_id: str) -> str:
    key = f"{spec.project_id}:{spec.name}:{spec.version}:{owner_id}"
    return f"scat-{hashlib.sha256(key.encode('utf-8')).hexdigest()[:16]}"


# -- HTTP routes ---------------------------------------------------------------

router = APIRouter(prefix="/umrs/scientific-catalogs", tags=["scientific-catalogs"])


class CatalogSpec(BaseModel):
    """Create shape: one catalog version row (immutable membership)."""

    project_id: str = Field(..., min_length=1)
    name: str = Field(..., min_length=1)
    version: str = Field(default="0.1.0", min_length=1)
    description: str = Field(default="")
    ontology: str = Field(default="")
    member_graph_ids: list[str] = Field(default_factory=list)
    skill_ids: list[str] = Field(default_factory=list)
    mcp_server_ids: list[str] = Field(default_factory=list)
    embedding_model_ids: list[str] = Field(default_factory=list)
    contributors: list[str] = Field(default_factory=list)
    licence: str = Field(default="")
    provenance: dict[str, Any] = Field(default_factory=dict)
    change_note: str = Field(default="")
    catalog_id: str = Field(default="")


class CatalogPatch(BaseModel):
    """Metadata-only patch; membership is pinned at version creation."""

    description: str | None = None
    ontology: str | None = None
    contributors: list[str] | None = None
    licence: str | None = None
    provenance: dict[str, Any] | None = None


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
