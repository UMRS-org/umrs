"""Scientific Catalog store — the foundation persistence layer (D-39).

The store is persistence, not routing: it lives here so foundation
modules (jobs monitor, adapters) can pin and read catalog rows without
importing the api package (ADR-053 — jobs must not import umrs.api).
The api module (``umrs.api.scientific_catalogs``) re-exports everything
for import compatibility.

Invariants pinned:

- one row per (project_id, name, version); creation of the same triple is
  idempotent (existing row returned as-is).
- version rows are immutable in their membership; PATCH touches
  description and metadata fields only — a new version is a new row.
- retire is a tombstone, never a DELETE.
- pins are append-only (D-43); unknown catalog ids raise CatalogNotFound,
  never a silent empty payload.

psycopg2-shaped sync store over the 0101 table (satellite pattern).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from pydantic import BaseModel, Field


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
    # D-24: the trust class each member MCP server carries (resolved from
    # the live registry by the route; {server_id: trust_class}; absent
    # member → "" — never invented).
    mcp_trust_classes: dict[str, str] = Field(default_factory=dict)
    embedding_model_ids: list[str] = Field(default_factory=list)
    contributors: list[str] = Field(default_factory=list)
    licence: str = Field(default="")
    provenance: dict[str, Any] = Field(default_factory=dict)
    change_note: str = Field(default="")
    redaction_rules: list[dict[str, str]] = Field(
        default_factory=list,
        description='D-46 export rules: [{"target": manifest_field, "action": "drop"|"hash"}]',
    )
    catalog_id: str = Field(default="")


_COLUMNS = (
    "catalog_id, project_id, name, version, description, ontology,"
    " member_graph_ids, skill_ids, mcp_server_ids, embedding_model_ids,"
    " mcp_trust_classes, contributors, licence, provenance, change_note,"
    " redaction_rules, owner_id, status, created_at, updated_at, access"
)


class CatalogPatch(BaseModel):
    """Metadata-only patch; membership is pinned at version creation."""

    description: str | None = None
    ontology: str | None = None
    contributors: list[str] | None = None
    licence: str | None = None
    provenance: dict[str, Any] | None = None
    redaction_rules: list[dict[str, str]] | None = None


class CatalogNotFound(Exception):
    """Raised when a catalog id does not exist (never silent)."""


class ScientificCatalogStore:
    """psycopg2-backed durable store over the 0101 registry."""

    def __init__(self, conn: Any) -> None:
        self._conn = conn

    # -- row helpers ------------------------------------------------------

    @staticmethod
    def _row_dict(row: Any) -> dict[str, Any]:
        """Read a catalog row; pre-0108 rows (19 cols, no declaration
        column) read the declaration as the honest empty dict; pre-0116
        rows (20 cols) read the ADR-080 access list as the honest []."""
        if len(row) < 20:
            return ScientificCatalogStore._row_dict_legacy(row)
        payload = {
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
            "mcp_trust_classes": _as_dict(row[10]),
            "contributors": _as_list(row[11]),
            "licence": row[12],
            "provenance": _as_dict(row[13]),
            "change_note": row[14],
            "redaction_rules": _as_list(row[15]),
            "owner_id": row[16],
            "status": row[17],
            "created_at": row[18].isoformat() if row[18] else None,
            "updated_at": row[19].isoformat() if row[19] else None,
            "access": _as_list(row[20]) if len(row) > 20 else [],
        }
        return payload

    @staticmethod
    def _row_dict_legacy(row: Any) -> dict[str, Any]:
        """Pre-0108 row shape (no mcp_trust_classes column)."""
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
            "mcp_trust_classes": {},
            "contributors": _as_list(row[10]),
            "licence": row[11],
            "provenance": _as_dict(row[12]),
            "change_note": row[13],
            "redaction_rules": _as_list(row[14]),
            "owner_id": row[15],
            "status": row[16],
            "created_at": row[17].isoformat() if row[17] else None,
            "updated_at": row[18].isoformat() if row[18] else None,
            "access": [],
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

    def create(self, *, spec: CatalogSpec, owner_id: str) -> dict[str, Any]:
        """Ensuring-shape: identical (project, name, version) returns the
        existing row; a mismatched re-create is a typed conflict."""
        existing = self._row_by_triple(spec.project_id, spec.name, spec.version)
        if existing is not None:
            return existing
        catalog_id = spec.catalog_id or _mint_catalog_id(spec, owner_id)
        provenance = self._pin_embedding_models(spec)
        cur = self._conn.cursor()
        cur.execute(
            """INSERT INTO scientific_catalogs (
                   catalog_id, project_id, name, version, description, ontology,
                   member_graph_ids, skill_ids, mcp_server_ids,
                   embedding_model_ids, mcp_trust_classes,
                   contributors, licence, provenance,
                   change_note, redaction_rules, owner_id
               ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                         %s, %s, %s, %s)
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
                json.dumps(spec.mcp_trust_classes),
                json.dumps(spec.contributors),
                spec.licence,
                json.dumps(provenance),
                spec.change_note,
                json.dumps(spec.redaction_rules),
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

    def list_shared(self) -> list[dict[str, Any]]:
        """Active rows carrying a non-empty ADR-080 access list — sharing
        candidates; the route filters by access_admits before disclosure."""
        cur = self._conn.cursor()
        cur.execute(
            f"SELECT {_COLUMNS} FROM scientific_catalogs"
            " WHERE status != 'retired' AND access != '[]'::jsonb"
            " ORDER BY project_id, name, created_at"
        )
        return [self._row_dict(row) for row in cur.fetchall()]

    def set_access(self, catalog_id: str, access: list[str]) -> dict[str, Any]:
        """Replace the row's ADR-080 access list (validated upstream)."""
        self.get(catalog_id)  # raises CatalogNotFound
        cur = self._conn.cursor()
        cur.execute(
            """UPDATE scientific_catalogs
               SET access = %s::jsonb, updated_at = now()
               WHERE catalog_id = %s""",
            (json.dumps(list(access)), catalog_id),
        )
        self._conn.commit()
        return self.get(catalog_id)

    def update_metadata(
        self, catalog_id: str, *, patch: CatalogPatch
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
                   redaction_rules = COALESCE(%s, redaction_rules),
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
                json.dumps(patch.redaction_rules)
                if patch.redaction_rules is not None
                else None,
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

    def pins_by_ref(
        self, ref: str, *, project_id: str | None = None
    ) -> list[dict[str, Any]]:
        """Inverse pin index (D-45): which catalog versions pinned ``ref``.

        Scans the append-only pins arrays for an exact ref match. Retired
        versions are included — a retired catalog's pin is still provenance.
        """
        cur = self._conn.cursor()
        where = " WHERE provenance->'pins' @> %s::jsonb"
        params: list[Any] = [json.dumps([{"ref": ref}])]
        if project_id:
            where += " AND project_id = %s"
            params.append(project_id)
        cur.execute(
            "SELECT catalog_id, project_id, provenance FROM scientific_catalogs"
            f"{where} ORDER BY catalog_id",
            tuple(params),
        )
        out: list[dict[str, Any]] = []
        for catalog_id, pid, prov in cur.fetchall():
            for pin in (prov or {}).get("pins", []):
                if pin.get("ref") == ref:
                    out.append(
                        {
                            "catalog_id": catalog_id,
                            "project_id": pid,
                            "kind": pin.get("kind", ""),
                            "note": pin.get("note", ""),
                        }
                    )
        return out

    def _pin_embedding_models(self, spec: CatalogSpec) -> dict[str, Any]:
        """D-11/D-43: resolve each embedding model id against the model
        registry at create time and stamp the pin into provenance.

        Fail-open by design: an id the registry does not know (e.g. a
        sentence-transformers name pinned from a bundle) records
        ``registry_resolved: False`` — the pin is honest, creation never
        blocked; a registry read failure records the same shape with
        ``registry_unreachable: True``.
        """
        if not spec.embedding_model_ids:
            return dict(spec.provenance)
        pins: list[dict[str, Any]] = []
        try:
            cur = self._conn.cursor()
            for model_id in spec.embedding_model_ids:
                cur.execute(
                    "SELECT 1 FROM model_catalog_entries"
                    " WHERE model_id = %s OR huggingface_id = %s OR name = %s"
                    " LIMIT 1",
                    (model_id, model_id, model_id),
                )
                pins.append(
                    {"id": model_id, "registry_resolved": cur.fetchone() is not None}
                )
        except Exception:  # noqa: BLE001 — pin is best-effort, never a gate
            pins = [
                {"id": m, "registry_resolved": False, "registry_unreachable": True}
                for m in spec.embedding_model_ids
            ]
        provenance = dict(spec.provenance)
        provenance["embedding_models"] = pins
        return provenance


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


def _mint_catalog_id(spec: CatalogSpec, owner_id: str) -> str:
    key = f"{spec.project_id}:{spec.name}:{spec.version}:{owner_id}"
    return f"scat-{hashlib.sha256(key.encode('utf-8')).hexdigest()[:16]}"
