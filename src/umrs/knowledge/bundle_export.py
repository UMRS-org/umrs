"""Server-side .kb bundle export logic for knowledge portability (INV-13).

Dumps all knowledge (artifacts, IRs, nodes/edges, embeddings via paged cursor,
catalog, and A2L lineage mappings) into r4-kb v1.x format inside an S3 temp
bucket.  Computes a SHA-256 content hash of the manifest and generates an
HMAC signature using the site secret key.

Tables dumped:
  - knowledge_artifacts  → artifacts.jsonl
  - knowledge_ir         → documents.jsonl (IR content as documents)
  - knowledge_nodes      → nodes.jsonl
  - knowledge_edges      → edges.jsonl
  - memory_vectors       → embeddings.jsonl (paged cursor)
  - catalog artifacts    → catalog.jsonl
  - knowledge_a2l_links  → a2l_links.jsonl (knowledge → A2L mapping)
  - a2l_artifact_refs    → a2l_refs.jsonl (A2L artifact identity metadata)

The a2l_links and a2l_refs tables are backward-compatible: older bundle
readers (v1.0) that do not know these tables simply ignore them.

Downloaded via Gateway with auth, validated on client with signature.sig.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .bundle_reader import ExportBundleWriter
from .store import KnowledgeStore

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

_HMAC_ALGORITHM = "sha256"
_SIGNATURE_FILENAME = "signature.sig"
_PAGE_SIZE = 500
_BUNDLE_FORMAT = "r4-kb"
_BUNDLE_VERSION = "1.2"
_GENERATOR_NAME = "umrs"

# Default embedding vector cursor (pgvector stores as text, use id as cursor)
_DEFAULT_EMBEDDING_CURSOR = "0"


class BundleIntegrityError(RuntimeError):
    """Raised when integrity requirements for an export cannot be met."""


def _read_umrs_version() -> str:
    """Installed umrs package version, or "unknown" when not installed."""
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as pkg_version

    try:
        return pkg_version("umrs")
    except PackageNotFoundError:
        return "unknown"


def _pin(
    *,
    model: str,
    model_path: str,
    revision: str,
    dimensions: int,
    pending: bool = False,
) -> dict[str, Any]:
    """Embedding trust pin block (ADR-032 amendment 2026-09)."""
    return {
        "model": model,
        "model_path": model_path,
        "revision": revision,
        "dimensions": dimensions,
        "distance": "cosine",
        "normalization": "unknown",
        "pending": pending,
    }


def _pending_pin() -> dict[str, Any]:
    """Pin emitted when the vector pipeline cannot be identified honestly."""
    return _pin(model="", model_path="", revision="", dimensions=0, pending=True)


def _model_basename(model_path: str) -> str:
    """Canonical-ish name from a pre-staged path (falls back to the value)."""
    if not model_path:
        return ""
    return model_path.rstrip("/").rsplit("/", 1)[-1]


def _json_list(value: Any) -> list[Any]:
    """JSONB column → list (asyncpg returns decoded; tolerate raw strings)."""
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except ValueError:
            return []
        return parsed if isinstance(parsed, list) else []
    return []


def _json_dict(value: Any) -> dict[str, Any]:
    """JSONB column → dict (same tolerance as ``_json_list``)."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except ValueError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


# ── Result DTO ─────────────────────────────────────────────────────────────────


@dataclass
class KnowledgeExportResult:
    """Result of a KNOWLEDGE_EXPORT job attempt."""

    bundle_path: str = ""  # S3 key of the exported bundle
    project_id: str = ""
    artifact_count: int = 0
    document_count: int = 0
    node_count: int = 0
    edge_count: int = 0
    embedding_count: int = 0
    catalog_count: int = 0
    manifest_hash: str = ""  # SHA-256 of manifest.json
    signature: str = ""  # HMAC-SHA256 of manifest
    degraded: bool = False
    degraded_reason: str = ""


# ── Export Engine ──────────────────────────────────────────────────────────────


class KnowledgeExportEngine:
    """Serializes knowledge tables → r4-kb v1.x bundle directory.

    Writes to a local directory first, then optionally uploads to S3.
    """

    def __init__(
        self,
        store: KnowledgeStore,
        output_dir: Path,
        secret_key: str = "",
        allow_unsigned: bool = False,
        populate_embedding_pin: bool = True,
    ) -> None:
        self._store = store
        self._output_dir = output_dir
        self._secret_key = secret_key
        self._allow_unsigned = allow_unsigned
        self._populate_embedding_pin = populate_embedding_pin
        self._manifest: dict[str, Any] = {}

    async def export_project(
        self,
        project_id: str,
    ) -> KnowledgeExportResult:
        """Dump all knowledge for *project_id* into the output directory.

        Returns a result with counts and integrity hash/signature.
        """
        start = time.monotonic()
        if not self._secret_key and not self._allow_unsigned:
            raise BundleIntegrityError(
                "bundle signing key is not configured (UMRS_SECRET_KEY); refusing "
                "to produce an unsigned bundle — configure the key or pass "
                "allow_unsigned=True"
            )
        if not self._secret_key:
            logger.warning(
                "PRODUCING UNSIGNED BUNDLE for project=%s (allow_unsigned opt-out)",
                project_id,
            )
        logger.info("Starting knowledge export for project=%s", project_id)

        writer = ExportBundleWriter(self._output_dir / project_id)

        table_counts = await self._dump_core_tables(project_id, writer)
        link_rows, ref_rows = await self._dump_a2l_tables(project_id, writer)
        table_counts["a2l_links"] = (len(link_rows), "a2l_links.jsonl")
        table_counts["a2l_refs"] = (len(ref_rows), "a2l_refs.jsonl")

        # 8. Build manifest (r4-kb v1.2 — ADR-032 amendment 2026-09)
        skills, mcp_configs = await self._resolve_context_references(project_id)
        embedding_pin = await self._resolve_embedding_pin(project_id)
        reconciliation_snapshot = self._resolve_reconciliation_snapshot(project_id)
        catalog_record = await self._resolve_catalog_record(project_id)
        self._manifest = self._build_manifest(
            project_id,
            table_counts,
            embedding_pin,
            skills,
            mcp_configs,
            reconciliation_snapshot,
            catalog_record,
        )
        await writer.write_manifest(self._manifest)

        # 8. Compute hash & signature
        manifest_bytes = json.dumps(self._manifest, sort_keys=True).encode("utf-8")
        manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
        signature = self._sign_manifest(manifest_bytes)

        self._manifest["hash"] = manifest_hash
        self._manifest["signature"] = signature
        await writer.write_manifest(self._manifest)
        await writer.write_signature(signature)

        elapsed = time.monotonic() - start
        logger.info(
            "Knowledge export completed: project=%s time=%.2fs artifacts=%d docs=%d nodes=%d edges=%d embeds=%d catalog=%d",
            project_id,
            elapsed,
            table_counts["artifacts"][0],
            table_counts["documents"][0],
            table_counts["nodes"][0],
            table_counts["edges"][0],
            table_counts["embeddings"][0],
            table_counts["catalog"][0],
        )

        return KnowledgeExportResult(
            bundle_path=str(self._output_dir / project_id),
            project_id=project_id,
            artifact_count=table_counts["artifacts"][0],
            document_count=table_counts["documents"][0],
            node_count=table_counts["nodes"][0],
            edge_count=table_counts["edges"][0],
            embedding_count=table_counts["embeddings"][0],
            catalog_count=table_counts["catalog"][0],
            manifest_hash=manifest_hash,
            signature=signature,
        )

    # ── Serialization helpers ─────────────────────────────────────────────────

    def _serialize_artifact(self, artifact: Any) -> dict[str, Any]:
        row = {
            "artifact_id": artifact.artifact_id,
            "content_hash": artifact.content_hash,
            "kind": artifact.kind,
            "project_id": artifact.project_id,
            "source_ref": artifact.source_ref,
            "size_bytes": artifact.size_bytes,
            "mime_type": artifact.mime_type,
            "status": artifact.status,
            "metadata": artifact.metadata,
        }
        if artifact.created_at:
            row["created_at"] = artifact.created_at.isoformat()
        if artifact.updated_at:
            row["updated_at"] = artifact.updated_at.isoformat()
        return row

    def _serialize_ir(self, ir: Any) -> dict[str, Any]:
        return {
            "ir_id": ir.ir_id,
            "artifact_id": ir.artifact_id,
            "ir_type": ir.ir_type,
            "version": ir.version,
            "content": ir.content,
            "content_hash": ir.content_hash,
            "status": ir.status,
            "created_at": ir.created_at.isoformat() if ir.created_at else None,
        }

    def _serialize_node(self, node: Any) -> dict[str, Any]:
        row = {
            "node_id": node.node_id,
            "project_id": node.project_id,
            "label": node.label,
            "node_type": node.node_type,
            "content": node.content,
            "content_hash": node.content_hash,
            "source_ir": node.source_ir,
        }
        if node.created_at:
            row["created_at"] = node.created_at.isoformat()
        if node.updated_at:
            row["updated_at"] = node.updated_at.isoformat()
        return row

    def _serialize_edge(self, edge: Any) -> dict[str, Any]:
        row = {
            "edge_id": edge.edge_id,
            "project_id": edge.project_id,
            "source_id": edge.source_id,
            "target_id": edge.target_id,
            "relation_type": edge.relation_type,
            "content": edge.content,
        }
        if edge.created_at:
            row["created_at"] = edge.created_at.isoformat()
        return row

    async def _dump_core_tables(
        self, project_id: str, writer: ExportBundleWriter
    ) -> dict[str, tuple[int, str]]:
        """Write artifacts/documents/nodes/edges/embeddings/catalog tables."""
        artifacts, artifact_total = await self._store.list_artifacts(
            project_id=project_id,
            status="active",
            page=1,
            page_size=_PAGE_SIZE,
        )
        artifact_rows = [self._serialize_artifact(a) for a in artifacts]
        await writer.write_artifacts(artifact_rows)
        logger.info(
            "Exported %d/%d artifacts for project=%s",
            len(artifact_rows),
            artifact_total,
            project_id,
        )

        irs = await self._store.list_irs(project_id=project_id)
        doc_rows = [self._serialize_ir(ir) for ir in irs]
        await writer.write_documents(doc_rows)
        logger.info(
            "Exported %d IR documents for project=%s", len(doc_rows), project_id
        )

        nodes = await self._store.list_nodes(project_id=project_id)
        node_rows = [self._serialize_node(n) for n in nodes]
        await writer.write_nodes(node_rows)
        logger.info("Exported %d nodes for project=%s", len(node_rows), project_id)

        edges = await self._store.list_edges(project_id=project_id)
        edge_rows = [self._serialize_edge(e) for e in edges]
        await writer.write_edges(edge_rows)
        logger.info("Exported %d edges for project=%s", len(edge_rows), project_id)

        emb_rows, emb_total = await self._list_embeddings_paged(
            store=self._store,
            project_id=project_id,
            cursor=_DEFAULT_EMBEDDING_CURSOR,
            page_size=_PAGE_SIZE,
        )
        await writer.write_embeddings(emb_rows)
        logger.info(
            "Exported %d/%d embeddings for project=%s",
            len(emb_rows),
            emb_total,
            project_id,
        )

        cat_rows = await self._list_catalog(project_id=project_id)
        await writer.write_catalog(cat_rows)
        logger.info(
            "Exported %d catalog entries for project=%s", len(cat_rows), project_id
        )

        return {
            "artifacts": (artifact_total, "artifacts.jsonl"),
            "documents": (len(doc_rows), "documents.jsonl"),
            "nodes": (len(node_rows), "nodes.jsonl"),
            "edges": (len(edge_rows), "edges.jsonl"),
            "embeddings": (emb_total, "embeddings.jsonl"),
            "catalog": (len(cat_rows), "catalog.jsonl"),
        }

    async def _dump_a2l_tables(
        self, project_id: str, writer: ExportBundleWriter
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Write A2L lineage tables (best-effort; empty on missing tables)."""
        link_rows = await self._fetch_optional_rows(
            writer,
            write="a2l_links",
            query=(
                "SELECT * FROM knowledge_a2l_links WHERE knowledge_artifact_id IN "
                "(SELECT id FROM knowledge_artifacts WHERE project_id = $1) "
                "ORDER BY derived_at"
            ),
            project_id=project_id,
            table_what="knowledge_a2l_links",
        )
        ref_rows = await self._fetch_optional_rows(
            writer,
            write="a2l_refs",
            query=(
                "SELECT * FROM a2l_artifact_refs WHERE artifact_id IN "
                "(SELECT id FROM knowledge_artifacts WHERE project_id = $1) "
                "ORDER BY synced_at"
            ),
            project_id=project_id,
            table_what="a2l_artifact_refs",
        )
        return link_rows, ref_rows

    async def _fetch_optional_rows(
        self,
        writer: ExportBundleWriter,
        *,
        write: str,
        query: str,
        project_id: str,
        table_what: str,
    ) -> list[dict[str, Any]]:
        """Fetch an optional A2L table; warn-and-empty on failure."""
        try:
            rows = await self._store._pool.fetch(query, project_id)  # type: ignore[union-attr]
            out = [dict(r) for r in rows]
            write_one = getattr(writer, f"write_{write}")
            await write_one(out)
            logger.info("Exported %d %s for project=%s", len(out), write, project_id)
            return out
        except Exception:
            logger.warning("%s unavailable for project=%s", table_what, project_id)
            return []

    def _sign_manifest(self, manifest_bytes: bytes) -> str:
        """HMAC-SHA256 signature of manifest content."""
        if not self._secret_key:
            return ""
        return hmac.new(
            self._secret_key.encode("utf-8"),
            manifest_bytes,
            hashlib.sha256,
        ).hexdigest()

    # ── Manifest (v1.2) ─────────────��─────────────────────────────────────────

    def _build_manifest(
        self,
        project_id: str,
        table_counts: dict[str, tuple[int, str]],
        embedding_pin: dict[str, Any],
        skills: list[dict[str, Any]] | None = None,
        mcp_configs: list[dict[str, Any]] | None = None,
        reconciliation_snapshot: dict[str, Any] | None = None,
        catalog_record: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Assemble the r4-kb v1.2 manifest (ADR-032 amendment 2026-09).

        ``catalog`` carries the D-39 catalog row (when the project has one);
        ``ownership`` is the ADR-032 ownership block — sourced from the
        catalog row, never invented (empty when no catalog exists).
        """
        catalog = catalog_record or {}
        return {
            "format": _BUNDLE_FORMAT,
            "version": _BUNDLE_VERSION,
            "project_id": project_id,
            "generator": _GENERATOR_NAME,
            "generator_version": _read_umrs_version(),
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "unsigned": not bool(self._secret_key),
            "skills": skills or [],
            "mcp_configs": mcp_configs or [],
            "catalog": catalog,
            "ownership": {
                "contributors": list(catalog.get("contributors") or []),
                "licence": str(catalog.get("licence") or ""),
            },
            "collision_policy": "abort",
            "provenance_canonical": False,
            "reconciliation_snapshot": reconciliation_snapshot,
            "embedding": embedding_pin,
            "tables": {
                name: {"count": count, "file": filename}
                for name, (count, filename) in table_counts.items()
            },
        }

    async def _resolve_catalog_record(self, project_id: str) -> dict[str, Any]:
        """Latest active D-39 catalog row for the project (best-effort).

        Read back verbatim into the manifest so an import can re-register
        the catalog (D-40 bundle = catalog object). {} when the project has
        no catalog row or the registry is unreachable.
        """
        try:
            row = await self._store._pool.fetchrow(  # type: ignore[union-attr]
                "SELECT name, version, description, ontology, licence, contributors,"
                " provenance FROM scientific_catalogs"
                " WHERE project_id = $1 AND status = 'active'"
                " ORDER BY created_at DESC LIMIT 1",
                project_id,
            )
            if row is None:
                return {}
            return {
                "name": str(row["name"] or ""),
                "version": str(row["version"] or ""),
                "description": str(row["description"] or ""),
                "ontology": str(row["ontology"] or ""),
                "licence": str(row["licence"] or ""),
                "contributors": _json_list(row["contributors"]),
                "provenance": _json_dict(row["provenance"]),
            }
        except Exception:  # noqa: BLE001 — catalog block is best-effort
            return {}

    def _resolve_reconciliation_snapshot(
        self, project_id: str
    ) -> dict[str, Any] | None:
        """Best-effort reconciliation snapshot summary (WP-18 attach leg)."""
        try:
            from .reconcile_snapshot import summarize_reconcile_snapshot

            return summarize_reconcile_snapshot(project_id)
        except Exception as exc:  # noqa: BLE001 - snapshot is best-effort metadata
            logger.warning(
                "reconciliation snapshot skipped for %s: %s", project_id, exc
            )
            return None

    async def _resolve_context_references(
        self, project_id: str
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """By-reference skills + MCP configs (ratified bundle contract).

        Best-effort, never crashes the export: empty lists when the
        registries are unreachable.
        """
        skill_rows = await self._fetch_refs(
            "SELECT name, version, scope FROM skills "
            "WHERE scope = 'system' OR (scope = 'project' AND project_id = $1)",
            project_id,
        )
        mcp_rows = await self._fetch_refs(
            "SELECT server_id, name, version FROM mcp_servers "
            "WHERE project_id = $1 OR project_id IS NULL",
            project_id,
        )
        skills = [
            {
                "ref": f"{r.get('scope', 'user')}:{r.get('name', '')}",
                "version_pin": str(r.get("version") or ""),
                "scope": str(r.get("scope") or ""),
            }
            for r in skill_rows
        ]
        mcp_configs = [
            {
                "ref": str(r.get("name") or r.get("server_id") or ""),
                "version_pin": str(r.get("version") or ""),
            }
            for r in mcp_rows
        ]
        return skills, mcp_configs

    async def _fetch_refs(self, query: str, project_id: str) -> list[dict[str, Any]]:
        """Run one by-reference registry query; [] on any failure."""
        try:
            rows = await self._store._pool.fetch(query, project_id)  # type: ignore[union-attr]
            return [dict(r) for r in rows]
        except Exception:  # noqa: BLE001 — best-effort manifest metadata
            return []

    async def _resolve_embedding_pin(self, project_id: str) -> dict[str, Any]:
        """Resolve the vector-pipeline trust pin (best-effort, never crashes)."""
        if not self._populate_embedding_pin:
            return _pending_pin()
        pin = await self._pin_from_memory_vectors(project_id)
        if pin is None:
            pin = await self._pin_from_node_embeddings(project_id)
        return pin if pin is not None else _pending_pin()

    async def _pin_from_memory_vectors(self, project_id: str) -> dict[str, Any] | None:
        """Pin from the newest non-'default' memory_vectors lineage row (0041)."""
        try:
            row = await self._store._pool.fetchrow(  # type: ignore[union-attr]
                "SELECT model, model_path, model_version, embedding_dimension "
                "FROM memory_vectors WHERE project_id = $1 "
                "AND model IS NOT NULL AND model <> 'default' "
                "ORDER BY created_at DESC LIMIT 1",
                project_id,
            )
            if row is None:
                return None
            return _pin(
                model=str(row["model"] or ""),
                model_path=str(row["model_path"] or ""),
                revision=str(row["model_version"] or ""),
                dimensions=int(row["embedding_dimension"] or 0),
            )
        except Exception:  # noqa: BLE001 — pin is best-effort metadata
            return None

    async def _pin_from_node_embeddings(self, project_id: str) -> dict[str, Any] | None:
        """Pin by aggregating knowledge_node_embeddings (0075) of this project's nodes."""
        try:
            row = await self._store._pool.fetchrow(  # type: ignore[union-attr]
                "SELECT kne.model_name, kne.dimension, COUNT(*) AS n "
                "FROM knowledge_node_embeddings kne "
                "JOIN knowledge_nodes kn ON kn.node_id = kne.node_id "
                "WHERE kn.project_id = $1 "
                "GROUP BY kne.model_name, kne.dimension "
                "ORDER BY n DESC LIMIT 1",
                project_id,
            )
            if row is None:
                return None
            model_path = str(row["model_name"] or "")
            return _pin(
                model=_model_basename(model_path),
                model_path=model_path,
                revision="",
                dimensions=int(row["dimension"] or 0),
            )
        except Exception:  # noqa: BLE001 — pin is best-effort metadata
            return None

    # ── Paged embedding fetch ─────────────────────────────────────────────────

    async def _list_embeddings_paged(
        self,
        store: KnowledgeStore,
        project_id: str,
        cursor: str,
        page_size: int,
    ) -> tuple[list[dict[str, Any]], int]:
        """Fetch embeddings via paged cursor from memory_vectors.

        Uses a cursor-based pagination on the id column for large datasets.
        Returns (rows, total_count).
        """
        try:
            async with store._pool.acquire() as conn:
                # Get total count
                total_row = await conn.fetchrow(
                    "SELECT COUNT(*) FROM memory_vectors WHERE project_id = $1",
                    project_id,
                )
                total = int(total_row["count"]) if total_row else 0

                # Cursor pagination: rows where id > cursor
                rows = await conn.fetch(
                    """SELECT id, vector, metadata, document_id, created_at
                       FROM memory_vectors
                       WHERE project_id = $1 AND id > $2
                       ORDER BY id
                       LIMIT $3""",
                    project_id,
                    cursor,
                    page_size,
                )

                result = []
                for row in rows:
                    result.append(
                        {
                            "id": str(row["id"]),
                            "vector": list(row["vector"]) if row["vector"] else [],
                            "metadata": row["metadata"] or {},
                            "document_id": row["document_id"],
                            "created_at": row["created_at"].isoformat()
                            if row["created_at"]
                            else None,
                        }
                    )
                return result, total
        except Exception as exc:
            logger.warning("Embedding fetch failed (degraded): %s", exc)
            return [], 0

    async def _list_catalog(self, project_id: str) -> list[dict[str, Any]]:
        """Fetch catalog entries from the catalog_store or memory_vectors.

        Falls back to empty list if catalog table unavailable.
        """
        try:
            async with self._store._pool.acquire() as conn:
                rows = await conn.fetch(
                    """SELECT id, name, project_id, metadata, created_at
                       FROM knowledge_catalog
                       WHERE project_id = $1
                       ORDER BY created_at""",
                    project_id,
                )
                return [
                    {
                        "id": str(r["id"]),
                        "name": r["name"],
                        "project_id": r["project_id"],
                        "metadata": r["metadata"] or {},
                        "created_at": r["created_at"].isoformat()
                        if r["created_at"]
                        else None,
                    }
                    for r in rows
                ]
        except Exception as exc:
            logger.warning("Catalog fetch failed (degraded): %s", exc)
            return []


# ── S3 Upload Helper ───────────────────────────────────────────────────────────


async def upload_bundle_to_s3(
    source_dir: Path,
    s3_client: Any,
    bucket: str,
    key: str,
) -> str:
    """Recursively upload a bundle directory to S3.

    Returns the S3 key of the uploaded bundle.
    """
    import os

    key = key.rstrip("/")
    for root, _dirs, files in os.walk(source_dir):
        for fname in files:
            local_path = Path(root) / fname
            rel_path = local_path.relative_to(source_dir)
            s3_key = f"{key}/{rel_path}"
            try:
                s3_client.upload_file(str(local_path), bucket, s3_key)
            except Exception as exc:
                logger.warning("S3 upload failed for %s: %s", s3_key, exc)
    return key


# ── Signature Verification ─────────────────────────────────────────────────────


def verify_signature(
    manifest_path: Path,
    signature_path: Path,
    secret_key: str,
) -> bool:
    """Verify the HMAC signature of a manifest.

    The signature covers the manifest content *without* hash/signature
    fields (the version that was present when export was computed).
    """
    if not manifest_path.exists() or not signature_path.exists():
        return False

    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    stored_sig = signature_path.read_text(encoding="utf-8").strip()

    if not stored_sig:
        logger.warning("Empty signature file — tampering detected")
        return False

    # Reconstruct pre-hash manifest (same state when signature was computed)
    manifest_data.pop("hash", None)
    manifest_data.pop("signature", None)
    manifest_bytes = json.dumps(manifest_data, sort_keys=True).encode("utf-8")

    computed_sig = hmac.new(
        secret_key.encode("utf-8"),
        manifest_bytes,
        hashlib.sha256,
    ).hexdigest()

    return hmac.compare_digest(computed_sig, stored_sig)


def verify_bundle_integrity(
    bundle_dir: Path,
    secret_key: str,
) -> bool:
    """Verify manifest content-hash and HMAC signature of a bundle directory.

    The manifest contains its own SHA-256 hash (stored before the hash field
    was added to the manifest dict).  We reconstruct the pre-hash manifest
    content by removing *hash* and *signature* fields, then compare.

    Returns True if both the content hash and HMAC signature match.
    """
    manifest_path = bundle_dir / "manifest.json"
    signature_path = bundle_dir / _SIGNATURE_FILENAME

    if not manifest_path.exists():
        logger.warning("No manifest found in bundle: %s", bundle_dir)
        return False

    manifest_bytes = manifest_path.read_bytes()
    manifest_data = json.loads(manifest_bytes)

    # 1. Check stored hash matches content (reconstruct manifest without
    # hash/signature to avoid circular dependency).
    stored_hash = manifest_data.pop("hash", "")
    manifest_data.pop("signature", None)
    if stored_hash:
        reconstructed = json.dumps(manifest_data, sort_keys=True).encode("utf-8")
        computed_hash = hashlib.sha256(reconstructed).hexdigest()
        if stored_hash != computed_hash:
            logger.warning("Manifest hash mismatch — tampering detected")
            return False

    # 2. Check HMAC signature on the full manifest (re-read with fields).
    return verify_signature(manifest_path, signature_path, secret_key)


__all__ = [
    "KnowledgeExportEngine",
    "KnowledgeExportResult",
    "upload_bundle_to_s3",
    "verify_bundle_integrity",
    "verify_signature",
]
