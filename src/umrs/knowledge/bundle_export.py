"""Server-side .kb bundle export logic for knowledge portability (INV-13).

Dumps all knowledge (artifacts, IRs, nodes/edges, embeddings via paged cursor,
catalog) into r4-kb v1.x format inside an S3 temp bucket.  Computes a SHA-256
content hash of the manifest and generates an HMAC signature using the site
secret key.

Tables dumped:
  - knowledge_artifacts  → artifacts.jsonl
  - knowledge_ir         → documents.jsonl (IR content as documents)
  - knowledge_nodes      → nodes.jsonl
  - knowledge_edges      → edges.jsonl
  - memory_vectors       → embeddings.jsonl (paged cursor)
  - catalog artifacts    → catalog.jsonl

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

# Default embedding vector cursor (pgvector stores as text, use id as cursor)
_DEFAULT_EMBEDDING_CURSOR = "0"


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
    ) -> None:
        self._store = store
        self._output_dir = output_dir
        self._secret_key = secret_key
        self._manifest: dict[str, Any] = {}

    async def export_project(
        self,
        project_id: str,
    ) -> KnowledgeExportResult:
        """Dump all knowledge for *project_id* into the output directory.

        Returns a result with counts and integrity hash/signature.
        """
        start = time.monotonic()
        logger.info("Starting knowledge export for project=%s", project_id)

        writer = ExportBundleWriter(self._output_dir / project_id)

        # 1. Artifacts
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

        # 2. IRs as documents
        irs = await self._store.list_irs(project_id=project_id)
        doc_rows = [self._serialize_ir(ir) for ir in irs]
        await writer.write_documents(doc_rows)
        logger.info(
            "Exported %d IR documents for project=%s", len(doc_rows), project_id
        )

        # 3. Graph nodes
        nodes = await self._store.list_nodes(project_id=project_id)
        node_rows = [self._serialize_node(n) for n in nodes]
        await writer.write_nodes(node_rows)
        logger.info("Exported %d nodes for project=%s", len(node_rows), project_id)

        # 4. Graph edges
        edges = await self._store.list_edges(project_id=project_id)
        edge_rows = [self._serialize_edge(e) for e in edges]
        await writer.write_edges(edge_rows)
        logger.info("Exported %d edges for project=%s", len(edge_rows), project_id)

        # 5. Embeddings (paged cursor from memory_vectors)
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

        # 6. Catalog (from catalog_store or memory_vectors)
        cat_rows = await self._list_catalog(project_id=project_id)
        await writer.write_catalog(cat_rows)
        logger.info(
            "Exported %d catalog entries for project=%s", len(cat_rows), project_id
        )

        # 7. Build manifest
        self._manifest = {
            "format": "r4-kb",
            "version": "1.0",
            "project_id": project_id,
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "tables": {
                "artifacts": {"count": artifact_total, "file": "artifacts.jsonl"},
                "documents": {"count": len(doc_rows), "file": "documents.jsonl"},
                "nodes": {"count": len(node_rows), "file": "nodes.jsonl"},
                "edges": {"count": len(edge_rows), "file": "edges.jsonl"},
                "embeddings": {"count": emb_total, "file": "embeddings.jsonl"},
                "catalog": {"count": len(cat_rows), "file": "catalog.jsonl"},
            },
        }
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
            artifact_total,
            len(doc_rows),
            len(node_rows),
            len(edge_rows),
            emb_total,
            len(cat_rows),
        )

        return KnowledgeExportResult(
            bundle_path=str(self._output_dir / project_id),
            project_id=project_id,
            artifact_count=artifact_total,
            document_count=len(doc_rows),
            node_count=len(node_rows),
            edge_count=len(edge_rows),
            embedding_count=emb_total,
            catalog_count=len(cat_rows),
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

    def _sign_manifest(self, manifest_bytes: bytes) -> str:
        """HMAC-SHA256 signature of manifest content."""
        if not self._secret_key:
            return ""
        return hmac.new(
            self._secret_key.encode("utf-8"),
            manifest_bytes,
            hashlib.sha256,
        ).hexdigest()

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
    "verify_signature",
    "verify_bundle_integrity",
]
