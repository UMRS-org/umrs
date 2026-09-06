"""K4: .kb bundle import/restore with manifest verification.

Restores an exported knowledge bundle (``r4-kb`` v1.1) into a project:
artifacts, IR documents, graph nodes/edges, embeddings, catalog rows,
and A2L bridge links/refs.

Integrity: when a secret key is configured, the bundle must pass
``verify_bundle_integrity`` (content hash + HMAC) before any write.

Idempotency: every restore step is re-runnable — artifacts dedup by
content hash, IRs are write-once, nodes/edges upsert, SQL restores use
``ON CONFLICT DO NOTHING``. Rows that fail to parse/insert are skipped
individually (row-level fault isolation), never aborting the run.

Retargeting: ``target_project_id`` rewrites every row's project scope —
the DR/migration path (restore source bundle into a new project).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from .bundle_export import verify_bundle_integrity
from .store import KnowledgeEdge, KnowledgeNode

logger = logging.getLogger(__name__)

_BUNDLE_FORMAT = "r4-kb"
_ACCEPTED_VERSIONS = {"1.0", "1.1", "1.2"}
_SIGNATURE_FILENAME = "signature.sig"

_TABLE_FILES: dict[str, str] = {
    "artifacts": "artifacts.jsonl",
    "documents": "documents.jsonl",
    "nodes": "nodes.jsonl",
    "edges": "edges.jsonl",
    "embeddings": "embeddings.jsonl",
    "catalog": "catalog.jsonl",
    "a2l_links": "a2l_links.jsonl",
    "a2l_refs": "a2l_refs.jsonl",
}


class BundleFormatError(ValueError):
    """Raised for integrity or format violations during restore."""


@dataclass
class KnowledgeImportResult:
    """Counts + integrity outcome of one restore run."""

    project_id: str = ""
    verified: bool = False
    artifact_count: int = 0
    document_count: int = 0
    node_count: int = 0
    edge_count: int = 0
    embedding_count: int = 0
    catalog_count: int = 0
    a2l_link_count: int = 0
    a2l_ref_count: int = 0
    skipped_count: int = 0
    degraded: bool = False
    degraded_reason: str = ""
    duration_seconds: float = 0.0
    unsigned: bool = False
    embedding_pin: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_id": self.project_id,
            "verified": self.verified,
            "artifact_count": self.artifact_count,
            "document_count": self.document_count,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "embedding_count": self.embedding_count,
            "catalog_count": self.catalog_count,
            "a2l_link_count": self.a2l_link_count,
            "a2l_ref_count": self.a2l_ref_count,
            "skipped_count": self.skipped_count,
            "degraded": self.degraded,
            "degraded_reason": self.degraded_reason,
            "duration_seconds": self.duration_seconds,
            "unsigned": self.unsigned,
            "embedding_pin": self.embedding_pin,
        }


def _iter_rows(path: Path) -> Iterator[dict[str, Any]]:
    """Yield JSONL rows, tolerating blank lines."""
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def _checked_manifest(bundle_path: Path) -> dict[str, Any]:
    """Load and validate manifest.json (format / version / project_id)."""
    manifest_path = bundle_path / "manifest.json"
    if not manifest_path.exists():
        raise BundleFormatError(f"no manifest.json in {bundle_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BundleFormatError(f"unreadable manifest: {exc}") from exc
    fmt = manifest.get("format")
    if fmt != _BUNDLE_FORMAT:
        raise BundleFormatError(f"unsupported bundle format: {fmt!r}")
    version = manifest.get("version")
    if version not in _ACCEPTED_VERSIONS:
        raise BundleFormatError(f"unsupported bundle version: {version!r}")
    if not manifest.get("project_id"):
        raise BundleFormatError("manifest missing project_id")
    return manifest


def _vector_literal(vector: Any) -> str:
    """Render a list of floats as a pgvector literal."""
    if not isinstance(vector, list):
        return "[]"
    return "[" + ",".join(str(v) for v in vector) + "]"


def _artifact_kwargs(row: dict[str, Any], project_id: str) -> dict[str, Any]:
    """Normalize an artifact row for ``KnowledgeStore.insert_artifact``."""
    return {
        "artifact_id": str(row["artifact_id"]),
        "content_hash": str(row.get("content_hash") or ""),
        "kind": str(row.get("kind") or "unknown"),
        "project_id": project_id,
        "source_ref": str(row.get("source_ref") or ""),
        "size_bytes": int(row.get("size_bytes") or 0),
        "mime_type": str(row.get("mime_type") or ""),
        "metadata": row.get("metadata") or {},
    }


def _ir_kwargs(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize an IR row for ``KnowledgeStore.insert_ir``."""
    return {
        "ir_id": str(row["ir_id"]),
        "artifact_id": str(row.get("artifact_id") or ""),
        "ir_type": str(row.get("ir_type") or "unknown"),
        "content": row.get("content") or {},
        "content_hash": str(row.get("content_hash") or ""),
        "version": int(row.get("version") or 1),
    }


async def _restore_jsonl_rows(
    path: Path,
    insert_one: Any,
) -> tuple[int, int]:
    """Restore rows via ``insert_one(row) -> bool``; row-fault-isolated.

    Returns ``(restored, skipped)``.
    """
    restored = skipped = 0
    if not path.exists():
        return restored, skipped
    for row in _iter_rows(path):
        try:
            ok = await insert_one(row)
        except (KeyError, TypeError, ValueError):
            skipped += 1
            continue
        if ok:
            restored += 1
        else:
            skipped += 1
    return restored, skipped


def _read_stored_signature(bundle_path: Path) -> str:
    """Signature carrier for the signed-marker decision.

    ``signature.sig`` is canonical: it is what ``verify_signature`` checks,
    and what the exporter always produces when a key is configured. A bundle
    whose manifest carries a signature field but has no ``signature.sig``
    (truncated package) counts as unsigned, matching the legacy behavior.
    """
    sig_path = bundle_path / _SIGNATURE_FILENAME
    if sig_path.exists():
        return sig_path.read_text(encoding="utf-8").strip()
    return ""


class KnowledgeImportEngine:
    """Restore engine (mirror of ``KnowledgeExportEngine``)."""

    def __init__(self, *, store: Any, secret_key: str = "") -> None:
        self._store = store
        self._secret_key = secret_key

    async def import_project(
        self,
        bundle_dir: str | Path,
        *,
        target_project_id: str | None = None,
        verify: bool = True,
        allow_unsigned: bool = False,
    ) -> KnowledgeImportResult:
        """Restore one bundle directory. Raises on integrity/format errors.

        Unsigned bundles are refused unless ``allow_unsigned=True``, regardless
        of ``verify`` (fail-closed per ADR-032 amendment 2026-09). ``verify``
        only gates the HMAC check for signed bundles.
        """
        t0 = time.monotonic()
        bundle_path = Path(bundle_dir)
        result = KnowledgeImportResult()

        manifest = _checked_manifest(bundle_path)
        embedding = manifest.get("embedding") or {}
        result.embedding_pin = bool(embedding.get("model")) and not embedding.get(
            "pending", False
        )
        self._enforce_integrity(bundle_path, verify, allow_unsigned, result)

        source_project = str(manifest["project_id"])
        target = target_project_id or source_project
        result.project_id = target
        if target != source_project:
            logger.info("Retargeting bundle restore: %s -> %s", source_project, target)

        (
            result.artifact_count,
            result.skipped_count,
        ) = await self._restore_artifacts(bundle_path, target)
        docs, doc_skipped = await self._restore_documents(bundle_path, target)
        result.document_count = docs
        result.skipped_count += doc_skipped
        result.node_count = await self._restore_nodes(bundle_path, target)
        result.edge_count = await self._restore_edges(bundle_path, target)
        (
            result.embedding_count,
            result.catalog_count,
        ) = await self._restore_sql_tables(bundle_path, target, result)
        (
            result.a2l_link_count,
            result.a2l_ref_count,
        ) = await self._restore_a2l(bundle_path, target)

        result.duration_seconds = time.monotonic() - t0
        logger.info(
            "Knowledge import completed: project=%s artifacts=%d docs=%d "
            "nodes=%d edges=%d skipped=%d verified=%s",
            target,
            result.artifact_count,
            result.document_count,
            result.node_count,
            result.edge_count,
            result.skipped_count,
            result.verified,
        )
        return result

    def _enforce_integrity(
        self,
        bundle_path: Path,
        verify: bool,
        allow_unsigned: bool,
        result: KnowledgeImportResult,
    ) -> None:
        """Fail-closed integrity gate (ADR-032 amendment 2026-09)."""
        stored_sig = _read_stored_signature(bundle_path)
        if not stored_sig:
            if not allow_unsigned:
                raise BundleFormatError(
                    "bundle is unsigned; pass allow_unsigned=True to restore anyway"
                )
            logger.warning("Restoring UNSIGNED bundle %s (allow_unsigned)", bundle_path)
            result.unsigned = True
            return
        if not verify:
            return
        if not self._secret_key:
            if not allow_unsigned:
                raise BundleFormatError(
                    "cannot verify signed bundle: UMRS_SECRET_KEY is not configured; "
                    "set the key or pass allow_unsigned=True"
                )
            logger.warning(
                "Skipping signature check on signed bundle %s (no key; allow_unsigned)",
                bundle_path,
            )
            result.unsigned = True
            return
        if not verify_bundle_integrity(bundle_path, self._secret_key):
            raise BundleFormatError("bundle integrity verification failed")
        result.verified = True

    # ── Store-backed restores ───────────────────────────────────────────

    async def _restore_artifacts(
        self, bundle_path: Path, project_id: str
    ) -> tuple[int, int]:
        return await self._restore_identity_table(
            bundle_path / _TABLE_FILES["artifacts"],
            table="knowledge_artifacts",
            id_column="artifact_id",
            row_id_key="artifact_id",
            insert=lambda row: self._store.insert_artifact(
                **_artifact_kwargs(row, project_id)
            ),
        )

    async def _restore_identity_table(
        self,
        path: Path,
        *,
        table: str,
        id_column: str,
        row_id_key: str,
        insert: Any,
    ) -> tuple[int, int]:
        """Restore identity-bearing rows with ADR-032 #4 collision-abort.

        Same identity + same content_hash → skip (idempotent rerun).
        Same identity + different content_hash → abort with deterministic diff.
        """
        restored = skipped = 0
        if not path.exists():
            return restored, skipped
        for row in _iter_rows(path):
            outcome = await self._collision_decision(
                table, id_column, str(row[row_id_key]), str(row.get("content_hash") or "")
            )
            if outcome == "skip":
                skipped += 1
                continue
            if outcome == "abort":
                raise BundleFormatError(
                    f"knowledge identity collision: {table}/{row[row_id_key]}: existing "
                    f"row has a different content_hash — import aborted per ADR-032 #4 "
                    "(deterministic diff pending upgrade; no rows re-written)"
                )
            try:
                ok = await insert(row)
            except (KeyError, TypeError, ValueError):
                skipped += 1
                continue
            if ok:
                restored += 1
            else:
                skipped += 1
        return restored, skipped

    async def _collision_decision(
        self, table: str, id_column: str, row_id: str, new_hash: str
    ) -> str:
        """Resolve skip/insert/abort for one identity row (best-effort)."""
        pool = self._pool()
        if pool is None or not new_hash:
            return "insert"
        try:
            async with pool.acquire() as conn:
                existing = await conn.fetchval(
                    f"SELECT content_hash FROM {table} WHERE {id_column} = $1",
                    row_id,
                )
        except Exception:  # noqa: BLE001 — untreated databases fall back to inserts
            return "insert"
        if existing is None:
            return "insert"
        return "skip" if str(existing) == new_hash else "abort"

    async def _restore_documents(
        self, bundle_path: Path, project_id: str
    ) -> tuple[int, int]:
        del project_id  # IRs bind project through their artifact link
        return await _restore_jsonl_rows(
            bundle_path / _TABLE_FILES["documents"],
            lambda row: self._store.insert_ir(**_ir_kwargs(row)),
        )

    async def _restore_nodes(self, bundle_path: Path, project_id: str) -> int:
        count = 0
        path = bundle_path / _TABLE_FILES["nodes"]
        if not path.exists():
            return count
        for row in _iter_rows(path):
            outcome = await self._collision_decision(
                "knowledge_nodes",
                "node_id",
                str(row["node_id"]),
                str(row.get("content_hash") or ""),
            )
            if outcome == "abort":
                raise BundleFormatError(
                    f"knowledge identity collision: knowledge_nodes/{row['node_id']}: "
                    "existing row has a different content_hash — import aborted "
                    "per ADR-032 #4"
                )
            if outcome == "skip":
                continue
            try:
                await self._store.upsert_node(self._node_from_row(row, project_id))
                count += 1
            except (KeyError, TypeError, ValueError) as exc:
                logger.warning("Skipping malformed node row: %s", exc)
        return count

    @staticmethod
    def _node_from_row(row: dict[str, Any], project_id: str) -> KnowledgeNode:
        return KnowledgeNode(
            node_id=str(row["node_id"]),
            project_id=project_id,
            label=str(row.get("label") or ""),
            node_type=str(row.get("node_type") or "unknown"),
            content=row.get("content") or {},
            content_hash=str(row.get("content_hash") or ""),
            source_ir=str(row.get("source_ir") or ""),
        )

    async def _restore_edges(self, bundle_path: Path, project_id: str) -> int:
        count = 0
        path = bundle_path / _TABLE_FILES["edges"]
        if not path.exists():
            return count
        for row in _iter_rows(path):
            try:
                await self._store.upsert_edge(self._edge_from_row(row, project_id))
                count += 1
            except (KeyError, TypeError, ValueError) as exc:
                logger.warning("Skipping malformed edge row: %s", exc)
        return count

    @staticmethod
    def _edge_from_row(row: dict[str, Any], project_id: str) -> KnowledgeEdge:
        return KnowledgeEdge(
            edge_id=str(row["edge_id"]),
            project_id=project_id,
            source_id=str(row.get("source_id") or ""),
            target_id=str(row.get("target_id") or ""),
            relation_type=str(row.get("relation_type") or "related_to"),
            content=row.get("content") or {},
        )

    # ── SQL restores (embeddings + catalog) ─────────────────────────────

    async def _restore_sql_tables(
        self,
        bundle_path: Path,
        project_id: str,
        result: KnowledgeImportResult,
    ) -> tuple[int, int]:
        embeddings = await self._restore_embeddings(bundle_path, project_id, result)
        catalog = await self._restore_catalog(bundle_path, project_id, result)
        return embeddings, catalog

    def _pool(self) -> Any:
        return getattr(self._store, "_pool", None)

    def _mark_degraded(self, result: KnowledgeImportResult, note: str) -> None:
        result.degraded = True
        result.degraded_reason = (
            f"{result.degraded_reason}; {note}" if result.degraded_reason else note
        )

    async def _restore_embeddings(
        self,
        bundle_path: Path,
        project_id: str,
        result: KnowledgeImportResult,
    ) -> int:
        path = bundle_path / _TABLE_FILES["embeddings"]
        if not path.exists():
            return 0
        pool = self._pool()
        if pool is None:
            self._mark_degraded(result, "embeddings skipped (no DB pool)")
            return 0
        count = 0
        async with pool.acquire() as conn:
            for row in _iter_rows(path):
                try:
                    await conn.execute(
                        """INSERT INTO memory_vectors
                           (project_id, vector, metadata, document_id)
                           VALUES ($1, $2, $3, $4)
                           ON CONFLICT DO NOTHING""",
                        project_id,
                        _vector_literal(row.get("vector")),
                        row.get("metadata") or {},
                        row.get("document_id"),
                    )
                    count += 1
                except (KeyError, TypeError, ValueError) as exc:
                    logger.warning("Skipping malformed embedding row: %s", exc)
        return count

    async def _restore_catalog(
        self,
        bundle_path: Path,
        project_id: str,
        result: KnowledgeImportResult,
    ) -> int:
        path = bundle_path / _TABLE_FILES["catalog"]
        if not path.exists():
            return 0
        pool = self._pool()
        if pool is None:
            self._mark_degraded(result, "catalog skipped (no DB pool)")
            return 0
        count = 0
        async with pool.acquire() as conn:
            for row in _iter_rows(path):
                try:
                    await conn.execute(
                        """INSERT INTO knowledge_catalog
                           (name, project_id, metadata)
                           VALUES ($1, $2, $3)
                           ON CONFLICT DO NOTHING""",
                        str(row["name"]),
                        project_id,
                        row.get("metadata") or {},
                    )
                    count += 1
                except (KeyError, TypeError, ValueError) as exc:
                    logger.warning("Skipping malformed catalog row: %s", exc)
        return count

    # ── A2L bridge restores ─────────────────────────────────────────────

    async def _restore_a2l(self, bundle_path: Path, project_id: str) -> tuple[int, int]:
        links = await self._restore_a2l_links(bundle_path)
        refs = await self._restore_a2l_refs(bundle_path)
        return links, refs

    async def _restore_a2l_links(self, bundle_path: Path) -> int:
        path = bundle_path / _TABLE_FILES["a2l_links"]
        if not path.exists():
            return 0
        count = 0
        for row in _iter_rows(path):
            try:
                ok = await self._store.insert_knowledge_a2l_link(
                    knowledge_artifact_id=str(row["knowledge_artifact_id"]),
                    a2l_artifact_id=str(row.get("a2l_artifact_id") or ""),
                    job_id=str(row.get("job_id") or ""),
                    attempt_id=str(row.get("attempt_id") or ""),
                    edge_kind=str(row.get("edge_kind") or "derived"),
                )
            except (KeyError, TypeError, ValueError) as exc:
                logger.warning("Skipping malformed a2l link row: %s", exc)
                continue
            if ok:
                count += 1
        return count

    async def _restore_a2l_refs(self, bundle_path: Path) -> int:
        path = bundle_path / _TABLE_FILES["a2l_refs"]
        if not path.exists():
            return 0
        count = 0
        for row in _iter_rows(path):
            try:
                ok = await self._store.insert_a2l_ref(
                    artifact_id=str(row["artifact_id"]),
                    a2l_hash=str(row.get("a2l_hash") or ""),
                    a2l_id=str(row.get("a2l_id") or ""),
                    source=str(row.get("source") or "auto"),
                )
            except (KeyError, TypeError, ValueError) as exc:
                logger.warning("Skipping malformed a2l ref row: %s", exc)
                continue
            if ok:
                count += 1
        return count
