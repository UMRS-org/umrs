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

import asyncio
import json
import logging
import os
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import bundle_raw, canonical_identity
from .bundle_export import tables_pinned, verify_bundle_integrity
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
    "node_embeddings": "node_embeddings.jsonl",
}

# ADR-032 precondition #3: the destination trusts exactly one pipeline — the
# site's own knowledge embedder (the pre-staged directory KNOWLEDGE_EMBED
# writes with). Same env name as jobs/knowledge_adapters (not imported: the
# knowledge package never imports jobs).
_DEST_EMBED_MODEL_ENV = "UMRS_KNOWLEDGE_EMBED_MODEL"


class BundleFormatError(ValueError):
    """Raised for integrity or format violations during restore."""


@dataclass
class KnowledgeImportResult:
    """Counts + integrity outcome of one restore run."""

    project_id: str = ""
    verified: bool = False
    # The signature also covered every table's contents (bundles exported
    # since 2026-09-24 pin each table's sha256); False on older bundles.
    tables_verified: bool = False
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
    collision_checks_degraded: int = 0
    # V-05/D-23: consumed skills / mcp_configs references (r4-kb >= 1.2 refs).
    skill_refs_imported: int = 0
    skill_refs_existing: int = 0
    mcp_refs_present: int = 0
    mcp_refs_missing: list[str] = field(default_factory=list)
    reference_stores_deferred: bool = False
    # D-40/D-39 tie: the D-39 catalog row re-registered from the manifest block.
    catalog_registered: bool = False
    # D-41: source version when the bundle was upgraded on read ("" when 1.2).
    upgraded_from: str = ""
    # D-40: canonical node vectors (0075) imported under the destination's
    # trusted model / left for re-embedding (ADR-032 precondition #3).
    node_embedding_count: int = 0
    node_embeddings_pending: int = 0
    # ADR-032 raw-materials amendment: objects stored in the destination and
    # re-bound to their artifact / carried but not bound (the artifact row
    # deduped onto an existing one) / declined by the importer.
    raw_restored: int = 0
    raw_unbound: int = 0
    raw_declined: int = 0
    # ADR-032 #1: incoming artifacts that are the same logical document as an
    # existing one under another id (reported, never merged — federation is
    # outside v1).
    canonical_matches: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_id": self.project_id,
            "verified": self.verified,
            "tables_verified": self.tables_verified,
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
            "collision_checks_degraded": self.collision_checks_degraded,
            "skill_refs_imported": self.skill_refs_imported,
            "skill_refs_existing": self.skill_refs_existing,
            "mcp_refs_present": self.mcp_refs_present,
            "mcp_refs_missing": list(self.mcp_refs_missing),
            "reference_stores_deferred": self.reference_stores_deferred,
            "catalog_registered": self.catalog_registered,
            "node_embedding_count": self.node_embedding_count,
            "node_embeddings_pending": self.node_embeddings_pending,
            "upgraded_from": self.upgraded_from,
            "canonical_matches": list(self.canonical_matches),
            "raw_restored": self.raw_restored,
            "raw_unbound": self.raw_unbound,
            "raw_declined": self.raw_declined,
        }


def _iter_rows(path: Path) -> Iterator[dict[str, Any]]:
    """Yield JSONL rows, tolerating blank lines."""
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


_CURRENT_VERSION = "1.2"
_LEGACY_VERSIONS = {"1.0", "1.1"}


def _upgrade_backfill(manifest: dict[str, Any]) -> str:
    """Back-fill v1.2 blocks missing from a 1.0/1.1 manifest (D-41).

    Mutates ``manifest`` in place so the consume legs (references, catalog
    row, embedding pin) see one shape regardless of source version. Returns
    the original version when an upgrade happened, "" for a 1.2 bundle.
    """
    version = str(manifest.get("version") or "")
    if version not in _LEGACY_VERSIONS:
        return ""
    manifest.setdefault("skills", [])
    manifest.setdefault("mcp_configs", [])
    manifest.setdefault("catalog", {})
    manifest.setdefault("ownership", {"contributors": [], "licence": ""})
    if "embedding" not in manifest:
        manifest["embedding"] = {"model": "", "dimensions": 0, "pending": True}
    return version


def _write_upgraded_manifest(bundle_path: Path, manifest: dict[str, Any]) -> None:
    """Rewrite manifest.json at v1.2; the stale hash/signature is dropped
    (an upgraded bundle is unsigned until re-exported)."""
    upgraded = dict(manifest)
    upgraded["version"] = _CURRENT_VERSION
    upgraded["unsigned"] = True
    upgraded.pop("hash", None)
    upgraded.pop("signature", None)
    (bundle_path / "manifest.json").write_text(
        json.dumps(upgraded, indent=2), encoding="utf-8"
    )
    sig = bundle_path / _SIGNATURE_FILENAME
    if sig.exists():
        sig.unlink()
    logger.info(
        "Upgraded bundle manifest to v%s at %s (unsigned until re-export)",
        _CURRENT_VERSION,
        bundle_path,
    )


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
        "metadata": _metadata_with_identity(row),
    }


def _metadata_with_identity(row: dict[str, Any]) -> dict[str, Any]:
    """Artifact metadata carrying the exported canonical identity (ADR-032 #1);
    a missing or malformed block is dropped, never trusted."""
    metadata = canonical_identity.metadata_dict(row.get("metadata"))
    metadata.pop("canonical_identity", None)
    block = canonical_identity.from_row(row)
    if block is not None:
        metadata["canonical_identity"] = block
    return metadata


def _incoming_identities(path: Path) -> dict[str, list[str]]:
    """document_id -> incoming artifact ids, from the bundle's artifact rows."""
    incoming: dict[str, list[str]] = {}
    if not path.exists():
        return incoming
    for row in _iter_rows(path):
        block = canonical_identity.from_row(row)
        if block is not None:
            incoming.setdefault(block["document_id"], []).append(
                str(row.get("artifact_id") or "")
            )
    return incoming


def _match_pairs(
    incoming: dict[str, list[str]], local: dict[str, list[str]]
) -> list[dict[str, str]]:
    """Same logical document under a different local id (sorted, deterministic)."""
    pairs = [
        {"document_id": doc, "incoming_artifact_id": mine, "local_artifact_id": theirs}
        for doc, ids in incoming.items()
        for mine in ids
        for theirs in local.get(doc, [])
        if theirs != mine
    ]
    return sorted(pairs, key=lambda p: (p["document_id"], p["incoming_artifact_id"]))


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


def _row_summary(row: dict[str, Any], existing_hash: str) -> dict[str, Any]:
    """Deterministic identity summary of an existing row for abort diffs."""
    return {
        "content_hash": existing_hash[:12],
        "kind": str(row.get("kind") or ""),
        "label": str(row.get("label") or ""),
        "size_bytes": int(row.get("size_bytes") or 0),
    }


def _diff_message(diff: dict[str, Any] | None) -> str:
    """Deterministic abort message for an identity collision (ADR-032 #4)."""
    return (
        "knowledge identity collision — import aborted per ADR-032 #4 "
        "(re-import aborted safely; resolve the mismatch and retry): "
        + json.dumps(diff or {}, sort_keys=True)
    )


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

    def __init__(
        self,
        *,
        store: Any,
        secret_key: str = "",
        skill_store: Any = None,
        mcp_store: Any = None,
        catalog_store: Any = None,
        raw_writer: Any = None,
    ) -> None:
        self._store = store
        # ADR-032 raw-materials amendment: ``raw_writer(project_id, bytes) ->
        # object key`` stores one verified object in the destination's own
        # storage (the api layer supplies it; None = raw objects not restored).
        self._raw_writer = raw_writer
        self._secret_key = secret_key
        # V-05/D-23: reference consumers. None = references resolved-byside
        # (counted as deferred in the result, never a crash).
        self._skill_store = skill_store
        self._mcp_store = mcp_store
        # D-40/D-39 tie: catalog-row adapter (api layer supplies
        # register_from_bundle(spec_dict, owner_id=...) — never umrs.api here).
        self._catalog_store = catalog_store
        self._collision_degraded: int = 0
        self._degraded_seen: set[str] = set()

    async def import_project(
        self,
        bundle_dir: str | Path,
        *,
        target_project_id: str | None = None,
        verify: bool = True,
        allow_unsigned: bool = False,
        write_upgrade: bool = False,
        accept_raw: bool = True,
    ) -> KnowledgeImportResult:
        """Restore one bundle directory. Raises on integrity/format errors.

        Unsigned bundles are refused unless ``allow_unsigned=True``, regardless
        of ``verify`` (fail-closed per ADR-032 amendment 2026-09). ``verify``
        only gates the HMAC check for signed bundles.

        D-41: bundles older than 1.2 are upgraded on read (missing blocks
        back-filled); ``write_upgrade`` additionally rewrites the bundle's
        manifest.json at v1.2 (the old signature is dropped — upgraded
        bundles are unsigned until re-exported).
        """
        t0 = time.monotonic()
        bundle_path = Path(bundle_dir)
        result = KnowledgeImportResult()

        manifest = _checked_manifest(bundle_path)
        result.upgraded_from = _upgrade_backfill(manifest)
        embedding = manifest.get("embedding") or {}
        result.embedding_pin = bool(embedding.get("model")) and not embedding.get(
            "pending", False
        )
        self._enforce_integrity(bundle_path, verify, allow_unsigned, result)
        result.tables_verified = result.verified and tables_pinned(manifest)
        raw_rows = self._raw_rows(bundle_path, accept_raw, result)

        source_project = str(manifest["project_id"])
        target = target_project_id or source_project
        result.project_id = target
        if target != source_project:
            logger.info("Retargeting bundle restore: %s -> %s", source_project, target)

        result.canonical_matches = await self._canonical_matches(bundle_path, target)
        (
            result.artifact_count,
            result.skipped_count,
        ) = await self._restore_artifacts(bundle_path, target)
        docs, doc_skipped = await self._restore_documents(bundle_path, target)
        result.document_count = docs
        result.skipped_count += doc_skipped
        await self._restore_raw(bundle_path, raw_rows, target, result)
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
        self._consume_context_references(manifest, target, result)
        result.catalog_registered = self._register_catalog_row(manifest, target)
        if result.upgraded_from and write_upgrade:
            _write_upgraded_manifest(bundle_path, manifest)

        result.duration_seconds = time.monotonic() - t0
        result.collision_checks_degraded = self._collision_degraded
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

    def _raw_rows(
        self, bundle_path: Path, accept: bool, result: KnowledgeImportResult
    ) -> list[dict[str, Any]]:
        """Verified raw rows BEFORE any write: one bad object refuses the
        whole bundle. Declined (or no writer) → nothing is read or stored."""
        if not (bundle_path / bundle_raw.RAW_TABLE).is_file():
            return []
        if not accept or self._raw_writer is None:
            result.raw_declined = sum(
                1
                for line in (bundle_path / bundle_raw.RAW_TABLE).open()
                if line.strip()
            )
            return []
        try:
            return bundle_raw.verified_raw_rows(bundle_path)
        except (bundle_raw.RawBundleError, ValueError) as exc:
            raise BundleFormatError(f"raw objects refused: {exc}") from exc

    async def _restore_raw(
        self,
        bundle_path: Path,
        rows: list[dict[str, Any]],
        project_id: str,
        result: KnowledgeImportResult,
    ) -> None:
        """Store each object in the destination and re-bind its artifact's
        ``source_ref`` (the one deployment-local field; ids/hashes kept)."""
        for row in rows:
            content = bundle_raw.read_object(bundle_path, row)
            try:
                key = await asyncio.to_thread(self._raw_writer, project_id, content)
            except Exception as exc:  # noqa: BLE001 — named degraded, rows stay restored
                result.degraded = True
                result.degraded_reason = f"raw objects not stored: {exc}"
                logger.warning("raw object store failed for %s: %s", project_id, exc)
                return
            bound = await self._store.rebind_artifact_source(
                project_id, str(row.get("artifact_id") or ""), key
            )
            if bound:
                result.raw_restored += 1
            else:
                result.raw_unbound += 1

    async def _canonical_matches(
        self, bundle_path: Path, project_id: str
    ) -> list[dict[str, str]]:
        """ADR-032 #1: incoming documents already present under another id,
        looked up before the restore (a store without the lookup: none)."""
        incoming = _incoming_identities(bundle_path / _TABLE_FILES["artifacts"])
        lookup = getattr(self._store, "artifacts_by_canonical_document", None)
        if not incoming or lookup is None:
            return []
        try:
            local = await lookup(project_id, sorted(incoming))
        except Exception as exc:  # noqa: BLE001 — report degraded, the restore proceeds
            self._note_degraded("canonical_identity", f"lookup failed: {exc}")
            return []
        return _match_pairs(incoming, local)

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
            outcome, diff = await self._collision_decision(
                table,
                id_column,
                str(row[row_id_key]),
                str(row.get("content_hash") or ""),
            )
            if outcome == "skip":
                skipped += 1
                continue
            if outcome == "abort":
                raise BundleFormatError(_diff_message(diff))
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
    ) -> tuple[str, dict[str, Any] | None]:
        """Resolve skip/insert/abort (+ deterministic diff) for one identity row.

        When the identity lookup itself is unavailable, the run continues
        idempotently but is marked ``collision_checks_degraded`` so the
        operator learns identity verification was bypassed (WP-18).
        """
        pool = self._pool()
        if pool is None or not new_hash:
            return "insert", None
        try:
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    f"SELECT * FROM {table} WHERE {id_column} = $1",
                    row_id,
                )
        except Exception as exc:  # noqa: BLE001
            self._note_degraded(table, str(exc))
            return "insert", None
        if row is None:
            return "insert", None
        row_d = dict(row) if not isinstance(row, dict) else row
        if "content_hash" not in row_d:
            self._note_degraded(table, "identity row lookup returned no content_hash")
            return "insert", None
        existing_hash = str(row_d.get("content_hash") or "")
        if existing_hash == new_hash:
            return "skip", None
        diff = {
            "table": table,
            "id": row_id,
            "existing": _row_summary(row_d, existing_hash),
            "incoming_hash": new_hash[:12],
        }
        return "abort", diff

    def _note_degraded(self, table: str, reason: str) -> None:
        """Mark one table's collision check as degraded (once per run)."""
        if table in self._degraded_seen:
            return
        self._degraded_seen.add(table)
        self._collision_degraded += 1
        logger.warning(
            "collision checks degraded for %s — import proceeding idempotently "
            "(existing rows not verified against bundle): %s",
            table,
            reason,
        )

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
            outcome, diff = await self._collision_decision(
                "knowledge_nodes",
                "node_id",
                str(row["node_id"]),
                str(row.get("content_hash") or ""),
            )
            if outcome == "abort":
                raise BundleFormatError(_diff_message(diff))
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
        await self._restore_node_embeddings(bundle_path, result)
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

    async def _restore_node_embeddings(
        self, bundle_path: Path, result: KnowledgeImportResult
    ) -> None:
        """Import canonical node vectors only under the destination's trusted
        model; every other row stays pending re-embedding (precondition #3).

        Trusted = same model (basename — sites stage the same model under
        different paths) and a vector whose length equals its declared
        dimension. Rows land under the DESTINATION's model name so its own
        queries find them; existing rows are never overwritten.
        """
        path = bundle_path / _TABLE_FILES["node_embeddings"]
        if not path.exists():
            return
        rows = list(_iter_rows(path))
        dest_model = os.environ.get(_DEST_EMBED_MODEL_ENV, "").strip()
        pool = self._pool()
        trusted = [r for r in rows if _trusted_node_vector(r, dest_model)]
        result.node_embeddings_pending = len(rows) - len(trusted)
        if not trusted:
            return
        if pool is None:
            self._mark_degraded(result, "node embeddings skipped (no DB pool)")
            result.node_embeddings_pending = len(rows)
            return
        result.node_embedding_count = await _insert_node_vectors(
            pool, trusted, dest_model
        )
        result.node_embeddings_pending = len(rows) - result.node_embedding_count

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

    # ── Reference consumption (V-05 / D-23, wave 2026-09-22) ─────────────

    def _consume_context_references(
        self,
        manifest: dict[str, Any],
        target_project: str,
        result: KnowledgeImportResult,
    ) -> None:
        """Consume the manifest's ``skills`` / ``mcp_configs`` references.

        Export writes these by reference (``{ref, version_pin, scope}``); a
        restore must (a) register skill stubs the target project can resolve
        later and (b) resolve MCP refs against the registry so an import
        honestly reports what Hatchling still has to fetch. Missing stores
        defer (counted, never crash); existing rows are never stomped.
        """
        skills, mcps = _manifest_refs(manifest)
        if not skills and not mcps:
            return
        result.reference_stores_deferred = self._skill_store is None or (
            bool(mcps) and self._mcp_store is None
        )
        self._consume_skill_refs(skills, target_project, result)
        self._consume_mcp_refs(mcps, result)

    def _consume_skill_refs(
        self,
        skills: list[dict[str, Any]],
        target_project: str,
        result: KnowledgeImportResult,
    ) -> None:
        if self._skill_store is None:
            return
        imported, existing = _import_skill_refs(
            self._skill_store, skills, target_project
        )
        result.skill_refs_imported = imported
        result.skill_refs_existing = existing

    def _consume_mcp_refs(
        self, mcps: list[dict[str, Any]], result: KnowledgeImportResult
    ) -> None:
        if self._mcp_store is None:
            return
        present, missing = _resolve_mcp_refs(self._mcp_store, mcps)
        result.mcp_refs_present = present
        result.mcp_refs_missing = missing

    def _register_catalog_row(
        self, manifest: dict[str, Any], target_project: str
    ) -> bool:
        """Re-register the D-39 catalog row from the manifest ``catalog`` block.

        The catalog round-trips with the bundle (D-40: the bundle IS the
        catalog object). Create is idempotent on (project, name, version),
        so re-imports converge; absent block/store or catalog-less bundles
        are a quiet no-op.
        """
        if self._catalog_store is None:
            return False
        spec = _catalog_spec_from_manifest(manifest, target_project)
        if spec is None:
            return False
        try:
            # The store adapter (api layer) owns the CatalogSpec shape —
            # knowledge never imports umrs.api (ADR-053 plane fence).
            self._catalog_store.register_from_bundle(spec, owner_id="bundle-import")
        except Exception:  # noqa: BLE001 — catalog reg is best-effort beside the restore
            logger.warning("catalog row registration failed for %r", spec["name"])
            return False
        return True


def _manifest_refs(manifest: dict[str, Any]) -> tuple[list[dict], list[dict]]:
    """The manifest's by-reference skill + MCP blocks (dict rows only)."""
    skills = [r for r in manifest.get("skills") or [] if isinstance(r, dict)]
    mcps = [r for r in manifest.get("mcp_configs") or [] if isinstance(r, dict)]
    return skills, mcps


def _catalog_spec_from_manifest(
    manifest: dict[str, Any], target_project: str
) -> dict[str, Any] | None:
    """The D-39 catalog row to re-register, or None (no block / no name)."""
    block = manifest.get("catalog")
    name = str(block.get("name") or "").strip() if isinstance(block, dict) else ""
    if not name:
        return None
    fields = _catalog_fields(block, manifest.get("ownership") or {})
    return {"project_id": target_project, "name": name, **fields}


def _catalog_fields(block: dict[str, Any], ownership: dict[str, Any]) -> dict[str, Any]:
    """Descriptive catalog fields; contributors/licence fall back to ownership."""
    provenance = block.get("provenance")
    return {
        "version": str(block.get("version") or "0.1.0"),
        "description": str(block.get("description") or ""),
        "ontology": str(block.get("ontology") or ""),
        "contributors": [
            str(c) for c in _owned(block, ownership, "contributors") or []
        ],
        "licence": str(_owned(block, ownership, "licence") or ""),
        "provenance": provenance if isinstance(provenance, dict) else {},
    }


def _owned(block: dict[str, Any], ownership: dict[str, Any], key: str) -> Any:
    """Catalog block value, falling back to the manifest ownership block."""
    return block.get(key) or ownership.get(key)


def _parse_skill_ref(ref: dict[str, Any]) -> tuple[str, str, str]:
    """(scope, name, version_pin) from one manifest skill reference."""
    raw = str(ref.get("ref") or "")
    scope, _, name = raw.partition(":")
    if not name:  # ref without a scope prefix — treat as project-scoped
        scope, name = "project", raw
    return (scope.strip() or "project", name.strip(), str(ref.get("version_pin") or ""))


def _import_skill_refs(
    skill_store: Any, refs: list[dict[str, Any]], target_project: str
) -> tuple[int, int]:
    """Register disabled stubs for refs the target install does not have.

    Stubs carry ``manifest._bundle_ref`` so a later full registration (with
    the SKILL.md body) clearly supersedes them; a ref whose (scope, owner,
    name) row already exists is counted as existing, never stomped.
    """
    from r4_contracts.skills import SkillScope

    imported = 0
    existing = 0
    for ref in refs:
        scope_s, name, pin = _parse_skill_ref(ref)
        if not name:
            continue
        try:
            scope = SkillScope(scope_s)
        except ValueError:
            logger.warning(
                "Skipping skill ref with unknown scope %r: %s", scope_s, name
            )
            continue
        project_id = target_project if scope is SkillScope.PROJECT else None
        if (
            skill_store.get_skill_by_name(name, scope, owner_id="bundle-import")
            is not None
        ):
            existing += 1
            continue
        skill_store.register_skill(
            name=name,
            version=pin or "0.1.0",
            scope=scope,
            owner_id="bundle-import",
            project_id=project_id,
            skill_md="",
            manifest={"_bundle_ref": True, "version_pin": pin},
            enabled=False,
        )
        imported += 1
    return imported, existing


def _resolve_mcp_refs(
    mcp_store: Any, refs: list[dict[str, Any]]
) -> tuple[int, list[str]]:
    """Resolve MCP refs against the registry; missing ones are reported."""
    from umrs.mcp.store import MCPServerStoreError

    present = 0
    missing: list[str] = []
    for ref in refs:
        name = str(ref.get("ref") or "").strip()
        if not name:
            continue
        try:
            mcp_store.get_server_by_name(name)
            present += 1
        except MCPServerStoreError:
            missing.append(name)
        except Exception:  # noqa: BLE001 — registry faults never crash a restore
            logger.warning("MCP registry lookup failed for %r", name)
            missing.append(name)
    return present, missing


def _model_base(model: str) -> str:
    return str(model or "").rstrip("/").rsplit("/", 1)[-1]


def _trusted_node_vector(row: dict[str, Any], dest_model: str) -> bool:
    """Precondition #3 pin check for one exported node vector."""
    vector = row.get("embedding") or []
    return (
        bool(dest_model)
        and _model_base(row.get("model_name", "")) == _model_base(dest_model)
        and isinstance(vector, list)
        and len(vector) == int(row.get("dimension") or 0) > 0
    )


async def _insert_node_vectors(
    pool: Any, rows: list[dict[str, Any]], dest_model: str
) -> int:
    """Insert trusted rows under the destination model (idempotent; the
    node must have been restored — FK — or the row is skipped)."""
    count = 0
    async with pool.acquire() as conn:
        for row in rows:
            try:
                await conn.execute(
                    """INSERT INTO knowledge_node_embeddings
                       (node_id, model_name, dimension, embedding, content_hash)
                       VALUES ($1, $2, $3, $4::vector, $5)
                       ON CONFLICT (node_id, model_name) DO NOTHING""",
                    row["node_id"],
                    dest_model,
                    len(row["embedding"]),
                    _vector_literal(row["embedding"]),
                    row.get("content_hash") or "",
                )
                count += 1
            except Exception as exc:  # noqa: BLE001 — per-row fault isolation (FK, shape)
                logger.warning(
                    "Skipping node embedding %s: %s", row.get("node_id"), exc
                )
    return count
