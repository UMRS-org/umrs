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

Chunk text (ADR-093 D-12, MD-5): with the explicit ``chunk_text`` opt-in
(default off) every chunk node of an active artifact carries its full body
(``text``, md5-verified, at most 4,096 chars) and the manifest becomes v1.3
with ``chunk_text: true``. A project whose data class is not
``non_sensitive`` (an unset class counts as sensitive, ADR-062) exports it
only when the job's verified submitter is its owner (never on an
acknowledgement the spec carries); the export refuses otherwise. Without the opt-in
the bundle keeps the v1.2 shape: excerpts only, raw bytes only with
``include_raw``.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import bundle_raw, canonical_identity
from .bundle_reader import CHUNK_TEXT_MAX_CHARS, ExportBundleWriter, is_chunk_row
from .retrieval.hydrate import artifact_slice, stored_text
from .store import KnowledgeStore

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

_HMAC_ALGORITHM = "sha256"
_SIGNATURE_FILENAME = "signature.sig"
_PAGE_SIZE = 500
_BUNDLE_FORMAT = "r4-kb"
_BUNDLE_VERSION = "1.2"
# ADR-093 D-12: only an export with the chunk_text opt-in is v1.3.
_CHUNK_TEXT_VERSION = "1.3"
_GENERATOR_NAME = "umrs"

# Default embedding vector cursor (pgvector stores as text, use id as cursor)
_DEFAULT_EMBEDDING_CURSOR = "0"

# ADR-093 D-12 chunk text: the stored bodies (migration 0120; an absent
# column reads as none), the active artifacts' object keys, and the project
# facts of the MD-5 gate (projects.owner_id / data_class, 0095 + 0107).
_CHUNK_SEARCH_TEXT_SQL = (
    "SELECT node_id, search_text FROM knowledge_nodes "
    "WHERE project_id = $1 AND content->>'kind' = 'chunk' "
    "AND search_text IS NOT NULL"
)
# The source name is the one the embed job decoded the object under (a PPTX
# is extracted only by its name; hydrate.chunked_text).
_ARTIFACT_REFS_SQL = (
    "SELECT artifact_id, source_ref, metadata->>'source_name' "
    "FROM knowledge_artifacts WHERE project_id = $1 AND status = 'active'"
)
_PROJECT_FACTS_SQL = "SELECT owner_id, data_class FROM projects WHERE project_id = $1"
# Artifact texts held at once while slicing (chunks arrive grouped by artifact).
_DOCUMENT_CACHE = 8


class BundleIntegrityError(RuntimeError):
    """Raised when integrity requirements for an export cannot be met."""


class ChunkTextRefused(PermissionError):
    """A chunk-text export the project's data class does not allow (MD-5)."""


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


def _node_vector_row(r: Any) -> dict[str, Any]:
    """One knowledge_node_embeddings row → its bundle shape."""
    return {
        "node_id": r["node_id"],
        "model_name": r.get("model_name") or "",
        "dimension": int(r.get("dimension") or 0),
        "embedding": _parse_vector_text(r["embedding"]),
        "content_hash": r.get("content_hash") or "",
    }


def _parse_vector_text(value: Any) -> list[float]:
    """pgvector text form ``[0.1,0.2]`` (or an already-decoded list) → floats."""
    if isinstance(value, list):
        return [float(v) for v in value]
    text = str(value or "").strip().strip("[]")
    return [float(v) for v in text.split(",") if v.strip()]


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


# D-46 (2026-09-22): redaction targets map onto the manifest's catalog /
# ownership blocks (metadata level only; integrity hash covers the result).
_REDACT_PATHS: dict[str, tuple[tuple[str, ...], ...]] = {
    "contributors": (("catalog", "contributors"), ("ownership", "contributors")),
    "licence": (("catalog", "licence"), ("ownership", "licence")),
    "provenance": (("catalog", "provenance"),),
    "description": (("catalog", "description"),),
    "ontology": (("catalog", "ontology"),),
}


def _redact_value(value: Any, action: str) -> Any:
    """drop empties; hash replaces strings with sha256 fingerprints."""
    if action == "drop":
        return (
            [] if isinstance(value, list) else ({} if isinstance(value, dict) else "")
        )
    if isinstance(value, str):
        return f"sha256:{hashlib.sha256(value.encode('utf-8')).hexdigest()[:12]}"
    if isinstance(value, list):
        return [_redact_value(v, action) if isinstance(v, str) else v for v in value]
    return value


def _apply_redaction_rules(
    manifest: dict[str, Any], rules: list[dict[str, Any]] | None
) -> None:
    """Apply the catalog record's redaction rules to a manifest in place.

    Every application is recorded in ``manifest["redacted"]`` — an import
    reads honest removal instead of missing data. Unknown targets/actions
    are ignored (never a silent over-redaction).
    """
    applied: list[str] = []
    for rule in rules or []:
        if not isinstance(rule, dict):
            continue
        target, action = str(rule.get("target") or ""), rule.get("action")
        if action not in ("drop", "hash"):
            continue
        _redact_target(manifest, target, action)
        if target in _REDACT_PATHS:
            applied.append(f"{target}:{action}")
    manifest["redacted"] = applied


def _redact_target(manifest: dict[str, Any], target: str, action: str) -> None:
    """Apply one redaction action to every manifest path of ``target``."""
    for block, key in _REDACT_PATHS.get(target, ()):
        section = manifest.get(block)
        if isinstance(section, dict) and key in section:
            section[key] = _redact_value(section[key], action)


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


# ── Chunk text (ADR-093 D-12) ──────────────────────────────────────────────────


def _chunk_text_refusal(
    project_id: str, facts: tuple[str, str] | None, submitter: str
) -> str:
    """Why a chunk-text export of the project is refused ("" = allowed).

    ``non_sensitive`` projects export it on the flag; any other class (an
    unset one included) only when the job's verified submitter is the
    project's owner. Nothing the spec says counts: a client-written
    acknowledgement naming the owner unlocked a sensitive project for any
    member through POST /jobs (review chunk-data-2, 2026-10-10). A project
    with no readable registry row (unregistered, or the registry down)
    refuses (fail closed): its owner and class are unknown.
    """
    if facts is None:
        return (
            f"chunk text of {project_id!r} refused: no readable project registry "
            "row (owner, data class) — register the project or retry when the "
            "registry is reachable"
        )
    owner, data_class = facts
    if data_class == "non_sensitive":
        return ""
    if owner and submitter == owner:
        return ""
    return (
        f"project {project_id!r} is {data_class or 'sensitive (unset)'}: its chunk "
        "text leaves the installation only on its owner's explicit chunk_text flag "
        f"(the job's submitter is {submitter or 'unknown'!r})"
    )


def _owner_acknowledgement(
    facts: tuple[str, str] | None, submitter: str
) -> dict[str, Any]:
    """The manifest record of a sensitive project's release: the verified
    owner who submitted it ({} for a ``non_sensitive`` project)."""
    if facts is None or facts[1] == "non_sensitive":
        return {}
    return {
        "by": submitter,
        "data_class": facts[1] or "sensitive",
        "verified": "job submitter is the project owner",
        "at": datetime.now(timezone.utc).isoformat(),
    }


def _quiet_reader(
    reader: Callable[[str], bytes | None] | None,
) -> Callable[[str], bytes | None] | None:
    """The artifact reader with read faults turned into "unreadable" (None)."""
    if reader is None:
        return None

    def read(source_ref: str) -> bytes | None:
        try:
            return reader(source_ref)
        except Exception as exc:  # noqa: BLE001 — one unreadable object keeps its excerpt
            logger.warning("chunk text: artifact %s unreadable: %s", source_ref, exc)
            return None

    return read


def _artifact_of(row: dict[str, Any]) -> str:
    """A node row's ``content.artifact_id`` ("" when it has none)."""
    content = row.get("content")
    return str(content.get("artifact_id") or "") if isinstance(content, dict) else ""


def _header_of(content: dict[str, Any], search_text: str | None) -> str | None:
    """The C9 header a stored ``search_text`` starts with, when it has one."""
    header_chars = content.get("header_chars")
    if search_text and isinstance(header_chars, int) and header_chars > 0:
        return search_text[:header_chars]
    return None


class _ChunkTexts:
    """Full chunk bodies for a ``chunk_text`` export.

    A body is the ``search_text`` whose md5 equals ``content.chunk_md5``
    (after a C9 header), else the md5-verified slice of the artifact the
    embed job chunked — both through knowledge/retrieval/hydrate, the rule
    every reader shares. A chunk of an artifact that is not active, or with
    neither text, keeps its excerpt only.
    """

    def __init__(
        self,
        search_texts: dict[str, str],
        artifacts: dict[str, tuple[str, str]],
        reader: Callable[[str], bytes | None] | None,
    ) -> None:
        self._search_texts = search_texts
        self._artifacts = artifacts  # artifact_id -> (source_ref, source_name)
        self._reader = _quiet_reader(reader)
        self._documents: dict[str, str | None] = {}
        self.counts = dict.fromkeys(
            (
                "chunks",
                "search_text",
                "artifact",
                "missing",
                "no_artifact",
                "truncated",
            ),
            0,
        )

    def attach(self, row: dict[str, Any]) -> None:
        """Give one serialized node row its ``text`` when it is a chunk."""
        if not is_chunk_row(row):
            return
        self.counts["chunks"] += 1
        content = row["content"]
        artifact = self._artifacts.get(str(content.get("artifact_id") or ""))
        if artifact is None:
            self.counts["no_artifact"] += 1
            return
        search_text = self._search_texts.get(str(row.get("node_id") or ""))
        body = stored_text(content, search_text)
        source, header = "search_text", _header_of(content, search_text)
        if not body:
            body, source, header = self._slice(content, *artifact), "artifact", None
        if not body:
            self.counts["missing"] += 1
            return
        self.counts[source] += 1
        self._put(row, body, header)

    def _slice(self, content: dict[str, Any], ref: str, name: str) -> str | None:
        if ref not in self._documents and len(self._documents) >= _DOCUMENT_CACHE:
            self._documents.clear()
        return artifact_slice(content, ref, self._documents, self._reader, name)

    def _put(self, row: dict[str, Any], body: str, header: str | None) -> None:
        if len(body) > CHUNK_TEXT_MAX_CHARS:
            body = body[:CHUNK_TEXT_MAX_CHARS]
            row["text_truncated"] = True
            self.counts["truncated"] += 1
        row["text"] = body
        if header:
            row["text_header"] = header

    def summary(self, acknowledgement: dict[str, Any]) -> dict[str, Any]:
        """The manifest ``chunk_text_summary`` block."""
        block: dict[str, Any] = {**self.counts, "max_chars": CHUNK_TEXT_MAX_CHARS}
        if acknowledgement:
            block["acknowledgement"] = acknowledgement
        return block


# ── Result DTO ─────────────────────────────────────────────────────────────────


@dataclass
class KnowledgeExportResult:
    """Result of a KNOWLEDGE_EXPORT job attempt."""

    bundle_path: str = ""  # server-local bundle directory (export working copy)
    project_id: str = ""
    # D-40 transport: the project-scoped .kb.tar.gz object key (gateway
    # download route serves it; "" when not published).
    archive_key: str = ""
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
        artifact_reader: Callable[[str], bytes | None] | None = None,
        include_raw: bool = False,
        raw_acknowledgement: dict[str, Any] | None = None,
        chunk_text: bool = False,
        chunk_text_submitter: str = "",
    ) -> None:
        self._store = store
        # ADR-032 raw-materials amendment: opt-in carriage of artifact bytes;
        # the acknowledgement (sensitive projects) is recorded in the manifest.
        self._include_raw = include_raw
        self._raw_acknowledgement = raw_acknowledgement or {}
        # ADR-093 D-12 / MD-5: opt-in full chunk text (r4-kb v1.3), gated on
        # the job's verified submitter; the owner's release of a sensitive
        # project rides the manifest (set by the gate).
        self._chunk_text = chunk_text
        self._chunk_text_submitter = str(chunk_text_submitter or "")
        self._chunk_text_ack: dict[str, Any] = {}
        self._chunk_text_block: dict[str, Any] = {}
        self._artifact_rows: list[dict[str, Any]] = []
        # ADR-032 #1: reads one artifact's raw bytes by source_ref (None =
        # unreadable) so the export can map each row to its canonical identity.
        self._artifact_reader = artifact_reader
        self._identity_counts: dict[str, int] = {}
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
        await self._gate_chunk_text(project_id)
        logger.info("Starting knowledge export for project=%s", project_id)

        writer = ExportBundleWriter(self._output_dir / project_id)

        # ADR-032 #5: the checked snapshot is taken FIRST and gates what the
        # canonical plane contributes; without one the export stays the
        # documented ad-hoc read (reconciliation_snapshot: null).
        snapshot = self._resolve_export_snapshot(project_id)
        excluded = snapshot.excluded_node_ids if snapshot is not None else frozenset()
        table_counts = await self._dump_core_tables(project_id, writer, excluded)
        table_counts["node_embeddings"] = (
            await self._dump_node_embeddings(project_id, writer, excluded),
            "node_embeddings.jsonl",
        )
        link_rows, ref_rows = await self._dump_a2l_tables(project_id, writer)
        table_counts["a2l_links"] = (len(link_rows), "a2l_links.jsonl")
        table_counts["a2l_refs"] = (len(ref_rows), "a2l_refs.jsonl")
        catalog_record = await self._resolve_catalog_record(project_id)
        raw_block = self._dump_raw(writer, catalog_record, table_counts)

        # 8. Build manifest (r4-kb v1.2 — ADR-032 amendment 2026-09)
        skills, mcp_configs = await self._resolve_context_references(project_id)
        embedding_pin = await self._resolve_embedding_pin(project_id)
        reconciliation_snapshot = snapshot.summary if snapshot is not None else None
        self._manifest = self._build_manifest(
            project_id,
            table_counts,
            embedding_pin,
            skills,
            mcp_configs,
            reconciliation_snapshot,
            catalog_record,
        )
        self._manifest["raw"] = raw_block
        self._stamp_chunk_text(self._manifest)
        # The signature must cover the table CONTENTS, not only their names
        # and counts: each listed file's sha256 rides the signed manifest.
        stamp_table_digests(self._manifest, writer.path)
        # D-46: the catalog record's redaction rules apply to the manifest
        # BEFORE hashing — signatures always cover the redacted content.
        _apply_redaction_rules(self._manifest, catalog_record.get("redaction_rules"))
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

    def _dump_raw(
        self,
        writer: ExportBundleWriter,
        catalog_record: dict[str, Any],
        table_counts: dict[str, tuple[int, str]],
    ) -> dict[str, Any]:
        """Opt-in raw objects (ADR-032 raw-materials amendment); the manifest
        ``raw`` block. Not requested → ``{"included": False}``."""
        if not self._include_raw:
            return {"included": False}
        caps = bundle_raw.RawCaps.from_env()
        candidates, skipped = bundle_raw.plan_raw(
            self._artifact_rows,
            licence=str(catalog_record.get("licence") or ""),
            redaction_rules=catalog_record.get("redaction_rules"),
            caps=caps,
        )
        rows = bundle_raw.write_raw(
            writer.path, candidates, self._artifact_reader, caps, skipped
        )
        table_counts["raw_objects"] = (len(rows), bundle_raw.RAW_TABLE)
        extra = (
            {"acknowledgement": self._raw_acknowledgement}
            if self._raw_acknowledgement
            else {}
        )
        return bundle_raw.summary(rows, skipped, **extra)

    def _stamp_identity(self, row: dict[str, Any]) -> None:
        """ADR-032 #1: map the row to its canonical identity (``cid-1``);
        existing ids are untouched — the map travels, nothing is rewritten."""
        block = canonical_identity.carried(row.get("metadata"))
        if block is None:
            block = self._compute_identity(row)
        row["canonical_identity"] = block
        counts = self._identity_counts
        counts[block["normalization"]] = counts.get(block["normalization"], 0) + 1

    def _compute_identity(self, row: dict[str, Any]) -> dict[str, str]:
        """Identity from the artifact's bytes (unstamped rows only)."""
        raw = None
        size = int(row.get("size_bytes") or 0)
        ref = str(row.get("source_ref") or "")
        readable = (
            canonical_identity.stored_object(ref)
            and size <= canonical_identity.READ_CAP
        )
        if self._artifact_reader is not None and readable:
            raw = self._artifact_reader(ref)
        digest = str(row.get("content_hash") or "")
        return canonical_identity.identity_of(raw, digest).to_dict()

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
        self,
        project_id: str,
        writer: ExportBundleWriter,
        excluded_node_ids: frozenset[str] = frozenset(),
    ) -> dict[str, tuple[int, str]]:
        """Write artifacts/documents/nodes/edges/embeddings/catalog tables.

        ``excluded_node_ids`` (from the checked snapshot) drops ineligible
        canonical nodes and every edge touching one.
        """
        artifacts, artifact_total = await self._store.list_artifacts(
            project_id=project_id,
            status="active",
            page=1,
            page_size=_PAGE_SIZE,
        )
        artifact_rows = [self._serialize_artifact(a) for a in artifacts]
        for row in artifact_rows:
            self._stamp_identity(row)
        self._artifact_rows = artifact_rows
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

        node_count = await self._dump_nodes(project_id, writer, excluded_node_ids)

        edges = await self._store.list_edges(project_id=project_id)
        edge_rows = [
            self._serialize_edge(e)
            for e in edges
            if e.source_id not in excluded_node_ids
            and e.target_id not in excluded_node_ids
        ]
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
            "nodes": (node_count, "nodes.jsonl"),
            "edges": (len(edge_rows), "edges.jsonl"),
            "embeddings": (emb_total, "embeddings.jsonl"),
            "catalog": (len(cat_rows), "catalog.jsonl"),
        }

    async def _dump_nodes(
        self,
        project_id: str,
        writer: ExportBundleWriter,
        excluded_node_ids: frozenset[str],
    ) -> int:
        """Write nodes.jsonl (snapshot-excluded nodes dropped); chunk rows
        carry their full text only with the chunk_text opt-in."""
        nodes = await self._store.list_nodes(project_id=project_id)
        node_rows = [
            self._serialize_node(n) for n in nodes if n.node_id not in excluded_node_ids
        ]
        await self._attach_chunk_text(project_id, node_rows)
        await writer.write_nodes(node_rows)
        logger.info("Exported %d nodes for project=%s", len(node_rows), project_id)
        return len(node_rows)

    # ── Chunk text (ADR-093 D-12) ─────────────────────────────────────────────

    async def _gate_chunk_text(self, project_id: str) -> None:
        """MD-5: refuse BEFORE anything is written when the project's data
        class does not let its chunk text leave (no-op without the opt-in)."""
        if not self._chunk_text:
            return
        facts = await self._project_facts(project_id)
        submitter = self._chunk_text_submitter
        refusal = _chunk_text_refusal(project_id, facts, submitter)
        if refusal:
            raise ChunkTextRefused(refusal)
        self._chunk_text_ack = _owner_acknowledgement(facts, submitter)

    async def _project_facts(self, project_id: str) -> tuple[str, str] | None:
        """(owner_id, data_class) from the projects registry, or None."""
        try:
            row = await self._store._pool.fetchrow(_PROJECT_FACTS_SQL, project_id)  # type: ignore[union-attr]
        except Exception as exc:  # noqa: BLE001 — unreadable registry refuses (fail closed)
            logger.warning("project facts unreadable for %s: %s", project_id, exc)
            return None
        if row is None:
            return None
        return str(row["owner_id"] or ""), str(row["data_class"] or "")

    async def _attach_chunk_text(
        self, project_id: str, node_rows: list[dict[str, Any]]
    ) -> None:
        """Give every chunk row of an active artifact its md5-verified body."""
        if not self._chunk_text:
            return
        search_texts = await self._fetch_pairs(_CHUNK_SEARCH_TEXT_SQL, project_id)
        artifacts = {
            str(row[0]): (str(row[1] or ""), str(row[2] or ""))
            for row in await self._fetch_rows(_ARTIFACT_REFS_SQL, project_id)
        }
        texts = _ChunkTexts(search_texts, artifacts, self._artifact_reader)
        # Visit by artifact (rows change in place; nodes.jsonl keeps its
        # order) so each artifact is read at most once whatever the order.
        for row in sorted(node_rows, key=_artifact_of):
            texts.attach(row)
        self._chunk_text_block = texts.summary(self._chunk_text_ack)
        logger.info("Chunk text for project=%s: %s", project_id, texts.counts)

    async def _fetch_pairs(self, query: str, project_id: str) -> dict[str, str]:
        """``{first column: second column}`` of one read ({} when unreadable,
        e.g. ``search_text`` before migration 0120)."""
        rows = await self._fetch_rows(query, project_id)
        return {str(r[0]): str(r[1] or "") for r in rows}

    async def _fetch_rows(self, query: str, project_id: str) -> list[Any]:
        """The rows of one read with a first column ([] when unreadable)."""
        try:
            rows = await self._store._pool.fetch(query, project_id)  # type: ignore[union-attr]
        except Exception as exc:  # noqa: BLE001 — the slice fallback still applies
            logger.warning("chunk text input unreadable for %s: %s", project_id, exc)
            return []
        return [r for r in rows if r[0]]

    def _stamp_chunk_text(self, manifest: dict[str, Any]) -> None:
        """v1.3 + ``chunk_text: true`` only with the opt-in; otherwise the
        manifest keeps the v1.2 shape untouched."""
        if not self._chunk_text:
            return
        manifest["version"] = _CHUNK_TEXT_VERSION
        manifest["chunk_text"] = True
        manifest["chunk_text_summary"] = self._chunk_text_block

    async def _dump_node_embeddings(
        self,
        project_id: str,
        writer: ExportBundleWriter,
        excluded_node_ids: frozenset[str] = frozenset(),
    ) -> int:
        """Write the project's canonical node vectors (0075, KNOWLEDGE_EMBED).

        Rows of snapshot-excluded nodes stay out, like the nodes themselves.
        Best-effort: an unreadable table writes an empty file (0 rows).
        """
        try:
            rows = await self._store._pool.fetch(  # type: ignore[union-attr]
                "SELECT kne.node_id, kne.model_name, kne.dimension, "
                "kne.embedding::text AS embedding, kne.content_hash "
                "FROM knowledge_node_embeddings kne "
                "JOIN knowledge_nodes kn ON kn.node_id = kne.node_id "
                "WHERE kn.project_id = $1 ORDER BY kne.node_id, kne.model_name",
                project_id,
            )
        except Exception as exc:  # noqa: BLE001 — optional table, honest empty file
            logger.warning("node embeddings not exported for %s: %s", project_id, exc)
            rows = []
        out = [
            _node_vector_row(r)
            for r in rows
            if r.get("node_id")
            and r.get("embedding")
            and r["node_id"] not in excluded_node_ids
        ]
        await writer.write_node_embeddings(out)
        logger.info("Exported %d node embeddings for project=%s", len(out), project_id)
        return len(out)

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
            "identity": {
                "contract": canonical_identity.CONTRACT,
                "normalizations": dict(sorted(self._identity_counts.items())),
            },
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
                " provenance, redaction_rules FROM scientific_catalogs"
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
                "redaction_rules": _json_list(row["redaction_rules"]),
            }
        except Exception:  # noqa: BLE001 — catalog block is best-effort
            return {}

    def _resolve_export_snapshot(self, project_id: str) -> Any | None:
        """The checked reconciliation snapshot this export consumes (ADR-032
        #5), or None when no DSN is reachable / the scan fails."""
        try:
            from .reconcile_snapshot import take_export_snapshot

            return take_export_snapshot(project_id)
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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stamp_table_digests(manifest: dict[str, Any], bundle_dir: Path) -> None:
    """Add each present table file's sha256 to its ``tables`` entry."""
    for entry in (manifest.get("tables") or {}).values():
        path = bundle_dir / str(entry.get("file") or "")
        if entry.get("file") and path.is_file():
            entry["sha256"] = _file_sha256(path)


def _table_entry_intact(bundle_dir: Path, entry: Any) -> bool:
    """One ``tables`` entry: unpinned passes; pinned needs its plain-named
    file present with the pinned sha256."""
    expected = entry.get("sha256") if isinstance(entry, dict) else None
    if not expected:
        return True
    name = str(entry.get("file") or "")
    path = bundle_dir / name
    if not name or Path(name).name != name or not path.is_file():
        return False
    return _file_sha256(path) == expected


def tables_intact(bundle_dir: Path, manifest: dict[str, Any]) -> bool:
    """Every table the manifest pins by sha256 is present and unchanged
    (entries without a digest — pre-2026-09-24 bundles — are not checked)."""
    tables = manifest.get("tables") or {}
    return all(_table_entry_intact(bundle_dir, e) for e in tables.values())


def tables_pinned(manifest: dict[str, Any]) -> bool:
    """Whether the manifest pins its table contents (bundles since 2026-09-24)."""
    tables = manifest.get("tables") or {}
    return bool(tables) and all(
        isinstance(e, dict) and e.get("sha256") for e in tables.values()
    )


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
    if not verify_signature(manifest_path, signature_path, secret_key):
        return False
    # 3. The signed manifest pins each table file's sha256: a changed row in
    # any table fails verification (before 2026-09-24 only the manifest was
    # covered, so edited table rows still verified).
    if not tables_intact(bundle_dir, manifest_data):
        logger.warning("Bundle table content mismatch — tampering detected")
        return False
    return True


__all__ = [
    "ChunkTextRefused",
    "KnowledgeExportEngine",
    "KnowledgeExportResult",
    "stamp_table_digests",
    "tables_intact",
    "tables_pinned",
    "upload_bundle_to_s3",
    "verify_bundle_integrity",
    "verify_signature",
]
