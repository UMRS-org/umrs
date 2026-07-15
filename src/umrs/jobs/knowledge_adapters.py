"""Durable KNOWLEDGE_INGEST / KNOWLEDGE_UNDERSTAND job backends.

No imports from ``umrs.api`` or ``umrs.server`` — only stdlib + knowledge/
modules.

Backends:
  - ``KnowledgeIngestBackend``   -> resolve sources → S3 artifacts
  - ``KnowledgeUnderstandBackend`` -> parse artifacts → KnowledgeIR → graph
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from umrs.knowledge.artifact_storage import ArtifactStorage

import boto3

from .models import Job, JobAttempt, JobType
from .scheduler import OrchestratorBackend

# Re-export KnowledgeExportResult for consumers that need it.
from ..knowledge.bundle_export import KnowledgeExportResult  # noqa: F401

logger = logging.getLogger(__name__)

_KNOWLEDGE_EXTRACTOR_VERSION = "1.0.0"


# ── Events ───────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class KnowledgeEvent:
    """Audit-logged event for knowledge operations."""

    event_type: str  # create | ingest | verify | skip | fail
    job_id: str
    attempt_id: str
    artifact_id: Optional[str] = None
    ir_id: Optional[str] = None
    source_ref: Optional[str] = None
    details: str = ""
    timestamp: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            object.__setattr__(
                self, "timestamp", datetime.now(timezone.utc).isoformat()
            )

    def emit(self) -> "KnowledgeEvent":
        """Log and return self for chaining."""
        payload = {"event": self.event_type, "job_id": self.job_id}
        if self.artifact_id:
            payload["artifact_id"] = self.artifact_id
        if self.ir_id:
            payload["ir_id"] = self.ir_id
        if self.source_ref:
            payload["source_ref"] = self.source_ref
        if self.details:
            payload["details"] = self.details
        logger.info("KnowledgeEvent [%s]: %s", self.event_type, payload)
        return self


# ── Result DTOs ──────────────────────────────────────────────────────────────


@dataclass
class KnowledgeIngestResult:
    """Result of a KNOWLEDGE_INGEST job attempt."""

    artifact_ids: list[str] = field(default_factory=list)
    failed_items: list[dict[str, Any]] = field(default_factory=list)
    degraded: bool = False
    degraded_reason: str = ""


@dataclass
class KnowledgeUnderstandResult:
    """Result of a KNOWLEDGE_UNDERSTAND job attempt."""

    ir_ids: list[str] = field(default_factory=list)
    nodes_created: int = 0
    edges_created: int = 0
    already_ir: int = 0  # Skipped (idempotent)
    failed_items: list[dict[str, Any]] = field(default_factory=list)
    degraded: bool = False
    degraded_reason: str = ""


# ── Helpers ──────────────────────────────────────────────────────────────────


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _build_summary(parsed: dict[str, Any]) -> str:
    """Build a short summary string from a parsed code result."""
    parts: list[str] = []
    if parsed.get("modules"):
        parts.append(f"{len(parsed['modules'])} module(s)")
    if parsed.get("classes"):
        parts.append(f"{len(parsed['classes'])} class(es)")
    if parsed.get("functions"):
        parts.append(f"{len(parsed['functions'])} function(s)")
    if parsed.get("algorithms"):
        parts.append(f"{len(parsed['algorithms'])} algorithm(s)")
    if parsed.get("kernels"):
        parts.append(f"{len(parsed['kernels'])} kernel(s)")
    return " | ".join(parts) if parts else "no structure extracted"


def _get_pool() -> Any:
    """Get a Postgres asyncpg pool from env. Returns None if unavailable."""
    conn_str = os.getenv(
        "UMRS_DATA_CATALOG_DB",
        os.getenv("DATABASE_URL", ""),
    )
    if not conn_str:
        return None
    try:
        import asyncpg

        return asyncpg.create_pool(dsn=conn_str)  # type: ignore[return-value]
    except Exception:
        return None


# ── Backend Protocol ─────────────────────────────────────────────────────────

from .monitor import BackendStatusChecker, AttemptCheckResult  # noqa: E402


# ── KnowledgeIngestBackend ──────────────────────────────────────────────────


class KnowledgeIngestBackend(OrchestratorBackend):
    """Resolve sources → S3 artifacts. Tracks partial failures."""

    def __init__(self) -> None:
        self._s3_client: Any = None
        self._bucket: str = ""

    @property
    def backend_name(self) -> str:
        return "knowledge_ingest"

    def _init_s3(self, job_spec: dict[str, Any]) -> None:
        """Lazily initialize S3 client from env."""
        if self._s3_client is not None:
            return
        from botocore.config import Config as BotoConfig

        self._bucket = job_spec.get("bucket", "umrs-data")
        endpoint = os.getenv("S3_ENDPOINT_URL", "http://localhost:9000")
        key = os.getenv("S3_ACCESS_KEY", "")
        secret = os.getenv("S3_SECRET_KEY", "")
        use_ssl = os.getenv("S3_USE_SSL", "false").lower() == "true"

        try:
            self._s3_client = boto3.client(
                "s3",
                endpoint_url=endpoint,
                aws_access_key_id=key,
                aws_secret_access_key=secret,
                region_name=os.getenv("S3_REGION", "us-east-1"),
                config=BotoConfig(
                    signature_version="s3v4",
                    s3={"addressing_style": "path"},
                ),
                use_ssl=use_ssl,
            )
            self._s3_client.list_buckets()
        except Exception as exc:
            logger.warning("S3 unavailable — degraded mode: %s", exc)
            self._s3_client = None

    async def submit_job(
        self,
        job: Job,
        attempt: JobAttempt,
    ) -> str:
        self._init_s3(job.spec)

        source_refs: list[dict[str, Any]] = job.spec.get("sources", [])
        if not source_refs:
            raise ValueError("knowledge_ingest requires spec.sources")

        project_id = job.spec.get("project_id", job.project_id)
        allow_domains = job.spec.get("allow_domains")

        KnowledgeEvent(
            event_type="create",
            job_id=job.job_id,
            attempt_id=attempt.attempt_id,
            source_ref=job.spec.get("source_ref", ""),
            details=f"project={project_id} sources={len(source_refs)}",
        ).emit()

        if self._s3_client is None:
            result = KnowledgeIngestResult(
                failed_items=[
                    {"source": s.get("kind", "unknown"), "error": "S3 unavailable"}
                    for s in source_refs
                ],
                degraded=True,
                degraded_reason="S3 client initialization failed",
            )
            _write_result(attempt, result)
            return f"knowledge_ingest|mode=degraded|project={project_id}"

        return _run_ingest_sync(
            job,
            attempt,
            self._s3_client,
            self._bucket,
            source_refs,
            project_id,
            allow_domains,
        )


# ── KnowledgeUnderstandBackend ───────────────────────────────────────────────


class KnowledgeUnderstandBackend(OrchestratorBackend):
    """Parse artifacts lacking current-version IR → KnowledgeIR + graph."""

    def __init__(self) -> None:
        self._project_id: str = ""

    @property
    def backend_name(self) -> str:
        return "knowledge_understand"

    async def submit_job(
        self,
        job: Job,
        attempt: JobAttempt,
    ) -> str:
        project_id = self._project_id or job.spec.get("project_id", job.project_id)
        self._project_id = project_id

        sources: list[dict[str, Any]] = job.spec.get("sources", [])
        KnowledgeEvent(
            event_type="create",
            job_id=job.job_id,
            attempt_id=attempt.attempt_id,
            source_ref=job.spec.get("source_ref", ""),
            details=f"project={project_id}",
        ).emit()

        pool = _get_pool()
        degraded = False
        degraded_reason = ""

        if pool is None:
            degraded_reason = "Postgres pool unavailable"
            degraded = True
            logger.warning("KnowledgeUnderstand degraded: %s", degraded_reason)

        try:
            result = await _run_understand(
                job,
                attempt,
                project_id,
                sources,
                pool,
            )
        finally:
            if pool is not None:
                pool.close()
                await pool.wait_closed()

        if degraded:
            result.degraded = True
            result.degraded_reason = degraded_reason

        _write_result(attempt, result)

        return (
            f"knowledge_understand|project={project_id}|"
            f"ir_ids={len(result.ir_ids)}|already={result.already_ir}|fail={len(result.failed_items)}"
        )


# ── Internal: Ingest Runner (sync) ──────────────────────────────────────────


def _run_ingest_sync(
    job: Job,
    attempt: JobAttempt,
    s3_client: Any,
    bucket: str,
    source_refs: list[dict[str, Any]],
    project_id: str,
    allow_domains: Optional[list[str]],
) -> str:
    from umrs.knowledge.sources import SourceResolver
    from umrs.knowledge.artifact_storage import ArtifactStorage

    result = KnowledgeIngestResult()
    success_ids: list[str] = []

    for src_spec in source_refs:
        source_kind = src_spec.get("kind", "local_file")
        uri = src_spec.get("uri", src_spec.get("path", ""))
        if not uri:
            result.failed_items.append({"source": source_kind, "error": "no uri/path"})
            continue

        try:
            resolver = SourceResolver()
            resolved = resolver.resolve(source_kind, uri, allow_domains=allow_domains)

            if not resolved.resolved:
                result.failed_items.append(
                    {
                        "source": source_kind,
                        "uri": uri,
                        "error": "resolution failed",
                    }
                )
                continue

            storage = ArtifactStorage(s3_client, bucket)
            artifact_id = f"art-{uuid.uuid4().hex[:12]}"

            if resolved.kind == "local_file":
                _put_file_sync(
                    job.job_id,
                    attempt.attempt_id,
                    storage,
                    project_id,
                    resolved.path or "",
                    artifact_id,
                    result,
                    success_ids,
                )
            elif resolved.kind == "local_dir":
                for rf in resolved.files:
                    aid = (
                        f"{artifact_id}-"
                        f"{hashlib.md5(rf.absolute_path.encode()).hexdigest()[:8]}"
                    )
                    _put_file_sync(
                        job.job_id,
                        attempt.attempt_id,
                        storage,
                        project_id,
                        rf.absolute_path,
                        aid,
                        result,
                        success_ids,
                    )
            elif resolved.kind == "s3_uri":
                # S3 reference — register without copy
                success_ids.append(artifact_id)
                result.artifact_ids.append(artifact_id)
                KnowledgeEvent(
                    event_type="create",
                    job_id=job.job_id,
                    attempt_id=attempt.attempt_id,
                    artifact_id=artifact_id,
                    source_ref=uri,
                    details="s3 reference (no copy)",
                ).emit()
            elif resolved.kind in ("http_url", "git_repo"):
                _fetch_and_store_remote(
                    job.job_id,
                    attempt.attempt_id,
                    storage,
                    project_id,
                    resolved,
                    artifact_id,
                    result,
                    success_ids,
                )
            else:
                result.failed_items.append(
                    {
                        "source": source_kind,
                        "error": f"unsupported kind: {source_kind}",
                    }
                )

        except Exception as exc:
            result.failed_items.append(
                {
                    "source": source_kind,
                    "uri": uri,
                    "error": str(exc),
                }
            )

    # Store artifact metadata in Postgres (best-effort)
    _store_artifacts_in_db_sync(project_id, success_ids)

    KnowledgeEvent(
        event_type="ingest" if not result.failed_items else "fail",
        job_id=job.job_id,
        attempt_id=attempt.attempt_id,
        details=f"{len(success_ids)} artifacts | {len(result.failed_items)} failed",
    ).emit()

    _write_result(attempt, result)

    return (
        f"knowledge_ingest|project={project_id}|"
        f"artifacts={len(success_ids)}|failed={len(result.failed_items)}"
    )


def _put_file_sync(
    job_id: str,
    attempt_id: str,
    storage: ArtifactStorage,
    project_id: str,
    file_path: str,
    artifact_id: str,
    result: KnowledgeIngestResult,
    success_ids: list[str],
) -> None:
    """Read file and put to S3."""
    try:
        with open(file_path, "rb") as f:
            content = f.read()
    except OSError as exc:
        result.failed_items.append({"file": file_path, "error": str(exc)})
        return

    content_hash, size_bytes, _ = storage.put_artifact(project_id, content)
    success_ids.append(artifact_id)
    result.artifact_ids.append(artifact_id)

    # Best-effort: bridge to A2L identity space (ADR-017)
    _bridge_a2l_ref(artifact_id, content_hash, project_id)

    KnowledgeEvent(
        event_type="create",
        job_id=job_id,
        attempt_id=attempt_id,
        artifact_id=artifact_id,
        source_ref=file_path,
        details=f"hash={content_hash[:16]}... size={size_bytes}",
    ).emit()


def _fetch_and_store_remote(
    job_id: str,
    attempt_id: str,
    storage: ArtifactStorage,
    project_id: str,
    resolved: Any,
    artifact_id: str,
    result: KnowledgeIngestResult,
    success_ids: list[str],
) -> None:
    """Fetch remote content (http_url / git_repo), scan, store to S3."""
    from umrs.knowledge.egress_fetch import fetch_url

    url = resolved.uri or ""
    uri = url
    if not url:
        result.failed_items.append(
            {
                "source": resolved.kind,
                "uri": uri,
                "error": "no uri",
            }
        )
        return

    allow_domains = resolved.allow_domains if resolved.allow_domains else []
    fetch = fetch_url(url, allow_domains=allow_domains)

    if not fetch.content:
        result.failed_items.append(
            {
                "source": resolved.kind,
                "uri": uri,
                "error": fetch.error,
            }
        )
        return

    content_hash, size_bytes, _ = storage.put_artifact(project_id, fetch.content)
    success_ids.append(artifact_id)
    result.artifact_ids.append(artifact_id)

    # Best-effort: bridge to A2L identity space (ADR-017)
    _bridge_a2l_ref(artifact_id, content_hash, project_id)

    KnowledgeEvent(
        event_type="create",
        job_id=job_id,
        attempt_id=attempt_id,
        artifact_id=artifact_id,
        source_ref=uri,
        details=f"remote={resolved.kind} domain={fetch.url} hash={content_hash[:16]}... size={size_bytes}",
    ).emit()


def _store_artifacts_in_db_sync(
    project_id: str,
    artifact_ids: list[str],
) -> None:
    """Persist artifact metadata to Postgres (best-effort)."""
    try:
        pool = _get_pool()
        if pool is None:
            return

        async def _do() -> None:
            from umrs.knowledge.store import KnowledgeStore

            store = KnowledgeStore(pool)
            for aid in artifact_ids:
                try:
                    await store.insert_artifact(
                        artifact_id=aid,
                        content_hash="",
                        kind="document",
                        project_id=project_id,
                    )
                except Exception:
                    pass

        asyncio.run(_do())
        pool.close()
        asyncio.run(pool.wait_closed())
    except Exception:
        pass


# ── Internal: Understand Runner (async) ────────────────────────────────────


async def _run_understand(
    job: Job,
    attempt: JobAttempt,
    project_id: str,
    sources: list[dict[str, Any]],
    pool: Any,
) -> KnowledgeUnderstandResult:
    """Core understand loop: iterate artifacts, parse, build IR, upsert graph."""
    if pool is None:
        return KnowledgeUnderstandResult(degraded=True, degraded_reason="no pool")

    from umrs.knowledge.ir import build_knowledge_ir
    from umrs.knowledge.store import KnowledgeStore

    store = KnowledgeStore(pool)

    # Build list of artifacts to process
    artifacts_to_process: list[dict[str, Any]] = []
    if sources:
        artifacts_to_process = sources
    else:
        # List all active artifacts for project
        try:
            artifacts, _ = await store.list_artifacts(project_id)
            for art in artifacts:
                artifacts_to_process.append(
                    {
                        "artifact_id": art.artifact_id,
                        "content_hash": art.content_hash,
                        "kind": art.kind,
                        "source_ref": art.source_ref,
                    }
                )
        except Exception as exc:
            logger.error("Failed to list artifacts: %s", exc)
            artifacts_to_process = []

    result = KnowledgeUnderstandResult()

    for src in artifacts_to_process:
        artifact_id = src.get("artifact_id", "")
        content_hash = src.get("content_hash", "")
        kind = src.get("kind", "document")

        if not artifact_id or not content_hash:
            continue

        # Check for existing current-version IR (write-once idempotent)
        try:
            existing = await store._pool.fetchval(  # type: ignore[union-attr]
                "SELECT ir_id FROM knowledge_ir WHERE artifact_id = $1 "
                "AND status = 'current' LIMIT 1",
                artifact_id,
            )
            if existing:
                result.already_ir += 1
                KnowledgeEvent(
                    event_type="verify",
                    job_id=job.job_id,
                    attempt_id=attempt.attempt_id,
                    artifact_id=artifact_id,
                    details=f"already has IR: {existing}",
                ).emit()
                continue
        except Exception:
            pass

        # Parse artifact
        parsed = _parse_for_understand(artifact_id, content_hash, kind)
        if parsed is None:
            result.failed_items.append(
                {
                    "artifact_id": artifact_id,
                    "error": "parse returned None",
                }
            )
            continue

        # Build KnowledgeIR
        content_bytes = content_hash.encode() if content_hash else b""
        ir = build_knowledge_ir(
            project_id=project_id,
            artifact_id=artifact_id,
            artifact_type=kind,
            content=content_bytes,
            extracted=parsed,
            extractor_version=_KNOWLEDGE_EXTRACTOR_VERSION,
        )

        # Store IR (write-once idempotent)
        try:
            inserted = await store.insert_ir(
                ir_id=ir.ir_id,
                artifact_id=artifact_id,
                ir_type=kind,
                content=ir.to_dict(),
                content_hash=ir.content_hash,
            )
        except Exception as exc:
            logger.error("IR insert failed: %s: %s", artifact_id, exc)
            result.failed_items.append({"artifact_id": artifact_id, "error": str(exc)})
            continue

        if not inserted:
            result.already_ir += 1
            continue

        result.ir_ids.append(ir.ir_id)

        # Build and upsert graph nodes/edges
        nodes = _build_nodes(artifact_id, ir.ir_id, parsed, project_id)
        edges = _build_edges(nodes)

        try:
            for node in nodes:
                await store.upsert_node(node)
            for edge in edges:
                await store.upsert_edge(edge)
            result.nodes_created += len(nodes)
            result.edges_created += len(edges)
        except Exception as exc:
            logger.error("Node/edge upsert failed: %s: %s", artifact_id, exc)
            result.failed_items.append(
                {
                    "artifact_id": artifact_id,
                    "error": f"graph: {exc}",
                }
            )
            continue

        summary = _build_summary(parsed)
        KnowledgeEvent(
            event_type="ingest",
            job_id=job.job_id,
            attempt_id=attempt.attempt_id,
            artifact_id=artifact_id,
            ir_id=ir.ir_id,
            details=f"summary: {summary}",
        ).emit()

    return result


def _parse_for_understand(
    artifact_id: str,
    content_hash: str,
    kind: str,
) -> Optional[dict[str, Any]]:
    """Parse artifact for understanding. Returns extracted dict or None."""
    from umrs.knowledge import parse_code

    ext_map = {
        "code": ".py",
        "cpp": ".cpp",
        "cuda": ".cu",
        "document": ".md",
        "schema": ".json",
        "catalog": ".json",
    }
    ext = ext_map.get(kind, ".py")
    fake_name = f"{artifact_id}{ext}"

    try:
        content_bytes = content_hash.encode("utf-8")
        parsed = parse_code(fake_name, content_bytes)
        return {
            "modules": parsed.modules,
            "classes": [c.name for c in parsed.classes],
            "functions": [f.name for f in parsed.functions],
            "algorithms": parsed.algorithms,
            "dependencies": parsed.dependencies,
            "kernels": [k.name for k in parsed.kernels],
            "summary": parsed.summary,
            "language": parsed.language,
            "tags": parsed.tags,
        }
    except Exception as exc:
        logger.warning("Parse failed for %s: %s", artifact_id, exc)
        return None


def _build_nodes(
    artifact_id: str,
    ir_id: str,
    parsed: dict[str, Any],
    project_id: str,
) -> list:  # list[KnowledgeNode]
    """Build KnowledgeNode entries from parsed code output."""
    from umrs.knowledge.store import KnowledgeNode

    nodes: list = []

    file_id = f"node-file-{artifact_id}"
    nodes.append(
        KnowledgeNode(
            node_id=file_id,
            project_id=project_id,
            label=artifact_id,
            node_type="file",
            content={"artifact_id": artifact_id, "ir_id": ir_id},
            content_hash=_sha256_hex(f"file-{artifact_id}".encode()),
            source_ir=ir_id,
        )
    )

    for cls_name in parsed.get("classes", []):
        nid = f"node-class-{cls_name}-{artifact_id}"
        nodes.append(
            KnowledgeNode(
                node_id=nid,
                project_id=project_id,
                label=cls_name,
                node_type="class",
                content={"artifact_id": artifact_id, "parent": file_id},
                content_hash=_sha256_hex(f"class-{cls_name}-{artifact_id}".encode()),
                source_ir=ir_id,
            )
        )

    for fn_name in parsed.get("functions", []):
        nid = f"node-fn-{fn_name}-{artifact_id}"
        nodes.append(
            KnowledgeNode(
                node_id=nid,
                project_id=project_id,
                label=fn_name,
                node_type="function",
                content={"artifact_id": artifact_id},
                content_hash=_sha256_hex(f"fn-{fn_name}-{artifact_id}".encode()),
                source_ir=ir_id,
            )
        )

    return nodes


def _build_edges(nodes: list) -> list:  # list[KnowledgeEdge]
    """Create parent-child edges between nodes."""
    from umrs.knowledge.store import KnowledgeEdge

    edges: list = []
    for node in nodes:
        if node.node_type == "class":
            for n in nodes:
                if n.node_type == "file":
                    edges.append(
                        KnowledgeEdge(
                            edge_id=f"edge-{n.node_id}-contains-{node.node_id}",
                            project_id=node.project_id,
                            source_id=n.node_id,
                            target_id=node.node_id,
                            relation_type="contains",
                            content={"role": "parent"},
                        )
                    )
        elif node.node_type == "function":
            for n in nodes:
                if n.node_type == "class" and node.content.get("parent") == n.node_id:
                    edges.append(
                        KnowledgeEdge(
                            edge_id=f"edge-{n.node_id}-defines-{node.node_id}",
                            project_id=node.project_id,
                            source_id=n.node_id,
                            target_id=node.node_id,
                            relation_type="defines",
                            content={"role": "method"},
                        )
                    )
                    break
    return edges


# ── Result Persistence ───────────────────────────────────────────────────────


def _write_result(attempt: JobAttempt, result: Any) -> None:
    """Store job result in attempt metrics."""
    result_dict = {
        "artifact_ids": result.artifact_ids if hasattr(result, "artifact_ids") else [],
        "ir_ids": result.ir_ids if hasattr(result, "ir_ids") else [],
        "nodes_created": getattr(result, "nodes_created", 0),
        "edges_created": getattr(result, "edges_created", 0),
        "already_ir": getattr(result, "already_ir", 0),
        "failed_items": result.failed_items if hasattr(result, "failed_items") else [],
        "degraded": getattr(result, "degraded", False),
        "degraded_reason": getattr(result, "degraded_reason", ""),
    }
    attempt.metrics["result"] = json.dumps(result_dict)
    logger.info(
        "Job result written: type=%s degraded=%s",
        type(result).__name__,
        result_dict.get("degraded", False),
    )


# ── Status Checkers ──────────────────────────────────────────────────────────


class KnowledgeIngestStatusChecker(BackendStatusChecker):
    """Status checker for synchronous knowledge-ingest jobs."""

    @property
    def job_type(self) -> JobType:
        return JobType.KNOWLEDGE_INGEST

    async def check_status(
        self,
        attempt: JobAttempt,
        job: Job,
    ) -> AttemptCheckResult:
        if not attempt.backend_ref:
            return AttemptCheckResult.failed("No backend reference")
        if attempt.backend_ref.startswith("knowledge_ingest|"):
            return AttemptCheckResult.succeeded(metrics={"verified_after_submit": True})
        return AttemptCheckResult.failed("Unexpected backend reference")


class KnowledgeUnderstandStatusChecker(BackendStatusChecker):
    """Status checker for synchronous knowledge-understand jobs."""

    @property
    def job_type(self) -> JobType:
        return JobType.KNOWLEDGE_UNDERSTAND

    async def check_status(
        self,
        attempt: JobAttempt,
        job: Job,
    ) -> AttemptCheckResult:
        if not attempt.backend_ref:
            return AttemptCheckResult.failed("No backend reference")
        if attempt.backend_ref.startswith("knowledge_understand|"):
            return AttemptCheckResult.succeeded(metrics={"verified_after_submit": True})
        return AttemptCheckResult.failed("Unexpected backend reference")


# ── KnowledgeVerifyBackend ────────────────────────────────────────────────────


class KnowledgeVerifyBackend(OrchestratorBackend):
    """Run all verifiers → produce confidence report."""

    def __init__(self) -> None:
        self._project_id: str = ""

    @property
    def backend_name(self) -> str:
        return "knowledge_verify"

    async def submit_job(
        self,
        job: Job,
        attempt: JobAttempt,
    ) -> str:
        from umrs.knowledge.verifiers import get_all_verifiers
        from umrs.knowledge.verifiers.base import VerifierResult
        from umrs.knowledge.store import KnowledgeStore
        from umrs.knowledge.verification_store import (
            VerificationStore,
            VerificationReport,
        )

        project_id = self._project_id or job.spec.get("project_id", job.project_id)
        self._project_id = project_id

        KnowledgeEvent(
            event_type="verify",
            job_id=job.job_id,
            attempt_id=attempt.attempt_id,
            details=f"project={project_id} verifiers={len(get_all_verifiers())}",
        ).emit()

        pool = _get_pool()
        if pool is None:
            return (
                f"knowledge_verify|project={project_id}|degraded=no_pool|confidence=0.0"
            )

        try:
            store = KnowledgeStore(pool)
            vstore = VerificationStore(pool)
            verifiers = get_all_verifiers()
            findings: list[dict[str, Any]] = []

            for verifier in verifiers:
                try:
                    result: VerifierResult = verifier.run(store, project_id)
                    if result.found:
                        for f in result.findings:
                            findings.append(
                                {
                                    "verifier": verifier.name,
                                    "key": f.get("key", verifier.name),
                                    "severity": f.get("severity", "medium"),
                                    "message": f.get("message", ""),
                                }
                            )
                    KnowledgeEvent(
                        event_type="verify",
                        job_id=job.job_id,
                        attempt_id=attempt.attempt_id,
                        details=f"verifier={verifier.name} found={result.found}",
                    ).emit()
                except Exception as exc:
                    logger.warning("Verifier %s failed: %s", verifier.name, exc)
                    # Degraded — skip, don't fail the job

            # Compute confidence score
            confidence = self._compute_confidence(findings)

            # Write report to verification store
            report = VerificationReport(
                project_id=project_id,
                report_version=1,
                confidence=confidence,
                severity_counts=self._severity_counts(findings),
            )

            try:
                await vstore.create_report(report)
                KnowledgeEvent(
                    event_type="verify",
                    job_id=job.job_id,
                    attempt_id=attempt.attempt_id,
                    details=f"confidence={confidence:.2f} report_v{report.report_version}",
                ).emit()
            except Exception as report_exc:
                logger.warning("Failed to write verification report: %s", report_exc)

            _write_result(
                attempt,
                KnowledgeIngestResult(
                    artifact_ids=[],
                    failed_items=[],
                    degraded=False,
                ),
            )

            return (
                f"knowledge_verify|project={project_id}|"
                f"confidence={confidence:.2f}|findings={len(findings)}"
            )

        finally:
            if pool is not None:
                pool.close()
                await pool.wait_closed()

    def _compute_confidence(self, findings: list[dict[str, Any]]) -> float:
        """Compute confidence score 0.0-1.0 from findings.

        Severity weights: blocking=0.0, high=0.5, medium=0.8, low=0.95, info=1.0.
        Confidence = weighted average.
        """
        weights = {
            "blocking": 0.0,
            "high": 0.5,
            "medium": 0.8,
            "low": 0.95,
            "info": 1.0,
        }
        if not findings:
            return 1.0

        total_weight = 0.0
        count = 0
        for f in findings:
            severity = f.get("severity", "medium")
            w = weights.get(severity, 0.8)
            total_weight += w
            count += 1

        if count == 0:
            return 1.0
        return total_weight / count

    def _severity_counts(self, findings: list[dict[str, Any]]) -> dict[str, int]:
        counts: dict[str, int] = {
            "blocking": 0,
            "high": 0,
            "medium": 0,
            "low": 0,
            "info": 0,
        }
        for f in findings:
            severity = f.get("severity", "medium")
            if severity in counts:
                counts[severity] += 1
            else:
                counts["medium"] += 1
        return counts

    async def check_status(
        self,
        attempt: JobAttempt,
        job: Job,
    ) -> AttemptCheckResult:
        if not attempt.backend_ref:
            return AttemptCheckResult.failed("No backend reference")
        if attempt.backend_ref.startswith("knowledge_verify|"):
            return AttemptCheckResult.succeeded(metrics={"verified_after_submit": True})
        return AttemptCheckResult.failed("Unexpected backend reference")


class KnowledgeVerifyStatusChecker(BackendStatusChecker):
    """Status checker for synchronous knowledge-verify jobs."""

    @property
    def job_type(self) -> JobType:
        return JobType.KNOWLEDGE_VERIFY

    async def check_status(
        self,
        attempt: JobAttempt,
        job: Job,
    ) -> AttemptCheckResult:
        if not attempt.backend_ref:
            return AttemptCheckResult.failed("No backend reference")
        if attempt.backend_ref.startswith("knowledge_verify|"):
            return AttemptCheckResult.succeeded(metrics={"verified_after_submit": True})
        return AttemptCheckResult.failed("Unexpected backend reference")


# ── KnowledgeExportBackend ────────────────────────────────────────────────────


class KnowledgeExportBackend(OrchestratorBackend):
    """Export project knowledge to a .kb bundle directory."""

    def __init__(self) -> None:
        self._project_id: str = ""

    @property
    def backend_name(self) -> str:
        return "knowledge_export"

    async def submit_job(
        self,
        job: Job,
        attempt: JobAttempt,
    ) -> str:
        from umrs.knowledge.bundle_export import (
            KnowledgeExportEngine,
            verify_bundle_integrity,
        )
        from umrs.knowledge.store import KnowledgeStore

        project_id = self._project_id or job.spec.get("project_id", job.project_id)
        self._project_id = project_id

        KnowledgeEvent(
            event_type="create",
            job_id=job.job_id,
            attempt_id=attempt.attempt_id,
            details=f"project={project_id} export",
        ).emit()

        pool = _get_pool()
        degraded = False
        degraded_reason = ""

        if pool is None:
            degraded_reason = "Postgres pool unavailable"
            degraded = True
            logger.warning("KnowledgeExport degraded: %s", degraded_reason)

        try:
            if pool is not None:
                store = KnowledgeStore(pool)
                engine = KnowledgeExportEngine(
                    store=store,
                    output_dir=Path(
                        os.getenv(
                            "UMRS_KB_EXPORT_DIR",
                            str(Path(os.getcwd()) / ".kb_exports"),
                        )
                    ),
                    secret_key=os.getenv("UMRS_SECRET_KEY", ""),
                )
                result = await engine.export_project(project_id)
            else:
                result = KnowledgeExportResult(
                    project_id=project_id,
                    degraded=True,
                    degraded_reason=degraded_reason,
                )

            _write_result(attempt, result)

            # Verify integrity if not degraded
            if not result.degraded and result.bundle_path:
                ok = verify_bundle_integrity(
                    Path(result.bundle_path),
                    os.getenv("UMRS_SECRET_KEY", ""),
                )
                if not ok:
                    logger.error(
                        "Bundle integrity check failed for %s", result.bundle_path
                    )
                    result.degraded = True
                    result.degraded_reason = "bundle integrity check failed"

            return (
                f"knowledge_export|project={project_id}|"
                f"artifacts={result.artifact_count}|"
                f"embeds={result.embedding_count}|"
                f"hash={result.manifest_hash}"
            )

        finally:
            if pool is not None:
                pool.close()
                await pool.wait_closed()

        if degraded:
            result.degraded = True
            result.degraded_reason = degraded_reason

        return f"knowledge_export|project={project_id}|degraded={degraded_reason}"


class KnowledgeExportStatusChecker(BackendStatusChecker):
    """Status checker for synchronous knowledge-export jobs."""

    @property
    def job_type(self) -> JobType:
        return JobType.KNOWLEDGE_EXPORT

    async def check_status(
        self,
        attempt: JobAttempt,
        job: Job,
    ) -> AttemptCheckResult:
        if not attempt.backend_ref:
            return AttemptCheckResult.failed("No backend reference")
        if attempt.backend_ref.startswith("knowledge_export|"):
            return AttemptCheckResult.succeeded(metrics={"verified_after_submit": True})
        return AttemptCheckResult.failed("Unexpected backend reference")


# ── Auto-Ingestion Enqueuers (TASK-035) ───────────────────────────────────────


async def _resolve_succeeded_attempts(pool: Any, job: Any) -> list[Any]:
    """Resolve attempt IDs to inspect: commit pointer or latest success."""
    commit_id = getattr(job, "commit_attempt_id", None)
    if commit_id:
        return [commit_id]
    try:
        rows = await pool.fetch(
            "SELECT attempt_id FROM job_attempts WHERE job_id = $1 "
            "AND status = 'succeeded' ORDER BY attempt_no DESC LIMIT 1",
            job.job_id,
        )
        return [r["attempt_id"] for r in rows] if rows else []
    except Exception:
        return []


async def _artifacts_from_attempt_metrics(pool: Any, attempts: list[Any]) -> list[dict[str, Any]]:
    """Extract artifact dicts recorded in attempt metrics."""
    results: list[dict[str, Any]] = []
    for att_id in attempts:
        try:
            row = await pool.fetchval(
                "SELECT metrics FROM job_attempts WHERE attempt_id = $1",
                att_id,
            )
            metrics = row if isinstance(row, dict) else {}
            for art in metrics.get("artifacts", []):
                if isinstance(art, dict) and art.get("artifact_id"):
                    results.append(art)
        except Exception:
            pass
    return results


async def _artifacts_from_store(pool: Any, project_id: str) -> list[dict[str, Any]]:
    """Pull artifact refs from the knowledge_artifacts table."""
    try:
        rows = await pool.fetch(
            "SELECT artifact_id, content_hash, kind, source_ref "
            "FROM knowledge_artifacts WHERE project_id = $1 "
            "ORDER BY created_at DESC",
            project_id,
        )
    except Exception:
        return []
    return [
        {
            "artifact_id": r["artifact_id"],
            "content_hash": r["content_hash"] or "",
            "kind": r["kind"] or "document",
            "source_ref": r.get("source_ref", ""),
        }
        for r in rows
    ]


def _collect_artifact_ids(job: Any) -> list[dict[str, Any]]:
    """Collect artifact IDs from a successful job's attempt metrics.

    Returns a list of dicts with artifact_id, content_hash, kind, source_ref.
    """
    artifacts: list[dict[str, Any]] = []

    try:
        pool = _get_pool()
        if pool is None:
            return artifacts

        async def _collect() -> list[dict[str, Any]]:
            attempts = await _resolve_succeeded_attempts(pool, job)
            results = await _artifacts_from_attempt_metrics(pool, attempts)
            results.extend(await _artifacts_from_store(pool, job.project_id))
            return results

        try:
            artifacts = asyncio.run(_collect())
            pool.close()
            asyncio.run(pool.wait_closed())
        except Exception:
            pass
    except Exception:
        pass

    return artifacts


def _dedup_artifacts(
    artifact_refs: list[dict[str, Any]],
    project_id: str,
) -> tuple[list[dict[str, Any]], str]:
    """Filter artifact refs through content-hash dedup (artifact_storage).

    Returns (deduped_refs, bucket_name). Falls back to all refs when
    S3 is unavailable (degraded mode).
    """
    deduped: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    bucket = os.getenv("UMRS_DATA_S3_BUCKET", "umrs-data")

    try:
        from umrs.knowledge.artifact_storage import ArtifactStorage

        endpoint = os.getenv("S3_ENDPOINT_URL", "http://localhost:9000")
        key = os.getenv("S3_ACCESS_KEY", "")
        secret = os.getenv("S3_SECRET_KEY", "")

        s3_client = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=key,
            aws_secret_access_key=secret,
            use_ssl=False,
        )
        storage = ArtifactStorage(s3_client, bucket)

        for art_ref in artifact_refs:
            content_hash = art_ref.get("content_hash", "") or art_ref.get("sha256", "")

            if not content_hash or content_hash in seen_hashes:
                continue

            # Check if content already exists (dedup)
            if storage.artifact_exists(project_id, content_hash):
                seen_hashes.add(content_hash)
                logger.debug(
                    "Artifact %s already exists (dedup), skip",
                    content_hash[:16],
                )
                continue

            seen_hashes.add(content_hash)
            deduped.append(art_ref)
    except Exception as exc:
        # Storage unavailable — proceed anyway (degraded)
        logger.warning(
            "Dedup check skipped (S3 unavailable): %s — enqueuing all",
            exc,
        )
        deduped = artifact_refs

    return deduped, bucket


def _store_artifact_provenance(
    deduped: list[dict[str, Any]],
    project_id: str,
    job: Any,
    attempt: Any,
    workflow: str,
    step: str,
) -> None:
    """Store artifact metadata with provenance (best-effort)."""
    store_pool = _get_pool()
    if not store_pool:
        return
    try:
        from umrs.knowledge.store import KnowledgeStore

        store = KnowledgeStore(store_pool)

        async def _store_provenance():
            for art in deduped:
                try:
                    aid = art.get("artifact_id", f"art-{uuid.uuid4().hex[:8]}")
                    ch = art.get("content_hash", "") or art.get("sha256", "")
                    kind = art.get("kind", "document")
                    await store.insert_artifact(
                        artifact_id=aid,
                        content_hash=ch,
                        kind=kind,
                        project_id=project_id,
                        metadata={
                            "produced_by_job_id": job.job_id,
                            "produced_by_attempt_id": attempt.attempt_id
                            if hasattr(attempt, "attempt_id")
                            else "",
                            "workflow": workflow,
                            "step": step,
                        },
                    )
                except Exception:
                    pass

        try:
            asyncio.run(_store_provenance())
        except Exception:
            pass
    except Exception:
        pass


def _enqueue_ingest_job(
    deduped: list[dict[str, Any]],
    bucket: str,
    project_id: str,
    user_id: str,
    job: Any,
    attempt: Any,
    workflow: str,
    step: str,
) -> str | None:
    """Enqueue the KNOWLEDGE_INGEST job. Returns job ID or None."""
    from .models import JobType
    from .service import JobSubmission, get_job_service

    ingest_sources = [
        {
            "kind": art.get("kind", "document"),
            "uri": art.get("source_ref", ""),
            "artifact_id": art.get("artifact_id", ""),
            "content_hash": content_hash,
        }
        for art in deduped
        if (content_hash := (art.get("content_hash") or art.get("sha256", "")))
    ]

    if not ingest_sources:
        return None

    _store_artifact_provenance(deduped, project_id, job, attempt, workflow, step)

    job_id = get_job_service().submit_job(
        JobSubmission(
            project_id=project_id,
            user_id=user_id,
            job_type=JobType.KNOWLEDGE_INGEST,
            spec={
                "sources": ingest_sources,
                "bucket": bucket,
                "project_id": project_id,
            },
            priority=job.priority,
            max_attempts=3,
        ),
    )
    logger.info(
        "Enqueued KNOWLEDGE_INGEST: job=%s artifacts=%d",
        job_id,
        len(ingest_sources),
    )
    return job_id


def _list_project_artifact_sources(project_id: str) -> list[dict[str, Any]]:
    """List all artifact sources for a project (best-effort)."""
    store_pool = _get_pool()
    if not store_pool:
        return []
    try:
        from umrs.knowledge.store import KnowledgeStore

        store = KnowledgeStore(store_pool)

        async def _list_new():
            arts, _ = await store.list_artifacts(project_id)
            return [
                {
                    "artifact_id": a.artifact_id,
                    "content_hash": a.content_hash,
                    "kind": a.kind,
                    "source_ref": a.source_ref,
                }
                for a in arts
            ]

        try:
            return asyncio.run(_list_new())
        except Exception:
            return []
    except Exception:
        return []


def _enqueue_understand_job(
    project_id: str,
    user_id: str,
    job: Any,
) -> str | None:
    """Enqueue the KNOWLEDGE_UNDERSTAND job. Returns job ID or None."""
    from .models import JobType

    sources = _list_project_artifact_sources(project_id)
    if not sources:
        logger.info("No artifacts for KNOWLEDGE_UNDERSTAND: job=%s", job.job_id)
        return None

    from .service import JobSubmission, get_job_service

    us_job_id = get_job_service().submit_job(
        JobSubmission(
            project_id=project_id,
            user_id=user_id,
            job_type=JobType.KNOWLEDGE_UNDERSTAND,
            spec={
                "sources": sources,
                "project_id": project_id,
                "parent_job_id": job.job_id,
            },
            priority=job.priority,
            max_attempts=3,
        ),
    )
    logger.info(
        "Enqueued KNOWLEDGE_UNDERSTAND: job=%s artifacts=%d",
        us_job_id,
        len(sources),
    )
    return us_job_id


def _create_produced_by_edges(
    deduped: list[dict[str, Any]],
    project_id: str,
    user_id: str,
    job: Any,
    attempt: Any,
    workflow: str,
    step: str,
) -> None:
    """Create PRODUCED_BY edges in the knowledge graph (best-effort)."""
    from umrs.knowledge.store import KnowledgeEdge, KnowledgeNode, KnowledgeStore

    store_pool = _get_pool()
    if not (store_pool and deduped):
        return
    try:
        store = KnowledgeStore(store_pool)

        async def _create_edges():
            prod_entity_id = f"prod-job-{job.job_id}"
            job_type = getattr(job, "job_type", "unknown")
            # Create provenance node
            await store.upsert_node(
                KnowledgeNode(
                    node_id=prod_entity_id,
                    project_id=project_id,
                    label=f"job:{job.job_id}",
                    node_type="job",
                    content={
                        "job_id": job.job_id,
                        "job_type": job_type.value
                        if hasattr(job_type, "value")
                        else job_type,
                        "workflow": workflow,
                        "step": step,
                        "user_id": user_id,
                    },
                    content_hash=_sha256_hex(
                        f"provenance-{job.job_id}-{workflow}-{step}".encode()
                    ),
                    source_ir="",
                )
            )

            # Create PRODUCED_BY edges
            for art in deduped:
                aid = art.get("artifact_id", "")
                if not aid:
                    continue
                try:
                    await store.upsert_edge(
                        KnowledgeEdge(
                            edge_id=f"edge-{aid}-produced_by-{job.job_id}",
                            project_id=project_id,
                            source_id=aid,
                            target_id=prod_entity_id,
                            relation_type="produced_by",
                            content={
                                "workflow": workflow,
                                "step": step,
                                "attempt_id": attempt.attempt_id
                                if hasattr(attempt, "attempt_id")
                                else "",
                            },
                        )
                    )
                except Exception:
                    pass

        try:
            asyncio.run(_create_edges())
        except Exception:
            pass
    except Exception:
        pass


def enqueue_knowledge_ingest(
    job: Any,
    attempt: Any,
    workflow: str = "default",
    step: str = "main",
) -> list[str]:
    """Enqueue KNOWLEDGE_INGEST and KNOWLEDGE_UNDERSTAND jobs.

    Idempotent by content hash — checks artifact_storage first.
    Returns list of enqueued job IDs.

    Sets artifact metadata:
      - produced_by_job_id
      - workflow
      - step
    """
    enqueued: list[str] = []

    project_id = getattr(job, "project_id", "")
    user_id = getattr(job, "user_id", "")
    if not project_id or not user_id:
        return enqueued

    # 1. Collect artifact references from the successful run
    artifact_refs = _collect_artifact_ids(job)
    if not artifact_refs:
        logger.info("No artifacts found for auto-ingest: job=%s", job.job_id)
        return enqueued

    # 2. Filter through content-hash dedup (artifact_storage)
    deduped, bucket = _dedup_artifacts(artifact_refs, project_id)
    if not deduped:
        logger.info(
            "All artifacts deduplicated, nothing to ingest: job=%s",
            job.job_id,
        )
        return enqueued

    # 3. Enqueue KNOWLEDGE_INGEST job (with provenance metadata)
    try:
        ingest_job_id = _enqueue_ingest_job(
            deduped, bucket, project_id, user_id, job, attempt, workflow, step
        )
        if ingest_job_id:
            enqueued.append(ingest_job_id)
    except Exception as exc:
        logger.warning("Failed to enqueue KNOWLEDGE_INGEST: %s", exc)

    # 4. Enqueue KNOWLEDGE_UNDERSTAND job (always after ingest)
    try:
        us_job_id = _enqueue_understand_job(project_id, user_id, job)
        if us_job_id:
            enqueued.append(us_job_id)
    except Exception as exc:
        logger.warning("KnowledgeUnderstand enqueue failed: %s", exc)

    # 5. Create PRODUCED_BY edge in knowledge graph (best-effort)
    try:
        _create_produced_by_edges(
            deduped, project_id, user_id, job, attempt, workflow, step
        )
    except Exception as exc:
        logger.warning("PRODUCED_BY edge creation failed: %s", exc)

    return enqueued


def enqueue_ingest_failure(
    job_id: str,
    attempt_no: int,
    exit_reason: str,
    user_id: str,
    project_id: str,
) -> None:
    """Log that auto-ingest was skipped due to job failure.

    Creates a FAILURE node in the knowledge graph for traceability.
    """
    from umrs.knowledge.store import KnowledgeNode

    try:
        store_pool = _get_pool()
        if store_pool is None:
            return

        from umrs.knowledge.store import KnowledgeStore

        store = KnowledgeStore(store_pool)

        async def _log_failure() -> None:
            try:
                fail_entity_id = f"fail-{job_id}-{attempt_no}"
                await store.upsert_node(
                    KnowledgeNode(
                        node_id=fail_entity_id,
                        project_id=project_id,
                        label=f"job:{job_id} FAILED",
                        node_type="job_failure",
                        content={
                            "job_id": job_id,
                            "attempt_no": attempt_no,
                            "exit_reason": exit_reason,
                            "user_id": user_id,
                        },
                        content_hash=_sha256_hex(
                            f"failure-{job_id}-{exit_reason}".encode()
                        ),
                        source_ir="",
                    )
                )
                # Link the failed job to this failure node
                try:
                    from umrs.knowledge.store import KnowledgeEdge

                    await store.upsert_edge(
                        KnowledgeEdge(
                            edge_id=f"edge-{job_id}-failure-{fail_entity_id}",
                            project_id=project_id,
                            source_id=f"prod-job-{job_id}",
                            target_id=fail_entity_id,
                            relation_type="caused_by",
                            content={"reason": exit_reason},
                        )
                    )
                except Exception:
                    pass
            except Exception:
                pass

        try:
            asyncio.run(_log_failure())
        except Exception:
            pass
    except Exception:
        pass


# ── A2L Bridge (ADR-017) ──────────────────────────────────────────────────


def _bridge_a2l_ref(artifact_id: str, content_hash: str, project_id: str) -> None:
    """Best-effort: store artifact → A2L identity mapping in DB.

    Never raises — A2L unavailability must NOT fail knowledge jobs.
    """
    try:
        from umrs.knowledge.a2l_bridge import A2lBridge
        from umrs.knowledge.store import get_store
        from umrs.settings import Settings

        settings = Settings()
        bridge = A2lBridge(gateway_url=settings.gateway_url)

        # Resolve first (A2L may already know this hash)
        ref = bridge.resolve_a2l_hash(content_hash)
        if ref and ref.a2l_id:
            bridge.sync_a2l_ref(
                type(ref)(artifact_id=artifact_id, a2l_hash=content_hash, a2l_id=ref.a2l_id)
            )
            a2l_id = ref.a2l_id
        else:
            a2l_id = ""

        # Always persist UMRS→A2L mapping regardless of resolve result
        store = get_store()
        pool = store._pool
        if pool:
            async def _persist() -> None:  # noqa: D103
                await store.insert_a2l_ref(artifact_id, content_hash, a2l_id)

            asyncio.run(_persist())
    except Exception:
        # Best-effort: silently degrade — A2L absent or unavailable
        pass
