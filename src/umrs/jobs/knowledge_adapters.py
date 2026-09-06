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
import re
import threading
import uuid
from contextlib import asynccontextmanager
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

_KNOWLEDGE_EXTRACTOR_VERSION = "1.1.0"


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
        """Log and return self for chaining; also feed the SSE project feed."""
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
        # SSE producer (was unwired — audit follow-up): project-scoped fan-out
        # when the embedding backend set its project context; skips silently
        # otherwise (fail closed). events_sse imports this module at top
        # level, so the import here must stay runtime-local.
        try:
            from umrs.knowledge import events_sse

            events_sse.emit(self)
        except Exception:  # noqa: BLE001 — SSE must never break a job
            pass
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


@dataclass
class KnowledgeEmbedResult:
    """Result of a KNOWLEDGE_EMBED job attempt."""

    nodes_embedded: int = 0
    skipped: int = 0  # artifacts with no decodable text
    failed_items: list[dict[str, Any]] = field(default_factory=list)
    degraded: bool = False
    degraded_reason: str = ""


# ── Helpers ──────────────────────────────────────────────────────────────────


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fire_coroutine(coro: Any, label: str = "") -> None:
    """Run a coroutine from sync code, respecting the ambient loop.

    In a plain sync context, runs via asyncio.run(). Inside a running loop
    (e.g. the JobService engine thread), schedules as a fire-and-forget task
    instead of calling asyncio.run(), which would raise
    "asyncio.run() cannot be called from a running event loop".
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(coro)
        return

    async def _swallow() -> None:
        try:
            await coro
        except Exception:
            logger.warning("background task %s failed", label, exc_info=True)

    asyncio.get_running_loop().create_task(_swallow())


def _run_coroutine_sync(coro: Any, label: str = "") -> Any:
    """Run a coroutine from sync code and return its result.

    Without a running loop, uses ``asyncio.run`` (unchanged behaviour).
    With a running loop (e.g. when the post-completion hook fires from the
    JobService engine), executes the coroutine on a dedicated helper thread
    with its own event loop — calling ``asyncio.run`` in the engine thread
    would raise "asyncio.run() cannot be called from a running event loop".
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    outcome: dict[str, Any] = {}

    def _runner() -> None:
        try:
            outcome["value"] = asyncio.run(coro)
        except Exception as exc:  # noqa: BLE001 — re-raised below
            outcome["error"] = exc

    worker = threading.Thread(target=_runner, name=f"krc-{label}"[:20], daemon=True)
    worker.start()
    worker.join(timeout=120)
    if worker.is_alive():
        raise RuntimeError(f"coroutine {label!r} timed out")
    if "error" in outcome:
        raise outcome["error"]
    return outcome.get("value")


def _resolve_knowledge_dsn() -> str:
    """Resolve the knowledge PG DSN with durable-jobs-pool parity.

    Precedence: UMRS_DATA_CATALOG_DB / DATABASE_URL (full DSN), otherwise the
    component vars the jobs pool uses (DB_HOST/DB_PORT/UMRS_JOBS_DB/
    POSTGRES_*/DB_SUPERUSER*).  Mirrors api._asyncpg_pool._resolve_knowledge_dsn;
    duplicated locally because this module must not import from umrs.api.

    Returns "" when no database configuration is present at all: the
    component-var fallback requires at least one component env var, so
    unconfigured environments keep the previous "no pool" degradation
    semantics instead of fabricating a localhost DSN.
    """
    for var in ("UMRS_DATA_CATALOG_DB", "DATABASE_URL"):
        dsn = os.getenv(var, "")
        if dsn:
            return dsn
    component_vars = (
        "DB_HOST",
        "DB_PORT",
        "UMRS_JOBS_DB",
        "POSTGRES_PORT",
        "POSTGRES_DB",
        "DB_SUPERUSER",
        "DB_SUPERUSER_PASSWORD",
        "POSTGRES_USER",
        "POSTGRES_PASSWORD",
    )
    if not any(os.getenv(var) for var in component_vars):
        return ""
    host = os.getenv("DB_HOST", "127.0.0.1")
    port = os.getenv("DB_PORT", os.getenv("POSTGRES_PORT", "5432"))
    name = os.getenv("UMRS_JOBS_DB", os.getenv("POSTGRES_DB", "umrs_db"))
    user = os.getenv("DB_SUPERUSER", os.getenv("POSTGRES_USER", "postgres"))
    password = os.getenv(
        "DB_SUPERUSER_PASSWORD", os.getenv("POSTGRES_PASSWORD", "postgres")
    )
    return f"postgresql://{user}:{password}@{host}:{port}/{name}"


def _resolve_bucket(job_spec: Optional[dict[str, Any]] = None) -> str:
    """Resolve the data bucket: job spec > UMRS_DATA_S3_BUCKET > S3_BUCKET > default."""
    if job_spec and job_spec.get("bucket"):
        return str(job_spec["bucket"])
    return os.getenv("UMRS_DATA_S3_BUCKET") or os.getenv("S3_BUCKET") or "umrs-data"


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
    """Get a Postgres asyncpg pool from env. Returns None if unavailable.

    Uses the full DSN chain (full-DSN vars, else component vars) so deployed
    containers without DATABASE_URL still resolve the knowledge database.
    """
    conn_str = _resolve_knowledge_dsn()
    if not conn_str:
        return None
    try:
        import asyncpg

        return asyncpg.create_pool(dsn=conn_str)  # type: ignore[return-value]
    except Exception:
        return None


@asynccontextmanager
async def _open_pool() -> Any:
    """Yield an initialized asyncpg pool bound to the caller's event loop.

    asyncpg pools must be created — and fully consumed and closed — inside ONE
    event loop: a pool built by the sync ``_get_pool()`` is never initialized,
    so every fetch raises InterfaceError('pool is not initialized') (silently
    swallowed by the collectors' guards).  All async consumers use this helper
    and keep every DB operation inside its body.

    The pool is codec-aware: without codecs, asyncpg hands jsonb values out as
    ``str`` and REJECTS dict parameters for jsonb columns (DataError),
    which silently killed provenance/graph writes.
    """
    pool = None
    dsn = _resolve_knowledge_dsn()
    if dsn:
        try:
            import asyncpg

            pool = await asyncpg.create_pool(dsn=dsn, init=_init_json_codecs)
        except Exception as exc:
            logger.warning("Knowledge asyncpg pool unavailable: %s", exc)
            pool = None
    try:
        yield pool
    finally:
        if pool is not None:
            try:
                await pool.close()
            except Exception:  # noqa: BLE001
                pass
            try:
                await pool.wait_closed()
            except Exception:  # noqa: BLE001
                pass


async def _init_json_codecs(conn: Any) -> None:
    """Register symmetric JSON/JSONB codecs on an ad-hoc pool connection.

    ``dict`` values encode to JSON on write; jsonb values decode back to dict
    on read. ``_parse_metrics_jsonb`` deliberately stays in place: it remains
    the defensive fallback for pools/paths outside this helper.
    """
    for typename in ("json", "jsonb"):
        await conn.set_type_codec(
            typename,
            encoder=json.dumps,
            decoder=json.loads,
            schema="pg_catalog",
        )


def _resolve_s3_credentials() -> tuple[str, str]:
    """Resolve S3 credentials: env first, then the platform admin s3cfg.

    Deployed umrs-front containers carry only S3_SECRET_KEY (+ the mounted
    admin s3cfg at ``UMRS_ADMIN_S3_CONFIGURATION_FILE``); S3_ACCESS_KEY is
    absent, so job-side S3 clients must not fail to resolve.
    Returns ``(access_key, secret_key)`` — empty strings when unresolvable.
    """
    access = os.getenv("S3_ACCESS_KEY", "")
    secret = os.getenv("S3_SECRET_KEY", "")
    if access and secret:
        return access, secret
    cfg_file = os.getenv("UMRS_ADMIN_S3_CONFIGURATION_FILE", "/etc/umrs/s3cfg")
    try:
        import configparser

        p = configparser.ConfigParser()
        p.read(cfg_file)
        access = access or p.get("default", "access_key", fallback="")
        secret = secret or p.get("default", "secret_key", fallback="")
    except Exception as exc:
        logger.warning("s3cfg credential resolution failed: %s", exc)
    return access, secret


def _build_s3_client(job_spec: dict[str, Any]) -> tuple[Any, str]:
    """Build an S3 client from env (same config as the ingest backend).

    Returns ``(client_or_None, bucket)`` — None client means S3 unavailable
    and callers must fail the item explicitly (never fall back silently).
    """
    from botocore.config import Config as BotoConfig

    bucket = _resolve_bucket(job_spec)
    access_key, secret_key = _resolve_s3_credentials()
    try:
        client = boto3.client(
            "s3",
            endpoint_url=os.getenv("S3_ENDPOINT_URL", "http://localhost:9000"),
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=os.getenv("S3_REGION", "us-east-1"),
            config=BotoConfig(
                signature_version="s3v4", s3={"addressing_style": "path"}
            ),
            use_ssl=os.getenv("S3_USE_SSL", "false").lower() == "true",
        )
        client.list_buckets()
        return client, bucket
    except Exception as exc:
        logger.warning("S3 unavailable for understand fetch: %s", exc)
        return None, bucket


def _fetch_artifact_bytes(
    s3_client: Any, bucket: str, source_ref: str
) -> Optional[bytes]:
    """Download real artifact bytes from S3 by source_ref key.

    Returns None when unreadable — callers record the failure explicitly;
    we never substitute the content_hash hex string for content.
    """
    if s3_client is None or not source_ref:
        return None
    try:
        obj = s3_client.get_object(Bucket=bucket, Key=source_ref)
        return obj["Body"].read()
    except Exception as exc:
        logger.warning("S3 fetch failed for %s: %s", source_ref, exc)
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
        """Lazily initialize S3 client from env/admin-s3cfg (see _resolve_s3_credentials)."""
        if self._s3_client is not None:
            return
        from botocore.config import Config as BotoConfig

        # Shared resolution: spec > UMRS_DATA_S3_BUCKET > S3_BUCKET > default.
        # (The old hardcoded "umrs-data" default wrote to a bucket that does
        # not exist on live deployments — every put_object failed NoSuchBucket.)
        self._bucket = _resolve_bucket(job_spec)
        endpoint = os.getenv("S3_ENDPOINT_URL", "http://localhost:9000")
        key, secret = _resolve_s3_credentials()
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
        _sse_token = None
        try:  # SSE project attribution for every event in this attempt
            from umrs.knowledge import events_sse

            _sse_token = events_sse.set_project_ctx(project_id)
        except Exception:  # noqa: BLE001 — SSE context is best-effort
            pass
        try:
            return await self._submit_ingest(
                job, attempt, source_refs, project_id, allow_domains
            )
        finally:
            if _sse_token is not None:
                from umrs.knowledge import events_sse

                events_sse.reset_project_ctx(_sse_token)

    async def _submit_ingest(
        self,
        job: Job,
        attempt: JobAttempt,
        source_refs: list[dict[str, Any]],
        project_id: str,
        allow_domains: Optional[list[str]],
    ) -> str:

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

        result_str, success_descs = _run_ingest_sync(
            job,
            attempt,
            self._s3_client,
            self._bucket,
            source_refs,
            project_id,
            allow_domains,
        )

        # Chain KNOWLEDGE_UNDERSTAND so API-triggered ingests become queryable
        # without a manual understand call (mirrors the auto-ingest hook).
        # The child job inherits spec.graph_id — named-graph targets stay named.
        # CRITICAL: pass the acquisition descriptors explicitly. The artifact-
        # row insert is fire-and-forget; re-listing right here raced ahead of
        # the commit in production and produced understand jobs with stale,
        # empty source lists (verified live: ir_ids=0 on every chained job).
        # Best-effort: a chain failure must never fail the acquisition attempt.
        try:
            _enqueue_understand_job(
                project_id,
                getattr(job, "user_id", "") or "",
                job,
                sources=success_descs,
            )
        except Exception as exc:  # defensive — chain must never break ingest
            logger.warning("KNOWLEDGE_UNDERSTAND chain enqueue failed: %s", exc)

        return result_str


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
        _sse_token = None
        try:  # SSE project attribution for every event in this attempt
            from umrs.knowledge import events_sse

            _sse_token = events_sse.set_project_ctx(project_id)
        except Exception:  # noqa: BLE001 — SSE context is best-effort
            pass
        try:
            return await self._submit_understand(job, attempt, project_id)
        finally:
            if _sse_token is not None:
                from umrs.knowledge import events_sse

                events_sse.reset_project_ctx(_sse_token)

    async def _submit_understand(
        self,
        job: Job,
        attempt: JobAttempt,
        project_id: str,
    ) -> str:
        sources: list[dict[str, Any]] = job.spec.get("sources", [])
        KnowledgeEvent(
            event_type="create",
            job_id=job.job_id,
            attempt_id=attempt.attempt_id,
            source_ref=job.spec.get("source_ref", ""),
            details=f"project={project_id}",
        ).emit()

        s3_client, s3_bucket = _build_s3_client(job.spec)
        degraded = False
        degraded_reason = ""

        async with _open_pool() as pool:
            if pool is None:
                degraded_reason = "Postgres pool unavailable"
                degraded = True
                logger.warning("KnowledgeUnderstand degraded: %s", degraded_reason)

            result = await _run_understand(
                job,
                attempt,
                project_id,
                sources,
                pool,
                s3_client,
                s3_bucket,
            )

        if degraded:
            result.degraded = True
            result.degraded_reason = degraded_reason

        _write_result(attempt, result)

        # Chain the embedding stage (audit B3): graph nodes alone are blind to
        # the vector plane. Best-effort — the embed job degrades cleanly when
        # no site embedder is configured.
        if sources:
            try:
                _enqueue_embed_job(project_id, job.user_id or "", job, sources)
            except Exception:  # noqa: BLE001 — chaining must not fail understand
                logger.warning(
                    "KNOWLEDGE_EMBED enqueue failed for job=%s",
                    job.job_id,
                    exc_info=True,
                )

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
) -> tuple[str, list[dict[str, Any]]]:
    from umrs.knowledge.sources import SourceResolver
    from umrs.knowledge.artifact_storage import ArtifactStorage

    result = KnowledgeIngestResult()
    success_ids: list[str] = []
    success_descs: list[dict[str, Any]] = []

    for src_spec in source_refs:
        source_kind = src_spec.get("kind", "local_file")
        uri = src_spec.get("uri", src_spec.get("path", ""))
        if not uri:
            result.failed_items.append({"source": source_kind, "error": "no uri/path"})
            continue

        # inline_text: content travels in the spec — fixes client-local files
        # for remote gateways (the job runs server-side; local paths fail).
        if source_kind == "inline_text":
            _put_inline_text_sync(
                job.job_id,
                attempt.attempt_id,
                ArtifactStorage(s3_client, bucket),
                project_id,
                src_spec,
                f"art-{uuid.uuid4().hex[:12]}",
                result,
                success_ids,
                success_descs,
            )
            continue

        try:
            _ingest_one_source(
                job,
                attempt,
                s3_client,
                bucket,
                project_id,
                source_kind,
                uri,
                allow_domains,
                result,
                success_ids,
                success_descs,
            )
        except Exception as exc:
            logger.warning(
                "knowledge_ingest source failed: kind=%s uri=%s error=%s",
                source_kind,
                uri,
                exc,
            )
            result.failed_items.append(
                {
                    "source": source_kind,
                    "uri": uri,
                    "error": str(exc),
                }
            )

    # Store artifact metadata in Postgres (best-effort). Full descriptors
    # (content_hash + source_ref S3 key + kind) are required by the chained
    # KNOWLEDGE_UNDERSTAND job — bare artifact_ids are skipped at parse time.
    _store_artifacts_in_db_sync(project_id, success_descs)

    KnowledgeEvent(
        event_type="ingest" if not result.failed_items else "fail",
        job_id=job.job_id,
        attempt_id=attempt.attempt_id,
        details=f"{len(success_ids)} artifacts | {len(result.failed_items)} failed",
    ).emit()

    _write_result(attempt, result)

    result_str = (
        f"knowledge_ingest|project={project_id}|"
        f"artifacts={len(success_ids)}|failed={len(result.failed_items)}"
    )
    return result_str, success_descs


def _ingest_one_source(
    job: Job,
    attempt: JobAttempt,
    s3_client: Any,
    bucket: str,
    project_id: str,
    source_kind: str,
    uri: str,
    allow_domains: Optional[list[str]],
    result: KnowledgeIngestResult,
    success_ids: list[str],
    success_descs: list[dict[str, Any]],
) -> None:
    """Resolve + store one non-inline ingest source (per-source extract)."""
    from umrs.knowledge.sources import SourceResolver
    from umrs.knowledge.artifact_storage import ArtifactStorage

    resolver = SourceResolver()
    resolved = resolver.resolve(source_kind, uri, allow_domains=allow_domains)
    if not resolved.resolved:
        result.failed_items.append(
            {"source": source_kind, "uri": uri, "error": "resolution failed"}
        )
        return

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
            success_descs,
        )
    elif resolved.kind == "local_dir":
        for rf in resolved.files:
            aid = f"{artifact_id}-{hashlib.md5(rf.absolute_path.encode()).hexdigest()[:8]}"
            _put_file_sync(
                job.job_id,
                attempt.attempt_id,
                storage,
                project_id,
                rf.absolute_path,
                aid,
                result,
                success_ids,
                success_descs,
            )
    elif resolved.kind == "s3_uri":
        # S3 reference — register without copy
        success_ids.append(artifact_id)
        success_descs.append(
            {
                "artifact_id": artifact_id,
                "content_hash": "",
                "kind": resolved.kind,
                "source_ref": uri,
            }
        )
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
            success_descs,
        )
    else:
        result.failed_items.append(
            {"source": source_kind, "error": f"unsupported kind: {source_kind}"}
        )


_INLINE_TEXT_MAX_BYTES = 2 * 1024 * 1024


def _put_inline_text_sync(
    job_id: str,
    attempt_id: str,
    storage: ArtifactStorage,
    project_id: str,
    src_spec: dict[str, Any],
    artifact_id: str,
    result: KnowledgeIngestResult,
    success_ids: list[str],
    success_descs: list[dict[str, Any]],
) -> None:
    """Store client-supplied inline text as an artifact (no file I/O)."""
    uri = str(src_spec.get("uri") or "inline.txt")
    text = src_spec.get("content")
    if not isinstance(text, str) or not text.strip():
        result.failed_items.append({"source": "inline_text", "uri": uri, "error": "no content"})
        return
    content = text.encode("utf-8")
    if len(content) > _INLINE_TEXT_MAX_BYTES:
        result.failed_items.append(
            {"source": "inline_text", "uri": uri, "error": "content exceeds 2 MiB"}
        )
        return
    content_hash, size_bytes, _ = storage.put_artifact(project_id, content)
    success_ids.append(artifact_id)
    success_descs.append(
        {
            "artifact_id": artifact_id,
            "content_hash": content_hash,
            "kind": _artifact_kind_for_path(uri),
            "source_ref": storage.key(project_id, content_hash),
        }
    )
    result.artifact_ids.append(artifact_id)
    _bridge_a2l_ref(artifact_id, content_hash, project_id)
    KnowledgeEvent(
        event_type="create",
        job_id=job_id,
        attempt_id=attempt_id,
        artifact_id=artifact_id,
        source_ref=uri,
        details=f"inline hash={content_hash[:16]}... size={size_bytes}",
    ).emit()


def _put_file_sync(
    job_id: str,
    attempt_id: str,
    storage: ArtifactStorage,
    project_id: str,
    file_path: str,
    artifact_id: str,
    result: KnowledgeIngestResult,
    success_ids: list[str],
    success_descs: list[dict[str, Any]],
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
    success_descs.append(
        {
            "artifact_id": artifact_id,
            "content_hash": content_hash,
            "kind": _artifact_kind_for_path(file_path),
            # source_ref must be the S3 key: knowledge_understand fetches
            # bytes via ArtifactStorage.key-style lookups by source_ref.
            "source_ref": storage.key(project_id, content_hash),
        }
    )
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
    success_descs: list[dict[str, Any]],
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
    success_descs.append(
        {
            "artifact_id": artifact_id,
            "content_hash": content_hash,
            # Remote-scraped research content is overwhelmingly prose/markdown.
            "kind": "document",
            "source_ref": storage.key(project_id, content_hash),
        }
    )
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


_ARTIFACT_KIND_BY_SUFFIX = {
    ".py": "code",
    ".pyi": "code",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".hh": "cpp",
    ".h": "cpp",
    ".cu": "cuda",
    ".cuh": "cuda",
    ".json": "schema",
    ".yaml": "schema",
    ".yml": "schema",
}


def _artifact_kind_for_path(path: str) -> str:
    """Heuristic artifact kind for parser extension selection (default: document)."""
    return _ARTIFACT_KIND_BY_SUFFIX.get(Path(path).suffix.lower(), "document")


def _store_artifacts_in_db_sync(
    project_id: str,
    artifact_descs: list[dict[str, Any]],
) -> None:
    """Persist artifact metadata to Postgres (synchronous best-effort).

    Descriptors must carry content_hash + source_ref (S3 key) + kind — the
    chained KNOWLEDGE_UNDERSTAND job fetches and parses bytes via these.

    MUST be synchronous: knowledge_ir.artifact_id has an FK to
    knowledge_artifacts. A fire-and-forget insert raced the chained
    understand job in production — FK violation, swallowed as a "likely
    duplicate" IR insert, and content never became graph nodes.
    """
    if not _resolve_knowledge_dsn():
        return
    try:

        async def _do() -> None:
            from umrs.knowledge.store import KnowledgeStore

            async with _open_pool() as pool:
                if pool is None:
                    return
                store = KnowledgeStore(pool)
                for desc in artifact_descs:
                    try:
                        await store.insert_artifact(
                            artifact_id=desc["artifact_id"],
                            content_hash=desc.get("content_hash", ""),
                            kind=desc.get("kind", "document"),
                            project_id=project_id,
                            source_ref=desc.get("source_ref", ""),
                        )
                    except Exception:
                        pass

        _run_coroutine_sync(_do(), "persist_artifact_meta")
    except Exception:
        pass


# ── A2L Derive Helper ─────────────────────────────────────────────────────


def _derive_a2l_link(
    knowledge_artifact_id: str,
    a2l_artifact_id: str,
    job_id: str = "",
    attempt_id: str = "",
    edge_kind: str = "derived",
) -> None:
    """Best-effort: derive A2L link for a knowledge artifact.

    Logs a KnowledgeEvent warning on failure; never raises.
    """
    try:
        from umrs.knowledge.store import get_store

        store = get_store()
        pool = store._pool
        if pool:

            async def _persist() -> None:
                await store.insert_knowledge_a2l_link(
                    knowledge_artifact_id,
                    a2l_artifact_id,
                    job_id,
                    attempt_id,
                    edge_kind,
                )

            asyncio.run(_persist())
    except Exception:
        # Best-effort: silently degrade — A2L unavailable
        try:
            KnowledgeEvent(
                event_type="a2l_link_skipped",
                artifact_id=knowledge_artifact_id,
                job_id=job_id,
                attempt_id=attempt_id,
                details=f"a2l_artifact={a2l_artifact_id} edge={edge_kind} skipped",
            ).emit()
        except Exception:
            pass


# ── Internal: Understand Runner (async) ────────────────────────────────────


async def _run_understand(
    job: Job,
    attempt: JobAttempt,
    project_id: str,
    sources: list[dict[str, Any]],
    pool: Any,
    s3_client: Any,
    bucket: str,
) -> KnowledgeUnderstandResult:
    """Core understand loop: iterate artifacts, parse, build IR, upsert graph."""
    if pool is None:
        return KnowledgeUnderstandResult(degraded=True, degraded_reason="no pool")

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
    target_graph_id = str(job.spec.get("graph_id") or "")

    for src in artifacts_to_process:
        artifact_id = src.get("artifact_id", "")
        if not artifact_id or not src.get("content_hash"):
            continue
        await _understand_one_artifact(
            job,
            attempt,
            src,
            project_id,
            target_graph_id,
            store,
            s3_client,
            bucket,
            result,
        )

    return result


async def _existing_current_ir(store: Any, artifact_id: str) -> Optional[str]:
    """Current-version IR id for the artifact, or None (probe is fail-open)."""
    try:
        return await store._pool.fetchval(  # type: ignore[union-attr]
            "SELECT ir_id FROM knowledge_ir WHERE artifact_id = $1 "
            "AND status = 'current' LIMIT 1",
            artifact_id,
        )
    except Exception:
        return None


async def _understand_one_artifact(
    job: Job,
    attempt: JobAttempt,
    src: dict[str, Any],
    project_id: str,
    target_graph_id: str,
    store: Any,
    s3_client: Any,
    bucket: str,
    result: KnowledgeUnderstandResult,
) -> None:
    """Parse → IR → graph for one artifact (per-artifact extract)."""
    from umrs.knowledge.ir import build_knowledge_ir

    artifact_id = src["artifact_id"]
    kind = src.get("kind", "document")

    # Check for existing current-version IR (write-once idempotent)
    existing = await _existing_current_ir(store, artifact_id)
    if existing:
        result.already_ir += 1
        KnowledgeEvent(
            event_type="verify",
            job_id=job.job_id,
            attempt_id=attempt.attempt_id,
            artifact_id=artifact_id,
            details=f"already has IR: {existing}",
        ).emit()
        return

    # Fetch real artifact bytes — never parse the hash string
    content = _fetch_artifact_bytes(s3_client, bucket, src.get("source_ref", ""))
    if content is None:
        result.failed_items.append(
            {
                "artifact_id": artifact_id,
                "error": "artifact content unavailable (source_ref unreadable)",
            }
        )
        return

    # Parse artifact
    parsed = _parse_for_understand(artifact_id, content, kind)
    if parsed is None:
        result.failed_items.append({"artifact_id": artifact_id, "error": "parse returned None"})
        return

    # Build KnowledgeIR
    ir = build_knowledge_ir(
        project_id=project_id,
        artifact_id=artifact_id,
        artifact_type=kind,
        content=content,
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
        return

    if not inserted:
        result.already_ir += 1
        return

    result.ir_ids.append(ir.ir_id)

    # Build and upsert graph nodes/edges (into the target graph: the
    # requesting named graph when the ingest ran with graph_id in spec,
    # otherwise the project's default graph). Prose artifacts gain
    # semantic concept/entity nodes via graph-JSON import (W3) or
    # site-LLM extraction (W2).
    nodes, edges = _nodes_edges_for_artifact(
        artifact_id,
        ir.ir_id,
        kind,
        content,
        parsed,
        project_id,
        target_graph_id,
    )

    try:
        for node in nodes:
            await store.upsert_node(node)
        for edge in edges:
            await store.upsert_edge(edge)
        result.nodes_created += len(nodes)
        result.edges_created += len(edges)
    except Exception as exc:
        logger.error("Node/edge upsert failed: %s: %s", artifact_id, exc)
        result.failed_items.append({"artifact_id": artifact_id, "error": f"graph: {exc}"})
        return

    summary = _build_summary(parsed)
    KnowledgeEvent(
        event_type="ingest",
        job_id=job.job_id,
        attempt_id=attempt.attempt_id,
        artifact_id=artifact_id,
        ir_id=ir.ir_id,
        details=f"summary: {summary}",
    ).emit()

    # Best-effort: derive A2L link for UNDERSTAND step (ADR-017)
    _derive_a2l_link(
        artifact_id,
        a2l_artifact_id="",
        job_id=job.job_id,
        edge_kind="understand",
    )


def _parse_for_understand(
    artifact_id: str,
    content: bytes,
    kind: str,
) -> Optional[dict[str, Any]]:
    """Parse real artifact bytes for understanding. Returns extracted dict or None."""
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
        parsed = parse_code(fake_name, content)
        if parsed.error:
            # parse_code is fail-soft: Null-byte/undecodable sources surface
            # as ``result.error`` (Python 3.11+ raises SyntaxError for null
            # bytes; older Pythons raise ValueError through the outer path).
            # Either way the understand flow treats it as parse-rejected.
            logger.debug("Parse rejected for %s: %s", artifact_id, parsed.error)
            return None
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


def _node_id_scope(project_id: str, graph_id: str) -> str:
    """Id prefix for named graphs (default graph keeps legacy node ids)."""
    from umrs.knowledge.graph_registry import default_graph_id

    if not graph_id or graph_id == default_graph_id(project_id):
        return ""
    return f"{graph_id}-"


_PROSE_KINDS = {"document", "schema", "catalog"}
_EXCERPT_MAX = 2000


def _with_excerpt(nodes: list, kind: str, content: bytes) -> list:
    """Attach a bounded text excerpt to the file node for prose artifacts.

    Structure nodes (classes/functions/kernels) are code-oriented; document
    text has no structured representation — without an excerpt, prose phrases
    are simply not present in graph content and the K3 ILIKE query can never
    find the document. Code artifacts keep metadata-only file nodes.
    """
    if kind not in _PROSE_KINDS or not nodes:
        return nodes
    excerpt = content.decode("utf-8", errors="replace")[:_EXCERPT_MAX].strip()
    if excerpt:
        import dataclasses

        nodes = [
            dataclasses.replace(nodes[0], content={**nodes[0].content, "excerpt": excerpt}),
            *nodes[1:],
        ]
    return nodes


def _build_nodes(
    artifact_id: str,
    ir_id: str,
    parsed: dict[str, Any],
    project_id: str,
    graph_id: str = "",
) -> list:  # list[KnowledgeNode]
    """Build KnowledgeNode entries from parsed code output.

    When ``graph_id`` targets a NAMED graph (not the project default), node
    ids are prefixed with it so one artifact can live in multiple graphs
    without node_id collisions.
    """
    from umrs.knowledge.store import KnowledgeNode

    scope = _node_id_scope(project_id, graph_id)
    nodes: list = []

    file_id = f"node-file-{scope}{artifact_id}"
    nodes.append(
        KnowledgeNode(
            node_id=file_id,
            project_id=project_id,
            label=artifact_id,
            node_type="file",
            content={"artifact_id": artifact_id, "ir_id": ir_id},
            content_hash=_sha256_hex(f"file-{scope}{artifact_id}".encode()),
            source_ir=ir_id,
            graph_id=graph_id,
        )
    )

    for cls_name in parsed.get("classes", []):
        nid = f"node-class-{scope}{cls_name}-{artifact_id}"
        nodes.append(
            KnowledgeNode(
                node_id=nid,
                project_id=project_id,
                label=cls_name,
                node_type="class",
                content={"artifact_id": artifact_id, "parent": file_id},
                content_hash=_sha256_hex(f"class-{scope}{cls_name}-{artifact_id}".encode()),
                source_ir=ir_id,
                graph_id=graph_id,
            )
        )

    for fn_name in parsed.get("functions", []):
        nid = f"node-fn-{scope}{fn_name}-{artifact_id}"
        nodes.append(
            KnowledgeNode(
                node_id=nid,
                project_id=project_id,
                label=fn_name,
                node_type="function",
                content={"artifact_id": artifact_id},
                content_hash=_sha256_hex(f"fn-{scope}{fn_name}-{artifact_id}".encode()),
                source_ir=ir_id,
                graph_id=graph_id,
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
                            graph_id=node.graph_id,
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
                            graph_id=node.graph_id,
                        )
                    )
                    break
    return edges


# ── Semantic extraction (prose → concept/entity graph) ───────────────────────

# DB CHECK constraint allows only these node categories; semantic extraction
# maps every LLM/graph-JSON type onto one of them (raw label survives in
# content["semantic_type"] for UI coloring).
_ENTITY_TYPES = {
    "paper",
    "dataset",
    "software",
    "application",
    "person",
    "institution",
    "model",
    "code",
    "notebook",
    "figure",
    "result",
    "material",
}

_PROSE_EXTRACT_SYSTEM = (
    "You extract a knowledge graph from scientific/technical documents. "
    "Reply with a JSON object only, no prose outside the JSON. Schema:\n"
    '{"entities": [{"name": "short canonical name", "type": "one of: concept, '
    'method, technique, property, process, model, paper, dataset, software, '
    'application, person, material, finding"}], '
    '"relations": [{"source": "entity name", "target": "entity name", '
    '"relation": "short verb phrase"}]}\n'
    "Rules: at most 40 entities and 60 relations; names concise (≤6 words); "
    "only relations between listed entities; skip trivial/generic mentions."
)


def _semantic_db_type(semantic_type: str) -> str:
    """Semantic label → DB-allowed node_type (content keeps the raw label)."""
    return "entity" if semantic_type.strip().lower() in _ENTITY_TYPES else "concept"


def _slug_label(value: Any, *, max_len: int = 40) -> str:
    """Stable slug for node/edge id construction."""
    text = "".join(ch if ch.isalnum() else "-" for ch in str(value).lower())
    slug = "-".join(part for part in text.split("-") if part)
    return slug[:max_len] or "node"


def _first_str(*values: Any, default: str = "") -> str:
    """First non-empty trimmed string value (keeps or-chains out of CCN)."""
    for value in values:
        text = "" if value is None else str(value).strip()
        if text:
            return text
    return default


def _norm_graph_nodes(raw_nodes: list) -> list[dict[str, Any]]:
    """Raw payload nodes → canonical id/name/type dicts."""
    nodes: list[dict[str, Any]] = []
    for entry in raw_nodes:
        if not isinstance(entry, dict):
            continue
        name = _first_str(entry.get("name"), entry.get("label"), entry.get("id"))
        if not name:
            continue
        nodes.append(
            {
                "id": _first_str(entry.get("id"), default=name),
                "name": name,
                "type": _first_str(entry.get("type"), entry.get("category"), default="concept"),
            }
        )
    return nodes[:80]


def _norm_graph_edges(raw_edges: list) -> list[dict[str, Any]]:
    """Raw payload edges → canonical source/target/relation dicts."""
    edges: list[dict[str, Any]] = []
    for entry in raw_edges:
        if not isinstance(entry, dict):
            continue
        source = _first_str(entry.get("source"), entry.get("from"))
        target = _first_str(entry.get("target"), entry.get("to"))
        if not source or not target:
            continue
        edges.append(
            {
                "source": source,
                "target": target,
                "relation": _first_str(
                    entry.get("label"), entry.get("relation"), entry.get("type"), default="related"
                ),
            }
        )
    return edges[:160]


def _first_list(data: dict, *keys: str) -> list:
    """First key whose value is a list (keeps or-chains out of CCN)."""
    for key in keys:
        value = data.get(key)
        if isinstance(value, list):
            return value
    return []


def _normalize_graph_payload(data: Any) -> Optional[dict[str, Any]]:
    """Pre-built graph JSON → canonical {'nodes', 'edges'} shape, or None.

    Accepts nodes/entities + edges/relations/links arrays with the usual
    field aliases (id/name/label; source/target/from/to; type/label/relation).
    """
    if not isinstance(data, dict):
        return None
    raw_nodes = _first_list(data, "nodes", "entities")
    raw_edges = _first_list(data, "edges", "relations", "links")
    if not raw_nodes:
        return None
    nodes = _norm_graph_nodes(raw_nodes)
    if not nodes:
        return None
    return {"nodes": nodes, "edges": _norm_graph_edges(raw_edges)}


def _detect_graph_payload(text: str) -> Optional[dict[str, Any]]:
    """Artifact text that is itself a serialized knowledge graph (W3)."""
    stripped = text.strip()
    if not stripped.startswith("{"):
        return None
    try:
        return _normalize_graph_payload(json.loads(stripped))
    except (json.JSONDecodeError, ValueError):
        return None


def _llm_extract_document(text: str) -> Optional[dict[str, Any]]:
    """Site-LLM semantic extraction for prose artifacts (W2).

    Disabled with UMRS_KNOWLEDGE_LLM_EXTRACT=0; any failure degrades to the
    structural file-node view (the job still succeeds).
    """
    if os.environ.get("UMRS_KNOWLEDGE_LLM_EXTRACT", "1") in ("0", "false"):
        return None
    try:
        from umrs.knowledge.agent_executor import _default_litellm_chat

        reply = _default_litellm_chat(
            [
                {"role": "system", "content": _PROSE_EXTRACT_SYSTEM},
                {"role": "user", "content": text[:9000]},
            ],
            model=os.environ.get("UMRS_KNOWLEDGE_EXTRACT_MODEL", "gpt-oss"),
            base_url=os.environ.get("UMRS_AGENT_LITELLM_URL", "http://127.0.0.1:11434"),
            api_key=None,
            max_tokens=1800,
            timeout=60.0,
        )
    except Exception as exc:  # noqa: BLE001 — degraded structural view is fine
        logger.warning("LLM prose extraction failed: %s", exc)
        return None
    match = re.search(r"\{.*\}", reply, re.DOTALL)
    if match is None:
        return None
    try:
        raw = json.loads(match.group(0))
    except (json.JSONDecodeError, ValueError):
        return None
    if isinstance(raw, dict) and "entities" in raw and "nodes" not in raw:
        raw = {**raw, "nodes": raw.get("entities"), "edges": raw.get("relations")}
    return _normalize_graph_payload(raw)


def _semantic_nodes_edges(
    artifact_id: str,
    ir_id: str,
    project_id: str,
    graph_id: str,
    payload: dict[str, Any],
) -> tuple[list, list]:
    """Normalized graph payload → KnowledgeNode/KnowledgeEdge rows (W2/W3).

    Node ids carry the artifact suffix, so the same concept ingested from two
    documents stays distinct; 'mentions' edges link each semantic node to its
    source file node.
    """
    from umrs.knowledge.store import KnowledgeEdge, KnowledgeNode

    scope = _node_id_scope(project_id, graph_id)
    file_id = f"node-file-{scope}{artifact_id}"
    nodes: list = []
    node_id_by_name: dict[str, str] = {}
    for entry in payload.get("nodes", []):
        name = " ".join(str(entry.get("name", "")).split())[:120]
        semantic_type = str(entry.get("type", "concept")).strip().lower()
        slug = _slug_label(entry.get("id") or name)
        node_id = f"node-sem-{scope}{slug}-{artifact_id}"
        node_id_by_name[slug] = node_id
        node_id_by_name[_slug_label(name)] = node_id
        nodes.append(
            KnowledgeNode(
                node_id=node_id,
                project_id=project_id,
                label=name,
                node_type=_semantic_db_type(semantic_type),
                content={
                    "artifact_id": artifact_id,
                    "semantic_type": semantic_type,
                    "source": str(entry.get("id") or name),
                },
                content_hash=_sha256_hex(f"sem-{scope}{node_id}".encode()),
                source_ir=ir_id,
                graph_id=graph_id,
            )
        )
    edges: list = []
    for entry in payload.get("edges", []):
        src_slug = _slug_label(entry.get("source", ""))
        dst_slug = _slug_label(entry.get("target", ""))
        source_id = node_id_by_name.get(src_slug)
        target_id = node_id_by_name.get(dst_slug)
        if not source_id or not target_id or source_id == target_id:
            continue
        relation = _slug_label(entry.get("relation", "related"), max_len=40)
        edges.append(
            KnowledgeEdge(
                edge_id=f"edge-sem-{scope}{src_slug}-{relation}-{dst_slug}-{artifact_id}",
                project_id=project_id,
                source_id=source_id,
                target_id=target_id,
                relation_type=relation,
                content={"semantic": True, "artifact_id": artifact_id},
                graph_id=graph_id,
            )
        )
    for node in nodes:
        slug = node.node_id.removeprefix(f"node-sem-{scope}").rsplit(f"-{artifact_id}", 1)[0]
        edges.append(
            KnowledgeEdge(
                edge_id=f"edge-merge-{scope}{file_id.replace('node-file-', '')}-{slug}-{artifact_id}",
                project_id=project_id,
                source_id=file_id,
                target_id=node.node_id,
                relation_type="mentions",
                content={"semantic": True, "artifact_id": artifact_id},
                graph_id=graph_id,
            )
        )
    return nodes, edges


def _nodes_edges_for_artifact(
    artifact_id: str,
    ir_id: str,
    kind: str,
    content: bytes,
    parsed: dict[str, Any],
    project_id: str,
    target_graph_id: str,
) -> tuple[list, list]:
    """Structural nodes/edges plus, for prose artifacts, semantic ones.

    Order matters: graph-JSON artifacts (W3) import verbatim; ordinary prose
    without code structure falls back to site-LLM extraction (W2).
    """
    nodes = _build_nodes(artifact_id, ir_id, parsed, project_id, graph_id=target_graph_id)
    nodes = _with_excerpt(nodes, kind, content)
    edges = _build_edges(nodes)
    if kind not in _PROSE_KINDS:
        return nodes, edges
    text = content.decode("utf-8", errors="replace")
    payload = _detect_graph_payload(text)
    if payload is None and len(nodes) == 1:
        # Structural parse found nothing semantic → LLM extraction (W2).
        payload = _llm_extract_document(text)
    if payload is not None:
        sem_nodes, sem_edges = _semantic_nodes_edges(
            artifact_id, ir_id, project_id, target_graph_id, payload
        )
        nodes = [*nodes, *sem_nodes]
        edges = [*edges, *sem_edges]
    return nodes, edges


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


def _knowledge_ref_counts(ref: str, key: str) -> int:
    """Parse `key=N` out of a `type|k=v|k=v` backend_ref (0 when absent)."""
    for part in ref.split("|")[1:]:
        if part.startswith(f"{key}="):
            value = part.split("=", 1)[1]
            return int(value) if value.isdigit() else 0
    return 0


class KnowledgeIngestStatusChecker(BackendStatusChecker):
    """Status checker for synchronous knowledge-ingest jobs.

    Honesty rule (audit B7): an attempt where every source failed is a
    failure, not a success — the legacy prefix-only mapping reported
    ``failed=N`` jobs as ``succeeded`` while nothing was ingested.
    """

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
        ref = attempt.backend_ref
        if ref.startswith("knowledge_ingest|"):
            artifacts = _knowledge_ref_counts(ref, "artifacts")
            failed = _knowledge_ref_counts(ref, "failed")
            if artifacts == 0 and failed > 0:
                return AttemptCheckResult.failed(
                    f"ingest: all {failed} source(s) failed, 0 artifacts stored"
                )
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
        ref = attempt.backend_ref
        if ref.startswith("knowledge_understand|"):
            ir_ids = _knowledge_ref_counts(ref, "ir_ids")
            already = _knowledge_ref_counts(ref, "already")
            failed = _knowledge_ref_counts(ref, "fail")
            if ir_ids == 0 and already == 0 and failed > 0:
                return AttemptCheckResult.failed(
                    f"understand: all {failed} artifact(s) failed, no IR produced"
                )
            return AttemptCheckResult.succeeded(metrics={"verified_after_submit": True})
        return AttemptCheckResult.failed("Unexpected backend reference")


class KnowledgeEmbedStatusChecker(BackendStatusChecker):
    """Status checker for synchronous knowledge-embed jobs.

    ``mode=degraded`` refs (no embedder configured / pool unavailable) are
    documented no-ops: embed is an opt-in stage, so the attempt succeeds.
    Genuine failures (0 embedded with failing artifacts) fail honestly.
    """

    @property
    def job_type(self) -> JobType:
        return JobType.KNOWLEDGE_EMBED

    async def check_status(
        self,
        attempt: JobAttempt,
        job: Job,
    ) -> AttemptCheckResult:
        if not attempt.backend_ref:
            return AttemptCheckResult.failed("No backend reference")
        ref = attempt.backend_ref
        if ref.startswith("knowledge_embed|"):
            if "mode=degraded" in ref:
                return AttemptCheckResult.succeeded(
                    metrics={"verified_after_submit": True, "degraded": True}
                )
            embedded = _knowledge_ref_counts(ref, "embedded")
            failed = _knowledge_ref_counts(ref, "failed")
            if embedded == 0 and failed > 0:
                return AttemptCheckResult.failed(
                    f"embed: all {failed} artifact(s) failed, 0 vectors written"
                )
            return AttemptCheckResult.succeeded(metrics={"verified_after_submit": True})
        return AttemptCheckResult.failed("Unexpected backend reference")


# ── KnowledgeEmbedBackend ──────────────────────────────────────────────────────
#
# Audit B3 (2026-08-25): the durable ingest lane produced graph nodes but never
# embeddings — knowledge_node_embeddings existed (0075) with zero writers from
# the KNOWLEDGE_INGEST → KNOWLEDGE_UNDERSTAND chain, leaving the vector plane
# permanently blind to fresh ingestion. This backend closes the gap:
# artifact bytes → chunks → canonical chunk nodes (part_of the file node) →
# one vector row per chunk node.
#
# Offline invariant: the embedder is a pre-staged local SentenceTransformer
# directory named by UMRS_KNOWLEDGE_EMBED_MODEL. Names are never resolved via
# the network — without a pre-staged path the job completes as a documented
# degraded no-op (graph content is unaffected; embedding is opt-in per site).

_NODE_EMBED_MODEL_ENV = "UMRS_KNOWLEDGE_EMBED_MODEL"
_NODE_EMBED_CHUNK_TOKENS_ENV = "UMRS_KNOWLEDGE_EMBED_CHUNK_TOKENS"
_NODE_EMBED_MAX_CHUNKS_ENV = "UMRS_KNOWLEDGE_EMBED_MAX_CHUNKS"

_node_embedder: Any = None
_node_embedder_unavailable: bool = False


def _get_node_embedder() -> Any:
    """Resolve the site-local node embedder, or None when not pre-staged.

    Only local directories are accepted: model downloads at runtime are
    forbidden by the platform offline rule, so a bare HF id fails closed.
    """
    global _node_embedder, _node_embedder_unavailable
    if _node_embedder is not None:
        return _node_embedder
    if _node_embedder_unavailable:
        return None
    model_path = os.environ.get(_NODE_EMBED_MODEL_ENV, "").strip()
    if not model_path:
        return None
    if not os.path.isdir(model_path):
        logger.warning(
            "%s=%r is not a pre-staged model directory — node embedding "
            "disabled (runtime model downloads are forbidden)",
            _NODE_EMBED_MODEL_ENV,
            model_path,
        )
        _node_embedder_unavailable = True
        return None
    try:
        from langchain_community.embeddings import SentenceTransformerEmbeddings

        _node_embedder = SentenceTransformerEmbeddings(model_name=model_path)
    except Exception as exc:  # noqa: BLE001 — degrade, never crash the chain
        logger.warning("node embedder init failed (%s): %s", model_path, exc)
        _node_embedder_unavailable = True
        return None
    return _node_embedder


def _reset_node_embedder_for_tests() -> None:
    """Test hook: drop the cached embedder/unavailable flag between cases."""
    global _node_embedder, _node_embedder_unavailable
    _node_embedder = None
    _node_embedder_unavailable = False


async def _embed_one_artifact(
    store: Any,
    src: dict[str, Any],
    project_id: str,
    graph_id: str,
    embedder: Any,
    s3_client: Any,
    bucket: str,
    result: KnowledgeEmbedResult,
) -> None:
    """Chunk one artifact's text and write chunk nodes + vector rows."""
    from umrs.knowledge.store import KnowledgeEdge, KnowledgeNode
    from umrs.shared.utils import _chunk_text_without_nlp, add_document_prefix

    artifact_id = src.get("artifact_id", "")
    if not artifact_id:
        result.failed_items.append({"artifact_id": "", "error": "missing artifact_id"})
        return

    content = _fetch_artifact_bytes(s3_client, bucket, src.get("source_ref", ""))
    if content is None:
        result.failed_items.append(
            {"artifact_id": artifact_id, "error": "artifact content unavailable"}
        )
        return
    text = content.decode("utf-8", errors="ignore").strip()
    if not text:
        result.skipped += 1
        return

    tokenizer = getattr(getattr(embedder, "client", None), "tokenizer", None)
    max_tokens = int(os.environ.get(_NODE_EMBED_CHUNK_TOKENS_ENV, "256"))
    chunks = _chunk_text_without_nlp(text, tokenizer=tokenizer, max_tokens=max_tokens)
    if not chunks:
        result.skipped += 1
        return

    model_name = os.environ.get(_NODE_EMBED_MODEL_ENV, "").strip()
    chunk_texts = [c["text"] for c in chunks]
    vectors = embedder.embed_documents(add_document_prefix(chunk_texts, model_name))

    scope = _node_id_scope(project_id, graph_id)
    file_node_id = f"node-file-{scope}{artifact_id}"
    cap = int(os.environ.get(_NODE_EMBED_MAX_CHUNKS_ENV, "128"))
    written = 0
    for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
        if i >= cap:
            break
        chunk_id = f"node-chunk-{scope}{artifact_id}-{i}"
        chunk_md5 = hashlib.md5(chunk["text"].encode("utf-8")).hexdigest()
        await store.upsert_node(
            KnowledgeNode(
                node_id=chunk_id,
                project_id=project_id,
                label=f"chunk {i} of {artifact_id}",
                node_type="entity",
                content={
                    "kind": "chunk",
                    "artifact_id": artifact_id,
                    "chunk_index": i,
                    "chunk_start": chunk.get("start"),
                    "chunk_end": chunk.get("end"),
                    "chunk_md5": chunk_md5,
                    "excerpt": chunk["text"][:500],
                    "origin": "knowledge_embed",
                },
                content_hash=chunk_md5,
                graph_id=graph_id,
            )
        )
        await store.upsert_edge(
            KnowledgeEdge(
                edge_id=f"edge-{chunk_id}-part-of-{file_node_id}",
                project_id=project_id,
                source_id=chunk_id,
                target_id=file_node_id,
                relation_type="part_of",
                content={"kind": "part_of", "origin": "knowledge_embed"},
                graph_id=graph_id,
            )
        )
        await store.upsert_node_embedding(
            chunk_id,
            model_name,
            vector,
            content_hash=chunk_md5,
        )
        written += 1
    result.nodes_embedded += written


class KnowledgeEmbedBackend(OrchestratorBackend):
    """Chunk artifact text → canonical chunk nodes + node vectors (B3)."""

    @property
    def backend_name(self) -> str:
        return "knowledge_embed"

    async def submit_job(
        self,
        job: Job,
        attempt: JobAttempt,
    ) -> str:
        project_id = job.spec.get("project_id", job.project_id)
        _sse_token = None
        try:  # SSE project attribution for every attempt event
            from umrs.knowledge import events_sse

            _sse_token = events_sse.set_project_ctx(project_id)
        except Exception:  # noqa: BLE001 — SSE context is best-effort
            pass
        try:
            return await self._submit_embed(job, attempt, project_id)
        finally:
            if _sse_token is not None:
                from umrs.knowledge import events_sse

                events_sse.reset_project_ctx(_sse_token)

    async def _submit_embed(
        self,
        job: Job,
        attempt: JobAttempt,
        project_id: str,
    ) -> str:
        sources: list[dict[str, Any]] = job.spec.get("sources", [])
        graph_id = str(job.spec.get("graph_id") or "")

        result = KnowledgeEmbedResult()
        embedder = _get_node_embedder()
        if embedder is None:
            reason = f"no embedder (set {_NODE_EMBED_MODEL_ENV} to a pre-staged model dir)"
            logger.info("knowledge_embed degraded (no-op): %s", reason)
            result.degraded = True
            result.degraded_reason = reason
            _write_result(attempt, result)
            return f"knowledge_embed|mode=degraded|reason=no_embedder|project={project_id}"

        s3_client, s3_bucket = _build_s3_client(job.spec)

        from umrs.knowledge.store import KnowledgeStore

        async with _open_pool() as pool:
            if pool is None:
                logger.warning("knowledge_embed degraded: Postgres pool unavailable")
                result.degraded = True
                result.degraded_reason = "Postgres pool unavailable"
                _write_result(attempt, result)
                return f"knowledge_embed|mode=degraded|reason=no_pool|project={project_id}"

            store = KnowledgeStore(pool)
            for src in sources:
                try:
                    await _embed_one_artifact(
                        store,
                        src,
                        project_id,
                        graph_id,
                        embedder,
                        s3_client,
                        s3_bucket,
                        result,
                    )
                except Exception as exc:  # noqa: BLE001 — per-artifact isolation
                    logger.error("embed failed for %s: %s", src.get("artifact_id"), exc)
                    result.failed_items.append(
                        {"artifact_id": src.get("artifact_id", ""), "error": str(exc)}
                    )

        _write_result(attempt, result)
        return (
            f"knowledge_embed|project={project_id}|"
            f"embedded={result.nodes_embedded}|skipped={result.skipped}|"
            f"failed={len(result.failed_items)}"
        )


def _enqueue_embed_job(
    project_id: str,
    user_id: str,
    job: Any,
    sources: list[dict[str, Any]],
) -> str | None:
    """Enqueue the KNOWLEDGE_EMBED job chained to a completed understand.

    Mirrors _enqueue_understand_job: the descriptors travelled with the
    parent spec, so the embed stage re-processes exactly the artifacts the
    understand stage touched — never a whole-project re-list.
    """
    from .models import JobType

    if not sources:
        return None

    from .service import JobSubmission, get_job_service

    embed_job_id = get_job_service().submit_job(
        JobSubmission(
            project_id=project_id,
            user_id=user_id,
            job_type=JobType.KNOWLEDGE_EMBED,
            spec={
                "sources": sources,
                "project_id": project_id,
                "parent_job_id": job.job_id,
                "graph_id": job.spec.get("graph_id", ""),
            },
            priority=job.priority,
            max_attempts=3,
        ),
    )
    logger.info(
        "Enqueued KNOWLEDGE_EMBED: job=%s artifacts=%d",
        embed_job_id,
        len(sources),
    )
    return embed_job_id


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
                findings.extend(_run_verifier(verifier, store, project_id, job, attempt))

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


class _ExportIntegrityMixin:
    """Shared export+integrity flow for the export backend's submit path."""

    async def _export_with_integrity(
        self,
        pool: Any,
        project_id: str,
        degraded_reason: str,
        store_cls: Any,
        engine_cls: Any,
        verify_fn: Any,
        allow_unsigned: bool = False,
    ) -> KnowledgeExportResult:
        """Run the export (or degraded fallback) and post-verify the bundle."""
        if pool is None:
            return KnowledgeExportResult(
                project_id=project_id,
                degraded=True,
                degraded_reason=degraded_reason,
            )
        secret_key = os.getenv("UMRS_SECRET_KEY", "")
        engine = engine_cls(
            store=store_cls(pool),
            output_dir=Path(
                os.getenv(
                    "UMRS_KB_EXPORT_DIR",
                    str(Path(os.getcwd()) / ".kb_exports"),
                )
            ),
            secret_key=secret_key,
            allow_unsigned=allow_unsigned,
        )
        result = await engine.export_project(project_id)
        if secret_key and not result.degraded and result.bundle_path:
            ok = verify_fn(Path(result.bundle_path), secret_key)
            if not ok:
                logger.error(
                    "Bundle integrity check failed for %s", result.bundle_path
                )
                result.degraded = True
                result.degraded_reason = "bundle integrity check failed"
        return result


class KnowledgeExportBackend(OrchestratorBackend, _ExportIntegrityMixin):
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
        allow_unsigned = bool(job.spec.get("allow_unsigned", False))

        KnowledgeEvent(
            event_type="create",
            job_id=job.job_id,
            attempt_id=attempt.attempt_id,
            details=f"project={project_id} export",
        ).emit()

        pool = _get_pool()
        degraded_reason = ""

        if pool is None:
            degraded_reason = "Postgres pool unavailable"
            logger.warning("KnowledgeExport degraded: %s", degraded_reason)

        try:
            result = await self._export_with_integrity(
                pool,
                project_id,
                degraded_reason,
                KnowledgeStore,
                KnowledgeExportEngine,
                verify_bundle_integrity,
                allow_unsigned=allow_unsigned,
            )
            _write_result(attempt, result)

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


def _parse_metrics_jsonb(value: Any) -> dict[str, Any]:
    """Normalize a JSONB ``metrics`` value to a dict.

    Ad-hoc asyncpg pools (no registered type codecs) return JSON/JSONB
    columns as ``str``; app-layer pools return ``dict``.  Malformed values
    return ``{}`` — callers treat them as empty, never as crash-worthy.
    """
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            logger.warning("Malformed metrics JSON from job_attempts: %.80s", value)
            return {}
    return dict(value) if isinstance(value, dict) else {}


async def _artifacts_from_attempt_metrics(
    pool: Any, attempts: list[Any]
) -> list[dict[str, Any]]:
    """Extract artifact dicts recorded in attempt metrics."""
    results: list[dict[str, Any]] = []
    for att_id in attempts:
        try:
            row = await pool.fetchval(
                "SELECT metrics FROM job_attempts WHERE attempt_id = $1",
                att_id,
            )
            metrics = _parse_metrics_jsonb(row)
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
    if not _resolve_knowledge_dsn():
        return artifacts

    async def _collect() -> list[dict[str, Any]]:
        async with _open_pool() as pool:
            if pool is None:
                return []
            attempts = await _resolve_succeeded_attempts(pool, job)
            results = await _artifacts_from_attempt_metrics(pool, attempts)
            results.extend(await _artifacts_from_store(pool, job.project_id))
            return results

    try:
        collected = _run_coroutine_sync(_collect(), "collect-artifacts")
        if collected:
            artifacts = collected
    except Exception as exc:
        logger.warning(
            "Artifact collection failed for job %s: %s",
            getattr(job, "job_id", "?"),
            exc,
        )

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
    bucket = _resolve_bucket()

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
    if not _resolve_knowledge_dsn():
        return
    try:
        from umrs.knowledge.store import KnowledgeStore

        async def _store_provenance():
            async with _open_pool() as pool:
                if pool is None:
                    return
                store = KnowledgeStore(pool)
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
            _fire_coroutine(_store_provenance(), "artifact_provenance")
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
    if not _resolve_knowledge_dsn():
        return []
    try:
        from umrs.knowledge.store import KnowledgeStore

        async def _list_new() -> list[dict[str, Any]]:
            async with _open_pool() as pool:
                if pool is None:
                    return []
                store = KnowledgeStore(pool)
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
            return _run_coroutine_sync(_list_new(), "list-artifact-sources") or []
        except Exception as exc:
            logger.warning(
                "Artifact source listing failed for project %s: %s",
                project_id,
                exc,
            )
            return []
    except Exception:
        return []


def _enqueue_understand_job(
    project_id: str,
    user_id: str,
    job: Any,
    sources: list[dict[str, Any]] | None = None,
) -> str | None:
    """Enqueue the KNOWLEDGE_UNDERSTAND job. Returns job ID or None.

    ``sources`` carries the already-collected deduplicated descriptors (the
    preferential source: concurrent provenance persistence must not race a
    just-written table). Falls back to the project listing for callers that
    did not collect descriptors first.
    """
    from .models import JobType

    if sources is None:
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
                # Named-graph ingestion (0069): the parent ingest's target graph
                # (absent ⇒ default project graph)
                "graph_id": job.spec.get("graph_id", ""),
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
    from umrs.knowledge.store import KnowledgeNode, KnowledgeStore

    if not (deduped and _resolve_knowledge_dsn()):
        return
    try:
        # Named-graph ingestion (0069): provenance goes to the target graph.
        graph_id = str(job.spec.get("graph_id") or "")

        async def _create_edges():
            async with _open_pool() as pool:
                if pool is None:
                    logger.warning(
                        "produced_by edges skipped: knowledge pool unavailable "
                        "job=%s artifacts=%d",
                        job.job_id,
                        len(deduped),
                    )
                    return
                store = KnowledgeStore(pool)
                prod_entity_id = f"prod-job-{job.job_id}"
                job_type = getattr(job, "job_type", "unknown")
                job_type_value = (
                    job_type.value if hasattr(job_type, "value") else job_type
                )
                # Create provenance node. The semantic role stays in the
                # label; node_type must be one of ALLOWED_NODE_TYPES
                # ("entity" covers job/provenance roles) — anything else
                # violates knowledge_nodes_node_type_check.
                await store.upsert_node(
                    KnowledgeNode(
                        node_id=prod_entity_id,
                        project_id=project_id,
                        label=f"job:{job.job_id}",
                        node_type="entity",
                        graph_id=graph_id,
                        content={
                            "job_id": job.job_id,
                            "job_type": job_type_value,
                            "workflow": workflow,
                            "step": step,
                            "user_id": user_id,
                        },
                        content_hash=_sha256_hex(
                            f"provenance-{job.job_id}-{workflow}-{step}".encode()
                        ),
                        # NULL, not "": source_ir FKs into knowledge_ir and
                        # an empty string matches no ir_id
                        # (knowledge_nodes_source_ir_fkey).
                        source_ir=None,
                    )
                )

                # Artifact nodes: edge endpoints must resolve in
                # knowledge_nodes (FK knowledge_edges_source_id_fkey —
                # artifacts live in knowledge_artifacts, a different table).
                await _upsert_artifact_nodes(
                    store, deduped, project_id, job, attempt, workflow, step,
                    graph_id=graph_id,
                )

                # Create PRODUCED_BY edges
                for art in deduped:
                    aid = art.get("artifact_id", "")
                    if not aid:
                        continue
                    try:
                        await _upsert_produced_edge(
                            store,
                            aid,
                            prod_entity_id,
                            project_id,
                            job,
                            attempt,
                            workflow,
                            step,
                            graph_id=graph_id,
                        )
                    except Exception as edge_exc:
                        # Authorization constraint: never silently swallow a
                        # graph write failure — context needed for audit.
                        logger.warning(
                            "produced_by edge write failed "
                            "source_node=%s (artifact/entity) "
                            "target_node=%s (job/entity) "
                            "relation=produced_by error=%r",
                            aid,
                            prod_entity_id,
                            edge_exc,
                            exc_info=True,
                        )

        try:
            _fire_coroutine(_create_edges(), "produced_by_edges")
        except Exception:
            logger.warning(
                "produced_by edge scheduling failed job=%s workflow=%s step=%s",
                job.job_id,
                workflow,
                step,
                exc_info=True,
            )
    except Exception:
        logger.warning("_create_produced_by_edges aborted", exc_info=True)


async def _upsert_artifact_nodes(
    store: Any,
    deduped: list[dict[str, Any]],
    project_id: str,
    job: Any,
    attempt: Any,
    workflow: str,
    step: str,
    graph_id: str = "",
) -> None:
    """Upsert artifact entity nodes so produced_by edge endpoints resolve.

    ``knowledge_edges.source_id``/``target_id`` are FKs into
    ``knowledge_nodes``; artifact ids alone (knowledge_artifacts rows) do NOT
    satisfy that invariant. Each deduplicated artifact gets an ``entity``
    node carrying its provenance linkage.
    """
    from umrs.knowledge.store import KnowledgeNode

    for art in deduped:
        aid = art.get("artifact_id", "")
        if not aid:
            continue
        await store.upsert_node(
            KnowledgeNode(
                node_id=aid,
                project_id=project_id,
                label=f"artifact:{aid}",
                node_type="entity",
                content={
                    "content_hash": art.get("content_hash", ""),
                    "kind": art.get("kind", ""),
                    "source_ref": art.get("source_ref", ""),
                    "produced_by_job_id": job.job_id,
                    "produced_by_attempt_id": attempt.attempt_id
                    if hasattr(attempt, "attempt_id")
                    else "",
                    "workflow": workflow,
                    "step": step,
                },
                content_hash=_sha256_hex(f"artifact-node-{aid}".encode()),
                # NULL, not "" (knowledge_nodes_source_ir_fkey).
                source_ir=None,
                graph_id=graph_id,
            )
        )


async def _upsert_produced_edge(
    store: Any,
    aid: str,
    prod_entity_id: str,
    project_id: str,
    job: Any,
    attempt: Any,
    workflow: str,
    step: str,
    graph_id: str = "",
) -> None:
    """Upsert one artifact → job produced_by edge (best-effort)."""
    from umrs.knowledge.store import KnowledgeEdge

    await store.upsert_edge(
        KnowledgeEdge(
            edge_id=f"edge-{aid}-produced_by-{job.job_id}",
            project_id=project_id,
            source_id=aid,
            target_id=prod_entity_id,
            relation_type="produced_by",
            graph_id=graph_id,
            content={
                "workflow": workflow,
                "step": step,
                "attempt_id": attempt.attempt_id
                if hasattr(attempt, "attempt_id")
                else "",
            },
        )
    )


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

    return _enqueue_ingest_chain(deduped, bucket, project_id, user_id, job, attempt, workflow, step)


def _enqueue_ingest_chain(
    deduped: list[dict[str, Any]],
    bucket: str,
    project_id: str,
    user_id: str,
    job: Any,
    attempt: Any,
    workflow: str,
    step: str,
) -> list[str]:
    """Enqueue the ingest→understand chain + provenance edges (fail-soft)."""
    enqueued: list[str] = []

    # 3. Enqueue KNOWLEDGE_INGEST job (with provenance metadata)
    try:
        ingest_job_id = _enqueue_ingest_job(
            deduped, bucket, project_id, user_id, job, attempt, workflow, step
        )
        if ingest_job_id:
            enqueued.append(ingest_job_id)
    except Exception as exc:
        logger.warning("Failed to enqueue KNOWLEDGE_INGEST: %s", exc)

    # 4. Enqueue KNOWLEDGE_UNDERSTAND job (always after ingest). The
    # deduplicated descriptors are authoritative here: provenance persistence
    # is concurrent and must not gate UNDERSTAND construction on a just-written
    # table being visible (fire-and-forget ordering is not a contract).
    try:
        us_job_id = _enqueue_understand_job(project_id, user_id, job, deduped)
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
        if not _resolve_knowledge_dsn():
            return

        from umrs.knowledge.store import KnowledgeEdge, KnowledgeStore

        async def _log_failure() -> None:
            try:
                fail_entity_id = f"fail-{job_id}-{attempt_no}"
                async with _open_pool() as pool:
                    if pool is None:
                        return
                    store = KnowledgeStore(pool)
                    # The caused_by edge references the produced-by job node;
                    # upsert it first so the FK resolves (same invariant as
                    # produced_by edges).
                    await store.upsert_node(
                        KnowledgeNode(
                            node_id=f"prod-job-{job_id}",
                            project_id=project_id,
                            label=f"job:{job_id}",
                            node_type="entity",
                            content={"job_id": job_id, "user_id": user_id},
                            content_hash=_sha256_hex(f"provenance-{job_id}".encode()),
                            source_ir=None,
                        )
                    )
                    await store.upsert_node(
                        KnowledgeNode(
                            node_id=fail_entity_id,
                            project_id=project_id,
                            label=f"job:{job_id} FAILED",
                            node_type="entity",
                            content={
                                "job_id": job_id,
                                "attempt_no": attempt_no,
                                "exit_reason": exit_reason,
                                "user_id": user_id,
                            },
                            content_hash=_sha256_hex(
                                f"failure-{job_id}-{exit_reason}".encode()
                            ),
                            source_ir=None,
                        )
                    )
                    # Link the failed job to this failure node
                    try:
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
                    except Exception as edge_exc:
                        logger.warning(
                            "caused_by edge write failed "
                            "source_node=%s (job/entity) "
                            "target_node=%s (failure/entity) "
                            "relation=caused_by error=%r",
                            f"prod-job-{job_id}",
                            fail_entity_id,
                            edge_exc,
                            exc_info=True,
                        )
            except Exception:
                logger.warning(
                    "ingest failure graph write aborted job=%s attempt_no=%s",
                    job_id,
                    attempt_no,
                    exc_info=True,
                )

        try:
            _fire_coroutine(_log_failure(), "ingest_failure_log")
        except Exception:
            logger.warning(
                "ingest failure log scheduling failed job=%s", job_id, exc_info=True
            )
    except Exception:
        logger.warning("enqueue_ingest_failure aborted", exc_info=True)


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
                type(ref)(
                    artifact_id=artifact_id, a2l_hash=content_hash, a2l_id=ref.a2l_id
                )
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
