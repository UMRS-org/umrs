"""Knowledge re-index backend (stability S10 follow-up: dispatcher wire-up).

The durable pattern registered with ``JobType.KNOWLEDGE_REINDEX``:

1. **sweep** — stale/uncurrent active artifacts under the spec's target
   model (`umrs.knowledge.embedding_metadata.select_stale_artifacts`);
2. **compute** — the canonical INDEX_BUILD seam
   (`umrs.server.sql.maintain_pgvector_index`), never re-implemented
   embeddings (EMBEDDING_BATCH dispatch is unimplemented by design, #216);
3. **stamp** — swept rows marked current (`stamp_embeddings_current`).

Execution is synchronous inside ``submit_job`` — mirroring
``IndexBuildBackend``: once the call returns, the work is done, and the
status checker only validates the returned ``backend_ref``.

Payload contract (validated at submit time):

    {"database": str, "table": str, "target_model": str,
     "index_name": str | None, "embedding_dim": int > 0 | None,
     "batch_limit": int > 0 (default 500)}
"""

from __future__ import annotations

import logging

from umrs.jobs.models import Job, JobAttempt, JobType
from umrs.knowledge.embedding_metadata import (
    EmbeddingMetadataError,
    select_stale_artifacts,
    stamp_embeddings_current,
    validate_reindex_payload,
)

from .monitor import AttemptCheckResult, BackendStatusChecker
from .scheduler import OrchestratorBackend

logger = logging.getLogger(__name__)


def validate_knowledge_reindex_spec(spec: dict) -> dict:
    """Submit-time contract enforcement for KNOWLEDGE_REINDEX jobs."""
    base = validate_reindex_payload(spec)
    database = str(spec.get("database") or "").strip() if isinstance(spec, dict) else ""
    table = str(spec.get("table") or "").strip() if isinstance(spec, dict) else ""
    _require_nonempty(database, "knowledge_reindex requires spec.database")
    _require_nonempty(table, "knowledge_reindex requires spec.table")
    dim = spec.get("embedding_dim")
    limit = spec.get("batch_limit", 500)
    _require_positive_int(dim, "embedding_dim", allow_none=True)
    _require_positive_int(limit, "batch_limit")
    return {
        **base,
        "database": database,
        "table": table,
        "embedding_dim": dim,
        "batch_limit": limit,
    }


def _require_nonempty(value: str, message: str) -> None:
    if not value:
        raise EmbeddingMetadataError(message)


def _require_positive_int(value, name: str, *, allow_none: bool = False) -> None:
    if allow_none and value is None:
        return
    if not isinstance(value, int) or value <= 0:
        raise EmbeddingMetadataError(f"{name} must be a positive int")


class KnowledgeReindexBackend(OrchestratorBackend):
    """Sweep → maintain → stamp, synchronously per attempt."""

    @property
    def backend_name(self) -> str:
        return "knowledge_reindex"

    async def submit_job(self, job: Job, attempt: JobAttempt) -> str:
        spec = validate_knowledge_reindex_spec(job.spec or {})

        stale = _sweep(spec)
        if not stale:
            logger.info("reindex no-op: no stale artifacts for job=%s", job.job_id)
            return _backend_ref(spec, swept=0, stamped=0)

        from umrs.server.sql import maintain_pgvector_index  # noqa: boundary — lazy server.sql composition (sweep)

        metrics = maintain_pgvector_index(
            spec["database"],
            spec["table"],
            rebuild=True,
            index_name=spec.get("index_name") or None,
        )
        stamped = _stamp(spec, stale)
        logger.info(
            "reindex done: job=%s attempt=%s swept=%d stamped=%d action=%s",
            job.job_id,
            attempt.attempt_id,
            len(stale),
            stamped,
            metrics.get("action", "rebuild"),
        )
        return _backend_ref(spec, swept=len(stale), stamped=stamped)


def _sweep(spec: dict) -> list[str]:
    with _artifact_conn() as conn:
        return select_stale_artifacts(
            conn, spec["target_model"], limit=int(spec.get("batch_limit", 500))
        )


def _stamp(spec: dict, stale: list[str]) -> int:
    with _artifact_conn() as conn:
        return stamp_embeddings_current(
            conn,
            stale,
            model=spec["target_model"],
            dim=spec.get("embedding_dim"),
        )


def _artifact_conn():
    """One-shot connection for the metadata plane (admin connection path,
    same as the durable jobs runtime)."""
    from contextlib import contextmanager

    from umrs.shared.pg import build_durable_job_db_pool_from_env

    @contextmanager
    def _managed():
        pool = build_durable_job_db_pool_from_env()
        conn = pool.getconn()
        try:
            with conn:
                yield conn
        finally:
            pool.putconn(conn)
            pool.closeall()

    return _managed()


def _backend_ref(spec: dict, *, swept: int, stamped: int) -> str:
    index = spec.get("index_name") or "default"
    return (
        f"knowledge_reindex|database={spec['database']}|table={spec['table']}|"
        f"index={index}|model={spec['target_model']}|swept={swept}|stamped={stamped}"
    )


def _parse_ref(ref: str) -> dict[str, str]:
    """Local pipe-ref parser (adapters._parse_backend_ref twin without the
    import cycle)."""
    out: dict[str, str] = {}
    for part in ref.split("|"):
        if "=" in part:
            key, _, value = part.partition("=")
            out[key] = value
    return out


class KnowledgeReindexStatusChecker(BackendStatusChecker):
    """Status checker for synchronous knowledge-reindex jobs."""

    @property
    def job_type(self) -> JobType:
        return JobType.KNOWLEDGE_REINDEX

    async def check_status(
        self,
        attempt: JobAttempt,
        job: Job,
    ) -> AttemptCheckResult:
        backend_ref = attempt.backend_ref
        if not backend_ref:
            return AttemptCheckResult.failed("No backend reference")
        if not backend_ref.startswith("knowledge_reindex|"):
            return AttemptCheckResult.failed("Unexpected backend reference")
        parsed = _parse_ref(backend_ref)
        if "database" not in parsed or "table" not in parsed:
            return AttemptCheckResult.failed("Missing reindex metadata")
        return AttemptCheckResult.succeeded(
            metrics={"swept": parsed.get("swept"), "stamped": parsed.get("stamped")}
        )
