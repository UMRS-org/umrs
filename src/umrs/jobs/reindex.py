"""Knowledge re-index backend (stability S10 follow-up: dispatcher wire-up).

The durable pattern registered with ``JobType.KNOWLEDGE_REINDEX``:

1. **resolve** — the spec's ``target_model`` must be this site's staged
   knowledge embedder (``UMRS_KNOWLEDGE_EMBED_MODEL``, matched by basename);
   a site cannot re-embed with a model it does not run — refused by name;
2. **sweep** — stale/uncurrent active artifacts under that model
   (`umrs.knowledge.embedding_metadata.select_stale_artifacts`);
3. **re-embed** — one KNOWLEDGE_EMBED job per project over the swept
   artifacts (the embedding-job sandbox lane when the site opted in); the
   embed job stamps each artifact current only after its vectors are
   written. This job never stamps: until 2026-09-24 it marked swept rows
   current WITHOUT recomputing a single vector;
4. **maintain** — the canonical INDEX_BUILD seam
   (`umrs.shared.pgvector_index.maintain_pgvector_index`).

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
    validate_reindex_payload,
)

from .monitor import AttemptCheckResult, BackendStatusChecker
from .scheduler import OrchestratorBackend

logger = logging.getLogger(__name__)

# Same env name as the KNOWLEDGE_EMBED backend (knowledge_adapters).
_SITE_EMBED_MODEL_ENV = "UMRS_KNOWLEDGE_EMBED_MODEL"


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
        spec["target_model"] = resolve_site_embed_model(spec["target_model"])

        stale = _sweep(spec)
        if not stale:
            logger.info("reindex no-op: no stale artifacts for job=%s", job.job_id)
            return _backend_ref(spec, swept=0, enqueued=0)

        enqueued = _enqueue_reembed(job, _stale_sources(stale))

        from umrs.shared.pgvector_index import (
            maintain_pgvector_index,
        )

        metrics = maintain_pgvector_index(
            spec["database"],
            spec["table"],
            rebuild=True,
            index_name=spec.get("index_name") or None,
        )
        logger.info(
            "reindex: job=%s attempt=%s swept=%d re-embed jobs=%d action=%s",
            job.job_id,
            attempt.attempt_id,
            len(stale),
            len(enqueued),
            metrics.get("action", "rebuild"),
        )
        return _backend_ref(spec, swept=len(stale), enqueued=len(enqueued))


def _model_base(model: str) -> str:
    return str(model).rstrip("/").rsplit("/", 1)[-1]


def resolve_site_embed_model(target_model: str) -> str:
    """The site embedder's model name when ``target_model`` names it (by
    basename — sites stage models under their own paths); refused by name
    otherwise, since re-embedding needs the model this site actually runs."""
    from umrs.knowledge.text_embedder import configured_text_embedder

    site = configured_text_embedder()
    if _model_base(site) != _model_base(target_model):
        raise EmbeddingMetadataError(
            f"KNOWLEDGE_REINDEX target_model {target_model!r} is not this site's "
            f"embedder ({_SITE_EMBED_MODEL_ENV}={site!r}); re-embedding needs the "
            "model the site stages"
        )
    return site


def _stale_sources(stale: list[str]) -> dict[str, list[dict[str, str]]]:
    """Swept artifact ids → KNOWLEDGE_EMBED source descriptors per project."""
    with _artifact_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT artifact_id, project_id, source_ref FROM knowledge_artifacts "
            "WHERE artifact_id = ANY(%s)",
            (list(stale),),
        )
        rows = cur.fetchall()
    by_project: dict[str, list[dict[str, str]]] = {}
    for artifact_id, project_id, source_ref in rows:
        by_project.setdefault(str(project_id), []).append(
            {"artifact_id": str(artifact_id), "source_ref": str(source_ref or "")}
        )
    return by_project


def _enqueue_reembed(
    job: Job, by_project: dict[str, list[dict[str, str]]]
) -> list[str]:
    """One KNOWLEDGE_EMBED job per project (its default graph)."""
    from umrs.knowledge.graph_registry import default_graph_id

    from .service import JobSubmission, get_job_service

    service = get_job_service()
    return [
        service.submit_job(
            JobSubmission(
                project_id=project_id,
                user_id=job.user_id,
                job_type=JobType.KNOWLEDGE_EMBED,
                spec={
                    "sources": sources,
                    "project_id": project_id,
                    "parent_job_id": job.job_id,
                    "graph_id": default_graph_id(project_id),
                    "reason": "knowledge_reindex",
                },
                priority=job.priority,
                max_attempts=3,
            )
        )
        for project_id, sources in sorted(by_project.items())
    ]


def _sweep(spec: dict) -> list[str]:
    with _artifact_conn() as conn:
        return select_stale_artifacts(
            conn, spec["target_model"], limit=int(spec.get("batch_limit", 500))
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


def _backend_ref(spec: dict, *, swept: int, enqueued: int) -> str:
    index = spec.get("index_name") or "default"
    return (
        f"knowledge_reindex|database={spec['database']}|table={spec['table']}|"
        f"index={index}|model={spec['target_model']}|swept={swept}|reembed_jobs={enqueued}"
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
            metrics={
                "swept": parsed.get("swept"),
                "reembed_jobs": parsed.get("reembed_jobs"),
            }
        )
