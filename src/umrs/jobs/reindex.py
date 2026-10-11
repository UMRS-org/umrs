"""Knowledge re-index backend (stability S10 follow-up: dispatcher wire-up).

The durable pattern registered with ``JobType.KNOWLEDGE_REINDEX``:

1. **resolve** — the spec's ``target_model`` must be this site's staged
   knowledge embedder (``UMRS_KNOWLEDGE_EMBED_MODEL``, matched by basename);
   a site cannot re-embed with a model it does not run — refused by name;
2. **sweep** — stale/uncurrent active artifacts under that model
   (`umrs.knowledge.embedding_metadata.select_stale_artifacts`);
3. **re-embed** — one KNOWLEDGE_EMBED job per graph of the job's project
   over the swept artifacts (the embedding-job sandbox lane when the site
   opted in); the embed job stamps each artifact current only after its
   vectors are written. This job never stamps: until 2026-09-24 it marked
   swept rows current WITHOUT recomputing a single vector;
4. **maintain** — the canonical INDEX_BUILD seam
   (`umrs.shared.pgvector_index.maintain_pgvector_index`), except on
   ``knowledge_node_embeddings``: its 0075 ``vector`` column has no
   dimension, so pgvector cannot build an ANN index on it (the HNSW gate,
   EMBEDDING-POLICY) and the call failed every reindex after it had already
   enqueued the re-embed — a retry enqueued them again (ADR-093 D-11).

Same-model repair (ADR-090 D-4, ADR-093 D-11, 2026-10-10): the sweep also
selects artifacts cut by another chunk scheme than the site's
(``UMRS_KNOWLEDGE_CHUNK_SCHEME``), and each artifact is re-embedded into the
graph(s) its chunks live in, so a re-chunk replaces them in place (the embed
writer deletes the orphans).

Scope (review chunk-data-5, 2026-10-10): a reindex sweeps the job's own
project only — any member can submit one, and a sweep of every project
re-chunked other tenants' knowledge under that member's name. A spec naming
another project is refused; the maintainer re-chunks a site with one job per
project. Each re-embed descriptor carries the artifact's ``kind`` and
``metadata`` (its source name decides PDF/PPTX extraction), and a retry
that finds a batch already enqueued goes on with the next one.

One pass, converging (ADR-093 D-11, wave F, 2026-10-11): a reindex makes
ONE pass over the whole stale set in keyset pages of ``batch_limit``
artifacts — one KNOWLEDGE_EMBED job per (graph, page), so ``batch_limit`` is
the most sources one embed job carries — and stops at the first short page.
It skips artifacts that a queued or running embed or understand job already
holds (another pass, the ingest chain, an earlier attempt of this job), and
a per-project transaction advisory lock serialises passes: a second pass
while one runs fails transiently by name before any sweep and is retried.
It still does not wait for its embed jobs; it completes when the pass is
enqueued. Before, one job swept at most ``batch_limit`` artifacts and the
store-computed idempotency key refused an identical repeat for ever (409;
the old job returned silently over ZMQ), so a project larger than
``batch_limit`` could not converge — ``models.default_idempotency_key`` now
gives each reindex submitted without a key a fresh one.

Payload contract (validated at submit time):

    {"database": str, "table": str, "target_model": str,
     "index_name": str | None, "embedding_dim": int > 0 | None,
     "batch_limit": int > 0 (default 500),
     "project_id": str | None, "chunk_scheme": str | None}

``chunk_scheme``, when given, must be the site's scheme: the re-embed
writes the scheme the site runs, never the one a spec asks for.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from umrs.jobs.models import Job, JobAttempt, JobStatus, JobType
from umrs.knowledge.embedding_metadata import (
    EmbeddingMetadataError,
    HeldBy,
    select_stale_artifacts,
    validate_reindex_payload,
)

from .monitor import AttemptCheckResult, BackendStatusChecker
from .scheduler import OrchestratorBackend

logger = logging.getLogger(__name__)

# Same env name as the KNOWLEDGE_EMBED backend (knowledge_adapters).
_SITE_EMBED_MODEL_ENV = "UMRS_KNOWLEDGE_EMBED_MODEL"

#: The unfinished jobs whose sources a pass does not sweep again.
HELD_BY = HeldBy(
    job_types=(JobType.KNOWLEDGE_EMBED.value, JobType.KNOWLEDGE_UNDERSTAND.value),
    statuses=(JobStatus.QUEUED.value, JobStatus.RUNNING.value),
)
#: The class half of the two-key advisory lock (``pg_try_advisory_xact_lock(
#: int4, int4)``): its key space is disjoint from the one-key job-id locks of
#: ``store._advisory_lock_key``. The second key is ``hashtext`` of the project.
_PASS_LOCK_CLASS = 0x52494458  # "RIDX"
_PASS_LOCK_SQL = "SELECT pg_try_advisory_xact_lock(%s, hashtext(%s))"


class ReindexBusy(RuntimeError):
    """Another KNOWLEDGE_REINDEX pass holds the project: transient, retried."""


@dataclass
class _PassTotals:
    swept: int = 0
    enqueued: int = 0
    pages: int = 0


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
        **_scope_fields(spec),
    }


def _scope_fields(spec: dict) -> dict:
    """The optional sweep scope: ``project_id`` and a ``chunk_scheme`` name
    (checked against the site at submit, see ``resolve_site_chunk_scheme``)."""
    from umrs.knowledge.chunking import resolve_scheme

    project = str(spec.get("project_id") or "").strip()
    scheme = spec.get("chunk_scheme")
    try:
        scheme = resolve_scheme(scheme) if scheme else None
    except ValueError as exc:
        raise EmbeddingMetadataError(str(exc)) from exc
    return {"project_id": project or None, "chunk_scheme": scheme}


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
        spec["project_id"] = resolve_job_project(job, spec.get("project_id"))
        spec["target_model"] = resolve_site_embed_model(spec["target_model"])
        spec["chunk_scheme"] = resolve_site_chunk_scheme(spec.get("chunk_scheme"))

        with _artifact_conn() as conn:
            _lock_project(conn, spec["project_id"])
            totals = _pass(job, conn, spec)
        if not totals.swept:
            logger.info("reindex no-op: no stale artifacts for job=%s", job.job_id)
            return _backend_ref(spec, totals)

        action = _maintain_index(spec)
        logger.info(
            "reindex: job=%s attempt=%s swept=%d pages=%d re-embed jobs=%d index=%s",
            job.job_id,
            attempt.attempt_id,
            totals.swept,
            totals.pages,
            totals.enqueued,
            action,
        )
        return _backend_ref(spec, totals)


def _lock_project(conn, project_id: str) -> None:
    """Take the project's pass lock for this transaction, or fail by name
    (transient: the attempt is retried) before any sweep. The try variant:
    a blocking wait would stall the scheduler loop."""
    cur = conn.cursor()
    cur.execute(_PASS_LOCK_SQL, (_PASS_LOCK_CLASS, f"knowledge_reindex:{project_id}"))
    row = cur.fetchone()
    if not (row and row[0]):
        raise ReindexBusy(
            f"another KNOWLEDGE_REINDEX pass holds project {project_id!r}; "
            "this one is retried"
        )


def _pass(job: Job, conn, spec: dict) -> _PassTotals:
    """One keyset-paged pass over the project's stale, unheld artifacts:
    one KNOWLEDGE_EMBED job per (graph, page); stops at the first short or
    empty page."""
    limit = int(spec.get("batch_limit", 500))
    totals, after = _PassTotals(), None
    while True:
        page = _sweep(conn, spec, after)
        if page:
            totals.pages += 1
            totals.swept += len(page)
            batches = _stale_sources(conn, page, spec["project_id"])
            totals.enqueued += len(_enqueue_reembed(job, batches))
            after = page[-1]
        if len(page) < limit:
            return totals


# Vector tables whose column has no dimension (0075): no ANN index can exist
# on them, so there is nothing to maintain — nearest-node reads scan exactly.
_UNTYPED_VECTOR_TABLES = frozenset({"knowledge_node_embeddings"})


def _maintain_index(spec: dict) -> str:
    """Rebuild the table's pgvector index; skipped for an untyped column."""
    if spec["table"] in _UNTYPED_VECTOR_TABLES:
        return "skipped (dimension-less vector column, 0075)"
    from umrs.shared.pgvector_index import maintain_pgvector_index

    metrics = maintain_pgvector_index(
        spec["database"],
        spec["table"],
        rebuild=True,
        index_name=spec.get("index_name") or None,
    )
    return str(metrics.get("action", "rebuild"))


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


def resolve_job_project(job: Job, requested: str | None) -> str:
    """The project a reindex sweeps: the job's own (verified at submit). A
    spec naming another project is refused by name, and so is a job with
    no project — there is no all-projects sweep from a job."""
    own = str(getattr(job, "project_id", "") or "").strip()
    if not own:
        raise EmbeddingMetadataError(
            "KNOWLEDGE_REINDEX sweeps the job's project and this job has none"
        )
    if requested and requested != own:
        raise EmbeddingMetadataError(
            f"KNOWLEDGE_REINDEX spec project_id {requested!r} is not the job's "
            f"project {own!r}: a reindex re-embeds its own project only"
        )
    return own


def resolve_site_chunk_scheme(requested: str | None) -> str:
    """The site's chunk scheme; a spec naming another one is refused by name
    (the re-embed cuts with the site's scheme, so the sweep would never
    converge)."""
    from umrs.knowledge.chunking import CHUNK_SCHEME_ENV, configured_scheme

    try:
        site = configured_scheme()
    except ValueError as exc:
        raise EmbeddingMetadataError(str(exc)) from exc
    if requested and requested != site:
        raise EmbeddingMetadataError(
            f"KNOWLEDGE_REINDEX chunk_scheme {requested!r} is not this site's "
            f"({CHUNK_SCHEME_ENV}={site!r}); set the site scheme, then reindex"
        )
    return site


# The graphs an artifact lives in: those holding its file node (the embed
# writer's part_of edges need it there) or its chunks (one read per batch),
# in the artifact's own project only — ``content.artifact_id`` is node data,
# and another project's node naming it must not add a target.
_CHUNK_GRAPHS_SQL = (
    "SELECT DISTINCT content->>'artifact_id', graph_id FROM knowledge_nodes "
    "WHERE (node_type = 'file' OR content->>'kind' = 'chunk') "
    "AND project_id = %s AND content->>'artifact_id' = ANY(%s)"
)
_STALE_ROWS_SQL = (
    "SELECT artifact_id, project_id, source_ref, kind, metadata "
    "FROM knowledge_artifacts WHERE artifact_id = ANY(%s) AND project_id = %s"
)
# The descriptor metadata the embed path reads: the source name (PDF/PPTX
# extraction, the header title's fallback) and the title.
_DESCRIPTOR_METADATA = ("source_name", "title", "staged_from")

# (project_id, graph_id) -> KNOWLEDGE_EMBED source descriptors
_Batches = dict[tuple[str, str], list[dict[str, Any]]]


def _graphs_by_artifact(cur, ids: list[str], project_id: str) -> dict[str, set[str]]:
    cur.execute(_CHUNK_GRAPHS_SQL, (project_id, ids))
    graphs: dict[str, set[str]] = {}
    for artifact_id, graph_id in cur.fetchall():
        graphs.setdefault(str(artifact_id), set()).add(str(graph_id or ""))
    return graphs


def _as_metadata(value: Any) -> dict[str, Any]:
    """A JSONB column as a dict (psycopg2 decodes it; a text form is parsed)."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return {}
    return value if isinstance(value, dict) else {}


def _descriptor(row: tuple) -> dict[str, Any]:
    """One KNOWLEDGE_EMBED source, shaped as the ingest chain shapes it: the
    embed path reads ``kind`` and ``metadata.source_name`` to decide how the
    bytes become text (without the name a PPTX was read as its zip bytes,
    review chunk-data-1)."""
    artifact_id, _project, source_ref, kind, metadata = row
    meta = _as_metadata(metadata)
    return {
        "artifact_id": str(artifact_id),
        "source_ref": str(source_ref or ""),
        "kind": str(kind or ""),
        "metadata": {k: meta[k] for k in _DESCRIPTOR_METADATA if meta.get(k)},
    }


def _stale_sources(conn, stale: list[str], project_id: str) -> _Batches:
    """Swept artifact ids of ``project_id`` → KNOWLEDGE_EMBED sources per
    (project, graph).

    An artifact is re-embedded into each graph that holds its file node or
    its chunks (its project's default graph when none does, or only legacy
    ``''`` rows): the writer replaces chunks by id within one graph scope,
    so re-embedding a named-graph artifact into the default graph would
    leave the old chunks where they are, and its part_of edges would point
    at a file node that is not there.
    """
    from umrs.knowledge.graph_registry import default_graph_id

    cur = conn.cursor()
    cur.execute(_STALE_ROWS_SQL, (list(stale), project_id))
    rows = cur.fetchall()
    graphs = _graphs_by_artifact(cur, [str(row[0]) for row in rows], project_id)
    default = default_graph_id(project_id)
    batches: _Batches = {}
    for row in rows:
        source = _descriptor(row)
        targets = {g or default for g in graphs.get(source["artifact_id"], {""})}
        for graph_id in sorted(targets):
            batches.setdefault((project_id, graph_id), []).append(source)
    return batches


def _enqueue_reembed(job: Job, batches: _Batches) -> list[str]:
    """One KNOWLEDGE_EMBED job per (project, graph). A batch whose job is
    already enqueued — this attempt's retry, while that embed job still
    runs — counts as enqueued, and the next batches still go (review
    chunk-data-11: the first duplicate ended every retry)."""
    from .service import get_job_service

    service = get_job_service()
    return [
        _submit_batch(service, job, project_id, graph_id, sources)
        for (project_id, graph_id), sources in sorted(batches.items())
    ]


def _submit_batch(
    service: Any,
    job: Job,
    project_id: str,
    graph_id: str,
    sources: list[dict[str, Any]],
) -> str:
    """Submit one re-embed job; its id (the existing job's when it is
    already enqueued)."""
    from .service import JobSubmission
    from .store import DuplicateJobError

    try:
        return service.submit_job(
            JobSubmission(
                project_id=project_id,
                user_id=job.user_id,
                job_type=JobType.KNOWLEDGE_EMBED,
                spec={
                    "sources": sources,
                    "project_id": project_id,
                    "parent_job_id": job.job_id,
                    "graph_id": graph_id,
                    "reason": "knowledge_reindex",
                },
                priority=job.priority,
                max_attempts=3,
            )
        )
    except DuplicateJobError as exc:
        existing = str(getattr(exc, "existing_job_id", "") or "")
        logger.info(
            "reindex %s: the re-embed of (%s, %s) is already enqueued (%s)",
            job.job_id,
            project_id,
            graph_id,
            existing or "unknown job",
        )
        return existing


def _sweep(conn, spec: dict, after: str | None) -> list[str]:
    """One page of the pass: stale, unheld artifacts after ``after``."""
    return select_stale_artifacts(
        conn,
        spec["target_model"],
        limit=int(spec.get("batch_limit", 500)),
        chunk_scheme=spec.get("chunk_scheme"),
        project_id=spec.get("project_id"),
        after=after,
        held_by=HELD_BY,
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


def _backend_ref(spec: dict, totals: _PassTotals) -> str:
    index = spec.get("index_name") or "default"
    scheme = spec.get("chunk_scheme") or "fixed-v0"
    return (
        f"knowledge_reindex|database={spec['database']}|table={spec['table']}|"
        f"index={index}|model={spec['target_model']}|scheme={scheme}|"
        f"swept={totals.swept}|reembed_jobs={totals.enqueued}|pages={totals.pages}"
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
                "pages": parsed.get("pages"),
            }
        )
