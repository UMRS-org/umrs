"""``GET /umrs/jobs/{job_id}/receipt``: one job's results, joined at read time.

ADR-105 D-10 (slice A105-S7, 2026-11-24). A finished run leaves up to four
provenance records: the commit attempt's metrics (the stored outputs and
whether each is ingested), ``artifacts`` registry rows (0068), ``a2l_artifacts``
rows (0077) and the knowledge ingest it chained (``knowledge_artifacts``
metadata, the ingest and understand jobs). Nothing joined them; an agent
asked three routes and still missed the knowledge leg.

The receipt is a read-time join in the ``asset_index/fanout.py`` style: no
new table, no cache, every call re-reads. It carries references (ids, URIs,
hashes), never content, so a tool can hand an agent a receipt instead of a
blob. Each section degrades on its own: a store that cannot be read becomes
a named warning and an empty section, never a failed receipt.

Access is the job's own (owner or admin, ``_enforce_job_access``): another
user's or another project's job answers 404 like an unknown id, and every
joined row is further bound to the job's project. The route is the same in
every deploy profile (hosted, stand-alone, own HPC): it reads only the
server's own database through the job store's pool.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from ..jobs.models import Job, JobAttempt
from ..jobs.store import JobNotFoundError, JobStore
from .typed_errors import PlatformError, raise_typed

logger = logging.getLogger(__name__)

#: Rows per joined section; a receipt is a handle, not a dump.
SECTION_LIMIT = 200


class ReceiptAttempt(BaseModel):
    """The attempt whose results the job committed."""

    attempt_id: str
    attempt_no: int
    status: str
    backend: str
    ended_at: str | None = None
    exit_reason: str | None = None


class ReceiptOutput(BaseModel):
    """One stored output of the commit attempt."""

    name: str = ""
    source_ref: str = ""
    content_hash: str = ""
    size_bytes: int | None = None
    ingest: bool = Field(
        True, description="False: outside the steps' declared ingest selection"
    )


class ReceiptArtifact(BaseModel):
    """One artifact-registry row the job registered."""

    registry_id: str
    name: str
    uri: str
    size_bytes: int | None = None
    checksum: str | None = None


class ReceiptA2LArtifact(BaseModel):
    """One a2l artifact recorded for the job."""

    id: str
    type: str
    uri: str
    status: str


class ReceiptKnowledgeJob(BaseModel):
    """A knowledge ingest or understand job the run chained."""

    job_id: str
    job_type: str
    status: str


class ReceiptWarning(BaseModel):
    """A section that could not be read, and why."""

    section: str
    reason: str


class JobReceiptResponse(BaseModel):
    """A job's results as references (ADR-105 D-10)."""

    job_id: str
    job_type: str
    status: str
    project_id: str
    execution_id: str | None = None
    workflow_name: str | None = None
    workflow_version: str | None = None
    commit_attempt: ReceiptAttempt | None = None
    outputs: list[ReceiptOutput] = Field(default_factory=list)
    artifacts: list[ReceiptArtifact] = Field(default_factory=list)
    a2l_artifacts: list[ReceiptA2LArtifact] = Field(default_factory=list)
    knowledge_jobs: list[ReceiptKnowledgeJob] = Field(default_factory=list)
    knowledge_artifact_ids: list[str] = Field(default_factory=list)
    warnings: list[ReceiptWarning] = Field(default_factory=list)


# ── sections ────────────────────────────────────────────────────────────────

_ARTIFACTS_SQL = (
    "SELECT id::text, name, uri, size_bytes, checksum FROM artifacts "
    "WHERE job_id = %s AND COALESCE(project_id, '') = %s "
    "ORDER BY created_at, id LIMIT %s"
)
_A2L_SQL = (
    "SELECT id, type, uri, status::text FROM a2l_artifacts "
    "WHERE job_id = %s AND COALESCE(project_id, '') = %s AND deleted_at IS NULL "
    "ORDER BY created_at, id LIMIT %s"
)
_KNOWLEDGE_JOBS_SQL = (
    "SELECT job_id, job_type, status FROM jobs "
    "WHERE spec->>'parent_job_id' = %s AND project_id = %s "
    "AND job_type IN ('knowledge_ingest', 'knowledge_understand') "
    "ORDER BY created_at, job_id LIMIT %s"
)
_KNOWLEDGE_ARTIFACTS_SQL = (
    "SELECT artifact_id FROM knowledge_artifacts "
    "WHERE metadata->>'produced_by_job_id' = %s AND project_id = %s "
    "ORDER BY artifact_id LIMIT %s"
)


def _rows(conn: Any, sql: str, job: Job) -> list[tuple]:
    with conn.cursor() as cur:
        cur.execute(sql, (job.job_id, job.project_id or "", SECTION_LIMIT))
        return list(cur.fetchall())


def _artifacts(conn: Any, job: Job) -> list[ReceiptArtifact]:
    return [
        ReceiptArtifact(
            registry_id=r[0], name=r[1], uri=r[2], size_bytes=r[3], checksum=r[4]
        )
        for r in _rows(conn, _ARTIFACTS_SQL, job)
    ]


def _a2l(conn: Any, job: Job) -> list[ReceiptA2LArtifact]:
    return [
        ReceiptA2LArtifact(id=r[0], type=r[1], uri=r[2], status=r[3])
        for r in _rows(conn, _A2L_SQL, job)
    ]


def _knowledge_jobs(conn: Any, job: Job) -> list[ReceiptKnowledgeJob]:
    return [
        ReceiptKnowledgeJob(job_id=r[0], job_type=r[1], status=r[2])
        for r in _rows(conn, _KNOWLEDGE_JOBS_SQL, job)
    ]


def _knowledge_artifacts(conn: Any, job: Job) -> list[str]:
    return [str(r[0]) for r in _rows(conn, _KNOWLEDGE_ARTIFACTS_SQL, job)]


#: (response field, reader). Each runs on its own; a failure is a warning.
SECTIONS: tuple[tuple[str, Callable[[Any, Job], list]], ...] = (
    ("artifacts", _artifacts),
    ("a2l_artifacts", _a2l),
    ("knowledge_jobs", _knowledge_jobs),
    ("knowledge_artifact_ids", _knowledge_artifacts),
)


def _read_section(conn: Any, reader: Callable, job: Job) -> list:
    try:
        return reader(conn, job)
    finally:
        # Reads only: end the implicit transaction (a failed SELECT aborts
        # it, and the next section must not inherit that).
        conn.rollback()


def _joined(store: JobStore, job: Job) -> dict[str, Any]:
    """The database sections of the receipt plus their warnings."""
    out: dict[str, Any] = {"warnings": []}
    pool = getattr(store, "db_pool", None)
    if pool is None:
        out["warnings"].append(ReceiptWarning(section="*", reason="no database pool"))
        return out
    conn = pool.getconn()
    try:
        for name, reader in SECTIONS:
            try:
                out[name] = _read_section(conn, reader, job)
            except Exception as exc:  # noqa: BLE001 — one section, named
                logger.warning("job receipt section %s unreadable: %s", name, exc)
                out["warnings"].append(
                    ReceiptWarning(section=name, reason=type(exc).__name__)
                )
    finally:
        pool.putconn(conn)
    return out


# ── the commit attempt ──────────────────────────────────────────────────────


def _status(value: Any) -> str:
    return str(getattr(value, "value", value) or "")


def _commit_attempt(store: JobStore, job: Job) -> JobAttempt | None:
    if not job.commit_attempt_id:
        return None
    for attempt in store.get_attempts_for_job(job.job_id):
        if attempt.attempt_id == job.commit_attempt_id:
            return attempt
    return None


def _attempt_view(attempt: JobAttempt) -> ReceiptAttempt:
    return ReceiptAttempt(
        attempt_id=attempt.attempt_id,
        attempt_no=attempt.attempt_no,
        status=_status(attempt.status),
        backend=attempt.backend,
        ended_at=attempt.ended_at.isoformat() if attempt.ended_at else None,
        exit_reason=attempt.exit_reason,
    )


def _output(entry: dict[str, Any]) -> ReceiptOutput:
    size = entry.get("size_bytes")
    return ReceiptOutput(
        name=str(entry.get("name") or ""),
        source_ref=str(entry.get("source_ref") or entry.get("uri") or ""),
        content_hash=str(entry.get("content_hash") or entry.get("sha256") or ""),
        size_bytes=size if isinstance(size, int) else None,
        ingest=entry.get("ingest") is not False,
    )


def _outputs(attempt: JobAttempt | None) -> list[ReceiptOutput]:
    entries = (attempt.metrics or {}).get("artifacts") if attempt else None
    if not isinstance(entries, list):
        return []
    return [_output(e) for e in entries[:SECTION_LIMIT] if isinstance(e, dict)]


def _spec_text(job: Job, key: str) -> str | None:
    value = (job.spec or {}).get(key)
    return value if isinstance(value, str) and value else None


def build_receipt(store: JobStore, job: Job) -> JobReceiptResponse:
    """The receipt of a job the caller may read."""
    attempt = _commit_attempt(store, job)
    return JobReceiptResponse(
        job_id=job.job_id,
        job_type=_status(job.job_type),
        status=_status(job.status),
        project_id=job.project_id,
        execution_id=_spec_text(job, "execution_id"),
        workflow_name=_spec_text(job, "workflow_name"),
        workflow_version=_spec_text(job, "workflow_version"),
        commit_attempt=_attempt_view(attempt) if attempt else None,
        outputs=_outputs(attempt),
        **_joined(store, job),
    )


def register(router: APIRouter) -> None:
    """Add the receipt route to ``api.jobs``' ``/umrs/jobs`` router, at that
    module's end. It reuses that module's identity and access helpers,
    imported here so neither module imports the other at load time. A plain
    route on that router, not a nested one: FastAPI 0.138 keeps an included
    router as one opaque entry in ``router.routes``, which route readers
    (``route.path``) cannot read."""
    from .jobs import (
        CallerIdentity,
        _enforce_job_access,
        get_caller_identity,
        get_job_store,
    )

    @router.get(
        "/{job_id}/receipt",
        operation_id="job_receipt",
        response_model=JobReceiptResponse,
        summary="A job's results as references",
        description=(
            "ADR-105 D-10: the job, its commit attempt's stored outputs (and "
            "whether each is ingested), its artifact-registry and a2l rows, and "
            "the knowledge ingest and understand jobs it chained, joined at read "
            "time. References only (ids, URIs, hashes), never content; a section "
            "that cannot be read is named in warnings. Another user's job "
            "answers 404 like an unknown id."
        ),
    )
    def get_job_receipt(
        job_id: str,
        caller: CallerIdentity = Depends(get_caller_identity),
        store: JobStore = Depends(get_job_store),
    ) -> JobReceiptResponse:
        """The job's receipt; another user's job answers 404.
        # auth: verified
        """
        try:
            job = store.get_job(job_id)
        except JobNotFoundError:
            raise_typed(PlatformError.jobs("not_found", f"Job not found: {job_id}"))
        _enforce_job_access(caller, job, action="receipt")
        return build_receipt(store, job)
