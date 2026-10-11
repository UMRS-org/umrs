"""``DELETE /umrs/knowledge/artifacts/{artifact_id}`` (ADR-093 MD-16).

A user/CLI verb: one transaction over the artifact row and everything derived
from it (``umrs.knowledge.artifact_delete``), then the object-store bytes,
best-effort and reported. The rag facade's ``DELETE /umrs/rag/documents/{id}``
resolves its document id to an artifact and runs the same delete
(:func:`delete_rag_document`).

Gates, in order:

- agents never delete (``x-r4-actor: agent`` → 403): no recorded user
  decision covers a delete (``enforce_human_delete_authorization``);
- the project-writable gate (``access_scope.project_writable``) on the target
  project — the caller's verified project unless ``project_id`` names
  another, which only an admin may write; an anonymous caller never deletes;
- the artifact must belong to that project: a foreign, unknown or
  already-deleted id is 404 by name, never a cross-project delete and never a
  second "deleted".
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Query, Request
from starlette.concurrency import run_in_threadpool

from umrs.knowledge.access_scope import project_writable
from umrs.knowledge.artifact_delete import (
    ArtifactDeleteReport,
    ArtifactNotFound,
    AsyncpgTx,
    ObjectStore,
    delete_artifact,
    resolve_document,
)

from ._asyncpg_pool import knowledge_asyncpg_pool
from ._knowledge_mutation_guard import enforce_human_delete_authorization
from ._scope_identity import request_caller_scope
from .typed_errors import PlatformError, raise_typed

router = APIRouter(prefix="/umrs/knowledge", tags=["knowledge"])
logger = logging.getLogger(__name__)


class S3ObjectStore:
    """The knowledge object store, resolved the way ingest resolves it."""

    def __init__(self, client: Any, content_bucket: str, staging_bucket: str) -> None:
        self._client = client
        self.content_bucket = content_bucket
        self.buckets = frozenset(b for b in (content_bucket, staging_bucket) if b)

    def delete(self, bucket: str, key: str) -> None:
        self._client.delete_object(Bucket=bucket, Key=key)


def object_store() -> ObjectStore | None:
    """Ingest's client and bucket (``_build_s3_client``); uploads stage in
    ``S3_BUCKET``. None when the store is unreachable."""
    from umrs.jobs.knowledge_adapters import _build_s3_client
    from umrs.shared.storage.s3_config import get_s3_config

    client, bucket = _build_s3_client({})
    if client is None:
        return None
    return S3ObjectStore(client, bucket, get_s3_config().bucket)


def _target_project(request: Request, project_id: str | None) -> tuple[str, str]:
    """(target project, user id) once the project-writable gate passed."""
    caller = request_caller_scope(request)
    target = (project_id or "").strip() or caller.project_id
    if caller.anonymous or not target or not project_writable(target, caller):
        raise_typed(
            PlatformError.knowledge_guard(
                "forbidden", f"Access denied: no write scope for project {target!r}"
            )
        )
    return target, caller.user_id


async def run_delete(request: Request, artifact_id: str, project_id: str) -> ArtifactDeleteReport:
    """The delete over the knowledge pool; 404 by name, 500 = nothing deleted."""
    pool = await knowledge_asyncpg_pool(request)
    try:
        return await delete_artifact(pool, artifact_id, project_id, object_store)
    except ArtifactNotFound as exc:
        raise_typed(PlatformError.knowledge("not_found", str(exc)), cause=exc)
    except Exception as exc:  # noqa: BLE001 — the transaction rolled back: say so
        logger.exception("knowledge artifact delete failed: %s", artifact_id)
        raise_typed(
            PlatformError.knowledge(
                "failure",
                f"delete of artifact {artifact_id!r} failed; nothing was deleted: {exc}",
            ),
            cause=exc,
        )


def _audit(report: ArtifactDeleteReport, user_id: str, via: str, **extra: Any) -> None:
    """The deletion in the platform audit trail (its events stay, detached).

    TODO(ADR-093 MD-16): ``knowledge_events`` has no deletion type (0056's
    CHECK allows parsed/extracted/embedded/indexed/archived/failed/resumed and
    requires a job id), so the deletion event goes to the audit log, not to
    that table; a ``deleted`` event type is a migration for the maintainer.
    """
    from .audit_events import emit_audit

    emit_audit(
        "knowledge.artifact_deleted",
        actor=user_id,
        project_id=report.project_id,
        details={
            "artifact_id": report.artifact_id,
            "via": via,
            "counts": dict(report.counts),
            "objects": [(o["key"], o["outcome"]) for o in report.objects],
            **extra,
        },
    )


@router.delete("/artifacts/{artifact_id}")
async def delete_knowledge_artifact(
    artifact_id: str,
    request: Request,
    project_id: str | None = Query(
        None, description="Target project (default: the caller's verified project)"
    ),
) -> dict[str, Any]:
    """Delete one knowledge artifact and everything derived from it (MD-16).
    # auth: verified

    200 with the deleted rows per kind and one outcome per object. 404 when
    the project holds no artifact of that id (unknown, another project's, or
    already deleted). 403 for agents and for a project the caller may not
    write.
    """
    enforce_human_delete_authorization(request, operation="knowledge.artifact_delete")
    target, user_id = _target_project(request, project_id)
    report = await run_delete(request, artifact_id, target)
    _audit(report, user_id, "knowledge.artifacts")
    return report.to_dict()


# ── the rag facade's document delete ─────────────────────────────────────────


def ingest_job_artifacts(job_id: str, project_id: str) -> list[str]:
    """The artifacts a project's KNOWLEDGE_INGEST job stored (its committed
    attempt's ``metrics.artifacts``): the rag facade's central upload answers
    with the ingest job id as its document id."""
    from umrs.jobs.models import JobType

    from .jobs import get_optional_job_store

    store = get_optional_job_store()
    if store is None:
        return []
    try:
        job = store.get_job(job_id)
        if job.project_id != project_id or job.job_type != JobType.KNOWLEDGE_INGEST:
            return []
        attempts = store.get_attempts_for_job(job_id)
    except Exception:  # noqa: BLE001 — not a job of this project: not resolvable
        return []
    committed = [a for a in attempts if a.attempt_id == job.commit_attempt_id] or attempts
    found = [
        d.get("artifact_id") for a in committed for d in (a.metrics or {}).get("artifacts", [])
    ]
    return list(dict.fromkeys(str(aid) for aid in found if aid))


async def _resolve(pool: Any, document_id: str, project_id: str) -> tuple[str | None, str]:
    """(artifact id or None, why not) for a rag document id in the project."""
    async with pool.acquire() as conn:
        artifact_id = await resolve_document(AsyncpgTx(conn), document_id, project_id)
        if artifact_id:
            return artifact_id, ""
        from_job = await run_in_threadpool(ingest_job_artifacts, document_id, project_id)
        if len(from_job) == 1:
            return await resolve_document(AsyncpgTx(conn), from_job[0], project_id), ""
    if from_job:
        return None, (
            f"; it is an ingest job that stored {len(from_job)} artifacts "
            f"({', '.join(from_job)}): delete each with `r4 knowledge delete`"
        )
    return None, ""


async def delete_rag_document(
    request: Request, document_id: str, project_id: str, user_id: str
) -> dict[str, Any]:
    """Resolve a rag document id to its artifact and run the artifact delete.

    An id that resolves to no artifact of the project is 404 — never a
    "deleted" for something still retrievable.
    """
    pool = await knowledge_asyncpg_pool(request)
    artifact_id, why = await _resolve(pool, document_id, project_id)
    if not artifact_id:
        raise_typed(
            PlatformError.rag(
                "not_found",
                f"rag document {document_id!r} resolves to no knowledge artifact in "
                f"project {project_id!r}; nothing was deleted{why}",
            )
        )
    report = await run_delete(request, artifact_id, project_id)
    from .governance import record_resource_deleted

    record_resource_deleted("rag_documents", document_id, project_id=project_id)
    _audit(report, user_id, "rag.documents", document_id=document_id)
    return {"document_id": document_id, **report.to_dict()}


__all__ = [
    "S3ObjectStore",
    "delete_knowledge_artifact",
    "delete_rag_document",
    "ingest_job_artifacts",
    "object_store",
    "router",
    "run_delete",
]
