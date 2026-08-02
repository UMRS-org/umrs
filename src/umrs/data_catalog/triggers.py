"""Post-upload trigger helpers for DATA_INGESTION jobs."""

from __future__ import annotations

import logging
from typing import Any

from umrs.jobs.models import JobType, RestartPolicy, compute_idempotency_key

from .ingest import write_pending_stub

logger = logging.getLogger(__name__)


def trigger_post_upload_ingestion(
    s3_client: Any,
    bucket: str,
    *,
    s3_key: str,
    object_etag: str,
    project_id: str,
    owner: str,
    filename: str | None = None,
) -> str | None:
    """
    Write pending catalog stub and enqueue a DATA_INGESTION job.

    Returns job_id when enqueued, else None (job store unavailable or error).
    Never raises — upload path must remain successful.
    """
    acl_scope = project_id
    try:
        from umrs.api.catalog_store import get_optional_catalog_meta_store

        meta_store = get_optional_catalog_meta_store()
        if meta_store is None:
            logger.warning(
                "Catalog store unavailable — skipping pending stub for %s", s3_key
            )
        else:
            write_pending_stub(
                s3_key=s3_key,
                object_etag=object_etag,
                owner=owner,
                acl_scope=acl_scope,
                project_id=project_id,
                filename=filename,
                meta_store=meta_store,
            )
    except Exception:
        logger.exception("Failed to write pending catalog stub for %s", s3_key)

    try:
        job_id = _enqueue_ingestion_job(
            bucket=bucket,
            s3_key=s3_key,
            object_etag=object_etag,
            project_id=project_id,
            owner=owner,
        )
    except Exception:
        logger.exception("Failed to enqueue DATA_INGESTION for %s", s3_key)
        return None

    _dispatch_workflow_triggers(
        project_id=project_id,
        bucket=bucket,
        s3_key=s3_key,
        owner=owner,
    )
    return job_id


def _dispatch_workflow_triggers(
    *, project_id: str, bucket: str, s3_key: str, owner: str
) -> None:
    """W4 fan-out: ingest event -> subscribed workflows (best-effort).

    Never raises — a broken workflow subscription must not break the
    upload path (same philosophy as the ingestion enqueue above).
    """
    try:
        from umrs.server.workflow_triggers import dispatch_catalog_event

        execution_ids = dispatch_catalog_event(
            project_id=project_id,
            bucket=bucket,
            s3_key=s3_key,
            owner=owner,
        )
        if execution_ids:
            logger.info(
                "Ingest triggers launched %d workflow run(s) for %s",
                len(execution_ids),
                s3_key,
            )
    except Exception:
        logger.exception("Workflow trigger dispatch failed for %s", s3_key)


def _enqueue_ingestion_job(
    *,
    bucket: str,
    s3_key: str,
    object_etag: str,
    project_id: str,
    owner: str,
) -> str:
    from umrs.api.jobs import get_job_store

    store = get_job_store()
    spec = {
        "bucket": bucket,
        "s3_key": s3_key,
        "object_etag": object_etag,
        "project_id": project_id,
    }
    idempotency_key = compute_idempotency_key(
        project_id,
        owner,
        JobType.DATA_INGESTION,
        spec,
    )
    job_id = store.create_job(
        project_id=project_id,
        user_id=owner,
        job_type=JobType.DATA_INGESTION,
        spec=spec,
        idempotency_key=idempotency_key,
        restart_policy=RestartPolicy.ON_CRASH,
    )
    store.enqueue_job(job_id)
    return job_id
