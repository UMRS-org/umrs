"""Durable DATA_INGESTION job backend."""

from __future__ import annotations

import logging
import os
from typing import Any

import boto3
from botocore.config import Config as BotoConfig

from .models import Job, JobAttempt, JobType
from .monitor import AttemptCheckResult, BackendStatusChecker
from .scheduler import OrchestratorBackend

logger = logging.getLogger(__name__)


def _build_s3_client() -> Any:
    return boto3.client(
        "s3",
        endpoint_url=os.getenv("S3_ENDPOINT_URL", "http://localhost:9000"),
        aws_access_key_id=os.getenv("S3_ACCESS_KEY", ""),
        aws_secret_access_key=os.getenv("S3_SECRET_KEY", ""),
        region_name=os.getenv("S3_REGION", "us-east-1"),
        config=BotoConfig(
            signature_version="s3v4",
            s3={"addressing_style": "path"},
        ),
        use_ssl=os.getenv("S3_USE_SSL", "false").lower() == "true",
    )


class DataIngestionBackend(OrchestratorBackend):
    """Synchronous per-object ingestion (MVP), similar to INDEX_BUILD."""

    @property
    def backend_name(self) -> str:
        return "data_ingestion"

    async def submit_job(self, job: Job, attempt: JobAttempt) -> str:
        from umrs.data_catalog.ingest import IngestionError, ingest_object

        spec = job.spec
        bucket = (spec.get("bucket") or "").strip()
        s3_key = (spec.get("s3_key") or "").strip()
        object_etag = (spec.get("object_etag") or "").strip()
        project_id = (spec.get("project_id") or job.project_id).strip()
        if not bucket or not s3_key:
            raise ValueError("data_ingestion requires spec.bucket and spec.s3_key")

        from umrs.shared.catalog_store_access import (
            CatalogStoreUnavailable,
            get_catalog_meta_store,
        )

        try:
            meta_store = get_catalog_meta_store()
        except CatalogStoreUnavailable as exc:
            raise RuntimeError(str(exc)) from exc

        s3_client = _build_s3_client()
        try:
            metrics = ingest_object(
                s3_client,
                bucket,
                s3_key=s3_key,
                object_etag=object_etag,
                project_id=project_id,
                owner=job.user_id,
                acl_scope=project_id,
                meta_store=meta_store,
            )
        except IngestionError as exc:
            raise RuntimeError(str(exc)) from exc

        logger.info(
            "DATA_INGESTION completed: job=%s key=%s action=%s",
            job.job_id,
            s3_key,
            metrics.get("action"),
        )
        return (
            f"data_ingestion|bucket={bucket}|key={s3_key}|"
            f"action={metrics.get('action', 'unknown')}"
        )


class DataIngestionStatusChecker(BackendStatusChecker):
    """Status checker for synchronous data-ingestion jobs."""

    @property
    def job_type(self) -> JobType:
        return JobType.DATA_INGESTION

    async def check_status(
        self,
        attempt: JobAttempt,
        job: Job,
    ) -> AttemptCheckResult:
        if not attempt.backend_ref:
            return AttemptCheckResult.failed("No backend reference")
        if attempt.backend_ref.startswith("data_ingestion|"):
            return AttemptCheckResult.succeeded(
                metrics={"verified_after_submit": True},
            )
        return AttemptCheckResult.failed("Unexpected backend reference")
