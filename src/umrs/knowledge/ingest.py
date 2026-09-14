"""Batch knowledge ingestion orchestration.

``KnowledgeIngester`` discovers project objects in S3, filters unchanged entries,
runs ingestion in parallel, and reports progress. Per-object enrichment (parse,
perception, catalog metadata) is delegated to
``umrs.data_catalog.ingest.ingest_object`` — the same path used by upload jobs.
Perception is controlled by ``UMRS_DATA_CATALOG_ENABLE_PERCEPTION`` (same as upload).
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol

from ..data_catalog.ingest import IngestionError
from ..data_catalog.ingest import ingest_object as catalog_ingest_object
from ..data_catalog.pg_store import CatalogMetaStore
from ..data_catalog.schema import CatalogState

logger = logging.getLogger(__name__)

_DEFAULT_MAX_EXTRACT_BYTES = 10 * 1024 * 1024  # 10 MB


# ── Result Types ─────────────────────────────────────────────────────────────


@dataclass
class IngestionResult:
    """Result of ingesting a single object."""

    s3_key: str
    modality: str
    status: str  # "ingested" | "skipped" | "error"
    summary: str = ""
    tags: list[str] = field(default_factory=list)
    extracted_text: str = ""
    has_extracted_text: bool = False
    sampled_bytes: int = 0
    total_size: int = 0
    error: str | None = None
    logical_path: str | None = None


@dataclass
class BatchIngestionResult:
    """Result of ingesting a batch of objects."""

    total: int = 0
    ingested: int = 0
    skipped: int = 0
    errors: int = 0
    results: list[IngestionResult] = field(default_factory=list)
    duration_seconds: float = 0.0


# ── Progress Callback ────────────────────────────────────────────────────────


class ProgressCallback(Protocol):
    """Protocol for progress callbacks."""

    def __call__(
        self,
        current: int,
        total: int,
        s3_key: str,
        status: str,
    ) -> None:
        """Report progress.

        Args:
            current: Current object index (1-based)
            total: Total number of objects
            s3_key: S3 key of the current object
            status: Status of the current object ("ingested", "skipped", "error")
        """
        ...


# ── Ingester Configuration ──────────────────────────────────────────────────


@dataclass
class IngesterConfig:
    """Configuration for the KnowledgeIngester."""

    # Parallel processing
    max_workers: int = 4
    batch_size: int = 10

    # Extraction limits (batch path default; upload jobs use env unless overridden)
    max_extract_bytes: int = _DEFAULT_MAX_EXTRACT_BYTES

    # Duplicate detection
    skip_if_unchanged: bool = True

    # Incremental indexing
    incremental: bool = True

    # Resume capability
    resume_state_key: str = "_data_catalog/ingestion_state.json"

    # RAG indexing (reserved; not wired yet)
    enable_rag_indexing: bool = True


# ── Knowledge Ingester ─────────────────────────────────────────────────────


def _resolve_meta_store(meta_store: CatalogMetaStore | None) -> CatalogMetaStore:
    if meta_store is not None:
        return meta_store
    from umrs.api.catalog_store import get_catalog_meta_store

    return get_catalog_meta_store()


class KnowledgeIngester:
    """Batch orchestrator over the data-catalog ingestion pipeline.

    Usage:
        ingester = KnowledgeIngester(s3_client, bucket, config)
        result = ingester.ingest_project(project_id, owner)
    """

    def __init__(
        self,
        s3_client: Any,
        bucket: str,
        config: IngesterConfig | None = None,
        *,
        meta_store: CatalogMetaStore | None = None,
    ) -> None:
        self.s3_client = s3_client
        self.bucket = bucket
        self.config = config or IngesterConfig()
        self._meta_store = meta_store
        self._lock = threading.Lock()
        self._state: dict[str, Any] = {}

    @property
    def meta_store(self) -> CatalogMetaStore:
        if self._meta_store is None:
            self._meta_store = _resolve_meta_store(None)
        return self._meta_store

    def ingest_project(
        self,
        project_id: str,
        owner: str,
        *,
        prefix: str = "",
        progress_callback: ProgressCallback | None = None,
    ) -> BatchIngestionResult:
        """Ingest all objects in a project."""
        start_time = datetime.now(timezone.utc)

        objects = self._discover_objects(prefix)
        logger.info(
            "Discovered %d objects in project %s",
            len(objects),
            project_id,
        )

        if self.config.incremental:
            objects = self._filter_unchanged(project_id, objects)
            logger.info(
                "After incremental filter: %d objects to ingest",
                len(objects),
            )

        result = BatchIngestionResult(total=len(objects))
        if not objects:
            result.duration_seconds = (
                datetime.now(timezone.utc) - start_time
            ).total_seconds()
            return result

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {}
            for idx, obj in enumerate(objects):
                future = executor.submit(
                    self._ingest_one,
                    project_id=project_id,
                    owner=owner,
                    s3_key=obj["key"],
                    object_etag=obj["etag"],
                    progress_callback=progress_callback,
                    current=idx + 1,
                    total=len(objects),
                )
                futures[future] = obj

            for future in as_completed(futures):
                try:
                    res = future.result()
                    result.results.append(res)
                    if res.status == "ingested":
                        result.ingested += 1
                    elif res.status == "skipped":
                        result.skipped += 1
                    elif res.status == "error":
                        result.errors += 1
                except Exception as exc:
                    logger.exception("Ingestion failed for object")
                    result.errors += 1
                    result.results.append(
                        IngestionResult(
                            s3_key=futures[future]["key"],
                            modality="unknown",
                            status="error",
                            error=str(exc),
                        )
                    )

        result.duration_seconds = (
            datetime.now(timezone.utc) - start_time
        ).total_seconds()

        logger.info(
            "Ingestion complete: %d ingested, %d skipped, %d errors in %.2fs",
            result.ingested,
            result.skipped,
            result.errors,
            result.duration_seconds,
        )

        return result

    def ingest_object(
        self,
        project_id: str,
        owner: str,
        s3_key: str,
        object_etag: str,
    ) -> IngestionResult:
        """Ingest a single object."""
        return self._ingest_one(
            project_id=project_id,
            owner=owner,
            s3_key=s3_key,
            object_etag=object_etag,
            progress_callback=None,
            current=1,
            total=1,
        )

    # ── Internal Methods ────────────────────────────────────────────────────

    def _discover_objects(self, prefix: str) -> list[dict[str, str]]:
        """Discover all objects in the bucket."""
        objects = []
        paginator = self.s3_client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                if obj["Key"].startswith("_data_catalog/"):
                    continue
                objects.append(
                    {
                        "key": obj["Key"],
                        "etag": obj.get("ETag", "").strip('"'),
                        "size": obj.get("Size", 0),
                    }
                )
        return objects

    def _filter_unchanged(
        self,
        project_id: str,
        objects: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        """Filter out objects that haven't changed since last ingestion."""
        filtered = []
        for obj in objects:
            existing = self.meta_store.get(project_id, obj["key"])
            if (
                existing
                and existing.object_etag == obj["etag"]
                and existing.state == CatalogState.READY
            ):
                continue
            filtered.append(obj)
        return filtered

    def _ingest_one(
        self,
        *,
        project_id: str,
        owner: str,
        s3_key: str,
        object_etag: str,
        progress_callback: ProgressCallback | None,
        current: int,
        total: int,
    ) -> IngestionResult:
        """Ingest a single object via the data-catalog pipeline."""
        try:
            metrics = catalog_ingest_object(
                self.s3_client,
                self.bucket,
                s3_key=s3_key,
                object_etag=object_etag,
                project_id=project_id,
                owner=owner,
                acl_scope=project_id,
                max_extract_bytes=self.config.max_extract_bytes,
                force=not self.config.skip_if_unchanged,
                meta_store=self.meta_store,
            )
        except IngestionError as exc:
            logger.error("Catalog ingestion failed for %s: %s", s3_key, exc)
            result = _result_from_catalog_failure(
                self.meta_store,
                project_id,
                s3_key,
                error=str(exc),
            )
            if progress_callback:
                progress_callback(current, total, s3_key, "error")
            return result

        result = _result_from_metrics(
            self.meta_store,
            project_id,
            s3_key,
            metrics,
        )
        if progress_callback:
            progress_callback(current, total, s3_key, result.status)
        return result


# ── Convenience Function ────────────────────────────────────────────────────


def ingest_object(
    s3_client: Any,
    bucket: str,
    *,
    s3_key: str,
    object_etag: str,
    project_id: str,
    owner: str,
    max_extract_bytes: int | None = None,
    meta_store: CatalogMetaStore | None = None,
) -> dict[str, Any]:
    """Ingest a single object (convenience function). Returns a metrics dict."""
    config = IngesterConfig()
    if max_extract_bytes is not None:
        config.max_extract_bytes = max_extract_bytes

    ingester = KnowledgeIngester(s3_client, bucket, config, meta_store=meta_store)
    result = ingester.ingest_object(
        project_id=project_id,
        owner=owner,
        s3_key=s3_key,
        object_etag=object_etag,
    )

    return {
        "action": result.status,
        "s3_key": result.s3_key,
        "modality": result.modality,
        "summary": result.summary,
        "tags": result.tags,
        "has_extracted_text": result.has_extracted_text,
        "sampled_bytes": result.sampled_bytes,
        "total_size": result.total_size,
        "error": result.error,
    }


# ── Helpers ─────────────────────────────────────────────────────────────────


def _result_from_metrics(
    meta_store: CatalogMetaStore,
    project_id: str,
    s3_key: str,
    metrics: dict[str, Any],
) -> IngestionResult:
    """Map catalog metrics to a batch ``IngestionResult`` (reads meta for summary)."""
    action = metrics.get("action", "error")
    status = action if action in {"ingested", "skipped", "error"} else "error"
    meta = meta_store.get(project_id, s3_key)
    modality = metrics.get("modality") or (
        meta.detected_modality if meta else "unknown"
    )
    return IngestionResult(
        s3_key=s3_key,
        modality=modality,
        status=status,
        summary=meta.summary if meta else "",
        tags=list(meta.tags) if meta else [],
        has_extracted_text=bool(metrics.get("has_extracted_text")),
        sampled_bytes=int(metrics.get("sampled_bytes") or 0),
        total_size=int(metrics.get("total_size") or 0),
        logical_path=metrics.get("logical_path"),
    )


def _result_from_catalog_failure(
    meta_store: CatalogMetaStore,
    project_id: str,
    s3_key: str,
    *,
    error: str,
) -> IngestionResult:
    """Build an error result after catalog ingestion raised ``IngestionError``.

    The catalog pipeline writes ERROR metadata before raising, so we read it
    back for modality/summary instead of duplicating error-state logic here.
    """
    meta = meta_store.get(project_id, s3_key)
    return IngestionResult(
        s3_key=s3_key,
        modality=meta.detected_modality if meta else "unknown",
        status="error",
        summary=meta.summary if meta else "",
        tags=list(meta.tags) if meta else [],
        error=error,
    )
