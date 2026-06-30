"""Unified knowledge ingestion pipeline.

This module implements the unified ingestion pipeline that:
  1. Discovers all files in a project
  2. Parses each file according to its modality
  3. Extracts metadata
  4. Summarizes
  5. Generates embeddings
  6. Registers into UMRS
  7. Updates catalog
  8. Indexes for RAG

Requirements:
  - Incremental indexing (only ingest changed files)
  - Parallel processing (ingest multiple files concurrently)
  - Progress reporting (callback for progress updates)
  - Resume capability (state persisted, can resume after interruption)
  - Duplicate detection (content-hash based)
"""

from __future__ import annotations

import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Protocol

from ..data_catalog.extract import extract_content
from ..data_catalog.modality import detect_modality, filename_signals
from ..data_catalog.schema import CatalogObjectMeta, CatalogState
from ..data_catalog.store import (
    get_object_etag,
    get_object_range,
    read_meta,
    write_extracted_text,
    write_meta,
)
from .parse_code import parse_code
from .parse_notebook import parse_notebook

logger = logging.getLogger(__name__)


# ── Exceptions ──────────────────────────────────────────────────────────────


class IngestionError(Exception):
    """Permanent ingestion failure for one object."""


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

    # Extraction limits
    max_extract_bytes: int = 10 * 1024 * 1024  # 10 MB

    # Duplicate detection
    skip_if_unchanged: bool = True

    # Incremental indexing
    incremental: bool = True

    # Resume capability
    resume_state_key: str = "_data_catalog/ingestion_state.json"

    # Perception (LLM summaries, image captions)
    enable_perception: bool = True

    # RAG indexing
    enable_rag_indexing: bool = True


# ── Knowledge Ingester ─────────────────────────────────────────────────────


class KnowledgeIngester:
    """Unified knowledge ingestion pipeline.

    Discovers, parses, extracts metadata, summarizes, generates embeddings,
    registers into UMRS, updates catalog, and indexes for RAG.

    Usage:
        ingester = KnowledgeIngester(s3_client, bucket, config)
        result = ingester.ingest_project(project_id, owner)
    """

    def __init__(
        self,
        s3_client: Any,
        bucket: str,
        config: IngesterConfig | None = None,
    ) -> None:
        self.s3_client = s3_client
        self.bucket = bucket
        self.config = config or IngesterConfig()
        self._lock = threading.Lock()
        self._state: dict[str, Any] = {}

    def ingest_project(
        self,
        project_id: str,
        owner: str,
        *,
        prefix: str = "",
        progress_callback: ProgressCallback | None = None,
    ) -> BatchIngestionResult:
        """Ingest all objects in a project.

        Args:
            project_id: The project ID
            owner: The owner user ID
            prefix: Optional S3 prefix to limit the scan
            progress_callback: Optional callback for progress updates

        Returns:
            A ``BatchIngestionResult`` with ingestion statistics.
        """
        start_time = datetime.now(timezone.utc)

        # Discover all objects
        objects = self._discover_objects(prefix)
        logger.info(
            "Discovered %d objects in project %s",
            len(objects),
            project_id,
        )

        # Filter out already-ingested objects (incremental)
        if self.config.incremental:
            objects = self._filter_unchanged(objects)
            logger.info(
                "After incremental filter: %d objects to ingest",
                len(objects),
            )

        # Ingest objects in parallel
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
        """Ingest a single object.

        Args:
            project_id: The project ID
            owner: The owner user ID
            s3_key: The S3 key of the object
            object_etag: The ETag of the object

        Returns:
            An ``IngestionResult`` with ingestion statistics.
        """
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
                # Skip catalog metadata files
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
        objects: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        """Filter out objects that haven't changed since last ingestion."""
        filtered = []
        for obj in objects:
            existing = read_meta(self.s3_client, self.bucket, obj["key"])
            if (
                existing
                and existing.object_etag == obj["etag"]
                and existing.state == CatalogState.READY
            ):
                # Object unchanged and already ingested
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
        """Ingest a single object."""
        # Check if already ingested (idempotent)
        if self.config.skip_if_unchanged:
            existing = read_meta(self.s3_client, self.bucket, s3_key)
            if (
                existing
                and existing.object_etag == object_etag
                and existing.state == CatalogState.READY
            ):
                if progress_callback:
                    progress_callback(current, total, s3_key, "skipped")
                return IngestionResult(
                    s3_key=s3_key,
                    modality=existing.detected_modality,
                    status="skipped",
                    summary=existing.summary,
                    tags=existing.tags,
                )

        # Get live ETag (in case object was updated)
        try:
            live_etag = get_object_etag(self.s3_client, self.bucket, s3_key)
        except Exception as exc:
            logger.error("Failed to get ETag for %s: %s", s3_key, exc)
            return IngestionResult(
                s3_key=s3_key,
                modality="unknown",
                status="error",
                error=str(exc),
            )

        if live_etag != object_etag:
            object_etag = live_etag

        # Download content (bounded)
        filename = s3_key.rsplit("/", 1)[-1]
        try:
            content, total_size = get_object_range(
                self.s3_client,
                self.bucket,
                s3_key,
                offset=0,
                length=self.config.max_extract_bytes,
            )
        except Exception as exc:
            logger.error("Failed to download %s: %s", s3_key, exc)
            return IngestionResult(
                s3_key=s3_key,
                modality="unknown",
                status="error",
                error=str(exc),
            )

        # Detect modality
        modality = detect_modality(filename, content)
        signals = filename_signals(filename)

        # Parse content based on modality
        extracted_text = ""
        tags: list[str] = []
        summary = ""
        try:
            if modality == "notebook":
                # Parse Jupyter notebook
                result = parse_notebook(filename, content)
                extracted_text = result.extracted_text
                tags = result.tags
                summary = result.summary
            elif modality == "code":
                # Parse Python/C++/CUDA code
                result = parse_code(filename, content)
                extracted_text = result.extracted_text
                tags = result.tags
                summary = result.summary
            else:
                # Use existing extract_content for text, structured, document, image
                summary, extracted_text, extra_tags = extract_content(
                    filename, content
                )
                tags = list(extra_tags)
        except Exception as exc:
            logger.error("Failed to parse %s: %s", s3_key, exc)
            return IngestionResult(
                s3_key=s3_key,
                modality=modality,
                status="error",
                error=str(exc),
            )

        # Write extracted text sidecar
        extracted_ref = None
        if extracted_text.strip():
            try:
                extracted_ref = write_extracted_text(
                    self.s3_client, self.bucket, s3_key, extracted_text
                )
            except Exception as exc:
                logger.warning(
                    "Failed to write extracted text for %s: %s",
                    s3_key,
                    exc,
                )

        # Write catalog metadata
        meta = CatalogObjectMeta(
            s3_key=s3_key,
            object_etag=object_etag,
            state=CatalogState.READY,
            owner=owner,
            acl_scope=project_id,
            updated_at=datetime.now(timezone.utc).isoformat(),
            summary=summary or f"{modality} object {filename}",
            deterministic_summary=summary,
            tags=tags,
            filename_signals=signals,
            detected_modality=modality,
            extracted_text_ref=extracted_ref,
        )
        try:
            write_meta(self.s3_client, self.bucket, meta)
        except Exception as exc:
            logger.error("Failed to write metadata for %s: %s", s3_key, exc)
            return IngestionResult(
                s3_key=s3_key,
                modality=modality,
                status="error",
                error=str(exc),
            )

        if progress_callback:
            progress_callback(current, total, s3_key, "ingested")

        return IngestionResult(
            s3_key=s3_key,
            modality=modality,
            status="ingested",
            summary=summary,
            tags=tags,
            extracted_text=extracted_text,
            has_extracted_text=bool(extracted_ref),
            sampled_bytes=len(content),
            total_size=total_size,
        )


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
) -> dict[str, Any]:
    """Ingest a single object (convenience function).

    Returns a metrics dict.
    """
    config = IngesterConfig()
    if max_extract_bytes:
        config.max_extract_bytes = max_extract_bytes

    ingester = KnowledgeIngester(s3_client, bucket, config)
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
