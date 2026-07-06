"""Ingestion orchestration for a single S3 object."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any

from botocore.exceptions import ClientError

from .extract import extract_content
from .modality import detect_modality, filename_signals
from .paths import logical_path_for_object
from .pg_store import CatalogMetaStore
from .schema import CatalogObjectMeta, CatalogState
from .store import (
    get_object_etag,
    get_object_range,
    write_extracted_text,
)

logger = logging.getLogger(__name__)


class IngestionError(Exception):
    """Permanent ingestion failure for one object."""


def _resolve_meta_store(meta_store: CatalogMetaStore | None) -> CatalogMetaStore:
    if meta_store is not None:
        return meta_store
    from umrs.api.catalog_store import get_catalog_meta_store

    return get_catalog_meta_store()


def write_pending_stub(
    *,
    s3_key: str,
    object_etag: str,
    owner: str,
    acl_scope: str,
    project_id: str,
    filename: str | None = None,
    meta_store: CatalogMetaStore | None = None,
) -> CatalogObjectMeta:
    """Write minimal pending metadata so the object is catalog-listable immediately."""
    store = _resolve_meta_store(meta_store)
    meta = CatalogObjectMeta.pending(
        s3_key=s3_key,
        object_etag=object_etag,
        owner=owner,
        acl_scope=acl_scope,
        filename=filename,
    )
    store.upsert(meta, project_id)
    return meta


def ingest_object(
    s3_client: Any,
    bucket: str,
    *,
    s3_key: str,
    object_etag: str,
    project_id: str,
    owner: str,
    acl_scope: str | None = None,
    max_extract_bytes: int | None = None,
    force: bool = False,
    meta_store: CatalogMetaStore | None = None,
) -> dict[str, Any]:
    """
    Enrich one object: extract text/metadata and flip catalog entry to ``ready``.

    Idempotent when ``object_etag`` matches the stored meta. Returns metrics dict.

    Set ``force=True`` to re-run extraction even when the object is already ready.
    """
    store = _resolve_meta_store(meta_store)
    scope = acl_scope or project_id
    existing = store.get(project_id, s3_key)
    if (
        not force
        and existing
        and existing.object_etag == object_etag
        and existing.state == CatalogState.READY
    ):
        return {"action": "skipped", "reason": "already_ready", "s3_key": s3_key}

    try:
        live_etag = get_object_etag(s3_client, bucket, s3_key)
    except ClientError as exc:
        raise IngestionError(f"object missing: {s3_key}") from exc

    if live_etag != object_etag:
        object_etag = live_etag

    filename = s3_key.rsplit("/", 1)[-1]
    try:
        content, total_size = get_object_range(
            s3_client,
            bucket,
            s3_key,
            offset=0,
            length=max_extract_bytes or _max_extract_bytes(),
        )
    except ClientError as exc:
        _write_error(
            store,
            project_id=project_id,
            s3_key=s3_key,
            object_etag=object_etag,
            owner=owner,
            acl_scope=scope,
            note=f"download failed: {exc}",
        )
        raise IngestionError(str(exc)) from exc

    modality = detect_modality(filename, content)
    signals = filename_signals(filename)
    truncated_sample = len(content) < total_size

    try:
        summary, extracted_text, extra_tags = extract_content(filename, content)
    except Exception as exc:
        _write_error(
            store,
            project_id=project_id,
            s3_key=s3_key,
            object_etag=object_etag,
            owner=owner,
            acl_scope=scope,
            note=f"extract failed: {exc}",
            detected_modality=modality,
            filename_signals=signals,
        )
        raise IngestionError(str(exc)) from exc

    deterministic_summary = summary
    extracted_text, perception_tags, perception_summary, perception_model = (
        _apply_perception(
            s3_client,
            bucket,
            s3_key=s3_key,
            filename=filename,
            modality=modality,
            project_id=project_id,
            owner=owner,
            extracted_text=extracted_text,
            sampled_content=content,
            total_size=total_size,
        )
    )
    display_summary = _display_summary(
        deterministic_summary,
        perception_summary,
        modality=modality,
    )

    extracted_ref = None
    if extracted_text.strip():
        extracted_ref = write_extracted_text(s3_client, bucket, s3_key, extracted_text)

    tags = list(extra_tags) + perception_tags
    if truncated_sample:
        tags.append("catalog_sample_truncated")

    meta = CatalogObjectMeta(
        s3_key=s3_key,
        object_etag=object_etag,
        state=CatalogState.READY,
        owner=owner,
        acl_scope=scope,
        updated_at=_utc_now(),
        summary=display_summary or f"{modality} object {filename}",
        deterministic_summary=deterministic_summary,
        perception_summary=perception_summary,
        perception_model=perception_model,
        tags=tags,
        filename_signals=signals,
        detected_modality=modality,
        extracted_text_ref=extracted_ref,
    )
    store.upsert(meta, project_id)

    logical_path = logical_path_for_object(project_id, s3_key, modality)

    return {
        "action": "ingested",
        "s3_key": s3_key,
        "modality": modality,
        "logical_path": logical_path,
        "has_extracted_text": bool(extracted_ref),
        "sampled_bytes": len(content),
        "total_size": total_size,
    }


_SUMMARY_LIMIT = 500


def _apply_perception(
    s3_client: Any,
    bucket: str,
    *,
    s3_key: str,
    filename: str,
    modality: str,
    project_id: str,
    owner: str,
    extracted_text: str,
    sampled_content: bytes,
    total_size: int,
) -> tuple[str, list[str], str | None, str | None]:
    """
    Best-effort generative perception layered on deterministic extraction.

    Returns ``(extracted_text, extra_tags, perception_summary, perception_model)``.
    Never raises; on failure returns inputs unchanged plus ``perception_skipped``.
    """
    from .perception import perception_enabled, run_perception

    if not perception_enabled():
        return extracted_text, [], None, None

    image_bytes: bytes | None = None
    if modality == "image":
        image_bytes = _fetch_image_bytes(
            s3_client, bucket, s3_key, sampled_content, total_size
        )

    result = run_perception(
        project_id=project_id,
        filename=filename,
        modality=modality,
        extracted_text=extracted_text,
        image_bytes=image_bytes,
        user_id=owner,
    )
    if result is None:
        return extracted_text, ["perception_skipped"], None, None

    perception_summary = result.text[:_SUMMARY_LIMIT] or None
    if result.kind == "vision_caption":
        merged = (
            result.text
            if not extracted_text.strip()
            else f"{result.text}\n\n{extracted_text}"
        )
        return merged, ["vision_captioned", *result.tags], perception_summary, result.model
    return extracted_text, ["llm_summarized", *result.tags], perception_summary, result.model


def _display_summary(
    deterministic_summary: str,
    perception_summary: str | None,
    *,
    modality: str,
) -> str:
    """Greppable primary summary: structured keeps deterministic; others prefer perception."""
    if not perception_summary:
        return deterministic_summary
    if modality == "structured":
        return deterministic_summary or perception_summary
    return perception_summary or deterministic_summary


def _fetch_image_bytes(
    s3_client: Any,
    bucket: str,
    s3_key: str,
    sampled_content: bytes,
    total_size: int,
) -> bytes:
    """Return full image bytes (bounded) for captioning; reuse sample if complete."""
    if len(sampled_content) >= total_size:
        return sampled_content
    try:
        content, _ = get_object_range(
            s3_client,
            bucket,
            s3_key,
            offset=0,
            length=_max_image_bytes(),
        )
        return content
    except ClientError:
        return sampled_content


def _max_image_bytes() -> int:
    raw = os.getenv("UMRS_DATA_CATALOG_PERCEPTION_MAX_IMAGE_BYTES", str(4 * 1024 * 1024))
    try:
        return max(1, int(raw))
    except ValueError:
        return 4 * 1024 * 1024


def _write_error(
    store: CatalogMetaStore,
    *,
    project_id: str,
    s3_key: str,
    object_etag: str,
    owner: str,
    acl_scope: str,
    note: str,
    detected_modality: str = "unknown",
    filename_signals: list[str] | None = None,
) -> None:
    meta = CatalogObjectMeta(
        s3_key=s3_key,
        object_etag=object_etag,
        state=CatalogState.ERROR,
        owner=owner,
        acl_scope=acl_scope,
        updated_at=_utc_now(),
        detected_modality=detected_modality,
        filename_signals=filename_signals or [],
        error_note=note[:2000],
    )
    store.upsert(meta, project_id)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _max_extract_bytes() -> int:
    raw = os.getenv("UMRS_DATA_CATALOG_MAX_EXTRACT_BYTES", "1048576")
    try:
        parsed = int(raw)
    except ValueError:
        logger.warning("Invalid UMRS_DATA_CATALOG_MAX_EXTRACT_BYTES=%r", raw)
        return 1_048_576
    return max(1, parsed)
