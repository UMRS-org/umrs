"""Catalog metadata → knowledge artifact conversion helpers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .schema import CatalogObjectMeta
from .store import get_object_range, read_meta

logger = logging.getLogger(__name__)

_DEFAULT_MAX_CONTENT_BYTES = 10 * 1024 * 1024

_MODALITY_TO_ARTIFACT_TYPE: dict[str, str] = {
    "notebook": "notebook",
    "code": "code",
    "document": "paper",
    "structured": "dataset",
    "text": "text",
    "image": "image",
}


def modality_to_artifact_type(modality: str) -> str:
    """Map catalog modality to knowledge-store artifact type."""
    return _MODALITY_TO_ARTIFACT_TYPE.get(modality, "text")


def artifact_id_for_meta(meta: CatalogObjectMeta) -> str:
    """Stable artifact id for a catalog object."""
    filename = meta.s3_key.rsplit("/", 1)[-1]
    artifact_type = modality_to_artifact_type(meta.detected_modality)
    return f"{artifact_type}-{Path(filename).stem}-{hash(meta.s3_key) & 0xFFFFFF:X}"


def meta_to_artifact(
    meta: CatalogObjectMeta,
    content: bytes = b"",
    *,
    local_path: str | None = None,
) -> dict[str, Any]:
    """Build a knowledge-store artifact dict from catalog metadata."""
    filename = meta.s3_key.rsplit("/", 1)[-1]
    return {
        "artifact_id": artifact_id_for_meta(meta),
        "artifact_type": modality_to_artifact_type(meta.detected_modality),
        "filename": filename,
        "content": content,
        "summary": meta.summary,
        "tags": list(meta.tags),
        "metadata": {
            "path": local_path or meta.s3_key,
            "s3_key": meta.s3_key,
            "catalog_ref": meta.s3_key,
            "object_etag": meta.object_etag,
            "detected_modality": meta.detected_modality,
            "extracted_text_ref": meta.extracted_text_ref,
            "catalog_state": meta.state.value,
        },
    }


def load_artifact_content(
    s3_client: Any,
    bucket: str,
    s3_key: str,
    meta: CatalogObjectMeta,
    *,
    max_bytes: int = _DEFAULT_MAX_CONTENT_BYTES,
) -> bytes:
    """Load bytes for an artifact, preferring catalog extracted text."""
    if meta.extracted_text_ref:
        try:
            response = s3_client.get_object(Bucket=bucket, Key=meta.extracted_text_ref)
            return response["Body"].read()
        except Exception as exc:
            logger.warning(
                "Failed to read extracted text for %s: %s",
                s3_key,
                exc,
            )
    content, _ = get_object_range(
        s3_client,
        bucket,
        s3_key,
        offset=0,
        length=max_bytes,
    )
    return content


def read_artifact_text(
    s3_client: Any,
    bucket: str,
    s3_key: str,
    meta: CatalogObjectMeta | None = None,
    *,
    max_bytes: int = _DEFAULT_MAX_CONTENT_BYTES,
) -> str:
    """Read searchable text for a catalog object."""
    catalog_meta = meta or read_meta(s3_client, bucket, s3_key)
    if catalog_meta is None:
        raise KeyError(f"catalog metadata missing for {s3_key}")
    raw = load_artifact_content(
        s3_client,
        bucket,
        s3_key,
        catalog_meta,
        max_bytes=max_bytes,
    )
    return raw.decode("utf-8", errors="replace")
