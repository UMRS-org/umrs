"""Catalog metadata → knowledge artifact conversion helpers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .schema import CatalogObjectMeta

if TYPE_CHECKING:
    from .pg_store import CatalogMetaStore
from .store import get_object_range

logger = logging.getLogger(__name__)

_DEFAULT_MAX_CONTENT_BYTES = 10 * 1024 * 1024

# S3 codes for an object that is not there (head_object answers a bare 404).
_MISSING_OBJECT_CODES = frozenset({"404", "NoSuchKey", "NotFound"})


class CatalogObjectMissing(LookupError):
    """A catalogued object is gone from the object store (deleted out of
    band): the catalog row outlived its bytes. Callers answer it honestly
    instead of as a 500 (cycle-3 N6)."""

    def __init__(self, s3_key: str) -> None:
        super().__init__(
            f"Catalogued object {s3_key!r} is no longer in the object store "
            "(deleted outside UMRS); upload it again, or delete it "
            "(DELETE /umrs/files/{key}) to drop its catalog entry"
        )
        self.s3_key = s3_key


def _is_missing_object(exc: Exception) -> bool:
    error = getattr(exc, "response", None)
    if not isinstance(error, dict):
        return False
    return str(error.get("Error", {}).get("Code")) in _MISSING_OBJECT_CODES


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
    try:
        content, _ = get_object_range(
            s3_client,
            bucket,
            s3_key,
            offset=0,
            length=max_bytes,
        )
    except Exception as exc:
        if _is_missing_object(exc):
            raise CatalogObjectMissing(s3_key) from exc
        raise
    return content


def read_artifact_text(
    s3_client: Any,
    bucket: str,
    s3_key: str,
    meta: CatalogObjectMeta | None = None,
    *,
    max_bytes: int = _DEFAULT_MAX_CONTENT_BYTES,
    meta_store: CatalogMetaStore | None = None,
) -> str:
    """Read searchable text for a catalog object."""
    catalog_meta = meta
    if catalog_meta is None:
        if meta_store is None:
            from umrs.api.catalog_store import get_catalog_meta_store

            meta_store = get_catalog_meta_store()
        project_id = s3_key.split("/", 1)[0] if "/" in s3_key else ""
        catalog_meta = meta_store.get(project_id, s3_key)
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
