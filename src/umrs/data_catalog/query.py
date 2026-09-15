"""Catalog search and read (Postgres metadata + S3 bytes)."""

from __future__ import annotations

from typing import Any, Literal

from .pg_store import CatalogMetaStore
from .scoring import CatalogHit, allowed, score_meta, tokenize
from .store import get_object_range

ReadSource = Literal["auto", "raw", "extracted"]

# Backward-compatible aliases for tests
_allowed = allowed
_score_meta = score_meta
_tokenize = tokenize


def search_catalog(
    meta_store: CatalogMetaStore,
    *,
    query: str,
    project_id: str,
    limit: int = 20,
    modality: str | None = None,
    state: str | None = None,
    tag: str | None = None,
    prefix: str | None = None,
) -> list[CatalogHit]:
    """Lexical search over catalog metadata with project ACL filtering."""
    return meta_store.search(
        project_id,
        query,
        limit=limit,
        modality=modality,
        state=state,
        tag=tag,
        prefix=prefix,
    )


def resolve_read_source(meta, source: ReadSource) -> Literal["raw", "extracted"]:
    """Pick whether to read raw object bytes or the extracted-text sidecar."""
    if source == "raw":
        return "raw"
    if source == "extracted":
        return "extracted"
    if meta.detected_modality == "document" and meta.extracted_text_ref:
        return "extracted"
    return "raw"


def read_catalog_object(
    s3_client: Any,
    bucket: str,
    meta_store: CatalogMetaStore,
    *,
    s3_key: str,
    project_id: str,
    offset: int = 0,
    length: int = 1_048_576,
    source: ReadSource = "auto",
) -> dict[str, Any]:
    """
    Read catalog object content (bounded) after ACL check.

    ``source=auto`` reads extracted text for documents when ``extracted_text_ref``
    exists; otherwise reads raw object bytes.
    """
    meta = meta_store.get(project_id, s3_key)
    if meta is None or not allowed(meta, project_id):
        raise PermissionError("access denied for object")

    if not s3_key.startswith(f"{project_id}/"):
        raise PermissionError("access denied for object")

    resolved = resolve_read_source(meta, source)
    if resolved == "extracted":
        if not meta.extracted_text_ref:
            raise ValueError("no extracted text available for object")
        read_key = meta.extracted_text_ref
    else:
        read_key = s3_key

    chunk, total = get_object_range(
        s3_client,
        bucket,
        read_key,
        offset=offset,
        length=length,
    )
    truncated = offset + len(chunk) < total

    text: str | None = None
    if resolved == "extracted":
        text = chunk.decode("utf-8")
    else:
        try:
            text = chunk.decode("utf-8")
        except UnicodeDecodeError:
            text = None

    payload: dict[str, Any] = {
        "s3_key": s3_key,
        "offset": offset,
        "length": len(chunk),
        "total_size": total,
        "truncated": truncated,
        "text": text,
        "read_source": resolved,
        "meta": meta.to_dict(),
    }
    if resolved == "extracted":
        payload["extracted_text_ref"] = meta.extracted_text_ref
    return payload
