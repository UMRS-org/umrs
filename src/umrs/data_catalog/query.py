"""Catalog search and read (Postgres metadata + S3 bytes)."""

from __future__ import annotations

from typing import Any

from .pg_store import CatalogMetaStore
from .scoring import CatalogHit, allowed, score_meta, tokenize
from .store import get_object_range

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
) -> list[CatalogHit]:
    """Lexical search over catalog metadata with project ACL filtering."""
    return meta_store.search(project_id, query, limit=limit)


def read_catalog_object(
    s3_client: Any,
    bucket: str,
    meta_store: CatalogMetaStore,
    *,
    s3_key: str,
    project_id: str,
    offset: int = 0,
    length: int = 1_048_576,
) -> dict[str, Any]:
    """
    Read raw object bytes (bounded) after ACL check.

    Returns dict with content metadata; text decoded when possible.
    """
    meta = meta_store.get(project_id, s3_key)
    if meta is None or not allowed(meta, project_id):
        raise PermissionError("access denied for object")

    if not s3_key.startswith(f"{project_id}/"):
        raise PermissionError("access denied for object")

    chunk, total = get_object_range(
        s3_client,
        bucket,
        s3_key,
        offset=offset,
        length=length,
    )
    truncated = offset + len(chunk) < total

    text: str | None = None
    try:
        text = chunk.decode("utf-8")
    except UnicodeDecodeError:
        text = None

    return {
        "s3_key": s3_key,
        "offset": offset,
        "length": len(chunk),
        "total_size": total,
        "truncated": truncated,
        "text": text,
        "meta": meta.to_dict(),
    }
