"""Catalog search and read (MVP: grep over ``.meta.json`` files)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from botocore.exceptions import ClientError

from .constants import OBJECTS_PREFIX
from .rollup import _list_object_metas
from .schema import CatalogObjectMeta, CatalogState
from .store import get_object_range, read_meta


@dataclass
class CatalogHit:
    """One catalog search result."""

    meta: CatalogObjectMeta
    score: float


def search_catalog(
    s3_client: Any,
    bucket: str,
    *,
    query: str,
    project_id: str,
    limit: int = 20,
) -> list[CatalogHit]:
    """Lexical grep over catalog metadata with project ACL filtering."""
    tokens = _tokenize(query)
    if not tokens:
        return []

    hits: list[CatalogHit] = []
    for meta in _list_object_metas(s3_client, bucket):
        if not _allowed(meta, project_id):
            continue
        if meta.state == CatalogState.ERROR:
            continue
        score = _score_meta(meta, tokens)
        if score <= 0:
            continue
        hits.append(CatalogHit(meta=meta, score=score))

    hits.sort(key=lambda h: h.score, reverse=True)
    return hits[:limit]


def read_catalog_object(
    s3_client: Any,
    bucket: str,
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
    meta = read_meta(s3_client, bucket, s3_key)
    if meta is None or not _allowed(meta, project_id):
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


def _allowed(meta: CatalogObjectMeta, project_id: str) -> bool:
    if meta.acl_scope == project_id:
        return True
    return meta.s3_key.startswith(f"{project_id}/")


def _tokenize(query: str) -> list[str]:
    return [t for t in re.split(r"\W+", query.lower()) if t]


def _score_meta(meta: CatalogObjectMeta, tokens: list[str]) -> float:
    haystack = " ".join(
        [
            meta.summary,
            meta.deterministic_summary,
            meta.perception_summary or "",
            meta.detected_modality,
            meta.s3_key,
            " ".join(meta.tags),
            " ".join(meta.filename_signals),
        ]
    ).lower()
    if not haystack:
        return 0.0
    score = 0.0
    for token in tokens:
        if token in haystack:
            score += 1.0
        for signal in meta.filename_signals:
            if token in signal.lower():
                score += 0.5
    if meta.state == CatalogState.PENDING:
        score *= 0.5
    return score
