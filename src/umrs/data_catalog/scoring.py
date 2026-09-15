"""Catalog search scoring helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .schema import CatalogObjectMeta, CatalogState

_STRUCTURED_QUERY_HINTS = frozenset(
    {"csv", "tsv", "json", "column", "columns", "table", "batch", "yield", "data"}
)


@dataclass
class CatalogHit:
    """One catalog search result."""

    meta: CatalogObjectMeta
    score: float


def allowed(meta: CatalogObjectMeta, project_id: str) -> bool:
    if meta.acl_scope == project_id:
        return True
    return meta.s3_key.startswith(f"{project_id}/")


def tokenize(query: str) -> list[str]:
    return [t for t in re.split(r"\W+", query.lower()) if t]


def score_meta(
    meta: CatalogObjectMeta,
    tokens: list[str],
    *,
    query: str = "",
) -> float:
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
    tag_blob = " ".join(meta.tags).lower()
    for token in tokens:
        if token in haystack:
            score += 1.0
        if token in tag_blob:
            score += 0.75
        for signal in meta.filename_signals:
            signal_lower = signal.lower()
            if token in signal_lower:
                score += 0.5
            stem = signal_lower.rsplit(".", 1)[0]
            if token in stem.replace("-", " ").replace("_", " ").split():
                score += 0.5

    if meta.detected_modality == "structured":
        query_lower = query.lower()
        if any(hint in query_lower for hint in _STRUCTURED_QUERY_HINTS):
            score += 0.5

    if meta.state == CatalogState.READY:
        score += 0.25
    elif meta.state == CatalogState.PENDING:
        score *= 0.5

    return score
