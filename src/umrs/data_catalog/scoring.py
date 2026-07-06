"""Catalog search scoring helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .schema import CatalogObjectMeta, CatalogState


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


def score_meta(meta: CatalogObjectMeta, tokens: list[str]) -> float:
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
