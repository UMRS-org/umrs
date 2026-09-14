"""JSON model catalog reader (ADR-053, foundation home).

Jobs' adapters lazily imported `umrs.server.model_registry_handler._load_catalog`
(a back-edge into umrs.server); the readers/writers move to the foundation
layer. The server module re-exports them for api/server callers unchanged.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from umrs.shared.config import get_resolved_umrs_subdir

logger = logging.getLogger(__name__)


def _catalog_path() -> str:
    """Return path to the JSON model catalog file."""
    return get_resolved_umrs_subdir("model_catalog.json")


def load_catalog() -> dict[str, Any]:
    """Load model catalog from JSON file."""
    path = _catalog_path()
    if not os.path.exists(path):
        return {"models": []}
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def save_catalog(catalog: dict[str, Any]) -> None:
    """Save model catalog to JSON file (format-parity with retired server variant)."""
    path = _catalog_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(catalog, fh, indent=2, ensure_ascii=False, default=str)


__all__ = ["load_catalog", "save_catalog"]
