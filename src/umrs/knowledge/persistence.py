"""Local persistence helpers for Phase 4 ``ExtractedKnowledge``."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any
from urllib.parse import quote

logger = logging.getLogger(__name__)


def extracted_knowledge_dir(storage_path: str | Path) -> Path:
    """Directory for per-artifact structured knowledge JSON files."""
    return Path(storage_path) / "extracted_knowledge"


def _artifact_filename(artifact_id: str) -> str:
    return f"{quote(artifact_id, safe='')}.json"


def write_local_extracted_knowledge(
    persistence_dir: str | Path,
    artifact_id: str,
    knowledge: dict[str, Any],
) -> Path:
    """Persist structured knowledge to disk; return the file path."""
    directory = Path(persistence_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / _artifact_filename(artifact_id)
    path.write_text(
        json.dumps(knowledge, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path


def read_local_extracted_knowledge(
    persistence_dir: str | Path,
    artifact_id: str,
) -> dict[str, Any] | None:
    """Load structured knowledge from disk, or None if missing."""
    path = Path(persistence_dir) / _artifact_filename(artifact_id)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to read local extracted knowledge %s: %s", path, exc)
        return None
    return payload if isinstance(payload, dict) else None
