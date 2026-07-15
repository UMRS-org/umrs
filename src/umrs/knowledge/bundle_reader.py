"""ExportBundleWriter — writes r4-kb v1.x bundle directory.

Produces the flat-file format described in INV-13 / TASK-041:

    <project_id>/
    ├── manifest.json
    ├── artifacts.jsonl
    ├── documents.jsonl
    ├── nodes.jsonl
    ├── edges.jsonl
    ├── embeddings.jsonl
    ├── catalog.jsonl
    └── signature.sig
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ExportBundleWriter:
    """Writes knowledge tables → flat r4-kb v1.x bundle directory."""

    def __init__(self, output_dir: Path) -> None:
        self._dir = Path(output_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    async def write_artifacts(self, rows: list[dict[str, Any]]) -> None:
        """Write artifacts.jsonl."""
        path = self._dir / "artifacts.jsonl"
        await _write_jsonl(path, rows)
        logger.debug("Wrote %d artifact rows to %s", len(rows), path)

    async def write_documents(self, rows: list[dict[str, Any]]) -> None:
        """Write documents.jsonl (IR content)."""
        path = self._dir / "documents.jsonl"
        await _write_jsonl(path, rows)
        logger.debug("Wrote %d document rows to %s", len(rows), path)

    async def write_nodes(self, rows: list[dict[str, Any]]) -> None:
        """Write nodes.jsonl."""
        path = self._dir / "nodes.jsonl"
        await _write_jsonl(path, rows)
        logger.debug("Wrote %d node rows to %s", len(rows), path)

    async def write_edges(self, rows: list[dict[str, Any]]) -> None:
        """Write edges.jsonl."""
        path = self._dir / "edges.jsonl"
        await _write_jsonl(path, rows)
        logger.debug("Wrote %d edge rows to %s", len(rows), path)

    async def write_embeddings(self, rows: list[dict[str, Any]]) -> None:
        """Write embeddings.jsonl."""
        path = self._dir / "embeddings.jsonl"
        await _write_jsonl(path, rows)
        logger.debug("Wrote %d embedding rows to %s", len(rows), path)

    async def write_catalog(self, rows: list[dict[str, Any]]) -> None:
        """Write catalog.jsonl."""
        path = self._dir / "catalog.jsonl"
        await _write_jsonl(path, rows)
        logger.debug("Wrote %d catalog rows to %s", len(rows), path)

    async def write_manifest(self, manifest: dict[str, Any]) -> None:
        """Write manifest.json."""
        path = self._dir / "manifest.json"
        content = json.dumps(manifest, sort_keys=True, indent=2)
        path.write_text(content, encoding="utf-8")
        logger.debug("Wrote manifest to %s", path)

    async def write_a2l_links(self, rows: list[dict[str, Any]]) -> None:
        """Write a2l_links.jsonl (knowledge artifact → A2L mapping)."""
        path = self._dir / "a2l_links.jsonl"
        await _write_jsonl(path, rows)
        logger.debug("Wrote %d A2L link rows to %s", len(rows), path)

    async def write_a2l_refs(self, rows: list[dict[str, Any]]) -> None:
        """Write a2l_refs.jsonl (A2L artifact identity mappings)."""
        path = self._dir / "a2l_refs.jsonl"
        await _write_jsonl(path, rows)
        logger.debug("Wrote %d A2L ref rows to %s", len(rows), path)

    async def write_signature(self, signature: str) -> None:
        """Write signature.sig."""
        path = self._dir / "signature.sig"
        path.write_text(signature, encoding="utf-8")
        logger.debug("Wrote signature to %s", path)


async def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows as newline-delimited JSON."""
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
