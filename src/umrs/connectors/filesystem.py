"""Filesystem connector — indexes local/mounted directories.

This is the simplest connector: walk a directory tree, read files,
yield documents. Useful for HPC shared filesystems.
"""

from __future__ import annotations

import logging
import mimetypes
import os
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

from . import (
    Connector,
    ConnectorConfig,
    ConnectorType,
    Document,
    SyncCheckpoint,
    register_connector,
)

logger = logging.getLogger(__name__)

# Default extensions to index
_DEFAULT_EXTENSIONS = {
    ".txt",
    ".md",
    ".rst",
    ".py",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".cfg",
    ".ini",
    ".sh",
    ".bash",
    ".csv",
    ".tsv",
    ".html",
    ".htm",
    ".xml",
    ".tex",
    ".bib",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
    ".java",
    ".js",
    ".ts",
    ".go",
    ".rs",
    ".rb",
    ".r",
    ".jl",
    ".f90",
    ".f95",
    ".f03",
}

_MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB default


class FilesystemConnector(Connector):
    """Walk a local directory and yield documents."""

    @property
    def connector_type(self) -> ConnectorType:
        return ConnectorType.FILESYSTEM

    @property
    def name(self) -> str:
        return "Filesystem"

    def validate_config(self, config: ConnectorConfig) -> list[str]:
        errors: list[str] = []
        root = config.settings.get("root_path")
        if not root:
            errors.append("root_path is required in settings")
        elif not Path(root).is_dir():
            errors.append(f"root_path does not exist: {root}")
        return errors

    def fetch(
        self,
        config: ConnectorConfig,
        checkpoint: SyncCheckpoint | None = None,
    ) -> Iterator[Document]:
        root = Path(config.settings["root_path"])
        extensions = set(config.settings.get("extensions", _DEFAULT_EXTENSIONS))
        max_size = config.settings.get("max_file_size", _MAX_FILE_SIZE)
        last_sync = checkpoint.last_sync if checkpoint else None

        for dirpath, _dirs, filenames in os.walk(root):
            for fname in filenames:
                fpath = Path(dirpath) / fname
                if fpath.suffix.lower() not in extensions:
                    continue
                try:
                    stat = fpath.stat()
                except OSError:
                    continue

                if stat.st_size > max_size:
                    continue

                mtime = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
                if last_sync and mtime <= last_sync:
                    continue

                try:
                    content = fpath.read_text(encoding="utf-8", errors="replace")
                except OSError as exc:
                    logger.warning("Cannot read %s: %s", fpath, exc)
                    continue

                mime = mimetypes.guess_type(str(fpath))[0] or "text/plain"
                rel = fpath.relative_to(root)
                yield Document(
                    source_type=ConnectorType.FILESYSTEM,
                    source_id=str(rel),
                    title=str(rel),
                    content=content,
                    content_type=mime,
                    metadata={"absolute_path": str(fpath)},
                    last_modified=mtime,
                    size_bytes=stat.st_size,
                )

    def health_check(self, config: ConnectorConfig) -> bool:
        root = config.settings.get("root_path", "")
        return Path(root).is_dir()


register_connector(FilesystemConnector())
