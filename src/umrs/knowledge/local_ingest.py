"""Local directory helpers for ``ResearchProject.ingest_directory()``.

Without S3: classify and walk local files for direct in-memory ingest.

With S3 configured (Phase B):
  1. Upload recognized files under ``{project_id}/uploads/``
  2. ``ResearchProject`` runs ``KnowledgeIngester.ingest_project()`` (catalog enrichment)
  3. ``CatalogKnowledgeStore.refresh()`` exposes READY catalog objects to
     ``understand()`` / ``verify()`` on demand — no in-memory artifact copy.
"""

from __future__ import annotations

import logging
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_SKIP_DIR_NAMES = frozenset({
    ".git",
    "__pycache__",
    "node_modules",
    ".venv",
    "venv",
    "_data_catalog",
    ".pytest_cache",
    ".mypy_cache",
    ".tox",
    ".ruff_cache",
})


@dataclass(frozen=True)
class UploadedObject:
    """A local file uploaded to project-scoped S3."""

    s3_key: str
    object_etag: str
    local_path: Path
    artifact_type: str


def uploads_prefix(project_id: str) -> str:
    """S3 key prefix for project uploads."""
    return f"{project_id}/uploads"


def should_skip_relative_path(relative: Path) -> bool:
    """Return True if a path relative to an ingest root should be skipped."""
    for part in relative.parts:
        if part in _SKIP_DIR_NAMES:
            return True
        if part.startswith(".") and part not in {".", ".."}:
            return True
    return False


def classify_local_file(file_path: Path) -> str | None:
    """Return artifact type for a local file, or None if unsupported."""
    suffix = file_path.suffix.lower()
    if suffix == ".ipynb":
        return "notebook"
    if suffix in {
        ".py", ".pyx", ".cpp", ".cc", ".cu", ".cuh", ".rs", ".go", ".rb", ".js", ".ts",
    }:
        return "code"
    if suffix in {".pdf", ".pptx", ".ppt"}:
        return "paper"
    if suffix in {".md", ".txt", ".rst"}:
        return "text"
    if suffix in {".csv", ".tsv", ".json", ".yaml", ".yml"}:
        return "dataset"
    if suffix in {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}:
        return "image"
    return None


def build_s3_key(project_id: str, base_dir: Path, file_path: Path) -> str:
    """Build a project-scoped S3 key preserving path relative to *base_dir*."""
    relative = file_path.relative_to(base_dir).as_posix()
    return f"{uploads_prefix(project_id)}/{relative}"


def iter_ingestible_files(
    *directories: str | Path,
) -> list[tuple[Path, Path]]:
    """Return ``(base_dir, file_path)`` pairs for recognized, non-skipped files."""
    pairs: list[tuple[Path, Path]] = []
    for directory in directories:
        dir_path = Path(directory)
        if not dir_path.exists():
            logger.warning("Directory does not exist: %s", dir_path)
            continue
        for file_path in sorted(dir_path.rglob("*")):
            if not file_path.is_file():
                continue
            try:
                relative = file_path.relative_to(dir_path)
            except ValueError:
                continue
            if should_skip_relative_path(relative):
                continue
            if classify_local_file(file_path) is None:
                continue
            pairs.append((dir_path, file_path))
    return pairs


def upload_local_files(
    s3_client: Any,
    bucket: str,
    project_id: str,
    *directories: str | Path,
) -> list[UploadedObject]:
    """Upload ingestible local files to S3. Returns metadata for each upload."""
    uploaded: list[UploadedObject] = []
    for base_dir, file_path in iter_ingestible_files(*directories):
        artifact_type = classify_local_file(file_path)
        if artifact_type is None:
            continue
        s3_key = build_s3_key(project_id, base_dir, file_path)
        content = file_path.read_bytes()
        content_type, _ = mimetypes.guess_type(file_path.name)
        response = s3_client.put_object(
            Bucket=bucket,
            Key=s3_key,
            Body=content,
            ContentType=content_type or "application/octet-stream",
            Metadata={
                "project_id": project_id,
                "filename": file_path.name,
            },
        )
        object_etag = (response.get("ETag") or "").strip('"')
        uploaded.append(
            UploadedObject(
                s3_key=s3_key,
                object_etag=object_etag,
                local_path=file_path,
                artifact_type=artifact_type,
            )
        )
    return uploaded
