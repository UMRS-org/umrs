"""Local directory upload and catalog → knowledge-store bridge.

Used by ``ResearchProject.ingest_directory()`` when S3 is configured:
  1. Upload recognized files under ``{project_id}/uploads/``
  2. Run ``KnowledgeIngester.ingest_project()`` (catalog enrichment)
  3. Sync READY catalog entries into the in-memory knowledge store for
     ``understand()`` (Phase A bridge until understand reads catalog directly).
"""

from __future__ import annotations

import logging
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..data_catalog.schema import CatalogObjectMeta, CatalogState
from ..data_catalog.store import get_object_range, read_meta

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

_MODALITY_TO_ARTIFACT_TYPE: dict[str, str] = {
    "notebook": "notebook",
    "code": "code",
    "document": "paper",
    "structured": "dataset",
    "text": "text",
    "image": "image",
}

_DEFAULT_MAX_CONTENT_BYTES = 10 * 1024 * 1024


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


def modality_to_artifact_type(modality: str) -> str:
    """Map catalog modality to knowledge-store artifact type."""
    return _MODALITY_TO_ARTIFACT_TYPE.get(modality, "text")


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


def load_artifact_content(
    s3_client: Any,
    bucket: str,
    s3_key: str,
    meta: CatalogObjectMeta,
    *,
    max_bytes: int = _DEFAULT_MAX_CONTENT_BYTES,
) -> bytes:
    """Load bytes for an artifact, preferring catalog extracted text."""
    if meta.extracted_text_ref:
        try:
            response = s3_client.get_object(Bucket=bucket, Key=meta.extracted_text_ref)
            return response["Body"].read()
        except Exception as exc:
            logger.warning(
                "Failed to read extracted text for %s: %s",
                s3_key,
                exc,
            )
    content, _ = get_object_range(
        s3_client,
        bucket,
        s3_key,
        offset=0,
        length=max_bytes,
    )
    return content


def artifact_from_catalog_meta(
    meta: CatalogObjectMeta,
    content: bytes,
    *,
    local_path: str | None = None,
) -> dict[str, Any]:
    """Build a knowledge-store artifact dict from catalog metadata."""
    artifact_type = modality_to_artifact_type(meta.detected_modality)
    filename = meta.s3_key.rsplit("/", 1)[-1]
    artifact_id = (
        f"{artifact_type}-{Path(filename).stem}-"
        f"{hash(meta.s3_key) & 0xFFFFFF:X}"
    )
    return {
        "artifact_id": artifact_id,
        "artifact_type": artifact_type,
        "filename": filename,
        "content": content,
        "summary": meta.summary,
        "tags": list(meta.tags),
        "metadata": {
            "path": local_path or meta.s3_key,
            "s3_key": meta.s3_key,
            "catalog_ref": meta.s3_key,
            "object_etag": meta.object_etag,
            "detected_modality": meta.detected_modality,
            "extracted_text_ref": meta.extracted_text_ref,
        },
    }


def sync_catalog_artifacts(
    s3_client: Any,
    bucket: str,
    project_id: str,
    *,
    uploads: list[UploadedObject] | None = None,
    max_content_bytes: int = _DEFAULT_MAX_CONTENT_BYTES,
) -> list[dict[str, Any]]:
    """List READY catalog objects under project uploads and build artifact dicts."""
    prefix = f"{uploads_prefix(project_id)}/"
    local_by_key = {item.s3_key: item.local_path for item in (uploads or [])}
    artifacts: list[dict[str, Any]] = []

    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            s3_key = obj["Key"]
            if s3_key.startswith("_data_catalog/"):
                continue
            meta = read_meta(s3_client, bucket, s3_key)
            if meta is None or meta.state != CatalogState.READY:
                continue
            content = load_artifact_content(
                s3_client,
                bucket,
                s3_key,
                meta,
                max_bytes=max_content_bytes,
            )
            local_path = local_by_key.get(s3_key)
            artifacts.append(
                artifact_from_catalog_meta(
                    meta,
                    content,
                    local_path=str(local_path) if local_path else None,
                )
            )
    return artifacts
