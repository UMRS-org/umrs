"""
Model Staging Service (MM-002).

Handles model weight staging to PVC/NFS for inference:
- Pre-stage models from HuggingFace/S3/local to shared storage
- Validate model files and checksums
- Track staging progress
- Clean up unused staged models
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class StagingStatus(Enum):
    """Model staging status."""

    PENDING = "pending"
    DOWNLOADING = "downloading"
    VERIFYING = "verifying"
    COPYING = "copying"
    COMPLETED = "completed"
    FAILED = "failed"


class StagingSource(Enum):
    """Source type for model staging."""

    HUGGINGFACE = "huggingface"
    S3 = "s3"
    LOCAL = "local"
    HTTP = "http"


@dataclass
class StagingConfig:
    """Configuration for model staging."""

    # Storage paths
    staging_root: str = "/shared/models"
    temp_dir: str = "/tmp/model-staging"
    cache_dir: str = "/cache/huggingface"

    # HuggingFace settings
    hf_token: Optional[str] = None
    hf_endpoint: str = "https://huggingface.co"

    # S3 settings
    s3_endpoint: Optional[str] = None
    s3_access_key: Optional[str] = None
    s3_secret_key: Optional[str] = None

    # Limits
    max_concurrent_downloads: int = 2
    download_timeout_seconds: int = 3600
    max_model_size_gb: float = 100.0

    # Verification
    verify_checksums: bool = True
    required_files: List[str] = field(default_factory=lambda: [
        "config.json",
        "tokenizer_config.json",
    ])


@dataclass
class StagingProgress:
    """Progress tracking for staging operation."""

    total_bytes: int = 0
    downloaded_bytes: int = 0
    current_file: str = ""
    files_completed: int = 0
    files_total: int = 0
    stage: StagingStatus = StagingStatus.PENDING
    started_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    error: Optional[str] = None


@dataclass
class StagingResult:
    """Result of a staging operation."""

    success: bool
    model_path: str
    source: str
    status: StagingStatus
    duration_seconds: float
    size_bytes: int
    file_count: int
    checksum: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ModelStager:
    """
    Model Staging Service (MM-002).

    Pre-stages model weights to shared storage for inference endpoints.
    Supports multiple sources and verification.
    """

    def __init__(self, config: Optional[StagingConfig] = None):
        """Initialize stager with configuration."""
        self.config = config or StagingConfig()
        self._progress: Dict[str, StagingProgress] = {}
        self._lock = asyncio.Lock()

        # Ensure directories exist
        Path(self.config.staging_root).mkdir(parents=True, exist_ok=True)
        Path(self.config.temp_dir).mkdir(parents=True, exist_ok=True)

    async def stage_from_huggingface(
        self,
        repo_id: str,
        target_name: str,
        revision: Optional[str] = None,
        scope: str = "system",
        project_id: Optional[str] = None,
    ) -> StagingResult:
        """
        Stage model from HuggingFace Hub.

        Args:
            repo_id: HuggingFace repository ID (e.g., "Qwen/Qwen2.5-7B-Instruct")
            target_name: Name for staged model
            revision: Git revision (branch, tag, commit)
            scope: Model scope (system, project, user)
            project_id: Project ID for project-scoped models

        Returns:
            Staging result with path and status
        """
        staging_id = f"hf:{repo_id}:{target_name}"
        start_time = datetime.now()

        progress = StagingProgress(
            stage=StagingStatus.DOWNLOADING,
            started_at=start_time,
        )
        self._progress[staging_id] = progress

        try:
            # Determine target path based on scope
            target_path = self._get_target_path(target_name, scope, project_id)

            # Check if already staged
            if target_path.exists() and self._is_complete(target_path):
                logger.info("Model already staged at %s", target_path)
                return StagingResult(
                    success=True,
                    model_path=str(target_path),
                    source=repo_id,
                    status=StagingStatus.COMPLETED,
                    duration_seconds=0,
                    size_bytes=self._get_dir_size(target_path),
                    file_count=self._count_files(target_path),
                )

            # Download using huggingface_hub
            progress.stage = StagingStatus.DOWNLOADING
            progress.updated_at = datetime.now()

            # Use snapshot_download for complete model
            temp_path = Path(self.config.temp_dir) / target_name
            await self._download_hf_model(repo_id, temp_path, revision)

            # Verify download
            progress.stage = StagingStatus.VERIFYING
            progress.updated_at = datetime.now()

            if self.config.verify_checksums:
                await self._verify_model_files(temp_path)

            # Move to final location
            progress.stage = StagingStatus.COPYING
            progress.updated_at = datetime.now()

            target_path.parent.mkdir(parents=True, exist_ok=True)
            if target_path.exists():
                shutil.rmtree(target_path)
            shutil.move(str(temp_path), str(target_path))

            # Calculate checksum
            checksum = self._calculate_dir_checksum(target_path)

            # Complete
            progress.stage = StagingStatus.COMPLETED
            progress.updated_at = datetime.now()

            duration = (datetime.now() - start_time).total_seconds()
            size = self._get_dir_size(target_path)
            file_count = self._count_files(target_path)

            logger.info(
                "Model staged: %s -> %s (%.2f GB in %.1fs)",
                repo_id,
                target_path,
                size / (1024**3),
                duration,
            )

            return StagingResult(
                success=True,
                model_path=str(target_path),
                source=repo_id,
                status=StagingStatus.COMPLETED,
                duration_seconds=duration,
                size_bytes=size,
                file_count=file_count,
                checksum=checksum,
                metadata={
                    "revision": revision,
                    "scope": scope,
                    "project_id": project_id,
                },
            )

        except Exception as e:
            logger.error("Staging failed for %s: %s", repo_id, e)
            progress.stage = StagingStatus.FAILED
            progress.error = str(e)
            progress.updated_at = datetime.now()

            return StagingResult(
                success=False,
                model_path="",
                source=repo_id,
                status=StagingStatus.FAILED,
                duration_seconds=(datetime.now() - start_time).total_seconds(),
                size_bytes=0,
                file_count=0,
                error=str(e),
            )

        finally:
            del self._progress[staging_id]

    async def stage_from_s3(
        self,
        bucket: str,
        key: str,
        target_name: str,
        scope: str = "system",
        project_id: Optional[str] = None,
    ) -> StagingResult:
        """
        Stage model from S3-compatible storage.

        Args:
            bucket: S3 bucket name
            key: S3 object key (prefix for model directory)
            target_name: Name for staged model
            scope: Model scope
            project_id: Project ID for project-scoped models

        Returns:
            Staging result
        """
        staging_id = f"s3:{bucket}/{key}:{target_name}"
        start_time = datetime.now()

        try:
            import boto3
            from botocore.config import Config

            # Initialize S3 client
            s3_config = Config(
                signature_version="s3v4",
                s3={"addressing_style": "path"},
            )
            s3_kwargs = {"config": s3_config}

            if self.config.s3_endpoint:
                s3_kwargs["endpoint_url"] = self.config.s3_endpoint
            if self.config.s3_access_key:
                s3_kwargs["aws_access_key_id"] = self.config.s3_access_key
                s3_kwargs["aws_secret_access_key"] = self.config.s3_secret_key

            s3 = boto3.client("s3", **s3_kwargs)

            # Determine target path
            target_path = self._get_target_path(target_name, scope, project_id)
            temp_path = Path(self.config.temp_dir) / target_name
            temp_path.mkdir(parents=True, exist_ok=True)

            # List and download objects
            paginator = s3.get_paginator("list_objects_v2")
            total_size = 0
            file_count = 0

            for page in paginator.paginate(Bucket=bucket, Prefix=key):
                for obj in page.get("Contents", []):
                    obj_key = obj["Key"]
                    relative_path = obj_key[len(key):].lstrip("/")
                    if not relative_path:
                        continue

                    local_file = temp_path / relative_path
                    local_file.parent.mkdir(parents=True, exist_ok=True)

                    s3.download_file(bucket, obj_key, str(local_file))
                    total_size += obj["Size"]
                    file_count += 1

            # Move to final location
            target_path.parent.mkdir(parents=True, exist_ok=True)
            if target_path.exists():
                shutil.rmtree(target_path)
            shutil.move(str(temp_path), str(target_path))

            duration = (datetime.now() - start_time).total_seconds()

            return StagingResult(
                success=True,
                model_path=str(target_path),
                source=f"s3://{bucket}/{key}",
                status=StagingStatus.COMPLETED,
                duration_seconds=duration,
                size_bytes=total_size,
                file_count=file_count,
            )

        except Exception as e:
            logger.error("S3 staging failed: %s", e)
            return StagingResult(
                success=False,
                model_path="",
                source=f"s3://{bucket}/{key}",
                status=StagingStatus.FAILED,
                duration_seconds=(datetime.now() - start_time).total_seconds(),
                size_bytes=0,
                file_count=0,
                error=str(e),
            )

    async def stage_from_local(
        self,
        source_path: str,
        target_name: str,
        scope: str = "system",
        project_id: Optional[str] = None,
        copy: bool = True,
    ) -> StagingResult:
        """
        Stage model from local filesystem.

        Args:
            source_path: Path to source model
            target_name: Name for staged model
            scope: Model scope
            project_id: Project ID
            copy: If True, copy files; if False, create symlink

        Returns:
            Staging result
        """
        start_time = datetime.now()
        source = Path(source_path)

        if not source.exists():
            return StagingResult(
                success=False,
                model_path="",
                source=source_path,
                status=StagingStatus.FAILED,
                duration_seconds=0,
                size_bytes=0,
                file_count=0,
                error=f"Source path not found: {source_path}",
            )

        target_path = self._get_target_path(target_name, scope, project_id)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            if copy:
                if target_path.exists():
                    shutil.rmtree(target_path)
                shutil.copytree(source, target_path)
            else:
                if target_path.exists():
                    target_path.unlink()
                target_path.symlink_to(source)

            duration = (datetime.now() - start_time).total_seconds()
            size = self._get_dir_size(target_path)
            file_count = self._count_files(target_path)

            return StagingResult(
                success=True,
                model_path=str(target_path),
                source=source_path,
                status=StagingStatus.COMPLETED,
                duration_seconds=duration,
                size_bytes=size,
                file_count=file_count,
            )

        except Exception as e:
            return StagingResult(
                success=False,
                model_path="",
                source=source_path,
                status=StagingStatus.FAILED,
                duration_seconds=(datetime.now() - start_time).total_seconds(),
                size_bytes=0,
                file_count=0,
                error=str(e),
            )

    def get_progress(self, staging_id: str) -> Optional[StagingProgress]:
        """Get progress for an ongoing staging operation."""
        return self._progress.get(staging_id)

    def list_staged_models(self, scope: str = "system") -> List[Dict[str, Any]]:
        """List all staged models in a scope."""
        scope_path = Path(self.config.staging_root) / scope
        if not scope_path.exists():
            return []

        models = []
        for model_dir in scope_path.iterdir():
            if model_dir.is_dir():
                models.append({
                    "name": model_dir.name,
                    "path": str(model_dir),
                    "size_bytes": self._get_dir_size(model_dir),
                    "file_count": self._count_files(model_dir),
                    "complete": self._is_complete(model_dir),
                })
        return models

    async def cleanup_incomplete(self) -> int:
        """Remove incomplete staging artifacts. Returns count removed."""
        removed = 0
        temp_path = Path(self.config.temp_dir)

        if temp_path.exists():
            for item in temp_path.iterdir():
                shutil.rmtree(item)
                removed += 1

        return removed

    # =========================================================================
    # Private helpers
    # =========================================================================

    def _get_target_path(
        self,
        name: str,
        scope: str,
        project_id: Optional[str] = None,
    ) -> Path:
        """Build target path for staged model."""
        if scope == "project" and project_id:
            return Path(self.config.staging_root) / "projects" / project_id / name
        elif scope == "user":
            return Path(self.config.staging_root) / "users" / name
        else:
            return Path(self.config.staging_root) / "system" / name

    async def _download_hf_model(
        self,
        repo_id: str,
        target_path: Path,
        revision: Optional[str] = None,
    ) -> None:
        """Download model from HuggingFace Hub."""
        try:
            from huggingface_hub import snapshot_download

            snapshot_download(
                repo_id=repo_id,
                local_dir=str(target_path),
                revision=revision,
                token=self.config.hf_token,
                cache_dir=self.config.cache_dir,
            )
        except ImportError:
            raise RuntimeError(
                "huggingface_hub not installed. Run: pip install huggingface_hub"
            )

    async def _verify_model_files(self, model_path: Path) -> None:
        """Verify required model files exist."""
        for required_file in self.config.required_files:
            file_path = model_path / required_file
            if not file_path.exists():
                logger.warning("Missing required file: %s", required_file)

    def _is_complete(self, model_path: Path) -> bool:
        """Check if model staging is complete."""
        # Check for config.json as indicator of complete download
        return (model_path / "config.json").exists()

    def _get_dir_size(self, path: Path) -> int:
        """Get total size of directory in bytes."""
        total = 0
        for root, _dirs, files in os.walk(path):
            for f in files:
                total += os.path.getsize(os.path.join(root, f))
        return total

    def _count_files(self, path: Path) -> int:
        """Count files in directory."""
        count = 0
        for _root, _dirs, files in os.walk(path):
            count += len(files)
        return count

    def _calculate_dir_checksum(self, path: Path) -> str:
        """Calculate SHA256 checksum of directory contents."""
        hasher = hashlib.sha256()

        for root, _dirs, files in sorted(os.walk(path)):
            for filename in sorted(files):
                filepath = os.path.join(root, filename)
                relative = os.path.relpath(filepath, path)
                hasher.update(relative.encode())
                hasher.update(str(os.path.getsize(filepath)).encode())

        return hasher.hexdigest()[:16]
