"""
Model Download Backend Adapter (U-821 - U-825).

Provides:
- ModelDownloadBackend: Submits model download jobs
- ModelDownloadStatusChecker: Tracks download progress
- HuggingFace download with token support
- Auto-registration in model catalog on completion
- Progress tracking in job_attempts.metrics

See DEVELOPMENT_OBJECTIVES.md Section 1.9 for design details.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

from .models import AttemptStatus, Job, JobAttempt, JobType
from .monitor import AttemptCheckResult, BackendStatusChecker
from .scheduler import OrchestratorBackend

if TYPE_CHECKING:
    from umrs.model_registry.models import ModelEntry, ModelScope, ServingProfile
    from umrs.model_registry.store import AccessContext, ModelStore, PathValidator

logger = logging.getLogger(__name__)


# =============================================================================
# Download Source Types
# =============================================================================


class DownloadSource(Enum):
    """Supported model download sources."""

    HUGGINGFACE = "huggingface"  # HuggingFace Hub
    S3 = "s3"  # S3-compatible storage
    HTTP = "http"  # Direct HTTP/HTTPS download
    LOCAL = "local"  # Copy from local path (admin only)


# =============================================================================
# Download Progress Tracking
# =============================================================================


@dataclass
class DownloadProgress:
    """Tracks download progress for a model download job."""

    total_files: int = 0
    downloaded_files: int = 0
    total_bytes: int = 0
    downloaded_bytes: int = 0
    current_file: Optional[str] = None
    started_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    error_message: Optional[str] = None
    status: str = "pending"  # pending, downloading, completed, failed

    @property
    def progress_percent(self) -> float:
        """Calculate download progress percentage."""
        if self.total_bytes == 0:
            return 0.0
        return min(100.0, (self.downloaded_bytes / self.total_bytes) * 100)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "total_files": self.total_files,
            "downloaded_files": self.downloaded_files,
            "total_bytes": self.total_bytes,
            "downloaded_bytes": self.downloaded_bytes,
            "current_file": self.current_file,
            "started_at": (
                self.started_at.isoformat() if self.started_at else None
            ),
            "updated_at": (
                self.updated_at.isoformat() if self.updated_at else None
            ),
            "progress_percent": self.progress_percent,
            "error_message": self.error_message,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DownloadProgress":
        """Create from dictionary."""
        return cls(
            total_files=data.get("total_files", 0),
            downloaded_files=data.get("downloaded_files", 0),
            total_bytes=data.get("total_bytes", 0),
            downloaded_bytes=data.get("downloaded_bytes", 0),
            current_file=data.get("current_file"),
            started_at=(
                datetime.fromisoformat(data["started_at"])
                if data.get("started_at")
                else None
            ),
            updated_at=(
                datetime.fromisoformat(data["updated_at"])
                if data.get("updated_at")
                else None
            ),
            error_message=data.get("error_message"),
            status=data.get("status", "pending"),
        )


# =============================================================================
# Download Task Context
# =============================================================================


@dataclass
class DownloadTaskContext:
    """Context for a download task (stored during async execution)."""

    job_id: str
    attempt_id: str
    source: DownloadSource
    source_id: str  # e.g., "meta-llama/Llama-3.2-8B"
    target_path: str
    model_name: str
    scope: str  # "system", "project", "user"
    owner_id: str  # project_id or user_id
    progress: DownloadProgress = field(default_factory=DownloadProgress)
    hf_token: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "job_id": self.job_id,
            "attempt_id": self.attempt_id,
            "source": self.source.value,
            "source_id": self.source_id,
            "target_path": self.target_path,
            "model_name": self.model_name,
            "scope": self.scope,
            "owner_id": self.owner_id,
            "progress": self.progress.to_dict(),
            "hf_token_set": bool(self.hf_token),  # Don't store token
            "metadata": self.metadata,
        }


# =============================================================================
# In-Memory Task Registry (for async download tracking)
# =============================================================================


class DownloadTaskRegistry:
    """
    Registry for tracking in-flight download tasks.

    Downloads run in background threads/processes; this registry
    provides status lookup for the status checker.
    """

    def __init__(self):
        self._tasks: Dict[str, DownloadTaskContext] = {}
        self._lock = asyncio.Lock()

    async def register(self, ctx: DownloadTaskContext) -> None:
        """Register a new download task."""
        async with self._lock:
            self._tasks[ctx.attempt_id] = ctx

    async def get(self, attempt_id: str) -> Optional[DownloadTaskContext]:
        """Get task context by attempt ID."""
        async with self._lock:
            return self._tasks.get(attempt_id)

    async def update_progress(
        self,
        attempt_id: str,
        progress: DownloadProgress,
    ) -> None:
        """Update progress for a task."""
        async with self._lock:
            if attempt_id in self._tasks:
                self._tasks[attempt_id].progress = progress

    async def remove(self, attempt_id: str) -> Optional[DownloadTaskContext]:
        """Remove and return a completed task."""
        async with self._lock:
            return self._tasks.pop(attempt_id, None)

    async def list_tasks(self) -> List[DownloadTaskContext]:
        """List all active tasks."""
        async with self._lock:
            return list(self._tasks.values())


# Global registry (singleton pattern)
_download_registry: Optional[DownloadTaskRegistry] = None


def get_download_registry() -> DownloadTaskRegistry:
    """Get or create the global download registry."""
    global _download_registry
    if _download_registry is None:
        _download_registry = DownloadTaskRegistry()
    return _download_registry


# =============================================================================
# HuggingFace Download Implementation
# =============================================================================


async def download_from_huggingface(
    repo_id: str,
    target_path: str,
    token: Optional[str] = None,
    revision: Optional[str] = None,
    progress_callback: Optional[Callable[[DownloadProgress], None]] = None,
) -> DownloadProgress:
    """
    Download a model from HuggingFace Hub.

    Args:
        repo_id: HuggingFace repository ID (e.g., "meta-llama/Llama-3.2-8B")
        target_path: Local path to download to
        token: HuggingFace API token (for gated models)
        revision: Specific revision/branch (default: main)
        progress_callback: Optional callback for progress updates

    Returns:
        DownloadProgress with final status

    Raises:
        RuntimeError: If download fails
    """
    progress = DownloadProgress(
        started_at=datetime.now(timezone.utc),
        status="downloading",
    )

    try:
        # Import huggingface_hub lazily to avoid startup cost
        from huggingface_hub import HfApi, snapshot_download
        from huggingface_hub.utils import (
            GatedRepoError,
            RepositoryNotFoundError,
            RevisionNotFoundError,
        )

    except ImportError as e:
        progress.status = "failed"
        progress.error_message = (
            "huggingface_hub not installed. "
            "Install with: pip install huggingface_hub"
        )
        return progress

    try:
        # Initialize HF API
        api = HfApi(token=token)

        # Get repository info for size estimation
        try:
            repo_info = api.repo_info(
                repo_id=repo_id,
                revision=revision,
                token=token,
            )

            # Estimate total size from siblings (files)
            if hasattr(repo_info, "siblings") and repo_info.siblings:
                progress.total_files = len(repo_info.siblings)
                progress.total_bytes = sum(
                    getattr(f, "size", 0) or 0 for f in repo_info.siblings
                )

        except RepositoryNotFoundError:
            progress.status = "failed"
            progress.error_message = f"Repository not found: {repo_id}"
            return progress
        except GatedRepoError:
            progress.status = "failed"
            progress.error_message = (
                f"Repository '{repo_id}' is gated. "
                "Provide a valid HuggingFace token with access."
            )
            return progress

        # Create target directory
        os.makedirs(target_path, exist_ok=True)

        # Track progress via callback
        def _progress_callback(
            file_name: str,
            downloaded_bytes: int,
            total_bytes: int,
        ):
            progress.current_file = file_name
            progress.downloaded_bytes = downloaded_bytes
            progress.updated_at = datetime.now(timezone.utc)
            if progress_callback:
                progress_callback(progress)

        # Download model snapshot
        # Note: huggingface_hub doesn't support per-file progress callback directly
        # We use the default behavior and update progress after completion
        local_dir = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: snapshot_download(
                repo_id=repo_id,
                local_dir=target_path,
                token=token,
                revision=revision,
                local_dir_use_symlinks=False,  # Copy files, don't symlink
            ),
        )

        # Calculate final size
        total_size = 0
        file_count = 0
        for root, _dirs, files in os.walk(target_path):
            for f in files:
                file_path = os.path.join(root, f)
                total_size += os.path.getsize(file_path)
                file_count += 1

        progress.downloaded_files = file_count
        progress.downloaded_bytes = total_size
        progress.total_files = file_count
        progress.total_bytes = total_size
        progress.status = "completed"
        progress.updated_at = datetime.now(timezone.utc)

        logger.info(
            f"Downloaded {repo_id} to {target_path}: "
            f"{file_count} files, {total_size:,} bytes"
        )

        return progress

    except RevisionNotFoundError:
        progress.status = "failed"
        progress.error_message = f"Revision not found: {revision or 'main'}"
        return progress
    except Exception as e:
        progress.status = "failed"
        progress.error_message = f"Download failed: {str(e)}"
        logger.exception(f"HuggingFace download failed for {repo_id}")
        return progress


# =============================================================================
# ModelDownloadBackend (U-821)
# =============================================================================


class ModelDownloadBackend(OrchestratorBackend):
    """
    Backend adapter for model download jobs.

    Submits model download tasks and tracks progress.
    Auto-registers models in the catalog upon successful download.
    """

    def __init__(
        self,
        model_store: "ModelStore",
        path_validator: "PathValidator",
        download_registry: Optional[DownloadTaskRegistry] = None,
        default_hf_token: Optional[str] = None,
    ):
        """
        Initialize adapter.

        Args:
            model_store: Model registry store for auto-registration
            path_validator: Validates target paths
            download_registry: Task registry (uses global if None)
            default_hf_token: Default HuggingFace token from environment
        """
        self.model_store = model_store
        self.path_validator = path_validator
        self.download_registry = download_registry or get_download_registry()
        self.default_hf_token = default_hf_token or os.environ.get("HF_TOKEN")

    @property
    def backend_name(self) -> str:
        return "model_download"

    async def submit_job(
        self,
        job: Job,
        attempt: JobAttempt,
    ) -> str:
        """
        Submit a model download job.

        Job spec should contain:
        - source: "huggingface", "s3", "http", or "local"
        - source_id: Repository ID or URL
        - target_path: Where to download (validated)
        - model_name: Name for the model in catalog
        - scope: "system", "project", or "user"
        - hf_token: Optional HuggingFace token (or use default)
        - revision: Optional revision/branch
        - metadata: Optional additional metadata

        Returns:
            Task ID (same as attempt_id for tracking)
        """
        spec = job.spec

        # Parse source type
        source_str = spec.get("source", "huggingface").lower()
        try:
            source = DownloadSource(source_str)
        except ValueError:
            raise ValueError(
                f"Invalid source '{source_str}'. "
                f"Supported: {[s.value for s in DownloadSource]}"
            )

        source_id = spec.get("source_id")
        if not source_id:
            raise ValueError("source_id is required")

        target_path = spec.get("target_path")
        if not target_path:
            raise ValueError("target_path is required")

        # Validate target path
        if not self.path_validator.is_allowed(target_path):
            from umrs.model_registry.store import PathNotAllowedError

            raise PathNotAllowedError(
                target_path,
                self.path_validator.allowed_patterns,
            )

        model_name = spec.get("model_name")
        if not model_name:
            # Default to last part of source_id
            model_name = source_id.split("/")[-1]

        scope = spec.get("scope", "user")

        # Determine owner based on scope
        if scope == "system":
            owner_id = "system"
        elif scope == "project":
            owner_id = job.project_id
        else:
            owner_id = job.user_id

        # Get HF token (from spec or default)
        hf_token = spec.get("hf_token") or self.default_hf_token

        # Create task context
        ctx = DownloadTaskContext(
            job_id=job.job_id,
            attempt_id=attempt.attempt_id,
            source=source,
            source_id=source_id,
            target_path=target_path,
            model_name=model_name,
            scope=scope,
            owner_id=owner_id,
            hf_token=hf_token,
            metadata=spec.get("metadata", {}),
        )

        # Register task
        await self.download_registry.register(ctx)

        # Start download in background
        asyncio.create_task(self._execute_download(ctx))

        logger.info(
            f"Submitted model download: job={job.job_id} "
            f"attempt={attempt.attempt_id} source={source_id} "
            f"target={target_path}"
        )

        return attempt.attempt_id

    async def _execute_download(self, ctx: DownloadTaskContext) -> None:
        """Execute the download task."""
        try:
            # Update progress callback
            async def update_progress(progress: DownloadProgress) -> None:
                await self.download_registry.update_progress(
                    ctx.attempt_id,
                    progress,
                )

            # Dispatch to source-specific downloader
            if ctx.source == DownloadSource.HUGGINGFACE:
                progress = await download_from_huggingface(
                    repo_id=ctx.source_id,
                    target_path=ctx.target_path,
                    token=ctx.hf_token,
                    revision=ctx.metadata.get("revision"),
                    progress_callback=lambda p: asyncio.create_task(
                        update_progress(p)
                    ),
                )
            elif ctx.source == DownloadSource.S3:
                progress = await self._download_from_s3(ctx)
            elif ctx.source == DownloadSource.HTTP:
                progress = await self._download_from_http(ctx)
            elif ctx.source == DownloadSource.LOCAL:
                progress = await self._copy_from_local(ctx)
            else:
                progress = DownloadProgress(
                    status="failed",
                    error_message=f"Unsupported source: {ctx.source.value}",
                )

            ctx.progress = progress
            await self.download_registry.update_progress(ctx.attempt_id, progress)

            # Auto-register on success (U-824)
            if progress.status == "completed":
                await self._auto_register_model(ctx, progress)

        except Exception as e:
            logger.exception(
                f"Download failed for {ctx.attempt_id}: {e}"
            )
            ctx.progress.status = "failed"
            ctx.progress.error_message = str(e)
            await self.download_registry.update_progress(
                ctx.attempt_id,
                ctx.progress,
            )

    async def _auto_register_model(
        self,
        ctx: DownloadTaskContext,
        progress: DownloadProgress,
    ) -> None:
        """
        Auto-register model in catalog after successful download (U-824).
        """
        from umrs.model_registry.models import (
            ModelEntry,
            ModelScope,
            ModelType,
            ServingProfile,
            generate_model_id,
        )
        from umrs.model_registry.store import AccessContext

        try:
            # Map scope string to enum
            scope = ModelScope(ctx.scope)

            # Create serving profile with defaults
            serving_profile = ServingProfile(
                gpu_memory_utilization=0.9,
                max_model_len=4096,
                tensor_parallel_size=1,
            )

            # Detect model type from metadata or default to LLM
            model_type = ModelType(ctx.metadata.get("model_type", "llm"))

            # Create model entry
            model_entry = ModelEntry(
                model_id=generate_model_id(),
                name=ctx.model_name,
                scope=scope,
                model_type=model_type,
                storage_path=ctx.target_path,
                source_uri=f"{ctx.source.value}://{ctx.source_id}",
                owner_id=ctx.owner_id,
                project_id=ctx.owner_id if scope == ModelScope.PROJECT else None,
                serving_profile=serving_profile,
                description=ctx.metadata.get("description", ""),
                tags=ctx.metadata.get("tags", []),
            )

            # Create access context for the owner
            if scope == ModelScope.SYSTEM:
                # System models are created by admin
                access_ctx = AccessContext(
                    user_id=ctx.owner_id,
                    is_admin=True,
                )
            else:
                access_ctx = AccessContext(
                    user_id=ctx.owner_id,
                    is_admin=False,
                    project_ids=[ctx.owner_id] if scope == ModelScope.PROJECT else [],
                )

            # Register in catalog
            self.model_store.create_model(model_entry, access_ctx)

            logger.info(
                f"Auto-registered model: {model_entry.name} "
                f"(id={model_entry.model_id}) at {ctx.target_path}"
            )

        except Exception as e:
            logger.warning(
                f"Failed to auto-register model {ctx.model_name}: {e}"
            )
            # Don't fail the download - model can be registered manually

    async def _download_from_s3(
        self,
        ctx: DownloadTaskContext,
    ) -> DownloadProgress:
        """Download from S3-compatible storage."""
        progress = DownloadProgress(
            started_at=datetime.now(timezone.utc),
            status="downloading",
        )

        try:
            # Parse S3 URI: s3://bucket/prefix
            # source_id should be like "bucket/prefix" or full "s3://bucket/prefix"
            source = ctx.source_id
            if source.startswith("s3://"):
                source = source[5:]

            parts = source.split("/", 1)
            if len(parts) < 2:
                raise ValueError(f"Invalid S3 path: {ctx.source_id}")

            bucket = parts[0]
            prefix = parts[1]

            # Use boto3 for S3 downloads
            import boto3
            from botocore.exceptions import ClientError

            s3 = boto3.client("s3")

            # List objects
            paginator = s3.get_paginator("list_objects_v2")
            pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

            files_to_download = []
            total_bytes = 0

            for page in pages:
                for obj in page.get("Contents", []):
                    key = obj["Key"]
                    size = obj["Size"]
                    # Remove prefix to get relative path
                    rel_path = key[len(prefix):].lstrip("/")
                    if rel_path:  # Skip the prefix itself
                        files_to_download.append((key, rel_path, size))
                        total_bytes += size

            progress.total_files = len(files_to_download)
            progress.total_bytes = total_bytes

            # Download files
            os.makedirs(ctx.target_path, exist_ok=True)

            for key, rel_path, size in files_to_download:
                local_path = os.path.join(ctx.target_path, rel_path)
                os.makedirs(os.path.dirname(local_path), exist_ok=True)

                progress.current_file = rel_path
                s3.download_file(bucket, key, local_path)

                progress.downloaded_files += 1
                progress.downloaded_bytes += size
                progress.updated_at = datetime.now(timezone.utc)

            progress.status = "completed"
            progress.updated_at = datetime.now(timezone.utc)
            return progress

        except Exception as e:
            progress.status = "failed"
            progress.error_message = f"S3 download failed: {str(e)}"
            return progress

    async def _download_from_http(
        self,
        ctx: DownloadTaskContext,
    ) -> DownloadProgress:
        """Download from HTTP/HTTPS URL."""
        progress = DownloadProgress(
            started_at=datetime.now(timezone.utc),
            status="downloading",
        )

        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.get(ctx.source_id) as resp:
                    if resp.status != 200:
                        raise RuntimeError(
                            f"HTTP {resp.status}: {resp.reason}"
                        )

                    # Get content length if available
                    total_size = resp.content_length or 0
                    progress.total_bytes = total_size
                    progress.total_files = 1

                    # Determine filename
                    filename = ctx.source_id.split("/")[-1].split("?")[0]
                    if not filename:
                        filename = "model.bin"

                    os.makedirs(ctx.target_path, exist_ok=True)
                    target_file = os.path.join(ctx.target_path, filename)

                    progress.current_file = filename

                    # Download with progress
                    downloaded = 0
                    chunk_size = 1024 * 1024  # 1MB chunks

                    with open(target_file, "wb") as f:
                        async for chunk in resp.content.iter_chunked(chunk_size):
                            f.write(chunk)
                            downloaded += len(chunk)
                            progress.downloaded_bytes = downloaded
                            progress.updated_at = datetime.now(timezone.utc)

                    progress.downloaded_files = 1
                    progress.status = "completed"
                    progress.updated_at = datetime.now(timezone.utc)
                    return progress

        except Exception as e:
            progress.status = "failed"
            progress.error_message = f"HTTP download failed: {str(e)}"
            return progress

    async def _copy_from_local(
        self,
        ctx: DownloadTaskContext,
    ) -> DownloadProgress:
        """Copy from local path (admin only)."""
        progress = DownloadProgress(
            started_at=datetime.now(timezone.utc),
            status="downloading",
        )

        try:
            source_path = ctx.source_id
            if not os.path.exists(source_path):
                raise FileNotFoundError(f"Source not found: {source_path}")

            # Calculate size
            total_size = 0
            file_count = 0
            if os.path.isfile(source_path):
                total_size = os.path.getsize(source_path)
                file_count = 1
            else:
                for root, _dirs, files in os.walk(source_path):
                    for f in files:
                        fp = os.path.join(root, f)
                        total_size += os.path.getsize(fp)
                        file_count += 1

            progress.total_files = file_count
            progress.total_bytes = total_size

            # Copy
            if os.path.isfile(source_path):
                os.makedirs(ctx.target_path, exist_ok=True)
                filename = os.path.basename(source_path)
                shutil.copy2(source_path, os.path.join(ctx.target_path, filename))
                progress.downloaded_files = 1
                progress.downloaded_bytes = total_size
            else:
                # Copy directory tree
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: shutil.copytree(
                        source_path,
                        ctx.target_path,
                        dirs_exist_ok=True,
                    ),
                )
                progress.downloaded_files = file_count
                progress.downloaded_bytes = total_size

            progress.status = "completed"
            progress.updated_at = datetime.now(timezone.utc)
            return progress

        except Exception as e:
            progress.status = "failed"
            progress.error_message = f"Local copy failed: {str(e)}"
            return progress


# =============================================================================
# ModelDownloadStatusChecker (U-823)
# =============================================================================


class ModelDownloadStatusChecker(BackendStatusChecker):
    """
    Status checker for model download jobs.

    Queries the download registry for progress updates.
    """

    def __init__(
        self,
        download_registry: Optional[DownloadTaskRegistry] = None,
    ):
        """
        Initialize checker.

        Args:
            download_registry: Task registry (uses global if None)
        """
        self.download_registry = download_registry or get_download_registry()

    @property
    def job_type(self) -> JobType:
        return JobType.MODEL_DOWNLOAD

    async def check_attempt(
        self,
        attempt: JobAttempt,
        job: Job,
    ) -> AttemptCheckResult:
        """Check status of a model download attempt."""
        task = await self.download_registry.get(attempt.attempt_id)

        if task is None:
            # Task not found - may have completed and been cleaned up
            return AttemptCheckResult(
                status="failed",
                exit_reason="Download task not found",
            )

        progress = task.progress

        if progress.status == "completed":
            return AttemptCheckResult(
                status="succeeded",
                metrics=progress.to_dict(),
            )
        elif progress.status == "failed":
            return AttemptCheckResult(
                status="failed",
                exit_reason=progress.error_message or "Download failed",
                metrics=progress.to_dict(),
            )
        else:
            # Still running
            return AttemptCheckResult(
                status="running",
                metrics=progress.to_dict(),
            )


# =============================================================================
# Factory Integration
# =============================================================================


def create_model_download_backend(
    model_store: "ModelStore",
    path_validator: "PathValidator",
) -> ModelDownloadBackend:
    """
    Create a ModelDownloadBackend with default settings.

    Args:
        model_store: Model registry store
        path_validator: Path validator

    Returns:
        Configured ModelDownloadBackend
    """
    return ModelDownloadBackend(
        model_store=model_store,
        path_validator=path_validator,
        download_registry=get_download_registry(),
        default_hf_token=os.environ.get("HF_TOKEN"),
    )


def create_model_download_status_checker() -> ModelDownloadStatusChecker:
    """Create a ModelDownloadStatusChecker with default settings."""
    return ModelDownloadStatusChecker(
        download_registry=get_download_registry(),
    )
