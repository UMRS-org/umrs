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
import logging
import os
import shutil
from collections.abc import Callable
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from .model_download_sources import download_from_huggingface
from .model_download_state import (
    DownloadProgress,
    DownloadSource,
    DownloadTaskContext,
    DownloadTaskRegistry,
    _download_task_context_from_attempt,
    get_download_registry,
    recover_download_registry_from_job_store,
)
from .models import Job, JobAttempt, JobType
from .monitor import AttemptCheckResult, BackendStatusChecker
from .scheduler import OrchestratorBackend

if TYPE_CHECKING:
    from umrs.model_registry.models import ModelCapabilities
    from umrs.model_registry.store import ModelStore, PathValidator

logger = logging.getLogger(__name__)


def _notify_download_progress(
    progress_callback: Callable[[DownloadProgress], None] | None,
    progress: DownloadProgress,
) -> None:
    if progress_callback:
        progress_callback(progress)


__all__ = [
    "DownloadProgress",
    "DownloadSource",
    "DownloadTaskContext",
    "DownloadTaskRegistry",
    "ModelDownloadBackend",
    "ModelDownloadStatusChecker",
    "create_model_download_backend",
    "create_model_download_status_checker",
    "download_from_huggingface",
    "get_download_registry",
    "recover_download_registry_from_job_store",
]


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
        model_store: ModelStore,
        path_validator: PathValidator,
        download_registry: DownloadTaskRegistry | None = None,
        default_hf_token: str | None = None,
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

    def _validate_target_path(self, target_path: str) -> None:
        """Validate a target path against the configured model policy."""
        if self.path_validator.is_allowed(target_path):
            return

        from umrs.model_registry.store import PathNotAllowedError

        raise PathNotAllowedError(
            target_path,
            self.path_validator.allowed_patterns,
        )

    def _create_download_context(
        self,
        job: Job,
        attempt: JobAttempt,
    ) -> DownloadTaskContext:
        """Create a download context from durable job and attempt records."""
        ctx = _download_task_context_from_attempt(
            job,
            attempt,
            hf_token=job.spec.get("hf_token") or self.default_hf_token,
        )
        self._validate_target_path(ctx.target_path)
        return ctx

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
        ctx = self._create_download_context(job, attempt)

        # Register task
        await self.download_registry.register(ctx)

        # Start download in background
        asyncio.create_task(self._execute_download(ctx))

        logger.info(
            f"Submitted model download: job={job.job_id} "
            f"attempt={attempt.attempt_id} source={ctx.source_id} "
            f"target={ctx.target_path}"
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
                    progress_callback=lambda p: asyncio.create_task(update_progress(p)),
                )
            elif ctx.source == DownloadSource.S3:
                progress = await self._download_from_s3(ctx)
            elif ctx.source == DownloadSource.HTTP:
                progress = await self._download_from_http(
                    ctx,
                    progress_callback=lambda p: asyncio.create_task(update_progress(p)),
                )
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
            logger.exception(f"Download failed for {ctx.attempt_id}: {e}")
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
            ModelScope,
            ModelStatus,
            ModelType,
            ServingProfile,
        )

        try:
            metadata = ctx.metadata or {}
            existing_model_id = metadata.get("model_id")
            if existing_model_id:
                self.model_store.update_status(existing_model_id, ModelStatus.AVAILABLE)
                # Auto-detect and update capabilities on existing model
                self._detect_and_update_capabilities(existing_model_id, ctx.target_path)
                logger.info(
                    "Marked model %s available after %s -> %s",
                    existing_model_id,
                    ctx.source_id,
                    ctx.target_path,
                )
                return

            scope = ModelScope(ctx.scope)

            serving_profile = ServingProfile.from_dict(
                metadata.get("serving_profile") or {}
            )
            model_type = ModelType(metadata.get("model_type", "llm"))

            self.model_store.create_model(
                name=ctx.model_name,
                scope=scope,
                storage_ref=ctx.target_path,
                model_type=model_type,
                owner_id=metadata.get("owner_id") if scope == ModelScope.USER else None,
                project_id=metadata.get("project_id")
                if scope == ModelScope.PROJECT
                else None,
                huggingface_id=metadata.get("huggingface_id"),
                description=metadata.get("description"),
                serving_profile=serving_profile,
                capabilities=self._detect_capabilities(ctx.target_path),
                status=ModelStatus.AVAILABLE,
                ctx=None,
            )

            logger.info(
                "Auto-registered model %s at %s after download completion",
                ctx.model_name,
                ctx.target_path,
            )

        except Exception as e:
            logger.warning(f"Failed to auto-register model {ctx.model_name}: {e}")
            # Don't fail the download - model can be registered manually

    def _detect_capabilities(self, model_path: str) -> ModelCapabilities | None:
        """Detect model capabilities from downloaded files."""
        try:
            from umrs.model_registry.detect import detect_capabilities

            return detect_capabilities(model_path)
        except Exception as e:
            logger.debug("Capability detection failed for %s: %s", model_path, e)
            return None

    def _detect_and_update_capabilities(self, model_id: str, model_path: str) -> None:
        """Detect capabilities and update existing model entry."""
        caps = self._detect_capabilities(model_path)
        if caps is None:
            return
        try:
            model = self.model_store.get_model(model_id)
            if model and not any(
                [
                    model.capabilities.supports_tool_calling,
                    model.capabilities.supports_vision,
                    model.capabilities.model_family,
                ]
            ):
                # Only update if capabilities are empty
                self.model_store.update_capabilities(model_id, caps)
                logger.info(
                    "Auto-detected capabilities for model %s: family=%s, tools=%s, vision=%s",
                    model_id,
                    caps.model_family,
                    caps.supports_tool_calling,
                    caps.supports_vision,
                )
        except Exception as e:
            logger.debug("Failed to update capabilities for %s: %s", model_id, e)

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
                    rel_path = key[len(prefix) :].lstrip("/")
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
        progress_callback: Callable[[DownloadProgress], None] | None = None,
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
                        raise RuntimeError(f"HTTP {resp.status}: {resp.reason}")

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
                    progress.updated_at = datetime.now(timezone.utc)
                    _notify_download_progress(progress_callback, progress)

                    # Download with progress
                    downloaded = 0
                    chunk_size = 1024 * 1024  # 1MB chunks

                    with open(target_file, "wb") as f:
                        async for chunk in resp.content.iter_chunked(chunk_size):
                            f.write(chunk)
                            downloaded += len(chunk)
                            progress.downloaded_bytes = downloaded
                            progress.updated_at = datetime.now(timezone.utc)
                            _notify_download_progress(progress_callback, progress)

                    progress.downloaded_files = 1
                    progress.status = "completed"
                    progress.updated_at = datetime.now(timezone.utc)
                    _notify_download_progress(progress_callback, progress)
                    return progress

        except Exception as e:
            progress.status = "failed"
            progress.error_message = f"HTTP download failed: {str(e)}"
            progress.updated_at = datetime.now(timezone.utc)
            _notify_download_progress(progress_callback, progress)
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
        download_registry: DownloadTaskRegistry | None = None,
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
    model_store: ModelStore,
    path_validator: PathValidator,
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
