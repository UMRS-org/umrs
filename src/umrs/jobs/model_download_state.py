"""State and recovery helpers for durable model download jobs."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from .models import AttemptStatus, Job, JobAttempt, JobStatus, JobType

logger = logging.getLogger(__name__)


class DownloadSource(Enum):
    """Supported model download sources."""

    HUGGINGFACE = "huggingface"
    S3 = "s3"
    HTTP = "http"
    LOCAL = "local"


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
    status: str = "pending"

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
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
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


@dataclass
class DownloadTaskContext:
    """Context for a download task."""

    job_id: str
    attempt_id: str
    source: DownloadSource
    source_id: str
    target_path: str
    model_name: str
    scope: str
    owner_id: str
    progress: DownloadProgress = field(default_factory=DownloadProgress)
    hf_token: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary without exposing secret token values."""
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
            "hf_token_set": bool(self.hf_token),
            "metadata": self.metadata,
        }


class DownloadTaskRegistry:
    """In-memory registry for in-flight download task contexts."""

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


_download_registry: Optional[DownloadTaskRegistry] = None


def get_download_registry() -> DownloadTaskRegistry:
    """Get or create the global download registry."""
    global _download_registry
    if _download_registry is None:
        _download_registry = DownloadTaskRegistry()
    return _download_registry


def _download_source_from_spec(spec: Dict[str, Any]) -> DownloadSource:
    """Resolve and validate the download source from a durable job spec."""
    source_str = spec.get("source", "huggingface").lower()
    try:
        return DownloadSource(source_str)
    except ValueError as exc:
        raise ValueError(
            f"Invalid source '{source_str}'. "
            f"Supported: {[source.value for source in DownloadSource]}"
        ) from exc


def _download_owner_id(job: Job, scope: str) -> str:
    """Resolve model owner from the immutable durable job scope."""
    if scope == "system":
        return "system"
    if scope == "project":
        return job.project_id
    return job.user_id


def _download_task_context_from_attempt(
    job: Job,
    attempt: JobAttempt,
    hf_token: Optional[str] = None,
    recovered: bool = False,
) -> DownloadTaskContext:
    """Rebuild task context from durable job and attempt records."""
    spec = job.spec
    source_id = spec.get("source_id")
    if not source_id:
        raise ValueError("source_id is required")

    target_path = spec.get("target_path")
    if not target_path:
        raise ValueError("target_path is required")

    scope = spec.get("scope", "user")
    metadata = dict(spec.get("metadata", {}))
    progress = _download_progress_from_attempt(attempt)
    if recovered:
        metadata["recovered_from_durable_job"] = True
        progress.status = "recovering"
        if not progress.error_message:
            progress.error_message = "Recovered from durable model_download attempt"

    return DownloadTaskContext(
        job_id=job.job_id,
        attempt_id=attempt.attempt_id,
        source=_download_source_from_spec(spec),
        source_id=source_id,
        target_path=target_path,
        model_name=spec.get("model_name") or source_id.split("/")[-1],
        scope=scope,
        owner_id=_download_owner_id(job, scope),
        progress=progress,
        hf_token=hf_token,
        metadata=metadata,
    )


def _download_progress_from_attempt(attempt: JobAttempt) -> DownloadProgress:
    """Restore the last stored download progress metrics for an attempt."""
    metrics = attempt.metrics if isinstance(attempt.metrics, dict) else {}
    try:
        progress = DownloadProgress.from_dict(metrics)
    except (TypeError, ValueError):
        progress = DownloadProgress()
    if not progress.status:
        progress.status = "running"
    return progress


def _recoverable_download_attempt(job_store: Any, job: Job) -> Optional[JobAttempt]:
    """Return the active durable model-download attempt, if one exists."""
    active_attempt_id = job.active_attempt_id
    if active_attempt_id:
        active_attempt = job_store.get_attempt(active_attempt_id)
        if active_attempt.status in {AttemptStatus.SUBMITTED, AttemptStatus.RUNNING}:
            return active_attempt
        return None

    attempts = job_store.get_attempts_for_job(job.job_id)
    for attempt in reversed(attempts):
        if attempt.status in {AttemptStatus.SUBMITTED, AttemptStatus.RUNNING}:
            return attempt
    return None


async def recover_download_registry_from_job_store(
    job_store: Any,
    download_registry: Optional[DownloadTaskRegistry] = None,
    limit: int = 1000,
) -> int:
    """Rebuild in-memory task contexts from durable active attempts."""
    registry = download_registry or get_download_registry()
    jobs = job_store.list_jobs(
        status=JobStatus.RUNNING,
        job_type=JobType.MODEL_DOWNLOAD,
        limit=limit,
    )

    recovered_count = 0
    for job in jobs:
        recovered_count += await _recover_download_job(job_store, registry, job)

    if recovered_count:
        logger.info("Recovered %d model_download task(s)", recovered_count)
    return recovered_count


async def _recover_download_job(
    job_store: Any,
    registry: DownloadTaskRegistry,
    job: Job,
) -> int:
    """Recover one durable model-download job into the registry."""
    try:
        attempt = _recoverable_download_attempt(job_store, job)
        if attempt is None:
            return 0
        context = _download_task_context_from_attempt(
            job,
            attempt,
            hf_token=None,
            recovered=True,
        )
        await registry.register(context)
        return 1
    except Exception as exc:
        logger.warning(
            "Skipping model_download recovery for job=%s: %s",
            job.job_id,
            exc,
        )
        return 0
