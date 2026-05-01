"""Source-specific model download helpers."""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from .model_download_state import DownloadProgress

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HuggingFaceDeps:
    """Lazy HuggingFace imports needed by the downloader."""

    api_class: Any
    snapshot_download: Any
    gated_repo_error: type[Exception]
    repository_not_found_error: type[Exception]
    revision_not_found_error: type[Exception]


def _failed_progress(message: str) -> DownloadProgress:
    """Create a failed progress object with a user-facing error."""
    return DownloadProgress(
        started_at=datetime.now(timezone.utc),
        status="failed",
        error_message=message,
    )


def _load_huggingface_deps() -> HuggingFaceDeps:
    """Load HuggingFace dependencies lazily to keep startup light."""
    from huggingface_hub import HfApi, snapshot_download
    from huggingface_hub.utils import (
        GatedRepoError,
        RepositoryNotFoundError,
        RevisionNotFoundError,
    )

    return HuggingFaceDeps(
        api_class=HfApi,
        snapshot_download=snapshot_download,
        gated_repo_error=GatedRepoError,
        repository_not_found_error=RepositoryNotFoundError,
        revision_not_found_error=RevisionNotFoundError,
    )


def _update_hf_size_estimate(
    progress: DownloadProgress,
    repo_info: Any,
) -> None:
    """Populate initial file and byte totals from HuggingFace metadata."""
    siblings = getattr(repo_info, "siblings", None)
    if not siblings:
        return
    progress.total_files = len(siblings)
    progress.total_bytes = sum(getattr(item, "size", 0) or 0 for item in siblings)


def _summarize_downloaded_tree(target_path: str) -> tuple[int, int]:
    """Return `(file_count, total_size)` for the downloaded model tree."""
    total_size = 0
    file_count = 0
    for root, _dirs, files in os.walk(target_path):
        for filename in files:
            file_path = os.path.join(root, filename)
            total_size += os.path.getsize(file_path)
            file_count += 1
    return file_count, total_size


def _optimize_lustre_striping(repo_id: str, target_path: str) -> None:
    """Best-effort Lustre striping optimization for large model weights."""
    try:
        from umrs.shared.lustre import optimize_model_striping

        migrated = optimize_model_striping(target_path)
        if migrated:
            logger.info(
                "Lustre striping optimized for %s: %d files re-striped",
                repo_id,
                len(migrated),
            )
    except Exception as exc:
        logger.warning("Lustre striping optimization skipped: %s", exc)


def _mark_completed(
    progress: DownloadProgress,
    file_count: int,
    total_size: int,
) -> None:
    """Mark a HuggingFace download progress object as complete."""
    progress.downloaded_files = file_count
    progress.downloaded_bytes = total_size
    progress.total_files = file_count
    progress.total_bytes = total_size
    progress.status = "completed"
    progress.updated_at = datetime.now(timezone.utc)


def _notify_progress(
    progress_callback: Optional[Callable[[DownloadProgress], None]],
    progress: DownloadProgress,
) -> None:
    """Notify callers of a progress update when a callback was supplied."""
    if progress_callback:
        progress_callback(progress)


def _prepare_huggingface_repo(
    deps: HuggingFaceDeps,
    progress: DownloadProgress,
    repo_id: str,
    token: Optional[str],
    revision: Optional[str],
) -> Optional[DownloadProgress]:
    """Fetch repository metadata and return failed progress on access errors."""
    try:
        api = deps.api_class(token=token)
        repo_info = api.repo_info(repo_id=repo_id, revision=revision, token=token)
        _update_hf_size_estimate(progress, repo_info)
        return None
    except deps.repository_not_found_error:
        return _failed_progress(f"Repository not found: {repo_id}")
    except deps.gated_repo_error:
        return _failed_progress(
            f"Repository '{repo_id}' is gated. "
            "Provide a valid HuggingFace token with access."
        )


async def _run_huggingface_snapshot(
    deps: HuggingFaceDeps,
    repo_id: str,
    target_path: str,
    token: Optional[str],
    revision: Optional[str],
) -> None:
    """Execute `snapshot_download` off the event loop."""
    await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: deps.snapshot_download(
            repo_id=repo_id,
            local_dir=target_path,
            token=token,
            revision=revision,
            local_dir_use_symlinks=False,
        ),
    )


def _finalize_huggingface_download(
    progress: DownloadProgress,
    repo_id: str,
    target_path: str,
) -> DownloadProgress:
    """Summarize downloaded files, optimize storage, and mark completion."""
    file_count, total_size = _summarize_downloaded_tree(target_path)
    _optimize_lustre_striping(repo_id, target_path)
    _mark_completed(progress, file_count, total_size)
    logger.info(
        "Downloaded %s to %s: %d files, %d bytes",
        repo_id,
        target_path,
        file_count,
        total_size,
    )
    return progress


async def download_from_huggingface(
    repo_id: str,
    target_path: str,
    token: Optional[str] = None,
    revision: Optional[str] = None,
    progress_callback: Optional[Callable[[DownloadProgress], None]] = None,
) -> DownloadProgress:
    """Download a model from HuggingFace Hub into a local target path."""
    progress = DownloadProgress(
        started_at=datetime.now(timezone.utc),
        status="downloading",
    )

    try:
        deps = _load_huggingface_deps()
    except ImportError:
        return _failed_progress(
            "huggingface_hub not installed. Install with: pip install huggingface_hub"
        )

    failure = _prepare_huggingface_repo(
        deps,
        progress,
        repo_id,
        token,
        revision,
    )
    if failure:
        return failure
    _notify_progress(progress_callback, progress)

    try:
        os.makedirs(target_path, exist_ok=True)
        await _run_huggingface_snapshot(
            deps,
            repo_id,
            target_path,
            token,
            revision,
        )
        progress = _finalize_huggingface_download(progress, repo_id, target_path)
        _notify_progress(progress_callback, progress)
        return progress
    except deps.revision_not_found_error:
        return _failed_progress(f"Revision not found: {revision or 'main'}")
    except Exception as exc:
        progress.status = "failed"
        progress.error_message = f"Download failed: {str(exc)}"
        logger.exception("HuggingFace download failed for %s", repo_id)
        return progress
