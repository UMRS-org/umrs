"""Download venue safety (6b-3a, 2026-09-09 data0 incident).

Two physical facts were ignored pre-flip and cost us a wedged host:

1. **Path-policy validation says nothing about the target's mount.** A model
   cache path can match the registry's allowed patterns AND still be the
   container overlay on data0 — where a 100+ GiB download became CoW
   upper-layer copies and the front disk hit 100% (uvicorn SIGKILLed).
2. **Free space was never probed.** A venue that can't fit the model whole
   looks healthy at byte zero and dies mid-week.

This module closes both before a single byte is written:

- ``assert_download_safe(path)`` walks /proc/self/mounts longest-prefix to
  find the target's filesystem: overlay / vfuse / ramfs-style union => Refuse
  (that's exactly the data0 wedge class). A second ``os.statvfs`` check
  enforces a free-space floor (site knob
  ``UMRS_DOWNLOAD_MIN_FREE_BYTES``, default 20 GiB; ``0`` disables).

- The **cancel bus**: ``register_cancel_listener(fn)`` / ``broadcast_cancel(job_id)``
  is the in-process broadcast the jobs cancel route drives (precedent: the
  WORKFLOW_RUN line in api/jobs.py). The model_download backend registers a
  listener per construction so a durable cancel is ALSO an interrupt, not
  only a status flip.

Python this way ≠ fabricate: HF's plain-snapshot path stays an opacity —
its one blocking executor call can't be pierced from asyncio; HF kills need
the subprocess redesign (named in the 6b-3 tail, WP-09 family). S3/HTTP/
local chunk loops honour the cancel mid-flight.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable

logger = logging.getLogger(__name__)

_UNION_LIKE_FS = frozenset({"overlay", "overlayfs", "vfsfuse", "fuse-overlayfs", "unionfs", "aufs"})
_MOUNT_FILE = "/proc/self/mounts"

DEFAULT_MIN_FREE_BYTES = 20 * 1024**3  # 20 GiB


class DownloadVenueError(RuntimeError):
    """Raised when the venue class is physically wrong for a model write."""


def _mount_entries() -> list[tuple[str, str]]:
    """(mount_point, fs_type) pairs from /proc/self/mounts."""
    entries: list[tuple[str, str]] = []
    try:
        with open(_MOUNT_FILE, "r", encoding="utf-8") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) >= 3:
                    entries.append((parts[1], parts[2]))
    except OSError:
        return [("/", "unknown")]
    return entries


def mount_type_for(path: str) -> str:
    """Best-effort fs type covering *path* (longest matching mount point)."""
    needle = os.path.normpath(os.path.abspath(path)) + "/"
    best_len, best_type = 0, "unknown"
    for mount_point, fs_type in _mount_entries():
        if needle.startswith(mount_point.rstrip("/") + "/") and len(mount_point) >= best_len:
            best_len, best_type = len(mount_point), fs_type
    return best_type


def free_bytes_for(path: str) -> int:
    """Free bytes on the target's filesystem (walk up to an existing ancestor)."""
    probe = os.path.abspath(path)
    while not os.path.exists(probe):
        parent = os.path.dirname(probe)
        if parent == probe:
            probe = "/"
            break
        probe = parent
    vp = os.statvfs(probe)
    return int(vp.f_bavail * vp.f_frsize)


def min_free_bytes() -> int:
    """Download free-space floor from the site knob (0s disabled)."""
    raw = os.environ.get("UMRS_DOWNLOAD_MIN_FREE_BYTES", "")
    if not raw:
        return DEFAULT_MIN_FREE_BYTES
    try:
        return max(0, int(raw))
    except ValueError:
        return DEFAULT_MIN_FREE_BYTES


def assert_download_safe(target_path: str) -> dict[str, object]:
    """Refuse overlay/unionfs targets and venues under the free-space floor.

    Returns a small diagnostic dict when safe (mount_type, free_bytes).
    Raises DownloadVenueError otherwise — raised BEFORE anything is written.
    """
    fs_type = mount_type_for(target_path)
    if fs_type in _UNION_LIKE_FS:
        raise DownloadVenueError(
            f"target {target_path!r} resolves onto the {fs_type!r} container "
            "union filesystem — a 100+ GiB model write there would wedge the "
            "host (live class 2026-09-09 data0). Choose a mounted model cache "
            "(VLLM_PATH_TO_HUGGINGFACE host bind) instead."
        )
    free = free_bytes_for(target_path)
    floor = min_free_bytes()
    if floor > 0 and free < floor:
        raise DownloadVenueError(
            f"target {target_path!r} has only {free} bytes free; venue floor "
            f"is {floor} (UMRS_DOWNLOAD_MIN_FREE_BYTES)."
        )
    return {"mount_type": fs_type, "free_bytes": free, "min_free_bytes": floor}


# -- cancel bus --------------------------------------------------------------

_CancelListener = Callable[[str], None]
_listeners: list[_CancelListener] = []


def register_cancel_listener(fn: _CancelListener) -> None:
    """Register an in-process listener for durable cancel events."""
    if fn not in _listeners:
        _listeners.append(fn)


def broadcast_cancel(job_id: str) -> int:
    """Fan out one durable cancel; returns listeners notified (never raises)."""
    notified = 0
    for listener in list(_listeners):
        try:
            listener(job_id)
            notified += 1
        except Exception:
            logger.exception("cancel listener failed (job=%s)", job_id)
    return notified


class DownloadCancelledError(RuntimeError):
    """Raised mid-download when the durable cancel marker is set."""
