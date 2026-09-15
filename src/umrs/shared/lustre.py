"""Lustre filesystem helpers for model weight striping.

On Lustre parallel filesystems, read throughput for large files scales
with the number of OSTs (Object Storage Targets) the file is striped
across.  Model weight shards downloaded via HuggingFace Hub default to
``stripe_count=1``, bottlenecking load time at ~100 MB/s per file.

This module provides:
- ``optimize_model_striping()`` — re-stripes large files (safetensors,
  bin, pt) across multiple OSTs using ``lfs migrate``.
- ``check_striping()`` — reports current striping for diagnostics.

Both are no-ops on non-Lustre filesystems.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# File extensions that benefit from multi-OST striping (large weight files).
_WEIGHT_EXTENSIONS = frozenset(
    {
        ".safetensors",
        ".bin",
        ".pt",
        ".pth",
        ".gguf",
        ".ggml",
    }
)

# Minimum file size (bytes) worth re-striping.  Files smaller than this
# load fast enough on a single OST.
_MIN_STRIPE_SIZE = 256 * 1024 * 1024  # 256 MiB

# Default stripe parameters.
_DEFAULT_STRIPE_COUNT = 4
_DEFAULT_STRIPE_SIZE = "4M"


def _has_lfs() -> bool:
    """Return True if the ``lfs`` CLI is available."""
    return shutil.which("lfs") is not None


@dataclass
class StripingInfo:
    """Lustre striping information for a single file."""

    path: str
    stripe_count: int
    stripe_size: int
    is_lustre: bool


def check_striping(path: str) -> StripingInfo | None:
    """Return the Lustre striping info for *path*, or None on non-Lustre."""
    if not _has_lfs():
        return None
    try:
        result = subprocess.run(
            ["lfs", "getstripe", "--stripe-count", "--stripe-size", path],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return None

        stripe_count = 0
        stripe_size = 0
        for line in result.stdout.splitlines():
            stripped = line.strip()
            if stripped.startswith("lmm_stripe_count:") or stripped.startswith(
                "stripe_count:"
            ):
                stripe_count = int(stripped.split(":")[-1].strip())
            elif stripped.startswith("lmm_stripe_size:") or stripped.startswith(
                "stripe_size:"
            ):
                stripe_size = int(stripped.split(":")[-1].strip())

        return StripingInfo(
            path=path,
            stripe_count=stripe_count,
            stripe_size=stripe_size,
            is_lustre=True,
        )
    except Exception:
        return None


def optimize_model_striping(
    model_dir: str,
    *,
    stripe_count: int = _DEFAULT_STRIPE_COUNT,
    stripe_size: str = _DEFAULT_STRIPE_SIZE,
    min_file_size: int = _MIN_STRIPE_SIZE,
    dry_run: bool = False,
) -> list[str]:
    """Re-stripe large model weight files for parallel I/O on Lustre.

    Parameters
    ----------
    model_dir:
        Path to the model directory.
    stripe_count:
        Number of OSTs to stripe across (default 4).
    stripe_size:
        Stripe unit size, e.g. ``"4M"`` (default ``"4M"``).
    min_file_size:
        Only re-stripe files larger than this (bytes).
    dry_run:
        If True, log what would be done but don't change anything.

    Returns
    -------
    List of file paths that were (or would be) re-striped.
    """
    if not _has_lfs():
        logger.debug("lfs not found — skipping Lustre striping for %s", model_dir)
        return []

    if not os.path.isdir(model_dir):
        logger.warning("Model directory does not exist: %s", model_dir)
        return []

    # Check if the directory is on Lustre.
    dir_info = check_striping(model_dir)
    if dir_info is None:
        logger.debug("%s is not on Lustre — skipping striping", model_dir)
        return []

    # First, set the directory default so *future* files inherit good striping.
    if not dry_run:
        try:
            subprocess.run(
                [
                    "lfs",
                    "setstripe",
                    "-c",
                    str(stripe_count),
                    "-S",
                    stripe_size,
                    model_dir,
                ],
                capture_output=True,
                text=True,
                timeout=10,
                check=True,
            )
            logger.info(
                "Set directory default stripe: %s (count=%d, size=%s)",
                model_dir,
                stripe_count,
                stripe_size,
            )
        except subprocess.CalledProcessError as e:
            logger.warning("Failed to set directory stripe: %s", e.stderr.strip())

    # Find weight files that need re-striping.
    migrated: list[str] = []

    for root, _dirs, files in os.walk(model_dir):
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext not in _WEIGHT_EXTENSIONS:
                continue

            fpath = os.path.join(root, fname)
            try:
                fsize = os.path.getsize(fpath)
            except OSError:
                continue

            if fsize < min_file_size:
                continue

            # Check current striping.
            info = check_striping(fpath)
            if info is None:
                continue
            if info.stripe_count >= stripe_count:
                logger.debug(
                    "Already well-striped (%d OSTs): %s",
                    info.stripe_count,
                    fpath,
                )
                continue

            # Re-stripe via lfs migrate.
            if dry_run:
                logger.info(
                    "[DRY RUN] Would migrate %s (%.1f GiB, stripe %d→%d)",
                    fpath,
                    fsize / (1024**3),
                    info.stripe_count,
                    stripe_count,
                )
            else:
                logger.info(
                    "Migrating %s (%.1f GiB, stripe %d→%d) ...",
                    fpath,
                    fsize / (1024**3),
                    info.stripe_count,
                    stripe_count,
                )
                try:
                    subprocess.run(
                        [
                            "lfs",
                            "migrate",
                            "-c",
                            str(stripe_count),
                            "-S",
                            stripe_size,
                            fpath,
                        ],
                        capture_output=True,
                        text=True,
                        timeout=600,  # large files can take minutes
                        check=True,
                    )
                except subprocess.CalledProcessError as e:
                    logger.warning(
                        "lfs migrate failed for %s: %s",
                        fpath,
                        e.stderr.strip(),
                    )
                    continue

            migrated.append(fpath)

    if migrated:
        logger.info(
            "Lustre striping optimized: %d files in %s (stripe_count=%d)",
            len(migrated),
            model_dir,
            stripe_count,
        )
    else:
        logger.debug("No files needed re-striping in %s", model_dir)

    return migrated
