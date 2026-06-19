#!/usr/bin/env python3
"""Reconcile orphaned Mini-RiVault endpoints after UMRS restart.

When UMRS restarts, its in-memory endpoint registry is wiped clean.  Any
Slurm jobs that are still running become "orphans" — the compute node is
serving requests, but UMRS no longer knows about the endpoint.

This script scans ``squeue`` for running mini-rivault jobs, reads the job
info file written by the sbatch script, and re-registers the endpoint in
the UMRS :class:`~umrs.orchestrator.endpoint_registry.EndpointRegistry`.

Usage (inside the UMRS container)::

    python3 /app/umrs-src/scripts/reconcile_orphan_endpoints.py
    python3 /app/umrs-src/scripts/reconcile_orphan_endpoints.py --dry-run
    python3 /app/umrs-src/scripts/reconcile_orphan_endpoints.py --project myproj

The script is idempotent: endpoints already registered are skipped.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

# Ensure umrs package is importable when run as a script
_script_dir = Path(__file__).resolve().parent
_src_root = _script_dir.parent / "src"
if _src_root.is_dir():
    sys.path.insert(0, str(_src_root))

logger = logging.getLogger("reconcile_orphans")

# ── Slurm interaction ──────────────────────────────────────────────

#: Regex to identify mini-rivault jobs from squeue output
_MINI_RIVAULT_RE = re.compile(r"mini.rivault|rivault", re.IGNORECASE)


def _run_squeue() -> list[dict[str, str]]:
    """Return running Slurm jobs for the current user.

    Each dict has keys: ``job_id``, ``name``, ``node``, ``state``.
    """
    try:
        result = subprocess.run(
            [
                "squeue",
                "--me",
                "--noheader",
                "--format=%i\t%j\t%N\t%T",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
    except FileNotFoundError:
        logger.error("squeue not found — this script must run on a Slurm login/compute node")
        return []
    except subprocess.CalledProcessError as exc:
        logger.error("squeue failed: %s", exc.stderr or exc)
        return []

    jobs: list[dict[str, str]] = []
    for line in result.stdout.strip().splitlines():
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        job_id, name, node, state = parts[0], parts[1], parts[2], parts[3]
        # Strip array/task suffix for display: "12345_2" → "12345"
        base_id = job_id.split("_")[0]
        jobs.append(
            {
                "job_id": base_id,
                "full_job_id": job_id,
                "name": name,
                "node": node,
                "state": state,
            }
        )
    return jobs


def _filter_rivault_jobs(jobs: list[dict[str, str]]) -> list[dict[str, str]]:
    """Filter to jobs whose name matches the mini-rivault pattern."""
    return [j for j in jobs if _MINI_RIVAULT_RE.search(j["name"])]


# ── Job info file parsing ──────────────────────────────────────────

def _find_job_info_file(job_id: str) -> Path | None:
    """Locate the job info file for a given Slurm job ID.

    The sbatch script writes to ``$JOB_INFO_DIR/rivault-job-<id>.info``.
    We search common directories if the env var is not set.
    """
    # Check JOB_INFO_DIR env var
    info_dir = os.environ.get("JOB_INFO_DIR")
    if info_dir:
        p = Path(info_dir) / f"rivault-job-{job_id}.info"
        if p.exists():
            return p

    # Search common locations
    search_dirs = [
        Path("/tmp/rivault-jobs"),
        Path("/var/run/rivault"),
        Path.home() / ".rivault" / "jobs",
    ]
    # Also search shared scratch directories
    for scratch in ["/scratch", "/lvs0", "/work"]:
        sp = Path(scratch)
        if sp.is_dir():
            search_dirs.append(sp / "rivault-jobs")

    for d in search_dirs:
        p = d / f"rivault-job-{job_id}.info"
        if p.exists():
            return p

    return None


def _parse_job_info(info_path: Path) -> dict[str, str]:
    """Parse a job info file into a dict of key=value pairs."""
    info: dict[str, str] = {}
    try:
        text = info_path.read_text()
    except OSError as exc:
        logger.warning("Could not read %s: %s", info_path, exc)
        return info

    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, _, value = line.partition("=")
            info[key.strip()] = value.strip()
    return info


# ─- Reconciliation ─────────────────────────────────────────────────

async def _reconcile(
    project_id: str | None = None,
    dry_run: bool = False,
) -> int:
    """Reconcile orphaned endpoints.

    Args:
        project_id: If provided, only register endpoints for this project.
        dry_run: If True, only report what would be done.

    Returns:
        Number of endpoints re-registered.
    """
    from umrs.orchestrator.endpoint_registry import (
        EndpointRegistry,
        EndpointStatus,
    )
    from umrs.orchestrator.registry_backend import get_backend

    # Get the registry singleton
    try:
        backend = get_backend()
    except Exception as exc:
        logger.warning("Failed to get registry backend: %s", exc)
        backend = None

    # Get or create the registry
    registry = EndpointRegistry(backend=backend)

    # Scan Slurm for running mini-rivault jobs
    all_jobs = _run_squeue()
    rivault_jobs = _filter_rivault_jobs(all_jobs)

    logger.info("Found %d mini-rivault job(s) in squeue", len(rivault_jobs))

    reRegistered = 0
    for job in rivault_jobs:
        job_id = job["job_id"]
        logger.info("Checking job %s (name=%s, state=%s)", job_id, job["name"], job["state"])

        # Check if already registered
        existing = registry.get_by_job_id(job_id)
        if existing is not None:
            logger.info("  → already registered (status=%s), skipping", existing.status.value)
            continue

        # Find and parse the job info file
        info_path = _find_job_info_file(job_id)
        if info_path is None:
            logger.warning("  → no job info file found for job %s, skipping", job_id)
            continue

        info = _parse_job_info(info_path)
        if not info:
            logger.warning("  → empty or invalid job info file: %s", info_path)
            continue

        # Extract endpoint information
        host = info.get("RIVAULT_NODE", job.get("node", "unknown"))
        port_str = info.get("RIVAULT_LITELLM_PORT", "11434")
        try:
            port = int(port_str)
        except ValueError:
            logger.warning("  → invalid port '%s' in %s, skipping", port_str, info_path)
            continue

        model = info.get("RIVAULT_MODEL_PATH", "")
        if not model:
            logger.warning("  → no model path in %s, skipping", info_path)
            continue

        # Determine status from job info
        raw_status = info.get("RIVAULT_STATUS", "unknown").lower()
        if raw_status in ("ready", "running"):
            status = EndpointStatus.READY
        elif raw_status in ("starting", "provisioning"):
            status = EndpointStatus.STARTING
        elif raw_status in ("failed", "error"):
            status = EndpointStatus.FAILED
        elif raw_status in ("stopped", "terminated"):
            logger.info("  → job %s is stopped/terminated, skipping", job_id)
            continue
        else:
            status = EndpointStatus.PROVISIONING

        # Determine project ID
        proj = project_id or info.get("RIVAULT_PROJECT_ID", "default")

        if dry_run:
            logger.info(
                "  → [DRY RUN] would register: job=%s model=%s host=%s:%d status=%s project=%s",
                job_id,
                model,
                host,
                port,
                status.value,
                proj,
            )
        else:
            try:
                endpoint = registry.register(
                    project_id=proj,
                    model=model,
                    job_id=job_id,
                    host=host,
                    port=port,
                    backend="slurm",
                    metadata={
                        "served_model_name": os.environ.get("VLLM_MODEL_NAME", model),
                        "tier": info.get("RIVAULT_TIER", "user"),
                        "owner_id": info.get("RIVAULT_OWNER_ID", ""),
                    },
                )
                # Update status to match what we found
                registry.update_status(job_id, status)

                logger.info(
                    "  → re-registered: job=%s model=%s host=%s:%d status=%s",
                    job_id,
                    model,
                    host,
                    port,
                    status.value,
                )
                reRegistered += 1
            except Exception as exc:
                logger.error("  → failed to register job %s: %s", job_id, exc)

    logger.info("Reconciliation complete: %d endpoint(s) re-registered", reRegistered)
    return reRegistered


# ── CLI ─────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reconcile orphaned Mini-RiVault endpoints after UMRS restart.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report what would be done, don't actually register endpoints.",
    )
    parser.add_argument(
        "--project",
        type=str,
        default=None,
        help="Project ID to register endpoints under (default: from job info or 'default').",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    count = asyncio.run(_reconcile(
        project_id=args.project,
        dry_run=args.dry_run,
    ))
    return 0 if count >= 0 else 1


if __name__ == "__main__":
    sys.exit(main())
