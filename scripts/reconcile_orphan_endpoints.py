#!/usr/bin/env python3
"""Reconcile orphaned Mini-RiVault endpoints after UMRS restart.

When UMRS restarts, its in-memory endpoint registry is wiped clean.  Any
Slurm jobs that are still running become "orphans" — the compute node is
serving requests, but UMRS no longer knows about the endpoint.

This script scans ``squeue`` for running mini-rivault jobs, reads the job
info file written by the sbatch script, and re-registers the endpoint in
the UMRS :class:`~umrs.orchestrator.endpoint_registry.EndpointRegistry`.

When running on a node without direct Slurm access (e.g. umrs-front), the
script uses the UMRS SSH ControlMaster to run ``squeue`` and read job info
files on the remote login node.  When running on a Slurm login node, it
uses local ``squeue`` directly.

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

from umrs.orchestrator.nodelist import first_node_hostname  # noqa: E402

logger = logging.getLogger("reconcile_orphans")

# ── Slurm interaction ──────────────────────────────────────────────

#: Delimiter used in ``squeue --format``.  A pipe survives SSH/shell quoting
#: cleanly (unlike a literal tab, which gets collapsed to the letter ``t`` when
#: the command is re-quoted by the SSH ControlMaster) and never appears in
#: Slurm job ids, names, node lists, or states.
_SQUEUE_DELIM = "|"
_SQUEUE_FORMAT = f"%i{_SQUEUE_DELIM}%j{_SQUEUE_DELIM}%N{_SQUEUE_DELIM}%T"

#: Regex to identify mini-rivault jobs from the squeue job name.  The sbatch
#: script names serve jobs ``serve.slurm`` (see orchestrator/slurm_script.py),
#: so match that in addition to the legacy ``mini-rivault``/``rivault`` names.
#: Non-rivault jobs that slip through are filtered out downstream because no
#: ``rivault-job-<id>.info`` file exists for them.
_MINI_RIVAULT_RE = re.compile(r"mini.rivault|rivault|serve", re.IGNORECASE)

#: Global SSH manager cached after first successful connection.
_ssh_manager: Any | None = None


async def _get_ssh_manager() -> Any | None:
    """Get or create an SSHControlManager from site config (slurm-remote).

    Returns None if the scheduler kind is not slurm-remote or SSH is not
    configured.
    """
    global _ssh_manager
    if _ssh_manager is not None:
        return _ssh_manager

    try:
        from umrs.orchestrator.ssh_control import SSHControlConfig, SSHControlManager
        from umrs.orchestrator.site_config import load_site_config
    except ImportError as exc:
        logger.debug("Cannot import SSH modules: %s", exc)
        return None

    try:
        site = load_site_config()
    except Exception as exc:
        logger.debug("Cannot load site config: %s", exc)
        return None

    if getattr(site.scheduler, "kind", "") != "slurm-remote":
        logger.debug("Scheduler kind is not slurm-remote, SSH not needed")
        return None

    extra = getattr(site, "extra", {}) or {}
    if not extra.get("ssh_target"):
        logger.debug("No ssh_target in site config")
        return None

    try:
        ssh_config = SSHControlConfig.from_site_config(extra)
        _ssh_manager = SSHControlManager(ssh_config)
        await _ssh_manager.ensure_connected()
        logger.info("SSH ControlMaster established to %s", ssh_config.ssh_target)
        return _ssh_manager
    except Exception as exc:
        logger.warning("Failed to establish SSH connection: %s", exc)
        return None


def _run_squeue_local() -> list[dict[str, str]]:
    """Run squeue locally. Returns empty list if squeue not available."""
    try:
        result = subprocess.run(
            [
                "squeue",
                "--me",
                "--noheader",
                f"--format={_SQUEUE_FORMAT}",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
    except FileNotFoundError:
        logger.debug("squeue not found locally")
        return []
    except subprocess.CalledProcessError as exc:
        logger.error("squeue failed: %s", exc.stderr or exc)
        return []

    return _parse_squeue_output(result.stdout)


def _parse_squeue_output(text: str) -> list[dict[str, str]]:
    """Parse squeue output (``squeue --format='%i|%j|%N|%T'``).

    Each dict has keys: ``job_id``, ``name``, ``node``, ``state``.
    """
    jobs: list[dict[str, str]] = []
    for line in text.strip().splitlines():
        parts = line.split(_SQUEUE_DELIM)
        if len(parts) < 4:
            continue
        job_id, name, node, state = parts[0], parts[1], parts[2], parts[3]
        # NodeList (%N) may contain commas/brackets for multi-node jobs; the
        # authoritative serving host comes from the job info file, so keep only
        # the first node as a fallback hint.
        node = node.split(",")[0]
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


async def _run_squeue_remote(ssh: Any) -> list[dict[str, str]]:
    """Run squeue via SSH ControlMaster. Returns empty list on failure."""
    try:
        result = await ssh.run_remote(
            f"squeue --me --noheader --format='{_SQUEUE_FORMAT}'",
            timeout=60,
        )
    except Exception as exc:
        logger.error("Remote squeue failed: %s", exc)
        return []

    if not result.ok:
        logger.error("Remote squeue returned rc=%d: %s", result.returncode, result.stderr)
        return []
    return _parse_squeue_output(result.stdout)


async def _run_squeue(ssh: Any | None = None) -> list[dict[str, str]]:
    """Return running Slurm jobs, trying local first then remote SSH."""
    # Try local first
    local_result = _run_squeue_local()
    if local_result:
        return local_result

    # Fall back to SSH
    if ssh is not None:
        return await _run_squeue_remote(ssh)

    logger.error("squeue not available locally and no SSH connection")
    return []


def _filter_rivault_jobs(jobs: list[dict[str, str]]) -> list[dict[str, str]]:
    """Filter to jobs whose name matches the mini-rivault pattern."""
    return [j for j in jobs if _MINI_RIVAULT_RE.search(j["name"])]


def _load_job_info_json(path: str) -> dict[str, dict[str, str]]:
    """Load pre-collected job info from a JSON file.

    Expected format::

        {
          "22016": {
            "host": "ar13n19",
            "port": "11434",
            "model": "GLM-5.2",
            "project_id": "rivault",
            "status": "ready"
          }
        }
    """
    import json

    with open(path) as fh:
        return json.load(fh)


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


async def _read_job_info_remote(ssh: Any, job_id: str) -> dict[str, str] | None:
    """Read job info file from remote login node via SSH.

    Uses the same path as :class:`~umrs.orchestrator.remote_slurm_backend.RemoteSlurmBackend`:
    ``{slurm_job_info_dir}/rivault-job-{job_id}.info`` where
    ``slurm_job_info_dir`` comes from site config's ``extra`` section.
    """
    try:
        from umrs.orchestrator.site_config import load_site_config

        site = load_site_config()
        info_dir = site.extra.get("slurm_job_info_dir", "/tmp") if site.extra else "/tmp"
    except Exception:
        info_dir = "/tmp"

    info_path = f"{info_dir}/rivault-job-{job_id}.info"
    try:
        content = await ssh.read_remote_file(info_path)
        if not content or not content.strip():
            logger.debug("  → empty or missing remote job info: %s", info_path)
            return None

        info: dict[str, str] = {}
        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                info[key.strip()] = value.strip()
        if info:
            logger.info("  → read job info from remote: %s", info_path)
            return info
    except Exception as exc:
        logger.debug("  → remote read failed for %s: %s", info_path, exc)

    logger.warning("  → could not read job info file remotely for job %s", job_id)
    return None


def _parse_job_info(info_path: Path) -> dict[str, str]:
    """Parse a local job info file into a dict of key=value pairs."""
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
    squeue_file: str | None = None,
    job_info_json: str | None = None,
) -> int:
    """Reconcile orphaned endpoints.

    Args:
        project_id: If provided, only register endpoints for this project.
        dry_run: If True, only report what would be done.
        squeue_file: Path to pre-collected squeue output (tab-separated).
        job_info_json: Path to JSON mapping job_id → {host, port, model, project_id, status}.

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

    # Set up SSH if needed (for remote squeue + remote job info reads)
    ssh: Any = None
    if not squeue_file:
        ssh = await _get_ssh_manager()

    # Scan Slurm for running mini-rivault jobs
    if squeue_file:
        logger.info("Reading squeue output from %s", squeue_file)
        all_jobs = _parse_squeue_output(Path(squeue_file).read_text())
    else:
        all_jobs = await _run_squeue(ssh)
    rivault_jobs = _filter_rivault_jobs(all_jobs)

    logger.info(
        "Found %d Slurm job(s), %d candidate mini-rivault job(s) in squeue",
        len(all_jobs),
        len(rivault_jobs),
    )

    # Load pre-collected job info map if provided
    job_info_map: dict[str, dict[str, str]] = {}
    if job_info_json:
        job_info_map = _load_job_info_json(job_info_json)
        logger.info("Loaded %d job info entries from %s", len(job_info_map), job_info_json)

    reRegistered = 0
    for job in rivault_jobs:
        job_id = job["job_id"]
        logger.info("Checking job %s (name=%s, state=%s)", job_id, job["name"], job["state"])

        # Check if already registered
        existing = registry.get_by_job_id(job_id)
        if existing is not None:
            logger.info("  → already registered (status=%s), skipping", existing.status.value)
            continue

        # Collect endpoint information — prefer job_info_map, then local files, then remote
        info: dict[str, str] = {}
        if job_id in job_info_map:
            info = job_info_map[job_id]
            logger.info("  → using info from --job-info-json")
        else:
            info_path = _find_job_info_file(job_id)
            if info_path is not None:
                info = _parse_job_info(info_path)
            elif ssh is not None:
                remote_info = await _read_job_info_remote(ssh, job_id)
                if remote_info:
                    info = remote_info

            if not info:
                logger.warning("  → no job info file found for job %s, skipping", job_id)
                continue

        # Extract endpoint information
        host = info.get("host") or info.get("RIVAULT_NODE", job.get("node", "unknown"))
        # Slurm reports multinode allocations as a compressed nodelist
        # (e.g. ``ng-dgx-m3-[14-15]``) which is not a resolvable hostname; reduce
        # it to the head node so tunneling/readiness probes can connect.
        host = first_node_hostname(host)
        port_str = info.get("port") or info.get("RIVAULT_LITELLM_PORT", "11434")
        try:
            port = int(port_str)
        except ValueError:
            logger.warning("  → invalid port '%s' for job %s, skipping", port_str, job_id)
            continue

        model = info.get("model") or info.get("RIVAULT_MODEL_PATH", "")
        if not model:
            logger.warning("  → no model name for job %s, skipping", job_id)
            continue

        # Determine status from job info
        raw_status = (info.get("status") or info.get("RIVAULT_STATUS", "unknown")).lower()
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
        proj = project_id or info.get("project_id") or info.get("RIVAULT_PROJECT_ID", "default")

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
                        "tier": info.get("tier") or info.get("RIVAULT_TIER", "user"),
                        "owner_id": info.get("owner_id") or info.get("RIVAULT_OWNER_ID", ""),
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
    parser.add_argument(
        "--squeue-file",
        type=str,
        default=None,
        help=(
            "Path to pre-collected squeue output (pipe-separated, same format as "
            "'squeue --format=\"%%i|%%j|%%N|%%T\"'). "
            "Use when squeue is not available locally."
        ),
    )
    parser.add_argument(
        "--job-info-json",
        type=str,
        default=None,
        help=(
            "Path to JSON file mapping job_id to {host, port, model, project_id, status}. "
            "Use when job info files are not accessible locally."
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    count = asyncio.run(_reconcile(
        project_id=args.project,
        dry_run=args.dry_run,
        squeue_file=args.squeue_file,
        job_info_json=args.job_info_json,
    ))
    return 0 if count >= 0 else 1


if __name__ == "__main__":
    sys.exit(main())
