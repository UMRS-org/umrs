"""Execution manifest assembly (P7a, G-P7-1).

Populates the r4-contracts provenance DTOs with real per-attempt data:
- ``EnvironmentSnapshot`` — runtime facts collected locally at attempt end
  (python/UMRS versions, container runtime/image when the venue records
  them, hostname, GPU devices).
- ``CodeVersionRef`` — the UMRS source tree's git revision + dirty flag
  (empty when the tree isn't a checkout — recorded, never fabricated).
- ``ExecutionManifest`` — provenance + environment + code + trigger.

Lives in ``umrs.jobs`` (foundation): the durable job store assembles and
persists these at attempt completion; nothing here may import umrs.api or
umrs.server.
"""

from __future__ import annotations

import logging
import os
import platform
import subprocess
from typing import Any

logger = logging.getLogger(__name__)


def _umrs_version() -> str:
    """Installed UMRS version, or 'src' when running uninstalled."""
    try:
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version("umrs")
        except PackageNotFoundError:
            return "src"
    except (ImportError, TypeError):  # pragma: no cover - exotic runtimes
        return "unknown"


def _git_field(repo_root: str, args: list[str]) -> str:
    """One read-only git query; empty on any failure (recorded, not invented)."""
    try:
        out = subprocess.run(
            ["git", "-C", repo_root, *args],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    if out.returncode != 0:
        return ""
    return out.stdout.strip()


def code_version_ref(repo_root: str | None = None) -> dict[str, Any]:
    """Git revision of the running source tree (CodeVersionRef shape)."""
    if repo_root is None:
        repo_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..")
        )
    commit = _git_field(repo_root, ["rev-parse", "HEAD"])
    branch = _git_field(repo_root, ["rev-parse", "--abbrev-ref", "HEAD"])
    dirty = bool(_git_field(repo_root, ["status", "--porcelain"])) if commit else False
    return {
        "repo_url": "",
        "commit_sha": commit,
        "branch": branch,
        "dirty": dirty,
        "tags": [],
    }


def build_model_pins(spec: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Declared model pins for the manifest (WP-10 round 2).

    One pin per well-known spec field; values are recorded VERBATIM with the
    field as source attribution — never resolved, defaulted, or invented.
    Absent fields contribute no pin. The active-version authority
    (VectorVersionStore) is not a job-store dependency, so venue lookup pins
    stay an open leg (named in wp10.md); spec pins are the durable substrate.
    """
    if not isinstance(spec, dict):
        return []
    pins: list[dict[str, Any]] = []
    for name_key, version_key in (
        ("model", "model_version"),
        ("model_name", "model_version"),
        ("embedding_model", "embedding_model_version"),
        ("rag_model", "rag_model_version"),
    ):
        name = spec.get(name_key)
        if not name:
            continue
        pin: dict[str, Any] = {"name": str(name), "source": f"spec.{name_key}"}
        version = spec.get(version_key)
        if version:
            pin["version"] = str(version)
        path = spec.get(f"{name_key}_path") or (
            spec.get("model_path") if name_key in ("model", "model_name") else None
        )
        if path:
            pin["path"] = str(path)
        # Same dedup predicate: one pin per distinct name (first source wins).
        if all(p["name"] != pin["name"] for p in pins):
            pins.append(pin)
    return pins


def environment_snapshot() -> dict[str, Any]:
    """Runtime facts for one completed attempt (EnvironmentSnapshot shape)."""
    gpus = [g for g in os.environ.get("NVIDIA_VISIBLE_DEVICES", "").split(",") if g]
    return {
        "python_version": platform.python_version(),
        "umrs_version": _umrs_version(),
        "r4_version": "",
        "site_config_hash": None,
        "env_vars": {
            "container_runtime": os.environ.get("UMRS_CONTAINER_RUNTIME", ""),
            "vllm_offline": os.environ.get("TRANSFORMERS_OFFLINE", ""),
        },
        "container_image": os.environ.get("UMRS_CONTAINER_IMAGE", ""),
        "container_runtime": os.environ.get("UMRS_CONTAINER_RUNTIME", ""),
        "hostname": platform.node(),
        "gpu_devices": gpus,
    }


def execution_manifest(
    *,
    provenance: dict[str, Any],
    trigger: str = "api",
    pipeline_name: str = "",
    tags: dict[str, str] | None = None,
    model_pins: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """One attempt's manifest, ready to persist under attempts.metrics.

    The trigger defaults to ``api`` — the honest value when nothing better
    is recorded; callers pass ``scheduled``/``webhook`` when known.
    ``model_pins`` is the verbatim spec-declared pin list (WP-10 round 2);
    an empty list is recorded honestly (nothing declared), never dropped.
    """
    from datetime import datetime, timezone

    return {
        "pipeline_name": pipeline_name,
        "pipeline_version": "",
        "trigger": trigger,
        "tags": tags or {},
        "created_at": datetime.now(timezone.utc).isoformat(),
        "trace_ids": [],
        "provenance": provenance,
        "model_pins": model_pins if model_pins is not None else [],
        "environment": environment_snapshot(),
        "code_versions": [code_version_ref()],
    }
