"""Per-gateway pool roster → single-provision sizing defaults.

A *pool roster* (the same YAML consumed by ``r4 pool up``) declares, for each
model a deployment cares about, the Slurm sizing required to serve it:
partition, node/GPU count, context length, TTL and vLLM env overrides. Without
this, a single-endpoint provision (the VS Code "play" button or ``r4 endpoint
up``) falls back to default sizing (1 node / 1 GPU) and lands large models on a
partition that cannot satisfy them — Slurm then holds the job ``PD
(MaxMemPerLimit)`` forever and the provision fails silently.

This module lets each deployment ("gateway") publish its roster via the
``UMRS_PROVISION_ROSTER`` environment variable (one path, or several separated
by ``os.pathsep``). When a model is provisioned individually, the API handler
consults the roster and fills *only* the request fields the caller left at their
defaults, so explicit API/CLI flags always win.

The roster schema matches ``r4 pool up``: a top-level ``defaults`` mapping plus
an ``endpoints`` list. Entries are keyed by their ``model`` field; the first
entry per model wins (the canonical sizing — named variants are reserved for
explicit ``pool up``).

This is a foundation module: stdlib + optional PyYAML only, no imports from
``umrs.api`` / ``umrs.server``.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

#: Environment variable naming one or more roster files (``os.pathsep``-joined).
ROSTER_ENV_VAR = "UMRS_PROVISION_ROSTER"

#: Sizing/serving fields lifted from a roster entry into provision defaults.
_ROSTER_FIELDS = (
    "partition",
    "node_count",
    "gpu_count",
    "max_model_len",
    "ttl_sec",
    "tier",
    "parallel_strategy",
)

# Cache keyed by the resolved roster path tuple; invalidated when any file's
# mtime changes so operators can edit the roster without restarting UMRS.
_cache: dict[tuple[str, ...], tuple[tuple[float, ...], dict[str, dict[str, Any]]]] = {}


def _roster_paths() -> list[str]:
    raw = os.getenv(ROSTER_ENV_VAR, "").strip()
    if not raw:
        return []
    return [p for p in (part.strip() for part in raw.split(os.pathsep)) if p]


def _parse_roster(path: str) -> list[dict[str, Any]]:
    """Return the merged ``defaults``+``endpoints`` entries for one roster file."""
    try:
        import yaml  # type: ignore
    except ImportError:
        logger.warning("PyYAML not installed; ignoring provision roster %s", path)
        return []
    try:
        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    except (OSError, yaml.YAMLError) as exc:  # type: ignore[attr-defined]
        logger.warning("Failed to read provision roster %s: %s", path, exc)
        return []
    if not isinstance(data, dict):
        logger.warning("Provision roster %s is not a YAML mapping; ignoring", path)
        return []
    defaults = data.get("defaults") or {}
    if not isinstance(defaults, dict):
        defaults = {}
    raw_entries = data.get("endpoints")
    if not isinstance(raw_entries, list):
        return []
    merged: list[dict[str, Any]] = []
    for entry in raw_entries:
        if isinstance(entry, dict) and entry.get("model"):
            merged.append({**defaults, **entry})
    return merged


def _build_index(paths: list[str]) -> dict[str, dict[str, Any]]:
    """Build a ``model`` → sizing-defaults map across all roster files.

    First entry per model wins (canonical sizing). Later files/entries do not
    override an already-seen model, keeping behaviour deterministic.
    """
    index: dict[str, dict[str, Any]] = {}
    for path in paths:
        for entry in _parse_roster(path):
            model = str(entry["model"])
            if model in index:
                continue
            defaults: dict[str, Any] = {}
            for field in _ROSTER_FIELDS:
                if field in entry and entry[field] is not None:
                    defaults[field] = entry[field]
            env_vars = entry.get("env_vars")
            if isinstance(env_vars, dict) and env_vars:
                defaults["env_vars"] = {str(k): str(v) for k, v in env_vars.items()}
            index[model] = defaults
    return index


def _load_index() -> dict[str, dict[str, Any]]:
    paths = _roster_paths()
    if not paths:
        return {}
    key = tuple(paths)
    try:
        mtimes = tuple(os.path.getmtime(p) if os.path.exists(p) else 0.0 for p in paths)
    except OSError:
        mtimes = tuple(0.0 for _ in paths)
    cached = _cache.get(key)
    if cached and cached[0] == mtimes:
        return cached[1]
    index = _build_index(paths)
    _cache[key] = (mtimes, index)
    if index:
        logger.info(
            "Loaded provision roster (%d models) from %s",
            len(index),
            os.pathsep.join(paths),
        )
    return index


def resolve_model_provision_defaults(model: str) -> dict[str, Any] | None:
    """Return roster sizing defaults for ``model`` or ``None`` if unlisted.

    The returned mapping may contain any of: ``partition``, ``node_count``,
    ``gpu_count``, ``max_model_len``, ``ttl_sec``, ``tier``,
    ``parallel_strategy`` and ``env_vars``. Callers apply each value only when
    the corresponding request field is still at its default.
    """
    index = _load_index()
    entry = index.get(model)
    return dict(entry) if entry else None
