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
_cache: dict[
    tuple[str, ...],
    tuple[tuple[float, ...], dict[tuple[str, str | None], dict[str, Any]]],
] = {}


def _file_hash(path: str) -> str:
    """Stable recipe content hash (sha256 of file bytes) for provenance."""
    import hashlib

    try:
        return hashlib.sha256(open(path, "rb").read()).hexdigest()
    except OSError:
        return ""


def _roster_paths() -> list[str]:
    raw = os.getenv(ROSTER_ENV_VAR, "").strip()
    if not raw:
        return []
    return [p for p in (part.strip() for part in raw.split(os.pathsep)) if p]


def _read_yaml_mapping(path: str) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError:
        logger.warning("PyYAML not installed; ignoring provision roster %s", path)
        return {}
    try:
        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    except (OSError, yaml.YAMLError) as exc:  # type: ignore[attr-defined]
        logger.warning("Failed to read provision roster %s: %s", path, exc)
        return {}
    if not isinstance(data, dict):
        logger.warning("Provision roster %s is not a YAML mapping; ignoring", path)
        return {}
    return data


def _merge_entries(data: dict[str, Any]) -> list[dict[str, Any]]:
    defaults = data.get("defaults")
    defaults = defaults if isinstance(defaults, dict) else {}
    raw_entries = data.get("endpoints")
    if not isinstance(raw_entries, list):
        return []
    return [
        {**defaults, **entry}
        for entry in raw_entries
        if isinstance(entry, dict) and entry.get("model")
    ]


def _entry_variant(entry: dict[str, Any]) -> str | None:
    variant = entry.get("name")  # variant = pool YAML entry name
    return variant if isinstance(variant, str) and variant else None


def _parse_roster(path: str) -> list[dict[str, Any]]:
    """Return the merged ``defaults``+``endpoints`` entries for one roster file."""
    return _merge_entries(_read_yaml_mapping(path))


def _build_index(paths: list[str]) -> dict[tuple[str, str | None], dict[str, Any]]:
    """Build a ``(model, variant)`` → sizing-defaults map across all roster files.

    First entry per ``(model, variant)`` wins (canonical sizing). Later
    files/entries do not override an already-seen key, keeping behaviour
    deterministic.

    The variant is derived from the entry's ``name`` field if present.
    Entries without a ``name`` use ``variant=None`` (the default variant).
    """
    index: dict[tuple[str, str | None], dict[str, Any]] = {}
    for path in paths:
        for entry in _parse_roster(path):
            model = str(entry["model"])
            variant = _entry_variant(entry)
            key = (model, variant)
            if key in index:
                continue
            defaults = _entry_defaults(entry, path)
            index[key] = defaults
            # The first entry for a model is also its *canonical* sizing, so a
            # provision that names only the model — `r4 endpoint up --model X`,
            # the VS Code play button — resolves to it via the (model, None)
            # fallback in resolve_model_provision_defaults().
            #
            # Without this, a roster whose entries all carry a `name:` has no
            # (model, None) key at all, so those provisions silently miss the
            # roster and fall back to 1-node/1-GPU defaults, dropping the
            # per-deployment vLLM overrides (VLLM_DOCKER_IMAGE, tool/reasoning
            # parsers, patch bind-mounts) with them. That is exactly the
            # failure this module exists to prevent, and it is silent: Slurm
            # just holds the undersized job pending. Named variants stay
            # addressable for explicit `pool up`.
            if variant is not None:
                index.setdefault((model, None), defaults)
            # Per-entry ``aliases`` provision alternate model-name lookups to
            # the same recipe. Needed when the catalog slug (registry row name
            # used for provisioning) differs from the canonical recipe key,
            # e.g. registry row ``qwen3.6-35b-a3b`` while the recipe is keyed
            # ``Qwen3.6-35B-A3B-NVFP4``. Without it the request silently falls
            # back to the default (wrong-image / 4k-ctx) plan exactly like the
            # missing-(model,None) case above.
            for alias in entry.get("aliases") or ():
                if not isinstance(alias, str) or not alias.strip():
                    continue
                a = alias.strip()
                index.setdefault((a, variant), defaults)
                index.setdefault((a, None), defaults)
    return index


def _entry_defaults(entry: dict[str, Any], path: str) -> dict[str, Any]:
    """Build the outward defaults dict for one entry, embedding the full
    recipe entry under "_full_entry" so the ServingPlan resolver sees
    recipe-controlled material (path, patches, engine, mounts)."""
    full_entry = dict(entry)
    full_entry["_recipe_file"] = path
    full_entry["_recipe_hash"] = _file_hash(path)
    defaults: dict[str, Any] = {}
    for field in _ROSTER_FIELDS:
        if field in entry and entry[field] is not None:
            defaults[field] = entry[field]
    env_vars = entry.get("env_vars")
    if isinstance(env_vars, dict) and env_vars:
        defaults["env_vars"] = {str(k): str(v) for k, v in env_vars.items()}
    defaults["_full_entry"] = full_entry
    return defaults


def _load_index() -> dict[tuple[str, str | None], dict[str, Any]]:
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
            "Loaded provision roster (%d entries) from %s",
            len(index),
            os.pathsep.join(paths),
        )
    return index


def resolve_model_provision_defaults(
    model: str,
    variant: str | None = None,
) -> dict[str, Any] | None:
    """Return roster sizing defaults for ``model`` (and optional ``variant``).

    Lookup order:
    1. Exact ``(model, variant)`` match
    2. ``(model, None)`` fallback (default variant for this model)

    The returned mapping may contain any of: ``partition``, ``node_count``,
    ``gpu_count``, ``max_model_len``, ``ttl_sec``, ``tier``,
    ``parallel_strategy`` and ``env_vars``. Callers apply each value only when
    the corresponding request field is still at its default.
    """
    index = _load_index()
    # 1. Exact variant match
    entry = index.get((model, variant))
    if not entry:
        # 2. Canonical variant fallback
        entry = index.get((model, None))
    if not entry:
        return None
    out = {k: v for k, v in entry.items() if k != "_full_entry"}
    return out


def resolve_model_serving_spec(
    model: str,
    variant: str | None = None,
    *,
    site: Any,
    project_id: str = "default",
    owner_id: str = "",
    storage_ref: str = "",
) -> Any:
    """Resolve a roster recipe into a typed ServingPlan (M2 canonical entry).

    Returns ``ResolvedEndpointSpec`` or ``None`` when no recipe matches.
    """
    from . import serving_spec

    index = _load_index()
    entry = index.get((model, variant)) or index.get((model, None))
    if not entry:
        return None
    full = entry.get("_full_entry") or entry
    return serving_spec.build_spec_from_legacy_entry(
        full,
        site=site,
        project_id=project_id,
        owner_id=owner_id,
        storage_ref=storage_ref or str(full.get("path") or ""),
    )


def roster_entries() -> list[dict[str, Any]]:
    """The full recipe entries of the published roster, one per indexed key.

    Same mtime-cached index as the provision defaults; each entry appears
    once even when it is reachable under several keys (variant, aliases).
    """
    seen: set[int] = set()
    entries: list[dict[str, Any]] = []
    for defaults in _load_index().values():
        full = defaults.get("_full_entry")
        if isinstance(full, dict) and id(full) not in seen:
            seen.add(id(full))
            entries.append(full)
    return entries


def resolve_model_recipe(
    model: str, variant: str | None = None
) -> dict[str, Any] | None:
    """The full recipe entry for ``model`` (same lookup as the defaults).

    ``resolve_model_provision_defaults`` returns only the sizing subset;
    recipe-level keys (``access``, ``keep_alive``, ``patches``) live here.
    """
    index = _load_index()
    entry = index.get((model, variant)) or index.get((model, None))
    if not entry:
        return None
    full = entry.get("_full_entry")
    return full if isinstance(full, dict) else None
