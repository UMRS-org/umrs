"""Per-caller host-path roots for endpoint launches (H-1, 2026-10-18).

A non-admin may bind into an endpoint container only host paths under its own
project and user data roots: the mount sources and a model path it names
itself. The site-wide prefixes (``endpoint_mounts.validate_mount_source``)
admit every user's home and every project's data, so they are not enough.

Shared by ``POST /umrs/endpoints/provision``, ``POST /umrs/endpoints/{job_id}/
upgrade`` and ``POST /umrs/jobs`` (``job_type=endpoint_provision``). The
check is lexical at request time on the UMRS host; the compute-node launcher
must re-check realpaths (symlink TOCTOU, paths only the node sees).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

# The defaults follow the model registry's per-scope roots (PathValidator
# DEFAULT_ALLOWED_PATHS: /shared/projects/{project_id}, /shared/users/
# {owner_id}, /home/{owner_id}) plus the /projects/{project_id} layout the
# site prefixes carry; a site names its own with a comma-separated
# UMRS_ENDPOINT_CALLER_MOUNT_ROOTS ({project_id}, {user_id} placeholders).
DEFAULT_CALLER_MOUNT_ROOTS = (
    "/home/{user_id}",
    "/shared/users/{user_id}",
    "/shared/projects/{project_id}",
    "/projects/{project_id}",
)


def _safe_path_segment(value: str) -> bool:
    """A placeholder value that names exactly one directory."""
    return (
        bool(value)
        and value not in (".", "..")
        and not any(ch in value for ch in ("/", "\\", "\x00"))
    )


def _expand_mount_root(template: str, values: dict[str, str]) -> Path | None:
    """*template* with its placeholders filled, or None when one is unsafe."""
    for key, value in values.items():
        token = "{" + key + "}"
        if token not in template:
            continue
        if not _safe_path_segment(value):
            return None
        template = template.replace(token, value)
    if not template.startswith("/") or "{" in template:
        return None
    return Path(template).resolve(strict=False)


def caller_mount_roots(project_id: str, user_id: str) -> list[Path]:
    """The caller's own mount roots on this site."""
    raw = os.environ.get("UMRS_ENDPOINT_CALLER_MOUNT_ROOTS")
    templates = raw.split(",") if raw is not None else DEFAULT_CALLER_MOUNT_ROOTS
    values = {"project_id": project_id, "user_id": user_id}
    roots = [_expand_mount_root(t.strip(), values) for t in templates]
    return [root for root in roots if root is not None]


def under_root(source: str, roots: list[Path]) -> bool:
    path = Path(source).resolve(strict=False)
    return any(path == root or root in path.parents for root in roots)


def foreign_sources(sources: list[str], project_id: str, user_id: str) -> list[str]:
    """The *sources* outside the caller's own roots."""
    roots = caller_mount_roots(project_id, user_id)
    return [source for source in sources if not under_root(source, roots)]


def _source_of(spec: Any) -> str | None:
    if not isinstance(spec, dict):
        return None
    return str(spec.get("source") or "").strip() or None


def raw_mount_sources(mounts: Any) -> list[str]:
    """Mount sources of a raw ``mounts`` dict (the job spec shape), read the
    way ``endpoint_mounts.normalize_mounts`` reads it (non-dict items are
    dropped there, so they are skipped here)."""
    if not isinstance(mounts, dict):
        return []
    mcp = mounts.get("mcp_servers")
    specs = [*(mcp if isinstance(mcp, list) else [])]
    specs += [mounts.get("project_data"), mounts.get("rag_index")]
    return [source for source in map(_source_of, specs) if source is not None]


def caller_named_host_path(resolved: str, model_entry: dict[str, Any] | None) -> bool:
    """True when *resolved* is a host path the caller named itself: not a
    remote URI and not the authorised catalog entry's own ``storage_ref``."""
    if "://" in resolved:
        return False
    catalog_ref = str((model_entry or {}).get("storage_ref") or "")
    if not catalog_ref or "://" in catalog_ref:
        return True
    catalog_path = Path(catalog_ref).resolve(strict=False)
    return catalog_path != Path(resolved).resolve(strict=False)


def model_path_in_scope(
    resolved: str,
    model_entry: dict[str, Any] | None,
    *,
    project_id: str,
    user_id: str,
) -> bool:
    """A model path the caller may bind: not caller-named, or under its roots."""
    if not caller_named_host_path(resolved, model_entry):
        return True
    return under_root(resolved, caller_mount_roots(project_id, user_id))


__all__ = [
    "DEFAULT_CALLER_MOUNT_ROOTS",
    "caller_mount_roots",
    "caller_named_host_path",
    "foreign_sources",
    "model_path_in_scope",
    "raw_mount_sources",
    "under_root",
]
