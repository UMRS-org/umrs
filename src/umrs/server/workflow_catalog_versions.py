"""Workflow catalog storage across versions (ADR-113 D-2, D-6, D-8).

Two jobs the catalog handler delegates here, both on the catalog dict it
loaded under its lock:

**First-start migration (D-8).** Before ADR-113 every version of an uploaded
workflow shared one directory, ``<extracted>/<owner>/<name>``, and a
re-upload extracted over it. :func:`migrate_catalog` moves that tree into
``<name>/<version>/`` for the row whose version it holds (the version in the
tree's own ``workflow.json``, else the highest SemVer among the rows), marks
the name's other rows ``code: "unavailable"`` (their files were overwritten;
running them fails by name) and records ``source_sha256`` on every row. It
runs once per catalog (``layout`` marker), under a cross-process file lock
(the fronts share one state dir), and is idempotent: a migrated row, a row
outside the extracted tree, a missing tree, are all left as they are.

**Removing a version's files (D-6).** :func:`retire_version_files` removes
the directory of a deleted version unless a queued or running run still
references it; then the path is kept in ``pending_removals`` and
:func:`sweep_pending_removals` removes it once no run does. Only directories
inside the owner's extracted tree are ever removed, never one another row
still uses.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import logging
import os
import shutil
import uuid
from collections.abc import Iterator
from typing import Any

from umrs.server.workflow_versions import (
    CODE_UNAVAILABLE,
    tree_digest_or_none,
    version_dir_name,
    version_sort_key,
)

logger = logging.getLogger(__name__)

#: ``catalog["layout"]`` once every row has its own version directory.
CATALOG_LAYOUT = 2
PENDING_KEY = "pending_removals"

_LIVE_JOB_STATUSES = ("planned", "queued", "running")
_TERMINAL_RUN = {"succeeded", "failed", "cancelled"}


# ── paths ──────────────────────────────────────────────────────────────────


def _owner_root(owner: str) -> str | None:
    from umrs.server.file_handling import workflows_root

    try:
        return workflows_root(owner) if owner else None
    except Exception:  # noqa: BLE001 — an unresolvable owner root: leave it
        return None


def _real(path: Any) -> str:
    return os.path.realpath(str(path)) if path else ""


def _inside(root: str, path: str) -> bool:
    return bool(root and path) and os.path.commonpath([root, path]) == root


def _legacy_dir(owner: str, name: str) -> str | None:
    """The pre-ADR-113 shared directory of (owner, name), or None."""
    root = _owner_root(owner)
    if not root or not name:
        return None
    path = os.path.realpath(os.path.join(root, name))
    return path if path != root and _inside(root, path) else None


@contextlib.contextmanager
def file_lock(path: str) -> Iterator[None]:
    """An exclusive ``flock`` on *path* (created on demand)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


# ── D-8: first-start migration ─────────────────────────────────────────────


def _groups(entries: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for entry in entries:
        key = (str(entry.get("user_id") or ""), str(entry.get("name") or ""))
        groups.setdefault(key, []).append(entry)
    return groups


def _tree_version(tree: str) -> str | None:
    """The ``version`` the shared tree's own workflow.json declares."""
    try:
        with open(os.path.join(tree, "workflow.json"), encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return None
    version = data.get("version") if isinstance(data, dict) else None
    return str(version) if version not in (None, "") else None


def _keeper_version(rows: list[dict[str, Any]], tree: str) -> str:
    """Whose files the shared tree holds: the version its workflow.json
    names when a row has it (the last upload), else the highest SemVer."""
    versions = [str(r.get("version") or "") for r in rows]
    declared = _tree_version(tree)
    if declared in versions:
        return str(declared)
    return max(versions, key=version_sort_key)


def _rebase_source(tree: str, old_root: str) -> None:
    """workflow.json's absolute ``source.path`` under *old_root* now points
    at the same place under *tree*."""
    path = os.path.join(tree, "workflow.json")
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return
    source = data.get("source") if isinstance(data, dict) else None
    if not isinstance(source, dict):
        return
    old = source.get("path")
    if not (isinstance(old, str) and os.path.isabs(old)):
        return
    real_old = os.path.realpath(old)
    if not _inside(old_root, real_old):
        return
    source["path"] = os.path.normpath(
        os.path.join(tree, os.path.relpath(real_old, old_root))
    )
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


def _move_tree(legacy: str, target: str, keep: set[str]) -> None:
    """Move *legacy*'s contents (except the *keep* version dirs) into
    *target*, through a staging dir so *target* appears complete."""
    staging = os.path.join(os.path.dirname(legacy), ".incoming", uuid.uuid4().hex)
    os.makedirs(staging)
    for item in sorted(os.listdir(legacy)):
        if item not in keep:
            shutil.move(os.path.join(legacy, item), os.path.join(staging, item))
    os.rename(staging, target)
    _rebase_source(target, legacy)


def _mark_unavailable(row: dict[str, Any]) -> None:
    row["code"] = CODE_UNAVAILABLE
    row["source_path"] = ""
    row["source_sha256"] = None


def _own_dirs(rows: list[dict[str, Any]], legacy: str) -> set[str]:
    """Version dirs inside *legacy* already owned by migrated rows."""
    return {
        os.path.basename(_real(r.get("source_path")))
        for r in rows
        if os.path.dirname(_real(r.get("source_path"))) == legacy
    }


def _shared_rows(
    rows: list[dict[str, Any]], legacy: str | None
) -> list[dict[str, Any]]:
    """The rows still pointing at the shared pre-ADR-113 directory."""
    if not legacy:
        return []
    return [r for r in rows if _real(r.get("source_path")) == legacy]


def _assign_tree(shared: list[dict[str, Any]], keep_version: str, target: str) -> None:
    """The keeper version's rows get the moved tree; the others no code."""
    has_tree = os.path.isfile(os.path.join(target, "workflow.json"))
    for row in shared:
        if has_tree and str(row.get("version") or "") == keep_version:
            row["source_path"] = target
            row["source_sha256"] = tree_digest_or_none(target)
        else:
            _mark_unavailable(row)


def _migrate_group(owner: str, name: str, rows: list[dict[str, Any]]) -> int:
    """Move one (owner, name)'s shared tree; returns rows changed."""
    legacy = _legacy_dir(owner, name)
    shared = _shared_rows(rows, legacy)
    if not legacy or not shared:
        return 0
    keep_version = _keeper_version(shared, legacy)
    target = os.path.join(legacy, version_dir_name(keep_version))
    has_tree = os.path.isfile(os.path.join(legacy, "workflow.json"))
    if has_tree and not os.path.exists(target):
        _move_tree(legacy, target, _own_dirs(rows, legacy))
    _assign_tree(shared, keep_version, target)
    logger.info(
        "workflow catalog migration: %s/%s keeps %s at %s; %d row(s) without code",
        owner,
        name,
        keep_version,
        target,
        sum(1 for r in shared if r.get("code") == CODE_UNAVAILABLE),
    )
    return len(shared)


def _digest_rows(entries: list[dict[str, Any]]) -> int:
    changed = 0
    for entry in entries:
        if entry.get("source_sha256") or entry.get("code") == CODE_UNAVAILABLE:
            continue
        digest = tree_digest_or_none(str(entry.get("source_path") or ""))
        if digest:
            entry["source_sha256"] = digest
            changed += 1
    return changed


def _migrate_groups(entries: list[dict[str, Any]]) -> tuple[int, int]:
    """(rows moved, groups failed); a failed group keeps its shared tree
    (it runs as before ADR-113) and is logged."""
    moved = failed = 0
    for (owner, name), rows in _groups(entries).items():
        try:
            moved += _migrate_group(owner, name, rows)
        except OSError:
            failed += 1
            logger.exception("workflow catalog migration failed for %s/%s", owner, name)
    return moved, failed


def migrate_catalog(catalog: dict[str, Any]) -> bool:
    """Bring a pre-ADR-113 catalog to one directory per version (module
    doc); True when the catalog changed (the caller saves it). Runs once
    (the ``layout`` marker) and is idempotent; a group that fails is logged
    and left in its shared directory."""
    if catalog.get("layout") == CATALOG_LAYOUT:
        return False
    entries = [e for e in catalog.get("workflows", []) if isinstance(e, dict)]
    moved, failed = _migrate_groups(entries)
    digested = _digest_rows(entries)
    catalog["layout"] = CATALOG_LAYOUT
    logger.info(
        "workflow catalog migrated to per-version directories: %d row(s) moved, "
        "%d digested, %d name(s) failed",
        moved,
        digested,
        failed,
    )
    return True


# ── D-6: deferred removal of a deleted version's files ─────────────────────


def _registry_references(name: str, version: str, digest: str | None) -> bool:
    """A non-terminal run of name@version in this process's registry."""
    from umrs.server.workflow_execution_registry import (
        WorkflowExecutionRegistry,
        get_execution_registry,
    )

    registry = get_execution_registry()
    # The in-memory map only: a durable registry's own list() loads every
    # stored row; other processes' runs are found through the job store.
    for record in WorkflowExecutionRegistry.list(registry):
        if record.status in _TERMINAL_RUN or record.workflow_name != name:
            continue
        if record.workflow_version == version or (
            digest and record.source_sha256 == digest
        ):
            return True
    return False


def _job_references(spec: dict[str, Any], path: str, name: str, version: str) -> bool:
    if _real(spec.get("source_path")) == path:
        return True
    return spec.get("workflow_name") == name and spec.get("workflow_version") == version


def _store_references(
    path: str, name: str, version: str, ignore: frozenset[str]
) -> bool:
    """A planned, queued or running durable workflow run naming the dir or
    name@version (every front shares the job store)."""
    from umrs.jobs.models import JobStatus, JobType
    from umrs.server.step_reaper import durable_job_store

    store = durable_job_store()
    if store is None:
        return False
    for status in _LIVE_JOB_STATUSES:
        jobs = store.list_jobs(
            status=JobStatus(status), job_type=JobType.WORKFLOW_RUN, limit=1000
        )
        for job in jobs:
            spec = job.spec if isinstance(job.spec, dict) else {}
            if job.job_id not in ignore and _job_references(spec, path, name, version):
                return True
    return False


def version_in_use(
    pending: dict[str, Any], ignore_job_ids: frozenset[str] = frozenset()
) -> bool:
    """Whether a queued or running run references the version; an
    unreadable registry or store counts as in use (removal waits)."""
    path = _real(pending.get("path"))
    name = str(pending.get("name") or "")
    version = str(pending.get("version") or "")
    try:
        return _registry_references(
            name, version, pending.get("source_sha256")
        ) or _store_references(path, name, version, ignore_job_ids)
    except Exception:  # noqa: BLE001 — fail safe: keep the files
        logger.warning("run references of %s unreadable; removal waits", path)
        return True


def _removable(path: str, owner: str, catalog: dict[str, Any]) -> bool:
    """Inside the owner's extracted tree, and no catalog row uses it."""
    root = _owner_root(owner)
    if not root or not _inside(root, path) or path == root:
        return False
    return not any(
        _real(e.get("source_path")) == path for e in catalog.get("workflows", [])
    )


def _remove_tree(path: str) -> None:
    shutil.rmtree(path, ignore_errors=True)
    parent = os.path.dirname(path)
    with contextlib.suppress(OSError):
        os.rmdir(parent)  # the name dir, once its last version is gone
    logger.info("workflow version files removed: %s", path)


def retire_version_files(
    catalog: dict[str, Any], removed: list[dict[str, Any]]
) -> list[str]:
    """Remove the files of the *removed* rows now, or keep them pending
    while a run references them. Returns the deferred paths; *catalog*
    gains them under ``pending_removals`` (the caller saves it)."""
    deferred: list[str] = []
    for row in removed:
        path = _real(row.get("source_path"))
        owner = str(row.get("user_id") or "")
        if not path or not os.path.isdir(path) or not _removable(path, owner, catalog):
            continue
        pending = {
            "path": path,
            "owner": owner,
            "name": row.get("name"),
            "version": row.get("version"),
            "source_sha256": row.get("source_sha256"),
        }
        if version_in_use(pending):
            catalog.setdefault(PENDING_KEY, []).append(pending)
            deferred.append(path)
        else:
            _remove_tree(path)
    return deferred


def sweep_pending_removals(
    catalog: dict[str, Any], *, ignore_job_ids: frozenset[str] = frozenset()
) -> bool:
    """Remove the pending version dirs no run references any more; True
    when the pending list changed (the caller saves the catalog)."""
    pending = list(catalog.get(PENDING_KEY) or [])
    if not pending:
        return False
    keep: list[dict[str, Any]] = []
    for item in pending:
        path = _real(item.get("path"))
        if not os.path.isdir(path):
            continue
        if not _removable(path, str(item.get("owner") or ""), catalog):
            continue  # re-registered meanwhile: the row owns it again
        if version_in_use(item, ignore_job_ids):
            keep.append(item)
        else:
            _remove_tree(path)
    catalog[PENDING_KEY] = keep
    return len(keep) != len(pending)


__all__ = [
    "CATALOG_LAYOUT",
    "PENDING_KEY",
    "file_lock",
    "migrate_catalog",
    "retire_version_files",
    "sweep_pending_removals",
    "version_in_use",
]
