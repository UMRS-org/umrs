"""Workflow export: the registered source tree of a workflow as a zip.

``GET /umrs/workflows/{name}/export`` (2026-11-07) lets anyone who may see a
workflow's ``/package`` (owner, system, an ADR-080 share, admin) take its
registered code home. The archive holds:

- the registered source tree (relative paths; symlinks are never followed
  and hard-linked files are never read, both are skipped and listed);
- ``workflow.json`` with an absolute ``source.path`` rewritten relative to
  the tree (the inverse of the upload rewrite in
  ``umrs.server.file_handling._mark_uploaded``), so the zip uploads again;
- ``EXPORT.json``: name, time, UMRS version, a sha256 + size row per file,
  the excluded paths and ``"images": "not included"``.

Secrets-shaped paths (:func:`export_excluded`) are left out and listed. A
registration whose source dir is gone exports ``workflow.json`` rebuilt from
the catalog row (``EXPORT.json`` says so) plus ``EXPORT.json`` only.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

#: Summed bytes of included files above which the export is refused (413).
MAX_EXPORT_BYTES = 200 * 1024 * 1024

_CHUNK = 1024 * 1024
_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)

#: Path components excluded when equal (case-insensitive) or as a prefix.
#: ``.git`` is exact so ``.gitignore``/``.github/`` ship; ``.env`` is a
#: prefix so ``.env.local``/``.envrc`` stay out (review fix 2026-11-07).
_EXCLUDED_COMPONENTS = frozenset({".git"})
_EXCLUDED_COMPONENT_PREFIXES = (".env", "__pycache__")
_EXCLUDED_NAMES = frozenset(
    {
        "credentials",
        "credentials.json",
        ".netrc",
        ".pgpass",
        "secrets.yaml",
        "secrets.yml",
    }
)
_EXCLUDED_NAME_PREFIXES = ("id_rsa", "id_ed25519")
_EXCLUDED_SUFFIXES = (".pem", ".key", ".p12", ".pfx")

WORKFLOW_JSON = "workflow.json"
EXPORT_JSON = "EXPORT.json"


class ExportTooLarge(Exception):
    """The included files exceed :data:`MAX_EXPORT_BYTES`."""

    def __init__(self, total: int, cap: int) -> None:
        super().__init__(
            f"Workflow export is {total} bytes, above the {cap}-byte export cap"
        )
        self.total = total
        self.cap = cap


@dataclass
class ExportResult:
    """A built archive (a temp file the caller owns) and its counts."""

    path: str
    files: int
    bytes: int
    excluded: int


@dataclass
class _Plan:
    files: list[tuple[str, str]] = field(default_factory=list)
    excluded: list[str] = field(default_factory=list)
    total_bytes: int = 0


def export_excluded(rel_path: str) -> bool:
    """True when *rel_path* (``/``-separated, relative) must stay out of an
    export: a component equal to ``.git`` or starting ``.env``/``__pycache__``,
    or a file named like a key, certificate bundle or credential store."""
    parts = [p.lower() for p in rel_path.replace(os.sep, "/").split("/") if p]
    if any(_excluded_component(part) for part in parts):
        return True
    name = parts[-1] if parts else ""
    return (
        name in _EXCLUDED_NAMES
        or name.startswith(_EXCLUDED_NAME_PREFIXES)
        or name.endswith(_EXCLUDED_SUFFIXES)
    )


def _excluded_component(part: str) -> bool:
    """*part* (lower-cased) is ``.git`` or starts an excluded prefix."""
    return part in _EXCLUDED_COMPONENTS or part.startswith(_EXCLUDED_COMPONENT_PREFIXES)


def _hard_linked(path: str) -> bool:
    """A regular file with another name somewhere (``st_nlink > 1``): its
    content may live outside the tree, so it is never read (2026-11-07)."""
    try:
        info = os.lstat(path)
    except OSError:
        return False
    return stat.S_ISREG(info.st_mode) and info.st_nlink > 1


def _rel(root: str, path: str) -> str:
    return os.path.relpath(path, root).replace(os.sep, "/")


def _skipped(full: str, rel: str) -> bool:
    return os.path.islink(full) or export_excluded(rel)


def _kept_dirs(root: str, dirpath: str, names: list[str], plan: _Plan) -> list[str]:
    kept: list[str] = []
    for name in sorted(names):
        full = os.path.join(dirpath, name)
        rel = _rel(root, full)
        if _skipped(full, rel):
            plan.excluded.append(rel)
        else:
            kept.append(name)
    return kept


def _excluded_file(full: str, rel: str) -> bool:
    """Not shipped: our own manifest name, a skipped path, a non-regular
    file or a hard-linked one."""
    if rel == EXPORT_JSON or _skipped(full, rel):
        return True
    return not os.path.isfile(full) or _hard_linked(full)


def _plan_file(root: str, full: str, plan: _Plan) -> None:
    rel = _rel(root, full)
    if rel == WORKFLOW_JSON and not os.path.islink(full):
        return  # rewritten and added on its own
    if _excluded_file(full, rel):
        plan.excluded.append(rel)
        return
    plan.files.append((rel, full))
    plan.total_bytes += os.path.getsize(full)


def _walk_source(root: str) -> _Plan:
    """Included files + excluded paths under *root*; symlinked dirs are not
    descended (``followlinks=False``) and are listed as excluded."""
    plan = _Plan()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = _kept_dirs(root, dirpath, dirnames, plan)
        for name in sorted(filenames):
            _plan_file(root, os.path.join(dirpath, name), plan)
    return plan


def _within(root: str, path: str) -> bool:
    return os.path.commonpath([root, path]) == root


def relative_source(workflow_json: dict[str, Any], root: str) -> dict[str, Any]:
    """*workflow_json* with an absolute ``source.path`` made relative to
    *root* (``.`` when it lies outside the exported tree)."""
    source = workflow_json.get("source")
    if not isinstance(source, dict):
        return workflow_json
    path = source.get("path")
    if not isinstance(path, str) or not os.path.isabs(path):
        return workflow_json
    real_root = os.path.realpath(root) if root else ""
    real = os.path.realpath(path)
    inside = bool(real_root) and _within(real_root, real)
    rel = _rel(real_root, real) if inside else "."
    return {**workflow_json, "source": {**source, "path": rel}}


def _read_source_workflow_json(root: str) -> dict[str, Any] | None:
    """The tree's own workflow.json; never through a symlink or a hard link.
    Unreadable or corrupt (not UTF-8, not JSON) is ``None``: the export then
    rebuilds it from the catalog row instead of failing with a 500
    (review fix 2026-11-07)."""
    path = os.path.join(root, WORKFLOW_JSON)
    if not root or os.path.islink(path) or not os.path.isfile(path):
        return None
    if _hard_linked(path):
        return None
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _catalog_definition(name: str, entry: dict[str, Any]) -> dict[str, Any]:
    """What the catalog row records of the definition (no source on disk)."""
    return {
        "name": name,
        "version": str(entry.get("version") or ""),
        "description": str(entry.get("description") or ""),
        "steps": list(entry.get("steps") or []),
        "requirements": list(entry.get("requirements") or []),
    }


def _workflow_json_bytes(
    name: str, entry: dict[str, Any], root: str
) -> tuple[bytes, str]:
    """(bytes, origin): ``source`` from the tree, else ``catalog``."""
    data = _read_source_workflow_json(root)
    if data is None:
        body, origin = _catalog_definition(name, entry), "catalog"
    else:
        body, origin = relative_source(data, root), "source"
    return (json.dumps(body, indent=2) + "\n").encode("utf-8"), origin


def _umrs_version() -> str:
    from ..jobs.manifest import _umrs_version as installed_version

    return installed_version()


def _copy_member(zf: zipfile.ZipFile, rel: str, full: str) -> dict[str, Any]:
    """Stream one file into *zf*, hashing what was actually read."""
    info = zipfile.ZipInfo.from_file(full, rel, strict_timestamps=False)
    info.compress_type = zipfile.ZIP_DEFLATED
    digest = hashlib.sha256()
    size = 0
    fd = os.open(full, os.O_RDONLY | _NOFOLLOW)
    with os.fdopen(fd, "rb") as src, zf.open(info, "w", force_zip64=True) as dst:
        for chunk in iter(lambda: src.read(_CHUNK), b""):
            digest.update(chunk)
            dst.write(chunk)
            size += len(chunk)
    return {"path": rel, "sha256": digest.hexdigest(), "size": size}


def _write_members(
    zf: zipfile.ZipFile, plan: _Plan, workflow_json: bytes
) -> list[dict[str, Any]]:
    zf.writestr(WORKFLOW_JSON, workflow_json)
    rows: list[dict[str, Any]] = [
        {
            "path": WORKFLOW_JSON,
            "sha256": hashlib.sha256(workflow_json).hexdigest(),
            "size": len(workflow_json),
        }
    ]
    written = len(workflow_json)
    for rel, full in plan.files:
        row = _copy_member(zf, rel, full)
        rows.append(row)
        written += int(row["size"])
        if written > MAX_EXPORT_BYTES:  # a file grew after the walk
            raise ExportTooLarge(written, MAX_EXPORT_BYTES)
    return rows


def _manifest(
    name: str,
    entry: dict[str, Any],
    rows: list[dict[str, Any]],
    plan: _Plan,
    origin: str,
) -> dict[str, Any]:
    return {
        "workflow": name,
        "version": str(entry.get("version") or ""),
        "exported_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "umrs_version": _umrs_version(),
        "workflow_json": origin,
        "files": rows,
        "excluded": plan.excluded,
        "images": "not included",
    }


def _write_archive(
    path: str, name: str, entry: dict[str, Any], plan: _Plan, wf: tuple[bytes, str]
) -> list[dict[str, Any]]:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        rows = _write_members(zf, plan, wf[0])
        manifest = _manifest(name, entry, rows, plan, wf[1])
        zf.writestr(EXPORT_JSON, json.dumps(manifest, indent=2) + "\n")
    return rows


def build_export_zip(name: str, entry: dict[str, Any]) -> ExportResult:
    """Build the export of catalog *entry* into a temp file.

    Raises :class:`ExportTooLarge` before anything is written when the
    included files exceed :data:`MAX_EXPORT_BYTES`. The caller deletes
    ``ExportResult.path`` once it is sent.
    """
    root = str(entry.get("source_path") or "")
    root = root if root and os.path.isdir(root) else ""
    plan = _walk_source(root) if root else _Plan()
    workflow_json = _workflow_json_bytes(name, entry, root)
    total = plan.total_bytes + len(workflow_json[0])
    if total > MAX_EXPORT_BYTES:
        raise ExportTooLarge(total, MAX_EXPORT_BYTES)
    fd, path = tempfile.mkstemp(prefix="umrs-export-", suffix=".zip")
    os.close(fd)
    try:
        rows = _write_archive(path, name, entry, plan, workflow_json)
    except BaseException:
        os.unlink(path)
        raise
    return ExportResult(
        path=path,
        files=len(rows),
        bytes=sum(int(row["size"]) for row in rows),
        excluded=len(plan.excluded),
    )


__all__ = [
    "MAX_EXPORT_BYTES",
    "ExportResult",
    "ExportTooLarge",
    "build_export_zip",
    "export_excluded",
    "relative_source",
]
