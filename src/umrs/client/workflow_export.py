"""``umrs workflow export NAME [-o FILE]`` client half (2026-11-07).

Downloads ``GET /gw/workflows/{name}/export`` through the gateway into a temp
file next to the destination, checks the archive against its ``EXPORT.json``
(every listed file present with the listed sha256 and size) and only then
renames it into place. Stdlib only.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from typing import Any

_CHUNK = 1024 * 1024


class ExportDownloadError(Exception):
    """The export could not be fetched, verified or saved."""


def _member_digest(zf: zipfile.ZipFile, name: str) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with zf.open(name) as member:
        for chunk in iter(lambda: member.read(_CHUNK), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _check_row(zf: zipfile.ZipFile, row: dict[str, Any]) -> None:
    name = str(row.get("path", ""))
    try:
        sha, size = _member_digest(zf, name)
    except KeyError as exc:
        raise ExportDownloadError(
            f"EXPORT.json lists {name!r}, absent from the zip"
        ) from exc
    if sha != row.get("sha256") or size != row.get("size"):
        raise ExportDownloadError(f"{name!r} does not match its EXPORT.json hash")


def verify_export_archive(path: str) -> dict[str, Any]:
    """The archive's EXPORT.json after every listed file is checked
    (``ExportDownloadError`` on a broken zip or a mismatch)."""
    try:
        with zipfile.ZipFile(path) as zf:
            manifest = json.loads(zf.read("EXPORT.json"))
            if not isinstance(manifest, dict):
                raise ExportDownloadError("EXPORT.json is not an object")
            for row in manifest.get("files") or []:
                _check_row(zf, row)
    except (zipfile.BadZipFile, KeyError, ValueError) as exc:
        raise ExportDownloadError(f"Not a valid workflow export: {exc}") from exc
    return dict(manifest)


def names_directory(path: str) -> bool:
    """*path* ends with a separator (``exports/``): the caller means a
    directory even when none exists yet (2026-11-07)."""
    seps = tuple(sep for sep in ("/", os.sep, os.altsep) if sep)
    return path.endswith(seps)


def export_destination(workflow_name: str, output: str | None) -> str:
    """``-o`` as given, ``<dir>/<name>.zip`` for a directory (an existing
    one, or a path ending in a separator, created with its parents), else
    ``./<name>.zip``."""
    target = output or f"{workflow_name}.zip"
    if names_directory(target):
        os.makedirs(target, exist_ok=True)
    if os.path.isdir(target):
        target = os.path.join(target, f"{workflow_name}.zip")
    return os.path.abspath(target)


def _export_request(
    gateway_url: str, workflow_name: str, api_key: str | None, project: str | None
) -> urllib.request.Request:
    quoted = urllib.parse.quote(workflow_name, safe="")
    url = f"{gateway_url.rstrip('/')}/gw/workflows/{quoted}/export"
    req = urllib.request.Request(url, method="GET")
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    if project:
        req.add_header("X-Project-ID", project)
    return req


def _http_error_message(exc: urllib.error.HTTPError) -> str:
    try:
        detail = json.loads(exc.read().decode("utf-8", errors="replace")).get("detail")
    except (OSError, ValueError, AttributeError):
        detail = None
    return f"HTTP {exc.code}: {detail or exc.reason}"


def _stream_to(req: urllib.request.Request, handle: Any) -> None:
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            for chunk in iter(lambda: resp.read(_CHUNK), b""):
                handle.write(chunk)
    except urllib.error.HTTPError as exc:
        raise ExportDownloadError(_http_error_message(exc)) from exc
    except urllib.error.URLError as exc:
        raise ExportDownloadError(f"Export request failed: {exc.reason}") from exc


def download_workflow_export(
    gateway_url: str,
    workflow_name: str,
    dest: str,
    *,
    api_key: str | None = None,
    project: str | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Fetch, verify and save the export at *dest*; returns EXPORT.json."""
    if os.path.exists(dest) and not overwrite:
        raise ExportDownloadError(f"{dest} exists (pass --force to overwrite)")
    req = _export_request(gateway_url, workflow_name, api_key, project)
    fd, tmp = tempfile.mkstemp(
        prefix=".umrs-export-", suffix=".zip", dir=os.path.dirname(dest)
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            _stream_to(req, handle)
        manifest = verify_export_archive(tmp)
        os.replace(tmp, dest)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return manifest


__all__ = [
    "ExportDownloadError",
    "download_workflow_export",
    "export_destination",
    "names_directory",
    "verify_export_archive",
]
