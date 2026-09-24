"""Portable ``.kb.tar.gz`` archives for knowledge bundles (D-40 transport leg).

A bundle is a directory (``manifest.json`` + tables). To leave or enter an
installation it travels as one gzip'd tar:

- export packs the bundle directory and uploads it under the exporting
  project's key prefix (``<project>/knowledge-exports/<job>.kb.tar.gz``),
  which the gateway's project-scoped download route serves as-is;
- import receives the archive's bytes and unpacks it into a private
  temporary directory with :func:`safe_extract` — never a caller-named path.

Extraction is hostile-input safe: regular files and directories only (no
links, devices or FIFOs), no absolute or ``..`` member paths, and caps on
member count and total unpacked bytes.
"""

from __future__ import annotations

import gzip
import shutil
import tarfile
import zlib
from pathlib import Path, PurePosixPath
from typing import IO

ARCHIVE_SUFFIX = ".kb.tar.gz"
_ARCNAME = "bundle"
MAX_MEMBERS = 100_000
MAX_UNPACKED_BYTES = 8 * 1024 * 1024 * 1024  # 8 GiB


class BundleArchiveError(ValueError):
    """The archive is not a safe, well-formed knowledge bundle."""


def export_archive_key(project_id: str, job_id: str) -> str:
    """Project-scoped object key the export job writes (download-route shape)."""
    return f"{project_id}/knowledge-exports/{job_id}{ARCHIVE_SUFFIX}"


def pack_bundle(bundle_dir: Path, dest: Path) -> Path:
    """Write ``bundle_dir`` as a gzip'd tar at ``dest`` (members under ``bundle/``)."""
    if not (bundle_dir / "manifest.json").is_file():
        raise BundleArchiveError(
            f"not a bundle directory (no manifest.json): {bundle_dir}"
        )
    with tarfile.open(dest, "w:gz") as tar:
        tar.add(str(bundle_dir), arcname=_ARCNAME)
    return dest


def _member_target(member: tarfile.TarInfo) -> PurePosixPath:
    """Validated relative path of one member (raises on anything unsafe)."""
    path = PurePosixPath(member.name)
    if path.is_absolute() or ".." in path.parts:
        raise BundleArchiveError(f"unsafe member path: {member.name!r}")
    if not (member.isfile() or member.isdir()):
        raise BundleArchiveError(
            f"unsupported member type (links/devices refused): {member.name!r}"
        )
    return path


def _checked_members(tar: tarfile.TarFile) -> list[tarfile.TarInfo]:
    members = tar.getmembers()
    if len(members) > MAX_MEMBERS:
        raise BundleArchiveError(
            f"archive has {len(members)} members (cap {MAX_MEMBERS})"
        )
    total = 0
    for member in members:
        _member_target(member)
        total += member.size
        if total > MAX_UNPACKED_BYTES:
            raise BundleArchiveError("archive unpacks beyond the size cap")
    return members


def _write_member(tar: tarfile.TarFile, member: tarfile.TarInfo, dest: Path) -> None:
    """Write one validated member under ``dest`` (content only, no metadata)."""
    target = dest.joinpath(*_member_target(member).parts)
    if member.isdir():
        target.mkdir(parents=True, exist_ok=True)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    source = tar.extractfile(member)
    if source is None:
        raise BundleArchiveError(f"unreadable member: {member.name!r}")
    with source, open(target, "wb") as out:
        shutil.copyfileobj(source, out)


def _bundle_root(dest: Path) -> Path:
    """The directory holding manifest.json: the top level or its single child."""
    if (dest / "manifest.json").is_file():
        return dest
    children = [p for p in dest.iterdir() if p.is_dir()]
    if len(children) == 1 and (children[0] / "manifest.json").is_file():
        return children[0]
    raise BundleArchiveError("archive holds no bundle (manifest.json not found)")


def safe_extract(fileobj: IO[bytes], dest: Path) -> Path:
    """Unpack a gzip'd tar from ``fileobj`` into ``dest``; return the bundle root.

    Every member is validated before anything is written; files are written
    by this function (never ``tar.extract``), so no member metadata (modes,
    owners, link targets) is honoured.
    """
    try:
        with tarfile.open(fileobj=fileobj, mode="r:gz") as tar:
            for member in _checked_members(tar):
                _write_member(tar, member, dest)
    except (tarfile.TarError, EOFError, zlib.error, gzip.BadGzipFile) as exc:
        raise BundleArchiveError(f"not a readable gzip'd tar archive: {exc}") from exc
    return _bundle_root(dest)


__all__ = [
    "ARCHIVE_SUFFIX",
    "BundleArchiveError",
    "export_archive_key",
    "pack_bundle",
    "safe_extract",
]
