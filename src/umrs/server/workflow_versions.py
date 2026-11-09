"""Workflow versions (ADR-113): SemVer order, the bundle digest, version refs.

Pure helpers, stdlib only, shared by the upload, the catalog, the run path
and the first-start migration:

- :func:`parse_semver` / :func:`is_semver`: the SemVer 2.0 grammar without
  build metadata (``MAJOR.MINOR.PATCH`` with an optional ``-prerelease``).
- :func:`version_sort_key`: SemVer precedence (a release above its
  prereleases, prerelease identifiers compared field by field). Catalog rows
  written before ADR-113 may carry a non-SemVer version (``"0.1"``, the old
  upload default): a dotted-numeric one sorts as if padded with zeros, just
  below the SemVer version it pads to; anything else sorts lowest.
- :func:`default_entry`: the highest version among catalog rows, releases
  before prereleases (D-3), rows whose code is unavailable last.
- :func:`tree_digest`: ``source_sha256``, one digest over the sorted relative
  paths and contents of a registered tree (D-2).
- :func:`split_version_ref`: ``name@1.2.3`` into its name and version.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Iterable
from typing import Any

#: ``code`` of a catalog row whose files are gone (D-8 migration).
CODE_UNAVAILABLE = "unavailable"

_NUM = r"0|[1-9]\d*"
_PRE_ID = r"(?:0|[1-9]\d*|\d*[A-Za-z-][0-9A-Za-z-]*)"
SEMVER_RE = re.compile(
    rf"(?P<major>{_NUM})\.(?P<minor>{_NUM})\.(?P<patch>{_NUM})"
    rf"(?:-(?P<pre>{_PRE_ID}(?:\.{_PRE_ID})*))?"
)
_LENIENT_RE = re.compile(r"(\d+)(?:\.(\d+))?(?:\.(\d+))?")
#: A version-like name suffix (``-v1.2.3``, ``_1.2``): a warning (D-1).
_NAME_SUFFIX_RE = re.compile(
    r"[-_]v?\d+(?:\.\d+){1,2}(?:-[0-9A-Za-z.-]+)?$", re.IGNORECASE
)
#: A version usable as one directory name as it is.
_SAFE_DIR_RE = re.compile(r"[0-9A-Za-z][0-9A-Za-z.+_-]{0,127}")

ERROR_HTTP_STATUS = {
    "workflows.invalid_version": 400,
    "workflows.version_exists": 409,
    "workflows.version_not_found": 404,
    "workflows.code_unavailable": 409,
    "workflows.source_mismatch": 409,
    "workflows.bad_request": 400,
    "workflows.failure": 500,
}


class WorkflowVersionError(ValueError):
    """A version refusal by name (``workflows.<reason>``)."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code

    @property
    def http_status(self) -> int:
        return ERROR_HTTP_STATUS.get(self.code, 400)


def parse_semver(version: Any) -> tuple[int, int, int, str | None] | None:
    """``(major, minor, patch, prerelease)`` of a SemVer string, else None."""
    if not isinstance(version, str):
        return None
    match = SEMVER_RE.fullmatch(version)
    if match is None:
        return None
    return (
        int(match["major"]),
        int(match["minor"]),
        int(match["patch"]),
        match["pre"],
    )


def is_semver(version: Any) -> bool:
    return parse_semver(version) is not None


def invalid_version_message(version: Any) -> str:
    shown = "missing" if version in (None, "") else repr(str(version)[:64])
    return (
        "workflow.json must carry `version`, a semantic version "
        f"MAJOR.MINOR.PATCH[-prerelease] (e.g. 1.0.0); got {shown}"
    )


def _prerelease_key(pre: str) -> tuple[tuple[int, Any], ...]:
    """Numeric identifiers below alphanumeric ones, numbers by value; a
    longer list wins over its own prefix (tuple order does exactly that)."""
    return tuple((0, int(p)) if p.isdigit() else (1, p) for p in pre.split("."))


def version_sort_key(version: Any) -> tuple[Any, ...]:
    """Ascending sort key: SemVer precedence, legacy rows below (module doc)."""
    text = str(version or "")
    parsed = parse_semver(text)
    if parsed is not None:
        major, minor, patch, pre = parsed
        release = (1, ()) if pre is None else (0, _prerelease_key(pre))
        return (1, major, minor, patch, *release, 1, text)
    lenient = _LENIENT_RE.fullmatch(text)
    if lenient is not None:
        parts = [int(p or 0) for p in lenient.groups()]
        return (1, *parts, 1, (), 0, text)
    return (0, 0, 0, 0, 0, (), 0, text)


def is_prerelease(version: Any) -> bool:
    parsed = parse_semver(version)
    return parsed is not None and parsed[3] is not None


def sort_versions_desc(versions: Iterable[Any]) -> list[str]:
    """Distinct versions, highest first."""
    unique = {str(v) for v in versions if v not in (None, "")}
    return sorted(unique, key=version_sort_key, reverse=True)


def entry_available(entry: dict[str, Any]) -> bool:
    return entry.get("code") != CODE_UNAVAILABLE


def default_entry(entries: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The default among *entries* (D-3): rows with code first, releases
    before prereleases, then the highest version."""
    if not entries:
        return None

    def rank(entry: dict[str, Any]) -> tuple[Any, ...]:
        version = entry.get("version")
        return (
            entry_available(entry),
            not is_prerelease(version),
            version_sort_key(version),
        )

    return max(entries, key=rank)


def split_version_ref(name: str) -> tuple[str, str | None]:
    """``name@1.2.3`` -> ``("name", "1.2.3")``; a plain name keeps None.
    Workflow names never hold ``@`` (``WORKFLOW_NAME_RE``)."""
    base, sep, version = str(name).rpartition("@")
    if not sep or not base or not version:
        return name, None
    return base, version


def requested_version(name: str, version: Any) -> tuple[str, str | None]:
    """The (name, version) a request names: an explicit ``version`` wins
    over a ``name@X`` shorthand; a conflict between the two is refused."""
    base, ref_version = split_version_ref(name)
    explicit = str(version).strip() if version not in (None, "") else None
    if explicit and ref_version and explicit != ref_version:
        raise WorkflowVersionError(
            "workflows.invalid_version",
            f"version {explicit!r} contradicts {name!r}",
        )
    return base, explicit or ref_version


def name_suffix_warning(name: str) -> str | None:
    """Advice when the name carries a version (D-1): a warning, never a refusal."""
    if not _NAME_SUFFIX_RE.search(str(name)):
        return None
    return (
        f"workflow name {name!r} ends in a version-like suffix: keep the name "
        "stable and put the version in workflow.json's `version` field "
        "(ADR-113 D-1)"
    )


def version_dir_name(version: str) -> str:
    """The directory one version's files live in: the version itself when it
    is a safe path segment (every SemVer is), else a stable digest name."""
    if _SAFE_DIR_RE.fullmatch(version) and version not in (".", ".."):
        return version
    return "legacy-" + hashlib.sha256(version.encode("utf-8")).hexdigest()[:12]


# ── source digest ──────────────────────────────────────────────────────────


def _normalized_workflow_json(root: str, raw: bytes) -> bytes:
    """workflow.json as the digest sees it: canonical JSON, ``source`` cut to
    its path relative to *root*. The upload rewrites ``source`` to the
    absolute path of the tree, so the digest of a bundle stays the same
    before and after that rewrite and wherever the tree lives."""
    try:
        data = json.loads(raw)
    except ValueError:
        return raw
    if not isinstance(data, dict):
        return raw
    source = data.get("source")
    if isinstance(source, dict) and isinstance(source.get("path"), str):
        real_root = os.path.realpath(root)
        resolved = os.path.realpath(os.path.join(real_root, source["path"]))
        if os.path.commonpath([real_root, resolved]) == real_root:
            data["source"] = {"path": os.path.relpath(resolved, real_root)}
    return json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _file_digest(root: str, rel: str) -> str:
    path = os.path.join(root, rel)
    with open(path, "rb") as handle:
        if rel == "workflow.json":
            return hashlib.sha256(
                _normalized_workflow_json(root, handle.read())
            ).hexdigest()
        digest = hashlib.sha256()
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
        return digest.hexdigest()


def tree_files(root: str) -> list[str]:
    """Regular files under *root* as sorted relative POSIX paths (links and
    special files are not part of a bundle)."""
    rows: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for filename in filenames:
            full = os.path.join(dirpath, filename)
            if os.path.isfile(full) and not os.path.islink(full):
                rows.append(os.path.relpath(full, root).replace(os.sep, "/"))
    return sorted(rows)


def tree_digest(root: str) -> str:
    """``source_sha256``: sha256 over each file's relative path and content
    digest, in sorted path order. Raises OSError when *root* is unreadable."""
    if not os.path.isdir(root):
        raise FileNotFoundError(root)
    digest = hashlib.sha256()
    for rel in tree_files(root):
        digest.update(rel.encode("utf-8") + b"\0")
        digest.update(_file_digest(root, rel).encode("ascii") + b"\n")
    return digest.hexdigest()


def tree_digest_or_none(root: str) -> str | None:
    try:
        return tree_digest(root) if root else None
    except OSError:
        return None


__all__ = [
    "CODE_UNAVAILABLE",
    "SEMVER_RE",
    "WorkflowVersionError",
    "default_entry",
    "entry_available",
    "invalid_version_message",
    "is_prerelease",
    "is_semver",
    "name_suffix_warning",
    "parse_semver",
    "requested_version",
    "sort_versions_desc",
    "split_version_ref",
    "tree_digest",
    "tree_digest_or_none",
    "tree_files",
    "version_dir_name",
    "version_sort_key",
]
