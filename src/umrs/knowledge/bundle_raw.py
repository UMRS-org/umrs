"""Opt-in raw source materials in a knowledge bundle (ADR-032 amendment
2026-09-24, raw-materials carriage).

Layout (additive to r4-kb v1.2): ``raw/<content_hash>`` holds one file per
distinct object; ``raw_objects.jsonl`` lists ``{artifact_id, content_hash,
size_bytes, mime_type, file}``. The table is pinned by sha256 in the signed
manifest (addendum 8), and every row pins its object's sha256, so the
signature covers the bytes transitively. Import recomputes every hash and
refuses the bundle on any mismatch or unlisted file.

Export eligibility: the catalog states a licence, the D-46 ``raw`` redaction
rule does not drop the artifact's kind, the object is readable, its bytes
hash to the row's content_hash, and it fits the per-object and total caps.
Everything else is skipped and counted by reason, never silently dropped.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import canonical_identity

RAW_DIR = "raw"
RAW_TABLE = "raw_objects.jsonl"

# The total default stays under the import route's 2 GiB upload cap
# (raw objects such as PDFs barely compress).
_OBJECT_MAX_DEFAULT = 64 * 1024 * 1024
_TOTAL_MAX_DEFAULT = 1536 * 1024 * 1024
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class RawBundleError(ValueError):
    """The bundle's raw objects do not match what its signed table pins."""


@dataclass(frozen=True)
class RawCaps:
    object_max: int = _OBJECT_MAX_DEFAULT
    total_max: int = _TOTAL_MAX_DEFAULT

    @classmethod
    def from_env(cls) -> RawCaps:
        """Site caps (can only tighten the defaults)."""
        return cls(
            object_max=_env_cap("UMRS_KB_RAW_OBJECT_MAX_BYTES", _OBJECT_MAX_DEFAULT),
            total_max=_env_cap("UMRS_KB_RAW_TOTAL_MAX_BYTES", _TOTAL_MAX_DEFAULT),
        )


def _env_cap(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, "") or default)
    except ValueError:
        return default
    return min(value, default) if value > 0 else default


def _count(skipped: dict[str, int], reason: str, n: int = 1) -> None:
    skipped[reason] = skipped.get(reason, 0) + n


def raw_redaction(rules: Iterable[Any] | None) -> tuple[bool, frozenset[str]]:
    """D-46 ``{"target": "raw", "action": "drop", "kinds": [...]}`` →
    (drop_all, dropped_kinds). A rule without ``kinds`` drops every object."""
    drop_all, kinds = False, set()
    for rule in rules or []:
        if not isinstance(rule, dict) or rule.get("target") != "raw":
            continue
        if rule.get("action") != "drop":
            continue
        listed = rule.get("kinds")
        if not listed:
            drop_all = True
        kinds.update(str(k) for k in listed or [])
    return drop_all, frozenset(kinds)


def plan_raw(
    rows: list[dict[str, Any]],
    *,
    licence: str,
    redaction_rules: Iterable[Any] | None,
    caps: RawCaps,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Candidate artifact rows (one per distinct content_hash) + skip counts."""
    skipped: dict[str, int] = {}
    if not licence.strip():
        _count(skipped, "no_licence", len(rows))
        return [], skipped
    drop_all, dropped = raw_redaction(redaction_rules)
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        reason = _ineligible(row, drop_all, dropped, caps, seen)
        if reason:
            _count(skipped, reason)
            continue
        seen.add(str(row["content_hash"]))
        out.append(row)
    return out, skipped


def _ineligible(
    row: dict[str, Any],
    drop_all: bool,
    dropped: frozenset[str],
    caps: RawCaps,
    seen: set[str],
) -> str:
    """The skip reason for one artifact row ("" = a candidate)."""
    reason = _without_object(row)
    if reason:
        return reason
    if str(row["content_hash"]) in seen:
        return "duplicate"
    if drop_all or str(row.get("kind") or "") in dropped:
        return "redacted"
    if int(row.get("size_bytes") or 0) > caps.object_max:
        return "over_object_cap"
    return ""


def _without_object(row: dict[str, Any]) -> str:
    if not _HEX64.match(str(row.get("content_hash") or "")):
        return "no_bytes"  # synthetic digest: nothing content-addressed to carry
    if not canonical_identity.stored_object(str(row.get("source_ref") or "")):
        return "no_object"  # minted without bytes (loop findings, legacy rows)
    return ""


def write_raw(
    bundle_dir: Path,
    candidates: list[dict[str, Any]],
    reader: Callable[[str], bytes | None] | None,
    caps: RawCaps,
    skipped: dict[str, int],
) -> list[dict[str, Any]]:
    """Write the objects that verify and fit; return the table rows."""
    rows: list[dict[str, Any]] = []
    total = 0
    (bundle_dir / RAW_DIR).mkdir(parents=True, exist_ok=True)
    for row in candidates:
        raw = reader(str(row.get("source_ref") or "")) if reader else None
        reason = _unwritable(raw, row, caps, total)
        if reason:
            _count(skipped, reason)
            continue
        assert raw is not None
        digest = str(row["content_hash"])
        (bundle_dir / RAW_DIR / digest).write_bytes(raw)
        total += len(raw)
        rows.append(
            {
                "artifact_id": str(row.get("artifact_id") or ""),
                "content_hash": digest,
                "size_bytes": len(raw),
                "mime_type": str(row.get("mime_type") or ""),
                "file": f"{RAW_DIR}/{digest}",
            }
        )
    with (bundle_dir / RAW_TABLE).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return rows


def _unwritable(
    raw: bytes | None, row: dict[str, Any], caps: RawCaps, total: int
) -> str:
    if raw is None:
        return "unreadable"
    if len(raw) > caps.object_max:
        return "over_object_cap"
    if total + len(raw) > caps.total_max:
        return "over_total_cap"
    if hashlib.sha256(raw).hexdigest() != row["content_hash"]:
        return "hash_mismatch"  # the stored object is not what the row names
    return ""


def summary(
    rows: list[dict[str, Any]], skipped: dict[str, int], **extra: Any
) -> dict[str, Any]:
    """The manifest ``raw`` block."""
    return {
        "included": True,
        "count": len(rows),
        "bytes": sum(int(r["size_bytes"]) for r in rows),
        "skipped": dict(sorted(skipped.items())),
        **extra,
    }


def verified_raw_rows(bundle_dir: Path) -> list[dict[str, Any]]:
    """The bundle's raw rows, each object re-hashed; RawBundleError on any
    mismatch, missing object or unlisted file under ``raw/``."""
    table = bundle_dir / RAW_TABLE
    if not table.is_file():
        if _present_objects(bundle_dir):
            raise RawBundleError("raw objects present without raw_objects.jsonl")
        return []
    rows = [
        json.loads(line)
        for line in table.read_text("utf-8").splitlines()
        if line.strip()
    ]
    listed = {_check_row(bundle_dir, row) for row in rows}
    unlisted = _present_objects(bundle_dir) - listed
    if unlisted:
        raise RawBundleError(f"unlisted raw objects: {sorted(unlisted)[:3]}")
    return rows


def _present_objects(bundle_dir: Path) -> set[str]:
    raw_dir = bundle_dir / RAW_DIR
    return {p.name for p in raw_dir.iterdir()} if raw_dir.is_dir() else set()


def _check_row(bundle_dir: Path, row: dict[str, Any]) -> str:
    digest = str(row.get("content_hash") or "")
    if not _HEX64.match(digest) or row.get("file") != f"{RAW_DIR}/{digest}":
        raise RawBundleError(f"malformed raw row: {row.get('file')!r}")
    path = bundle_dir / RAW_DIR / digest
    if not path.is_file():
        raise RawBundleError(f"raw object missing: {digest[:12]}")
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise RawBundleError(f"raw object hash mismatch: {digest[:12]}")
    return digest


def read_object(bundle_dir: Path, row: dict[str, Any]) -> bytes:
    return (bundle_dir / RAW_DIR / str(row["content_hash"])).read_bytes()


__all__ = [
    "RAW_DIR",
    "RAW_TABLE",
    "RawBundleError",
    "RawCaps",
    "plan_raw",
    "raw_redaction",
    "read_object",
    "summary",
    "verified_raw_rows",
    "write_raw",
]
