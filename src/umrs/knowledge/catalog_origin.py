"""Catalog origin identity (DEC-D39, slice S-D39a).

A bundle's manifest ``catalog`` block names the catalog at its source
installation (``catalog_id``) and that installation (``origin_fingerprint``).
The import records both on the new local row as ``origin_catalog_id`` and
``origin_fingerprint`` (migration 0144).

Both values come from a file someone handed us, so they are labels, never
keys: the import mints its own local ``catalog_id`` and never reuses the
bundle's. A value that does not match its strict shape is dropped (read as
"" = unknown origin), never stored as-is.

The fingerprint is a hash of the installation's random id
(``umrs_installation``); the id itself, hostnames and addresses never leave
the installation. Same behaviour hosted and self-hosted.

Pure module: stdlib only, no database, no ``umrs.api`` import (ADR-053).
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

FINGERPRINT_PREFIX = "inst-"
# A catalog id as minted by any UMRS (``scat-<16 hex>``) or set by a caller:
# printable, no whitespace, no path or markup characters, bounded.
_CATALOG_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:@-]{0,127}$")
_FINGERPRINT_RE = re.compile(r"^inst-[0-9a-f]{32}$")
_DOMAIN = b"umrs-installation:"


def installation_fingerprint(installation_id: str) -> str:
    """``inst-`` + 32 hex of a domain-separated SHA-256 of the id ("" for "")."""
    raw = str(installation_id or "").strip()
    if not raw:
        return ""
    digest = hashlib.sha256(_DOMAIN + raw.encode("utf-8")).hexdigest()
    return FINGERPRINT_PREFIX + digest[:32]


def clean_catalog_id(value: Any) -> str:
    """The value when it is a well-formed catalog id, else ""."""
    text = value if isinstance(value, str) else ""
    return text if _CATALOG_ID_RE.fullmatch(text) else ""


def clean_fingerprint(value: Any) -> str:
    """The value when it is a well-formed installation fingerprint, else ""."""
    text = value if isinstance(value, str) else ""
    return text if _FINGERPRINT_RE.fullmatch(text) else ""


def origin_from_block(block: Any) -> tuple[str, str]:
    """(origin_catalog_id, origin_fingerprint) from a manifest catalog block.

    ``origin_catalog_id`` is the block's ``catalog_id`` (the id at the
    installation that exported it); ``origin_fingerprint`` is the block's
    ``origin_fingerprint``. Malformed or absent values read as "".
    """
    if not isinstance(block, dict):
        return "", ""
    return (
        clean_catalog_id(block.get("catalog_id")),
        clean_fingerprint(block.get("origin_fingerprint")),
    )


MAX_CHANGE_NOTES = 256
MAX_NOTE_CHARS = 4000
MAX_VERSION_CHARS = 64


def _note_entry(entry: Any) -> dict[str, str] | None:
    if not isinstance(entry, dict):
        return None
    version = entry.get("version")
    note = entry.get("change_note")
    if not isinstance(version, str) or not version.strip():
        return None
    return {
        "version": version[:MAX_VERSION_CHARS],
        "change_note": note[:MAX_NOTE_CHARS] if isinstance(note, str) else "",
    }


def clean_change_notes(value: Any) -> list[dict[str, str]]:
    """A bundle's ``change_notes`` chain, bounded and shape-checked.

    Keeps at most the LAST ``MAX_CHANGE_NOTES`` well-formed entries
    (``{"version": non-empty str, "change_note": str}``), truncating long
    strings; anything else is dropped.
    """
    if not isinstance(value, list):
        return []
    entries = [e for e in (_note_entry(v) for v in value) if e is not None]
    return entries[-MAX_CHANGE_NOTES:]


def version_note(chain: list[dict[str, str]], version: str) -> str:
    """The change note recorded for ``version`` in a cleaned chain ("" if none)."""
    for entry in reversed(chain):
        if entry["version"] == version:
            return entry["change_note"]
    return ""
