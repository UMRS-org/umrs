"""Canonical content identity (ADR-032 precondition #1, contract ``cid-1``).

``physical_content_digest → canonical_content_identity → document_id``.

The normalization is chosen from the bytes alone (never from upload
metadata such as the MIME type, which differs between deployments):

- ``text-v1`` — the bytes decode as strict UTF-8 with no NUL: strip one
  leading BOM, Unicode NFC, CRLF / CR → LF, trailing spaces and tabs removed
  from every line, trailing newlines removed; identity = sha256 of the
  normalized UTF-8. The same text saved with a different line-ending or
  editor convention converges on one ``document_id``.
- ``bytes`` — anything else: identity = sha256 of the raw bytes.
- ``physical`` — the bytes were not readable at export time: identity = the
  stored physical digest (raw sha256). Weaker, never wrong: it can miss a
  text match, it never merges two different documents.

Existing rows keep their deployment-local ids; the identity is a map emitted
at export time and carried on import (never a rewrite in place).
"""

from __future__ import annotations

import hashlib
import unicodedata
from dataclasses import dataclass
from typing import Any

CONTRACT = "cid-1"
TEXT_V1 = "text-v1"
BYTES = "bytes"
PHYSICAL = "physical"
NORMALIZATIONS = (TEXT_V1, BYTES, PHYSICAL)

_BOM = b"\xef\xbb\xbf"
_DOCUMENT_ID_PREFIX = "kdoc-"


@dataclass(frozen=True)
class CanonicalIdentity:
    identity: str  # sha256 hex of the normalized payload
    normalization: str  # one of NORMALIZATIONS

    @property
    def document_id(self) -> str:
        return f"{_DOCUMENT_ID_PREFIX}{self.identity[:32]}"

    def to_dict(self) -> dict[str, str]:
        return {
            "contract": CONTRACT,
            "document_id": self.document_id,
            "identity": self.identity,
            "normalization": self.normalization,
        }


def normalize_text(raw: bytes) -> bytes | None:
    """The ``text-v1`` payload, or None when the bytes are not UTF-8 text."""
    if b"\x00" in raw:
        return None
    try:
        text = raw.removeprefix(_BOM).decode("utf-8")
    except UnicodeDecodeError:
        return None
    text = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip(" \t") for line in text.split("\n")]
    return "\n".join(lines).rstrip("\n").encode("utf-8")


def identity_of(raw: bytes | None, physical_digest: str) -> CanonicalIdentity:
    """Canonical identity of one artifact's content (``raw`` None = unreadable)."""
    if raw is None:
        return CanonicalIdentity(physical_digest, PHYSICAL)
    text = normalize_text(raw)
    if text is not None:
        return CanonicalIdentity(hashlib.sha256(text).hexdigest(), TEXT_V1)
    return CanonicalIdentity(hashlib.sha256(raw).hexdigest(), BYTES)


def from_row(row: dict[str, Any]) -> dict[str, str] | None:
    """The ``canonical_identity`` block of an exported artifact row, when it
    names this contract and a known normalization (anything else: None)."""
    block = row.get("canonical_identity")
    if not isinstance(block, dict) or block.get("contract") != CONTRACT:
        return None
    if block.get("normalization") not in NORMALIZATIONS:
        return None
    identity = str(block.get("identity") or "")
    if len(identity) != 64:
        return None
    return CanonicalIdentity(identity, str(block["normalization"])).to_dict()


__all__ = [
    "BYTES",
    "CONTRACT",
    "NORMALIZATIONS",
    "PHYSICAL",
    "TEXT_V1",
    "CanonicalIdentity",
    "from_row",
    "identity_of",
    "normalize_text",
]
