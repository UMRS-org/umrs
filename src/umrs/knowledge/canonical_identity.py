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

import codecs
import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Any

CONTRACT = "cid-1"
TEXT_V1 = "text-v1"
BYTES = "bytes"
PHYSICAL = "physical"
NORMALIZATIONS = (TEXT_V1, BYTES, PHYSICAL)

# Artifacts above this size are not read for an identity (export, backfill).
READ_CAP = 64 * 1024 * 1024

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


class IdentityStream:
    """:func:`identity_of` over bytes fed in parts, never holding the whole.

    Ingest copies a staged object of up to 512 MiB; reading it whole to stamp
    it took UMRS peak RSS past 3 GB (cycle-3 P1, 2026-10-22). ``text-v1`` is
    computed a line at a time: NFC never composes across a line break, so
    the normalized lines are the lines normalized. A line longer than
    ``_FLUSH_CHARS`` is flushed at a boundary between two ASCII characters
    (no composition and no reordering crosses one), keeping its trailing
    spaces and tabs back for the line's end. ``feed`` the parts in order,
    then ``identity`` once.
    """

    def __init__(self) -> None:
        self._raw = hashlib.sha256()
        self._text: Any = hashlib.sha256()  # None once the bytes are not text
        self._decoder = codecs.getincrementaldecoder("utf-8")()
        self._lead = b""  # the first bytes, until a leading BOM is decided
        self._carry = ""  # the current line, not yet normalized
        self._owed = 0  # line breaks owed before the next written character
        self._open = False  # the current line has written characters
        self._next_flush = _FLUSH_CHARS

    def feed(self, part: bytes) -> None:
        self._raw.update(part)
        if len(self._lead) < len(_BOM):
            self._lead += part
            if len(self._lead) < len(_BOM) and _BOM.startswith(self._lead):
                return
            part, self._lead = self._lead.removeprefix(_BOM), _BOM
        self._decode(part, final=False)

    def identity(self) -> CanonicalIdentity:
        """The identity of every byte fed."""
        if len(self._lead) < len(_BOM):  # fewer bytes than a BOM, all a BOM's start
            self._decode(self._lead, final=False)
        self._decode(b"", final=True)
        if self._text is None:
            return CanonicalIdentity(self._raw.hexdigest(), BYTES)
        return CanonicalIdentity(self._text.hexdigest(), TEXT_V1)

    def _decode(self, part: bytes, *, final: bool) -> None:
        if self._text is None:
            return
        try:
            if b"\x00" in part:
                raise UnicodeDecodeError("utf-8", part, 0, 1, "NUL")
            text = self._decoder.decode(part, final)
        except UnicodeDecodeError:
            self._text = None
            return
        self._lines(self._carry + text, final=final)

    def _lines(self, text: str, *, final: bool) -> None:
        hold = "\r" if not final and text.endswith("\r") else ""
        text = text[: len(text) - len(hold)].replace("\r\n", "\n").replace("\r", "\n")
        *lines, rest = text.split("\n")
        for line in lines:
            self._end_line(line)
        if final:
            self._end_line(rest)
            return
        self._carry = self._flush_long(rest) + hold

    def _flush_long(self, line: str) -> str:
        """The unwritten tail of a long open line (its safe head written)."""
        if len(line) < self._next_flush:
            return line
        match = _ASCII_PAIR_REVERSED.search(line[::-1])
        if match is None:
            self._next_flush = len(line) + _FLUSH_CHARS
            return line
        cut = len(line) - 1 - match.start()
        self._write(unicodedata.normalize("NFC", line[:cut]))
        self._next_flush = _FLUSH_CHARS
        return line[cut:]

    def _end_line(self, line: str) -> None:
        self._write(unicodedata.normalize("NFC", line).rstrip(" \t"))
        self._owed += 1  # the break after this line; trailing ones are never written
        self._open = False

    def _write(self, chunk: str) -> None:
        if not chunk:
            return
        if not self._open and self._owed:
            self._text.update(b"\n" * self._owed)
            self._owed = 0
        self._open = True
        self._text.update(chunk.encode("utf-8"))


# A text-v1 flush point, searched in the reversed line: an ASCII character
# that follows (in the line) an ASCII character other than a space or a tab
# (rstrip's set: a line's trailing whitespace is decided at its end).
_ASCII_PAIR_REVERSED = re.compile(r"[\x00-\x7f][\x00-\x08\x0a-\x1f\x21-\x7f]")
_FLUSH_CHARS = 1 << 20


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


def stored_object(source_ref: str) -> bool:
    """Whether ``source_ref`` names an object in artifact storage
    (``projects/<p>/knowledge/artifacts/<hash>``). Rows minted without bytes
    (``rag://`` legacy corpus rows, ``loop:`` findings) have none — there is
    nothing to fetch, so callers skip them instead of failing a read."""
    return source_ref.startswith("projects/") and "/knowledge/artifacts/" in source_ref


def metadata_dict(value: Any) -> dict[str, Any]:
    """Artifact metadata as a dict: pools without a JSON codec return JSONB
    as text; anything undecodable or non-object is empty."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return {}
    return dict(value) if isinstance(value, dict) else {}


def carried(metadata: Any) -> dict[str, str] | None:
    """The well-formed block an artifact's metadata already carries (stamped
    at ingest, by the backfill, or by an earlier import), else None."""
    return from_row(metadata_dict(metadata))


def stamp(raw: bytes, physical_digest: str) -> dict[str, dict[str, str]]:
    """Metadata fragment stamping bytes in hand at ingest time."""
    return stamp_identity(identity_of(raw, physical_digest))


def stamp_identity(identity: CanonicalIdentity) -> dict[str, dict[str, str]]:
    """Metadata fragment stamping an identity already computed (an
    :class:`IdentityStream` over bytes never held whole)."""
    return {"canonical_identity": identity.to_dict()}


__all__ = [
    "BYTES",
    "CONTRACT",
    "NORMALIZATIONS",
    "PHYSICAL",
    "READ_CAP",
    "TEXT_V1",
    "CanonicalIdentity",
    "IdentityStream",
    "carried",
    "from_row",
    "identity_of",
    "metadata_dict",
    "normalize_text",
    "stamp",
    "stamp_identity",
    "stored_object",
]
