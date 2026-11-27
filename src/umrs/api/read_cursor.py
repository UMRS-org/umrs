"""Opaque, query-bound list cursors (ADR-105 D-6, slice S6a).

A shaped list read answers ``next_cursor`` when more rows follow; the
caller passes it back as ``cursor=`` to read the next page. The token is
opaque to callers: base64url (no ``=`` padding) of compact canonical JSON

    {"k": <scope key>, "p": {<position fields>}, "v": 1}

serialised with ``json.dumps(..., sort_keys=True, separators=(",", ":"))``.

The cursor carries a *position only*. It never widens what a caller may
read: the route always derives the caller's scope from the verified
identity and binds the cursor to a ``scope_key`` of the effective
filters, so a cursor minted for another query, owner or project is
refused (``CursorError("cursor belongs to another query")``) instead of
being reinterpreted. Nothing secret is in a cursor; it is not signed.

Every failure raises ``CursorError`` with one of the two fixed messages
below; the routes answer them as a typed 400. Pure: no I/O.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
from collections.abc import Mapping
from typing import TypeGuard

CURSOR_VERSION = 1
#: Longest token ``decode_cursor`` accepts (characters).
MAX_CURSOR_LEN = 512

MALFORMED = "cursor is malformed"
OTHER_QUERY = "cursor belongs to another query"


class CursorError(ValueError):
    """A cursor that cannot be used; ``str(exc)`` is safe to return."""


def _is_position(value: object) -> TypeGuard[Mapping[str, str]]:
    """Whether ``value`` is a non-empty mapping of strings to strings."""
    return (
        isinstance(value, Mapping)
        and bool(value)
        and all(
            isinstance(key, str) and isinstance(item, str)
            for key, item in value.items()
        )
    )


def _decode_payload(token: str) -> object:
    """Decode the token's transport and JSON payload."""
    try:
        padded = token + "=" * (-len(token) % 4)
        decoded = base64.urlsafe_b64decode(padded)
        return json.loads(decoded.decode("utf-8"))
    except (binascii.Error, UnicodeDecodeError, ValueError) as exc:
        raise CursorError(MALFORMED) from exc


def _payload_parts(payload: object) -> tuple[str, Mapping[str, str]]:
    """Validate the payload shape and return its scope and position."""
    if not isinstance(payload, dict) or set(payload) != {"k", "p", "v"}:
        raise CursorError(MALFORMED)
    version = payload["v"]
    key = payload["k"]
    position = payload["p"]
    if (
        type(version) is not int
        or version != CURSOR_VERSION
        or not isinstance(key, str)
        or not _is_position(position)
    ):
        raise CursorError(MALFORMED)
    return key, position


def scope_key(filters: Mapping[str, str | None]) -> str:
    """A short, stable key for the effective filters of one list query.

    Drop the entries whose value is None, then hash the canonical JSON of
    the rest (``json.dumps(d, sort_keys=True, separators=(",", ":"))``,
    UTF-8) with SHA-256 and return the first 16 lowercase hex characters.
    Same filters in any insertion order give the same key; a different
    value (``""`` included) gives a different key.
    """
    effective = {key: value for key, value in filters.items() if value is not None}
    text = json.dumps(effective, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def encode_cursor(position: Mapping[str, str], scope: str) -> str:
    """The token for ``position`` under ``scope`` (format in the module doc).

    ``position`` must be a non-empty mapping of ``str`` keys to ``str``
    values, and ``scope`` a non-empty ``str``; otherwise raise
    ``CursorError(MALFORMED)``. Output alphabet: ``A-Z a-z 0-9 - _`` only.
    """
    if not _is_position(position) or not isinstance(scope, str) or not scope:
        raise CursorError(MALFORMED)
    payload = {"k": scope, "p": dict(position), "v": CURSOR_VERSION}
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return base64.urlsafe_b64encode(text.encode("utf-8")).decode("ascii").rstrip("=")


def decode_cursor(token: str, scope: str) -> dict[str, str]:
    """The position fields of ``token``; it must have been minted for ``scope``.

    Raise ``CursorError(MALFORMED)`` when ``token`` is not a ``str``, is
    empty, is longer than ``MAX_CURSOR_LEN``, has a character outside
    ``A-Z a-z 0-9 - _``, does not base64url-decode (after re-adding the
    ``=`` padding), is not UTF-8, is not JSON, is not a JSON object with
    exactly the keys ``k``, ``p`` and ``v``, has ``v`` other than the int
    ``CURSOR_VERSION`` (``true`` is not 1), has ``k`` not a ``str``, or has
    ``p`` not a non-empty object of string values.

    Raise ``CursorError(OTHER_QUERY)`` when the token is well formed but
    ``k`` differs from ``scope``.

    Return a new ``dict`` equal to ``p``.
    """
    if not isinstance(token, str) or not token or len(token) > MAX_CURSOR_LEN:
        raise CursorError(MALFORMED)
    if re.fullmatch(r"[A-Za-z0-9_-]+", token) is None:
        raise CursorError(MALFORMED)
    key, position = _payload_parts(_decode_payload(token))
    if key != scope:
        raise CursorError(OTHER_QUERY)
    return dict(position)
