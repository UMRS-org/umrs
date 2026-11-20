"""Pipe-separated backend references (U3 S1 item 3, ADR-117).

An attempt's ``backend_ref`` is ``<kind>|key=value|key=value``. This lenient
parser returns the ``key=value`` pairs and skips anything else (the kind
included). It has no imports, so any job module can use it without an
import cycle. ``adapters._parse_backend_ref`` is a different, strict parser
(it requires a kind and records it) and stays where it is.
"""

from __future__ import annotations


def parse_backend_ref(ref: str) -> dict[str, str]:
    """``"kind|a=1|b=2"`` → ``{"a": "1", "b": "2"}``; parts without ``=`` are skipped."""
    out: dict[str, str] = {}
    for part in ref.split("|"):
        key, sep, value = part.partition("=")
        if sep:
            out[key] = value
    return out


__all__ = ["parse_backend_ref"]
