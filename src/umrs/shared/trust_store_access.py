"""Process-global trust store accessor (ADR-053, jobs↔api cycle closure).

`umrs.api.companion` previously owned the singleton factory/global used both
by its routes and by jobs' archive cascade (`jobs/session_gc.py`, a
foundation layer). The singleton now lives in the shared layer; the api
module re-exports `get_trust_store`/`set_trust_store` unchanged so api
callers are byte-identical (step 1.1).
"""

from __future__ import annotations

from typing import Any

_trust_store = None


def set_trust_store(store: Any) -> None:
    """Install the process trust store (sidecar boot) — once."""
    global _trust_store
    _trust_store = store


def get_trust_store():
    """Return the installed trust store (or None when not wired)."""
    return _trust_store


__all__ = ["get_trust_store", "set_trust_store"]
