"""Process-local registry of in-flight research loop engines.

Companion / Bun rebuild ``ResearchProject`` per call, so cancel must reach
the engine that owns the run via this registry rather than ``project._loop_engine``.
"""

from __future__ import annotations

import threading
from typing import Any

_lock = threading.Lock()
_engines: dict[str, Any] = {}


def register_active_engine(run_id: str, engine: Any) -> None:
    if not run_id:
        return
    with _lock:
        _engines[run_id] = engine


def unregister_active_engine(run_id: str) -> None:
    if not run_id:
        return
    with _lock:
        _engines.pop(run_id, None)


def request_cancel_run(run_id: str) -> bool:
    """Signal cancel on the live engine for *run_id*. Returns True if found."""
    with _lock:
        engine = _engines.get(run_id)
    if engine is None:
        return False
    cancel = getattr(engine, "request_cancel", None)
    if callable(cancel):
        cancel()
        return True
    return False
