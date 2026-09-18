"""Process-wide job-cancel gateway (orchestrator-safe seam).

The registry/janitor/adapter layers live BELOW ``umrs.api`` and may not
import from it (Boundary-Check hard-fail — surfaced live 2026-09-18 when
the WP-22 reaper demanded a jobs cancel from inside orchestrator). The
api layer registers its store accessor here at import time; downstream
layers cancel through this module without ever importing api machinery.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .store import JobStore

logger = logging.getLogger(__name__)

_getter: Callable[[], "JobStore | None"] | None = None


def register_store_getter(getter: Callable[[], "JobStore | None"]) -> None:
    """The api layer wires its store accessor at import boot (idempotent)."""
    global _getter
    _getter = getter


def cancel_job_best_effort(durable_job_id: str, reason: str) -> bool:
    """Cancel a durable job through the registered seam.

    Best-effort: never raises. Returns False when the store accessor is
    not wired (e.g. orchestrator-in-isolation test contexts) so callers
    must keep their primary verdict in their own surface, not here.
    """
    if not durable_job_id:
        return False
    if _getter is None:
        logger.warning(
            "job cancel gateway: no store accessor registered; durable_job%s left alone",
            durable_job_id,
        )
        return False
    store = _getter()
    if store is None:
        logger.info(
            "job cancel gateway: store bound accessor returned None for durable_job%s",
            durable_job_id,
        )
        return False
    try:
        store.cancel_job(durable_job_id, reason=reason)
        logger.info(
            "job cancel gateway: durable_job%s cancelled (reason=%s)",
            durable_job_id,
            reason,
        )
        return True
    except Exception as exc:  # noqa: BLE001 — seam must never leak across layers
        logger.warning(
            "job cancel gateway: cancel_job(%s) failed: %s",
            durable_job_id,
            exc,
        )
        return False
