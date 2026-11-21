"""Durable job types an extension owns (U3 S3b, ADR-117 §6, u3a map D8).

The values stay in ``JobType`` (rows in the shared ``umrs_db`` carry them and
are never renamed), but only a process where the owning extension registered
the lane (``ExtensionContext.add_job_type``) can run them.

D8: a front without the lane does not claim such a row. The claim query skips
it, so it stays QUEUED for a front that has the extension (all fronts share
``umrs_db``; claims are venue-scoped), instead of failing in the backend
factory on every scheduler tick.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

logger = logging.getLogger(__name__)

# value -> owning extension. companion_worker joins as reserved-only in S4 (D6).
EXTENSION_JOB_TYPES: dict[str, str] = {"frontend_session": "r4"}


def unclaimable_here(wired: Iterable[str]) -> frozenset[str]:
    """Extension-owned job types this process has no lane for."""
    return frozenset(EXTENSION_JOB_TYPES) - frozenset(wired)


def build_extension_lanes(ctx: Any, db_pool: Any) -> dict[str, tuple[Any, Any]]:
    """value -> (backend, checker) for every registered lane that wires here.

    A lane whose builder returns None (feature off on this server) or raises
    is left out; the claim query then skips its rows (D8).
    """
    lanes: dict[str, tuple[Any, Any]] = {}
    for value, reg in ctx.job_types.items():
        try:
            lane = reg.build_lane(db_pool)
        except Exception as exc:  # noqa: BLE001 - a lane never blocks the runtime
            logger.warning("%s lane wiring skipped: %s", value, exc)
            continue
        if lane is not None:
            lanes[value] = lane
    skipped = sorted(unclaimable_here(lanes))
    if skipped:
        logger.info("extension job types not claimed on this server: %s", skipped)
    return lanes
