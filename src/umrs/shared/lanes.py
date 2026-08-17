"""Surface-lane registry — the single authority for lane classification.

Wave: UMRS Stability & Governed Surface Alignment, S5. This module owns
the family → lane mapping that the S4 premise tests assert against, and
the per-lane storage coordinates the S9 expand/contract migrations will
execute against.

Design decisions (recorded in the wave log 2026-08-17):

- Segregation unit for durable state is the **Postgres schema**
  (``lane_stable`` / ``lane_beta`` / ``lane_internal``) inside the
  existing database. Same DB, no second connection plan, and every table
  move S9 performs is a transaction-local ``ALTER TABLE ... SET SCHEMA``.
- S3/MinIO buckets stay per-user namespaced (the operative isolation
  today, per ``shared/bootstrap.py``); a lane dimension there doubles the
  bucket matrix without traffic justification (S2 evidence).
- Nothing changes behavior until ``UMRS_LANE_SCHEMAS_ENABLED=true``:
  schema resolution degrades to ``public`` for every lane, which is the
  current layout. The flag is the S9 rollout switch, not a product knob
  (S11 expiry policy applies).
"""

from __future__ import annotations

import os
from enum import Enum


class SurfaceLane(str, Enum):
    """Traffic lane for a mounted surface family (S2/S3 classification)."""

    STABLE = "v1-stable"
    BETA = "v1beta"
    INTERNAL = "internal"
    UNCLASSIFIED = "unclassified"


# Family key = first path segment after ``/umrs``. This table is the
# registry authority; umrs/tests/test_surface_characterization.py imports
# it instead of carrying a second copy. Classified entries are pinned to
# the S2/S3 record by tests/test_lane_registry.py — do not relane without
# a wave-log entry.
FAMILY_LANES: dict[str, SurfaceLane] = {
    # S2/S3-classified lanes
    "endpoints": SurfaceLane.STABLE,
    "models": SurfaceLane.STABLE,
    "llm": SurfaceLane.STABLE,
    "knowledge": SurfaceLane.BETA,
    "knowledge-graph": SurfaceLane.BETA,
    "k2": SurfaceLane.BETA,
    "loops": SurfaceLane.BETA,
    "research": SurfaceLane.BETA,
    "ops": SurfaceLane.INTERNAL,
    "skills": SurfaceLane.INTERNAL,
    "zmq": SurfaceLane.INTERNAL,
    # Mounted, no lane decision recorded yet — S5's placement list.
    "a2l": SurfaceLane.UNCLASSIFIED,
    "artifacts": SurfaceLane.UNCLASSIFIED,
    "catalog": SurfaceLane.UNCLASSIFIED,
    "chat": SurfaceLane.UNCLASSIFIED,
    "companion": SurfaceLane.UNCLASSIFIED,
    "components": SurfaceLane.UNCLASSIFIED,
    "egress": SurfaceLane.UNCLASSIFIED,
    "evidence": SurfaceLane.UNCLASSIFIED,
    "experiment-registry": SurfaceLane.UNCLASSIFIED,
    "files": SurfaceLane.UNCLASSIFIED,
    "jobs": SurfaceLane.UNCLASSIFIED,
    "mcp": SurfaceLane.UNCLASSIFIED,
    "pipelines": SurfaceLane.UNCLASSIFIED,
    "provenance": SurfaceLane.UNCLASSIFIED,
    "rag": SurfaceLane.UNCLASSIFIED,
    "rag-profiles": SurfaceLane.UNCLASSIFIED,
    "reflective-memory": SurfaceLane.UNCLASSIFIED,
    "scientific-memory": SurfaceLane.UNCLASSIFIED,
    "traces": SurfaceLane.UNCLASSIFIED,
    "uploads": SurfaceLane.UNCLASSIFIED,
    "usage": SurfaceLane.UNCLASSIFIED,
    "verification": SurfaceLane.UNCLASSIFIED,
    "vision": SurfaceLane.UNCLASSIFIED,
    "web": SurfaceLane.UNCLASSIFIED,
    "workflows": SurfaceLane.UNCLASSIFIED,
}


def lane_for_family(family: str) -> SurfaceLane:
    """Lane for a route family; undeclared families are UNCLASSIFIED."""
    return FAMILY_LANES.get(family, SurfaceLane.UNCLASSIFIED)


def family_of_umrs_path(path: str) -> str | None:
    """First segment after ``/umrs`` — the route family key, or None."""
    parts = [p for p in path.split("/") if p]
    if len(parts) < 2 or parts[0] != "umrs":
        return None
    return parts[1]


def lane_for_umrs_path(path: str) -> SurfaceLane:
    """Lane for a served ``/umrs/...`` path (family-level granularity)."""
    family = family_of_umrs_path(path)
    if family is None:
        return SurfaceLane.UNCLASSIFIED
    return lane_for_family(family)


# ── per-lane storage coordinates ────────────────────────────────────────────

_DEFAULT_SCHEMAS: dict[SurfaceLane, str] = {
    SurfaceLane.STABLE: "lane_stable",
    SurfaceLane.BETA: "lane_beta",
    SurfaceLane.INTERNAL: "lane_internal",
}

_COMPAT_SCHEMA = "public"  # current layout — identity mapping until S9
_ENV_ENABLE = "UMRS_LANE_SCHEMAS_ENABLED"
_ENV_SCHEMA_PREFIX = "UMRS_LANE_SCHEMA_"  # e.g. UMRS_LANE_SCHEMA_STABLE


def lane_schemas_enabled() -> bool:
    """Rollout switch: lane-schema resolution only when explicitly on."""
    return os.environ.get(_ENV_ENABLE, "").strip().lower() in ("1", "true", "yes")


def lane_db_schema(lane: SurfaceLane) -> str:
    """Postgres schema that owns a lane's tables.

    Identity mapping (``public``) until S9 flips the switch; per-lane
    overrides exist for staged rollouts (``UMRS_LANE_SCHEMA_<LANE>``).
    UNCLASSIFIED always lands on the compat schema — placement is
    deliberate, never accidental.
    """
    if not lane_schemas_enabled() or lane == SurfaceLane.UNCLASSIFIED:
        return _COMPAT_SCHEMA
    suffix = {
        SurfaceLane.STABLE: "STABLE",
        SurfaceLane.BETA: "BETA",
        SurfaceLane.INTERNAL: "INTERNAL",
    }[lane]
    return os.environ.get(_ENV_SCHEMA_PREFIX + suffix) or _DEFAULT_SCHEMAS[lane]
