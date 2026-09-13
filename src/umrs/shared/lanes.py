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

from enum import Enum

from umrs.shared import feature_flags


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
# a wave-log entry. 2026-09-13 (release lane decision, R-13): the S5
# placement list is empty — every mounted family carries a lane.
FAMILY_LANES: dict[str, SurfaceLane] = {
    # S2/S3-classified lanes
    "endpoints": SurfaceLane.STABLE,
    "models": SurfaceLane.STABLE,
    "llm": SurfaceLane.STABLE,
    "knowledge": SurfaceLane.BETA,
    # ADR-026 wave C: the sat facade's knowledge-graph surface is DELETED;
    # the canonical multigraph routes under knowledge/graph + knowledge/graphs.
    "k2": SurfaceLane.BETA,
    "loops": SurfaceLane.BETA,
    "research": SurfaceLane.BETA,
    # ADR-041 asset grants (WP-06) — user-facing v1beta, same lane as loops.
    "grants": SurfaceLane.BETA,
    # R1 secrets wave (a5ad437a): user-managed sources + their grants ride
    # the same /umrs/user-sources prefix — user-facing v1beta.
    "user-sources": SurfaceLane.BETA,
    "ops": SurfaceLane.INTERNAL,
    # R1 secrets wave: broker/lease machinery is ops-plane, never user-facing.
    "secrets": SurfaceLane.INTERNAL,
    "skills": SurfaceLane.INTERNAL,
    "zmq": SurfaceLane.INTERNAL,
    # ── Release lane decision, R-13 (2026-09-13) ──────────────────────────
    # Wave log: every family mounted by the sidecar now carries a lane;
    # UNCLASSIFIED is not a shippable lane (production-release-plan §3/§4
    # R-13). Decision rule: the router module's docstring/routes decide;
    # deviations from the maintainer's proposed list are justified inline.
    #
    # v1-stable — job/workflow/file/pipeline surfaces the r4 CLI and TUI
    # drive daily (durable jobs U-731+, workflow registry, presign U-402+,
    # pipeline definitions X-001).
    "jobs": SurfaceLane.STABLE,
    "workflows": SurfaceLane.STABLE,
    "files": SurfaceLane.STABLE,
    "pipelines": SurfaceLane.STABLE,
    # v1beta — user-facing knowledge / agent / memory / evidence surfaces
    # that are served but still evolving (ADR-012 chat, ADR-034 a2l ledger,
    # ADR-036 companion, K1 satellites, MEM-* memory routes, vision proxy).
    "mcp": SurfaceLane.BETA,
    "rag": SurfaceLane.BETA,
    "rag-profiles": SurfaceLane.BETA,
    "uploads": SurfaceLane.BETA,
    "companion": SurfaceLane.BETA,
    "chat": SurfaceLane.BETA,
    "catalog": SurfaceLane.BETA,
    "artifacts": SurfaceLane.BETA,
    "assets": SurfaceLane.BETA,
    "provenance": SurfaceLane.BETA,
    "traces": SurfaceLane.BETA,
    "verification": SurfaceLane.BETA,
    "evidence": SurfaceLane.BETA,
    "experiment-registry": SurfaceLane.BETA,
    "scientific-memory": SurfaceLane.BETA,
    "reflective-memory": SurfaceLane.BETA,
    "vision": SurfaceLane.BETA,
    "a2l": SurfaceLane.BETA,
    # Deviation (R-13): proposed internal; `r4 rag`/TUI read
    # /gw/components/{kind} to build pipelines — user-facing, so v1beta.
    "components": SurfaceLane.BETA,
    # Deviation (R-13): proposed internal; `r4 usage` (r4/_cli/usage.py)
    # reads /umrs/usage/* and /me is per-user — user-facing, so v1beta.
    "usage": SurfaceLane.BETA,
    # Deviation (R-13): proposed internal; r4-web + `r4 dashboard` consume
    # /umrs/web/* and r4-web ships as a beta product surface (§3) — v1beta.
    "web": SurfaceLane.BETA,
    # internal — operator plane: egress audit sink written by the gateway,
    # read by `r4 agent audit` (A-1204/A-1209); never a user API.
    "egress": SurfaceLane.INTERNAL,
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
_FLAG_ENABLE = "UMRS_LANE_SCHEMAS_ENABLED"
_FLAG_SCHEMA_PREFIX = "UMRS_LANE_SCHEMA_"  # e.g. UMRS_LANE_SCHEMA_STABLE


def lane_schemas_enabled() -> bool:
    """Rollout switch (S11 registry: expires 2026-09-30, owner stability-wave)."""
    return feature_flags.is_enabled(_FLAG_ENABLE)


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
    override = feature_flags.get_value(_FLAG_SCHEMA_PREFIX + suffix)
    return override or _DEFAULT_SCHEMAS[lane]
