"""Lane-registry authority tests (stability S5).

``umrs.shared.lanes`` is the single source for family → lane mapping and
per-lane storage coordinates. These tests pin the authority itself; the
mounted route table is asserted against it in
``test_surface_characterization.py``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from umrs.shared import lanes
from umrs.shared.lanes import SurfaceLane

# The S2/S3 classification record — do not relane without a wave-log entry.
CLASSIFIED_FIXED: dict[str, SurfaceLane] = {
    "endpoints": SurfaceLane.STABLE,
    "models": SurfaceLane.STABLE,
    "llm": SurfaceLane.STABLE,
    "knowledge": SurfaceLane.BETA,
    "k2": SurfaceLane.BETA,
    "loops": SurfaceLane.BETA,
    "research": SurfaceLane.BETA,
    "grants": SurfaceLane.BETA,
    # R1 secrets wave (a5ad437a): user-sources user-facing; broker internal.
    "user-sources": SurfaceLane.BETA,
    # ADR-051 (2026-09-14): self-service durable projects registry — BETA.
    "projects": SurfaceLane.BETA,
    "buckets": SurfaceLane.BETA,  # ADR-079 personal buckets (2026-10-02)
    # ADR-051 read model (2026-09-16): caller-attributed quota gauge — BETA.
    "quota": SurfaceLane.BETA,
    # ADR-086 (2026-10-05): circles, superseding the ADR-050 §3 groups authority — BETA.
    "circles": SurfaceLane.BETA,  # "groups" until umrs 696ed281
    # D-39 (2026-09-22): Scientific Catalog registry — BETA like projects.
    "scientific-catalogs": SurfaceLane.BETA,
    "ops": SurfaceLane.INTERNAL,
    "secrets": SurfaceLane.INTERNAL,
    "skills": SurfaceLane.INTERNAL,
    "zmq": SurfaceLane.INTERNAL,
    # ── 2026-09-13 release lane decision (R-13) — wave log ─────────────────
    # The 26 families of the S5 placement list received their lane from the
    # router docstrings/routes (see umrs/shared/lanes.py comment block for
    # the three inline-justified deviations: components/usage/web → v1beta).
    "jobs": SurfaceLane.STABLE,
    "workflows": SurfaceLane.STABLE,
    "files": SurfaceLane.STABLE,
    # 2026-10-18 (H-10.1 follow-up, wave ledger): "pipelines" (v1-stable) and
    # "verification" (v1beta) removed — their routes were deleted with their
    # retired clients (umrs/shared/lanes.py comments).
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
    "evidence": SurfaceLane.BETA,
    "experiment-registry": SurfaceLane.BETA,
    "scientific-memory": SurfaceLane.BETA,
    "reflective-memory": SurfaceLane.BETA,
    "vision": SurfaceLane.BETA,
    "a2l": SurfaceLane.BETA,
    "components": SurfaceLane.BETA,
    "usage": SurfaceLane.BETA,
    "web": SurfaceLane.BETA,
    "egress": SurfaceLane.INTERNAL,
}

# S5's placement list — families deliberately awaiting a lane decision.
# +1 2026-09-02: "assets" (asset index read-model, Review F4 / ADR-035 wave)
# (2026-09-13: R1 secrets wave lanes recorded — classified-only move, no
# placement-list delta)
# 2026-09-13 release lane decision (R-13): placement list emptied — 26 → 0.
# UNCLASSIFIED is not a shippable lane; a new family must land WITH a lane.
UNCLASSIFIED_COUNT = 0


# ── registry integrity ──────────────────────────────────────────────────────


def test_classified_fixed_points_pinned() -> None:
    classified = {
        family: lane
        for family, lane in lanes.FAMILY_LANES.items()
        if lane != SurfaceLane.UNCLASSIFIED
    }
    assert classified == CLASSIFIED_FIXED


def test_every_family_maps_to_a_lane_enum() -> None:
    bad = [f for f, lane in lanes.FAMILY_LANES.items() if not isinstance(lane, SurfaceLane)]
    assert not bad, f"families with non-enum lane values: {bad}"


def test_unclassified_placement_list_size() -> None:
    """Additions to the placement list are deliberate — update consciously."""
    unclassified = [f for f, lane in lanes.FAMILY_LANES.items() if lane == SurfaceLane.UNCLASSIFIED]
    assert len(unclassified) == UNCLASSIFIED_COUNT


def test_no_family_ships_unclassified() -> None:
    """R-13 (2026-09-13): UNCLASSIFIED is not a shippable lane for a mounted family."""
    unclassified = sorted(
        f for f, lane in lanes.FAMILY_LANES.items() if lane == SurfaceLane.UNCLASSIFIED
    )
    assert unclassified == [], f"families without a release lane: {unclassified}"


# ── path semantics ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("path", "family", "lane"),
    [
        ("/umrs/endpoints/42", "endpoints", SurfaceLane.STABLE),
        ("/umrs/models", "models", SurfaceLane.STABLE),
        ("/umrs/knowledge/graphs", "knowledge", SurfaceLane.BETA),
        ("/umrs/research/sessions/sess-1", "research", SurfaceLane.BETA),
        ("/umrs/ops/ping", "ops", SurfaceLane.INTERNAL),
        ("/umrs/companion/chat", "companion", SurfaceLane.BETA),
        ("/umrs/jobs/job-1", "jobs", SurfaceLane.STABLE),
        ("/umrs/egress/audit", "egress", SurfaceLane.INTERNAL),
        ("/health", None, SurfaceLane.UNCLASSIFIED),
        ("/umrs", None, SurfaceLane.UNCLASSIFIED),
        ("/umrs/not-a-family", "not-a-family", SurfaceLane.UNCLASSIFIED),
    ],
)
def test_path_semantics(path: str, family: str | None, lane: SurfaceLane) -> None:
    assert lanes.family_of_umrs_path(path) == family
    assert lanes.lane_for_umrs_path(path) == lane
    if family is not None:
        assert lanes.lane_for_family(family) == lane


# ── storage coordinates ─────────────────────────────────────────────────────


def test_schemas_identity_until_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("UMRS_LANE_SCHEMAS_ENABLED", raising=False)
    for lane in SurfaceLane:
        assert lanes.lane_db_schema(lane) == "public"


def test_schema_resolution_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UMRS_LANE_SCHEMAS_ENABLED", "true")
    assert lanes.lane_db_schema(SurfaceLane.STABLE) == "lane_stable"
    assert lanes.lane_db_schema(SurfaceLane.BETA) == "lane_beta"
    assert lanes.lane_db_schema(SurfaceLane.INTERNAL) == "lane_internal"
    # UNCLASSIFIED never silently lands in a lane schema.
    assert lanes.lane_db_schema(SurfaceLane.UNCLASSIFIED) == "public"


def test_schema_override_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UMRS_LANE_SCHEMAS_ENABLED", "1")
    monkeypatch.setenv("UMRS_LANE_SCHEMA_BETA", "staged_beta")
    assert lanes.lane_db_schema(SurfaceLane.BETA) == "staged_beta"
    assert lanes.lane_db_schema(SurfaceLane.STABLE) == "lane_stable"


def test_env_isolation(monkeypatch: pytest.MonkeyPatch) -> None:
    """The test-suite env must not leak the rollout switch."""
    monkeypatch.undo()
    assert "UMRS_LANE_SCHEMAS_ENABLED" not in os.environ


# ── migration scaffold ──────────────────────────────────────────────────────


def test_migration_0072_scaffold() -> None:
    path = Path(lanes.__file__).parents[1] / "migrations" / "0072_lane_schemas.sql"
    sql = path.read_text()
    for schema in ("lane_stable", "lane_beta", "lane_internal"):
        assert f"CREATE SCHEMA IF NOT EXISTS {schema};" in sql, (
            f"migration must create {schema} idempotently"
        )
