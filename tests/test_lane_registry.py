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
    "knowledge-graph": SurfaceLane.BETA,
    "k2": SurfaceLane.BETA,
    "loops": SurfaceLane.BETA,
    "research": SurfaceLane.BETA,
    "grants": SurfaceLane.BETA,
    "ops": SurfaceLane.INTERNAL,
    "skills": SurfaceLane.INTERNAL,
    "zmq": SurfaceLane.INTERNAL,
}

# S5's placement list — families deliberately awaiting a lane decision.
# +1 2026-09-02: "assets" (asset index read-model, Review F4 / ADR-035 wave)
UNCLASSIFIED_COUNT = 26


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


# ── path semantics ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("path", "family", "lane"),
    [
        ("/umrs/endpoints/42", "endpoints", SurfaceLane.STABLE),
        ("/umrs/models", "models", SurfaceLane.STABLE),
        ("/umrs/knowledge/graphs", "knowledge", SurfaceLane.BETA),
        ("/umrs/research/sessions/sess-1", "research", SurfaceLane.BETA),
        ("/umrs/ops/ping", "ops", SurfaceLane.INTERNAL),
        ("/umrs/companion/chat", "companion", SurfaceLane.UNCLASSIFIED),
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
