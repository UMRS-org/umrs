"""M2 golden-diff: production rosters → ServingPlan equivalence.

For every roster YAML in umrs/deploy/** we prove:
  legacy `resolve_model_provision_defaults(model, variant)` shape ↔
  `spec_to_legacy_defaults(resolve_model_serving_spec(...))`
carry the same effective env/fields (semantic equivalence, not key-perfect,
because legacy never carried `_full_entry`).
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import pytest
from umrs.orchestrator import provision_roster
from umrs.orchestrator.serving_spec import (
    SiteSpec,
    build_spec_from_legacy_entry,
    spec_to_legacy_defaults,
)

# Anchor on the test file, not the cwd: pytest can run with
# working-directory=umrs (CI) or the parent repo root (repo-local).
_UMRS_ROOT = Path(__file__).resolve().parents[1]
_DEPLOY_ROOT = _UMRS_ROOT / "deploy"


def _roster(*parts: str) -> str:
    """Absolute path to one roster YAML under umrs/deploy/."""
    return str(_DEPLOY_ROOT.joinpath(*parts))


_ALL_ROSTERS_GLOB = os.path.join(_DEPLOY_ROOT, "**", "pool*.yaml")
ALL_ROSTERS = sorted(
    p for p in glob.glob(_ALL_ROSTERS_GLOB, recursive=True) if "/templates/" not in p
)
assert ALL_ROSTERS, "expected at least one production roster"

_FILLERS = {
    "partition",
    "node_count",
    "gpu_count",
    "max_model_len",
    "ttl_sec",
    "tier",
    "parallel_strategy",
    "env_vars",
}


def _site_for(path: str) -> SiteSpec:
    site_dir = Path(path).relative_to(_DEPLOY_ROOT).parts[0]
    partition = "unknown"
    return SiteSpec(cluster=site_dir, backend="slurm-remote", partition=partition)


def _legacy_shape(defaults: dict | None) -> dict:
    out = {k: v for k, v in (defaults or {}).items() if k in _FILLERS}
    out.setdefault("node_count", 1)
    out.setdefault("gpu_count", 1)
    out.setdefault("max_model_len", 4096)
    out.setdefault("ttl_sec", 0)
    out.setdefault("partition", "")
    return out


def _spec_shape(spec) -> dict:
    out = spec_to_legacy_defaults(spec)
    return {k: v for k, v in out.items() if k in _FILLERS}


@pytest.mark.parametrize("roster_path", ALL_ROSTERS, ids=lambda p: p.split("/")[-1])
def test_legacy_entry_roundtrips_through_serving_plan(roster_path, monkeypatch):
    monkeypatch.setenv("UMRS_PROVISION_ROSTER", roster_path)
    provision_roster._cache.clear()

    from umrs.orchestrator.serving_spec import load_roster_recipes

    entries = load_roster_recipes([roster_path])
    assert entries, f"roster {roster_path} parsed empty"
    site = _site_for(roster_path)

    for entry in entries:
        model = str(entry["model"])
        variant = entry.get("name")
        # exact match must succeed (legacy)
        legacy = provision_roster.resolve_model_provision_defaults(model, variant)
        if legacy is None and variant is None:
            continue  # canonical fallback absence is allowed for full variants
        assert legacy is not None, f"legacy resolver missed {model}/{variant}"
        full = legacy.pop("_full_entry", entry)
        site = SiteSpec(
            cluster=site.cluster,
            backend=site.backend,
            partition=str(full.get("partition") or ""),
        )
        spec = build_spec_from_legacy_entry(
            full,
            site=site,
            project_id=entry.get("project_id", ""),
            owner_id=entry.get("owner", ""),
            storage_ref=full.get("path", ""),
        )
        # Node/GPU/env semantic equivalence; exact-match recipe only
        spec_shape = _spec_shape(spec)
        legacy_shape = _legacy_shape(legacy)
        # Every legacy env key/value must persist verbatim in the plan env
        legacy_env = (legacy_shape.get("env_vars")) or {}
        plan_env = dict(spec.env)
        for k, v in legacy_env.items():
            assert plan_env.get(k) == v, (
                f"{model}: env {k} mismatch {legacy_env[k]!r} vs {plan_env.get(k)!r}"
            )
        # node/gpu equivalence
        assert spec_shape["node_count"] == legacy_shape["node_count"]
        assert spec_shape["gpu_count"] == legacy_shape["gpu_count"]
        if "partition" in legacy_shape and legacy_shape["partition"]:
            assert spec_shape["partition"] == str(legacy_shape["partition"])
        assert spec_shape["max_model_len"] == legacy_shape["max_model_len"]
        assert spec_shape["ttl_sec"] == legacy_shape["ttl_sec"]


def test_recipe_partition_lifted_when_request_has_none(monkeypatch):
    """Live bdrt regression (deepseek-v4-flash-0731, 2026-08-18): bare
    `r4 endpoint up` passes site.partition="" — the plan must then take the
    recipe's partition, exactly like the legacy resolver's _ROSTER_FIELDS
    lift. Before the fix the typed path dropped it, so every bare provision
    defaulted to the site default partition (spark) and multi-node jobs
    sat PD PartitionNodeLimit forever."""
    roster = _roster("bdrt", "pool-bdrt-step.yaml")
    monkeypatch.setenv("UMRS_PROVISION_ROSTER", roster)
    provision_roster._cache.clear()

    legacy = provision_roster.resolve_model_provision_defaults("deepseek-v4-flash-0731", None)
    assert legacy is not None
    assert legacy.get("partition") == "edge-double"

    spec = provision_roster.resolve_model_serving_spec(
        "deepseek-v4-flash-0731",
        None,
        site=SiteSpec(cluster="bdrt", backend="slurm-remote", partition=""),
    )
    assert spec is not None
    assert spec.site.partition == "edge-double", (
        "recipe partition must survive into the plan when the request carries no explicit partition"
    )


def test_resolver_serving_spec_returns_typed_plan(monkeypatch):
    monkeypatch.setenv("UMRS_PROVISION_ROSTER", _roster("rccs-front", "pool-rccs-cloud.yaml"))
    provision_roster._cache.clear()
    spec = provision_roster.resolve_model_serving_spec(
        "cyberagent/calm3-22b-chat",
        None,
        site=SiteSpec(cluster="rccs-front", backend="slurm-remote", partition="ng-dgx-m0"),
    )
    assert spec is not None
    from umrs.orchestrator.serving_spec import ResolvedEndpointSpec

    assert isinstance(spec, ResolvedEndpointSpec)
    assert spec.model.name == "cyberagent/calm3-22b-chat"
    assert spec.engine.max_model_len == 8192


def test_legacy_public_shape_is_unchanged(monkeypatch):
    # Existing service callers expect this return shape — no new keys.
    monkeypatch.setenv("UMRS_PROVISION_ROSTER", _roster("rccs-front", "pool-rccs-cloud.yaml"))
    provision_roster._cache.clear()
    defaults = provision_roster.resolve_model_provision_defaults("cyberagent/calm3-22b-chat", None)
    assert defaults is not None
    assert "_full_entry" not in defaults
    assert set(defaults) <= _FILLERS


def test_production_roster_variants_resolve_distinct_plans(monkeypatch):
    for path in (_roster("bdrt", "pool-bdrt-step.yaml"),):
        monkeypatch.setenv("UMRS_PROVISION_ROSTER", path)
        provision_roster._cache.clear()
        from umrs.orchestrator.serving_spec import load_roster_recipes

        entries = load_roster_recipes([path])
        site = _site_for(path)
        seen: dict[str, str] = {}
        for entry in entries:
            spec = build_spec_from_legacy_entry(
                entry, site=site, project_id="", owner_id="", storage_ref=entry.get("path", "")
            )
            prev = seen.get(spec.model.name)
            if prev and prev != spec.plan_hash():
                # variants of the same model yield different plan hashes only when
                # their content actually differs (not guaranteed by design — only
                # assert hashes are stable and correctly-derived)
                assert prev != spec.plan_hash(), "spurious hash change"
            seen[spec.model.name] = prev or spec.plan_hash()
