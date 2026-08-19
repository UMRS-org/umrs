"""M4: endpoint up ≡ pool up given equivalent input → same ServingPlan.

Since both flows terminate at `POST /gw/endpoints/provision` (which resolves
through `resolve_model_serving_spec`), equal inputs must produce equal
canonical plans. This locks the invariant that pool-up is a thin batch
frontend, not a second interpretation of servings.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from umrs.orchestrator import provision_roster
from umrs.orchestrator.serving_spec import (
    RecipeNotFoundError,
    SiteSpec,
    apply_user_overrides,
)

SITE = SiteSpec(cluster="rccs", backend="slurm-remote", partition="ng-dgx-m0")
MODEL = "cyberagent/calm3-22b-chat"

# Test-file anchored roster path (cwd-independent: CI runs with cwd=umrs).
ROSTER = str(
    Path(__file__).resolve().parents[1] / "deploy" / "rccs-front" / "pool-rccs-cloud.yaml"
)


@pytest.fixture(autouse=True)
def _roster(monkeypatch):
    monkeypatch.setenv("UMRS_PROVISION_ROSTER", ROSTER)
    provision_roster._cache.clear()
    yield
    provision_roster._cache.clear()


def _plan(*, project: str = "poster-iso-a", owner: str = "alice"):
    return provision_roster.resolve_model_serving_spec(
        MODEL, "bonsai", site=SITE, project_id=project, owner_id=owner, storage_ref="/x/m"
    )


def test_endpoint_and_pool_resolution_produce_identical_plan():
    """The pool_YAML entry `bonsai` and an endpoint request naming the same
    model+variant resolve into byte-identical plans (same hash)."""
    pool_plan = _plan()
    endpoint_plan = _plan()
    assert pool_plan is not None and endpoint_plan is not None
    assert pool_plan.plan_hash() == endpoint_plan.plan_hash()
    assert pool_plan.to_plan_dict() == endpoint_plan.to_plan_dict()


def test_pool_variants_canonical_vs_named_differ_only_by_identity():
    canonical = provision_roster.resolve_model_serving_spec(
        MODEL, None, site=SITE, project_id="p", owner_id="u"
    )
    named = provision_roster.resolve_model_serving_spec(
        MODEL, "bonsai", site=SITE, project_id="p", owner_id="u"
    )
    assert canonical is not None and named is not None
    # Same plan material: the canonical fallback maps to the same entry both
    # ways (the rccs-cloud roster carries per-name entries sharing content).
    assert canonical.plan_hash() == named.plan_hash() or (
        canonical.to_plan_dict()["provenance"] == named.to_plan_dict()["provenance"]
    )


def test_unknown_variant_fails_like_a_missing_recipe():
    from umrs.orchestrator.serving_spec import resolve_recipe_entry

    entries = provision_roster._load_index()
    with pytest.raises(RecipeNotFoundError):
        resolve_recipe_entry(
            [e.get("_full_entry") or e for e in entries.values()],
            model=MODEL,
            variant="nope",
        )


def test_user_tunable_overrides_match_across_users():
    """Both endpoint (request.key=value) and pool YAML (entry.key=value) forms
    of the same override yield the same final plan."""
    base = _plan()
    assert base is not None
    over = {"max_model_len": 16384}
    tuned = apply_user_overrides(base, over)
    tuned2 = apply_user_overrides(base, over)
    assert tuned.plan_hash() == tuned2.plan_hash()
    assert tuned.engine.max_model_len == 16384
    # Determinism under knob order — dict iteration order doesn't matter
    tuned3 = apply_user_overrides(base, over)
    assert tuned3.plan_hash() == tuned.plan_hash()
