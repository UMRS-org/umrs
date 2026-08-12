"""M3: recipe resolver policy gate + plan-driven sizing in provision flow."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from umrs.api.endpoints import _current_cluster_name, _enforce_recipe_policy
from umrs.orchestrator import provision_roster
from umrs.orchestrator.serving_spec import ResolvedEndpointSpec, SiteSpec

SITE = SiteSpec(cluster="rccs", backend="slurm-remote", partition="ng-dgx-m0")


def test_enforce_recipe_policy_rejects_env_vars_for_non_admin():
    request = SimpleNamespace(env_vars={"VLLM_EXTRA_ARGS": "--flakes"}, container_image=None)
    with pytest.raises(HTTPException) as exc:
        _enforce_recipe_policy(False, request)
    assert exc.value.status_code == 403
    assert "env_vars" in str(exc.value.detail)


def test_enforce_recipe_policy_rejects_container_image_for_non_admin():
    request = SimpleNamespace(env_vars=None, container_image="docker.io/not-curated/x:latest")
    with pytest.raises(HTTPException) as exc:
        _enforce_recipe_policy(False, request)
    assert exc.value.status_code == 403
    assert "container_image" in str(exc.value.detail)


def test_enforce_recipe_policy_admits_admin_and_empty_request():
    request = SimpleNamespace(
        env_vars={"VLLM_EXTRA_ARGS": "--x"}, container_image="docker.io/x:latest"
    )
    _enforce_recipe_policy(True, request)
    _enforce_recipe_policy(False, SimpleNamespace(env_vars=None, container_image=None))


def test_cluster_name_is_deterministic_and_configurable(monkeypatch):
    monkeypatch.setenv("UMRS_CLUSTER", "rccs")
    assert _current_cluster_name() == "rccs"
    monkeypatch.delenv("UMRS_CLUSTER")
    name = _current_cluster_name()
    assert isinstance(name, str) and name


def test_plan_drives_sizing_for_curd_model(monkeypatch):
    monkeypatch.setenv("UMRS_PROVISION_ROSTER", "umrs/deploy/rccs-front/pool-rccs-cloud.yaml")
    provision_roster._cache.clear()
    serving_plan = provision_roster.resolve_model_serving_spec(
        "cyberagent/calm3-22b-chat",
        None,
        site=SITE,
        project_id="proj-a",
        owner_id="alice",
        storage_ref="/lvs0/x/calm3-22b-chat",
    )
    assert isinstance(serving_plan, ResolvedEndpointSpec)
    assert serving_plan.engine.max_model_len == 8192
    assert serving_plan.model.project_id == "proj-a"
    assert serving_plan.model.owner_id == "alice"
    assert serving_plan.provenance.recipe_file.endswith("pool-rccs-cloud.yaml")
    assert serving_plan.plan_hash()


def test_custom_model_without_recipe_gets_none_and_generic_fallback():
    plan = provision_roster.resolve_model_serving_spec(
        "user-owned/custom-7b",
        None,
        site=SITE,
        project_id="proj-c",
        owner_id="bob",
    )
    assert plan is None  # constrained generic fallback is the caller's lane


# ------------------------------------------------------------------------------
# M2.4 gate matrix additions (2026-08-12)
# ------------------------------------------------------------------------------


def test_flask_style_extra_args_via_env_vars_also_blocked_for_non_admin():
    """Extra-args cannot be smuggled through env_vars either (image-class risk)."""
    request = SimpleNamespace(
        env_vars={"VLLM_EXTRA_ARGS": "--speculative-config {} --enable-prefix-caching"},
        container_image=None,
        partition=None,
    )
    with pytest.raises(HTTPException) as exc:
        _enforce_recipe_policy(False, request)
    assert exc.value.status_code == 403
    assert "env_vars" in str(exc.value.detail)
    assert "recipe-controlled channel" in str(exc.value.detail)


def test_generic_plan_carries_no_recipe_artifacts(monkeypatch):
    """Custom model lane must NOT inherit ANY recipe material (M2.3)."""
    monkeypatch.setenv("UMRS_PROVISION_ROSTER", "umrs/deploy/rccs-front/pool-rccs-cloud.yaml")
    provision_roster._cache.clear()
    plan = provision_roster.resolve_model_serving_spec(
        "user-owned/custom-7b",
        None,
        site=SITE,
        project_id="proj-c",
        owner_id="bob",
    )
    assert plan is None

    from umrs.orchestrator import serving_spec as ss

    generic = ss.build_generic_spec(
        model="user-owned/custom-7b",
        served_name=None,
        storage_ref="/lvs0/x/user-owned-custom-7b",
        site=SITE,
        project_id="proj-c",
        owner_id="bob",
    )
    env = ss.spec_env_vars(generic)
    assert "VLLM_PATCHES" not in env
    assert not env.get("VLLM_EXTRA_ARGS"), "generic must not inherit extra args"
    assert "VLLM_REASONING_PARSER" not in env
    assert ss.patches_to_env(generic.patches) == {}


def test_partition_is_validated_against_site_partitions(monkeypatch):
    """Unauthorized partition rejected; listed partition accepted."""
    from umrs.api import endpoints as e

    fake_site = SimpleNamespace(scheduler=SimpleNamespace(partition="ng-dgx-m0,ng-dgx-m1"))
    monkeypatch.setattr(
        "umrs.orchestrator.site_config.load_site_config",
        lambda: fake_site,
    )
    e._validate_partition_allowed("ng-dgx-m1")
    with pytest.raises(HTTPException) as exc:
        e._validate_partition_allowed("ng-dgx-other")
    assert exc.value.status_code == 400
    assert "not in this site's scheduler partitions" in str(exc.value.detail)


def test_unknown_config_degrades_to_noop_for_partition():
    """When site config can't load, partition validation must not hard-fail."""
    from umrs.api import endpoints as e

    e._validate_partition_allowed("anything-goes")


def test_legacy_resolver_still_matches_plan_for_curated():
    import os

    os.environ["UMRS_PROVISION_ROSTER"] = "umrs/deploy/rccs-front/pool-rccs-cloud.yaml"
    provision_roster._cache.clear()
    legacy = provision_roster.resolve_model_provision_defaults("cyberagent/calm3-22b-chat", None)
    plan = provision_roster.resolve_model_serving_spec(
        "cyberagent/calm3-22b-chat",
        None,
        site=SITE,
        project_id="p",
        owner_id="u",
    )
    assert legacy is not None and plan is not None
    assert legacy["max_model_len"] == plan.engine.max_model_len
    assert legacy.get("node_count", 1) == plan.runtime.nodes
    assert legacy.get("gpu_count", 1) == plan.runtime.gpus_per_node
    assert (legacy.get("env_vars") or {}) == dict(plan.env)


def test_non_single_serving_mode_gets_501_not_silent_success():
    from umrs.api.endpoints import _enforce_serving_mode_supported

    request = SimpleNamespace(serving_mode="multi-lora")
    with pytest.raises(HTTPException) as exc:
        _enforce_serving_mode_supported(request)
    assert exc.value.status_code == 501
    assert "multi-lora" in str(exc.value.detail)


def test_co_located_serving_mode_gets_501():
    from umrs.api.endpoints import _enforce_serving_mode_supported

    with pytest.raises(HTTPException) as exc:
        _enforce_serving_mode_supported(SimpleNamespace(serving_mode="co-located"))
    assert exc.value.status_code == 501


def test_single_serving_mode_passes_gate():
    from umrs.api.endpoints import _enforce_serving_mode_supported

    _enforce_serving_mode_supported(SimpleNamespace(serving_mode="single"))
    _enforce_serving_mode_supported(SimpleNamespace(serving_mode=None))


def test_serving_mode_gate_is_before_warm_shortcircuit():
    """Live-verified bypass: with prefer_warm enabled and a ready endpoint
    cached, a multi-lora request used to return the cached single-model
    endpoint before reaching the 501 gate. Guard position is load-bearing:
    _enforce_serving_mode_supported must run before the
    '# Check for existing warm endpoint first' block in provision_endpoint."""
    import inspect
    import re

    from umrs.api import endpoints as ep

    src = inspect.getsource(ep.provision_endpoint)
    gate_pos = src.find("_enforce_serving_mode_supported(request)")
    warm_pos = src.find("Check for existing warm endpoint first")
    assert gate_pos != -1 and warm_pos != -1
    assert gate_pos < warm_pos, (
        "serving_mode gate must precede the warm-endpoint short-circuit"
    )
    # And it must not appear twice.
    assert len(re.findall(r"_enforce_serving_mode_supported\(request\)", src)) == 1


def test_recipe_policy_gate_before_warm_and_dedupe_returns():
    """The recipe-policy gate (admin-only env_vars/container_image → 403,
    invalid partition → 400) must precede BOTH early-reuse returns:
    warm-endpoint short-circuit AND already-provisioning dedupe 202.
    Bypass shape == the previously-fixed serving-mode class (#18)."""
    import inspect
    import re

    from umrs.api import endpoints as ep

    src = inspect.getsource(ep.provision_endpoint)
    sites = [m.start() for m in re.finditer(r"_enforce_recipe_policy", src)]
    assert len(sites) == 1, f"single clean call site expected: {sites}"
    gate_pos = sites[0]
    warm_pos = src.find("Check for existing warm endpoint first")
    dedupe_pos = src.find("# Check if there's already a provisioning endpoint")
    assert gate_pos < warm_pos, "gate must precede warm reuse"
    assert gate_pos < dedupe_pos, "gate must precede dedupe reuse"
