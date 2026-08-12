"""M1 unit tests for the typed ServingPlan (orchestrator.serving_spec)."""

from __future__ import annotations

import hashlib
import json

import pytest
from umrs.orchestrator.serving_spec import (
    RecipeNotFoundError,
    RecipePolicyError,
    RecipeSchemaError,
    ResolvedEndpointSpec,
    SiteSpec,
    apply_user_overrides,
    build_generic_spec,
    build_spec_from_recipe,
    load_roster_recipes,
    resolve_recipe_entry,
)

SITE_RCCS = SiteSpec(cluster="rccs", backend="slurm-remote", partition="ng-dgx-m0")
SITE_AI4S = SiteSpec(cluster="ai4s", backend="slurm-remote", partition="4n4gpu-p")


def _write_roster(tmp_path, content: str, name: str = "pool-rccs.yaml"):
    p = tmp_path / name
    p.write_text(content)
    return p


def _basic_recipe(tmp_path):
    return _write_roster(
        tmp_path,
        """\
defaults:
  tier: system
  timeout: 86400

endpoints:
  - name: qwen-agentworld
    model: Qwen-AgentWorld-35B-A3B
    path: /lvs0/rccs-dmp/masaru.nagaso/umrs_models/Qwen-AgentWorld-35B-A3B
    image: docker.io/vllm/vllm-openai:cu130-nightly-aarch64
    partition: ng-dgx-m0
    node_count: 1
    gpu_count: 1
    max_model_len: 262144
    engine:
      kv_cache_memory_bytes: 8589934592
      max_num_seqs: 4
      max_num_batched_tokens: 8192
      language_model_only: true
    patches:
      - source: /lvs0/rccs-dmp/masaru.nagaso/umrs_models/patches/qwen3_5.py
        target: /usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_5.py
        sha256: "9999999999999999999999999999999999999999999999999999999999999999"
""",
    )


def test_curated_model_resolves_complete_typed_plan(tmp_path):
    entries = load_roster_recipes([_basic_recipe(tmp_path).parent / "pool-rccs.yaml"])
    recipe = resolve_recipe_entry(entries, model="Qwen-AgentWorld-35B-A3B", variant=None)
    spec = build_spec_from_recipe(
        recipe,
        site=SITE_RCCS,
        project_id="poster-iso-a",
        owner_id="alice",
    )
    assert spec.model.name == "Qwen-AgentWorld-35B-A3B"
    assert spec.runtime.image.startswith("docker.io/vllm/")
    assert spec.runtime.nodes == 1 and spec.runtime.gpus_per_node == 1
    assert spec.engine.language_model_only is True
    assert spec.engine.kv_cache_memory_bytes == 8589934592
    assert spec.engine.max_num_batched_tokens == 8192
    assert spec.engine.max_model_len == 262144
    assert spec.patches and spec.patches[0].target.endswith("qwen3_5.py")
    assert spec.site.partition == "ng-dgx-m0"


def test_exact_variant_beats_canonical_fallback(tmp_path):
    roster = _write_roster(
        tmp_path,
        """\
endpoints:
  - name: m-variant
    model: M
    path: /x/m
    image: img:a
    partition: p
    node_count: 2
    gpu_count: 1
    max_model_len: 4096
    engine: {gpu_mem_util: 0.9}
  - name: m-default
    model: M
    path: /x/m
    image: img:b
    partition: p
    node_count: 1
    gpu_count: 1
    max_model_len: 4096
    engine: {gpu_mem_util: 0.9}
""",
    )
    entries = load_roster_recipes([roster])
    chosen = resolve_recipe_entry(entries, model="M", variant="m-variant")
    assert chosen["name"] == "m-variant"
    chosen_default = resolve_recipe_entry(entries, model="M", variant=None)
    assert chosen_default["name"] == "m-variant"  # first-insert order fallback when any non-name


def test_generic_model_produces_constrained_fallback():
    spec = build_generic_spec(
        model="user-x/model",
        served_name="user-x/model",
        storage_ref="/shared/users/alice/model",
        site=SITE_RCCS,
        project_id="proj-a",
        owner_id="alice",
    )
    assert spec.engine.extra_args == ()
    assert spec.patches == ()
    assert spec.runtime.image  # site default image only (no user image)
    assert spec.provenance.recipe_file is None
    assert spec.provenance.recipe_hash == ""


def test_missing_recipe_error_is_extractable():
    with pytest.raises(RecipeNotFoundError) as err:
        resolve_recipe_entry([], model="no/such", variant=None)
    assert "no/such" in str(err.value)


def test_invalid_recipe_rejected_with_field_context(tmp_path):
    roster = _write_roster(
        tmp_path,
        """\
endpoints:
  - name: broken
    model: B
    image: img:a
    partition: p
    node_count: 1
    gpu_count: 1
    max_model_len: 4096
    engine: {gpu_mem_util: 0.9}
""",
    )
    entries = load_roster_recipes([roster])
    recipe = resolve_recipe_entry(entries, model="B", variant=None)
    with pytest.raises(RecipeSchemaError) as err:
        build_spec_from_recipe(recipe, site=SITE_RCCS, project_id="p", owner_id="u")
    assert "path" in str(err.value)


def test_user_overrides_allowed_fields_only():
    spec = build_generic_spec(
        model="m",
        served_name="m",
        storage_ref="/shared/users/u/m",
        site=SITE_RCCS,
        project_id="p",
        owner_id="u",
    )
    tuned = apply_user_overrides(
        spec, {"partition": "ng-dgx-m1", "max_model_len": 8192, "gpu_count": 1}
    )
    assert tuned.site.partition == "ng-dgx-m1"
    assert tuned.engine.max_model_len == 8192
    with pytest.raises(RecipePolicyError):
        apply_user_overrides(spec, {"image": "evil/image:latest"})
    with pytest.raises(RecipePolicyError):
        apply_user_overrides(spec, {"env": {"VLLM_EXTRA_ARGS": "--x"}})
    with pytest.raises(RecipePolicyError):
        apply_user_overrides(spec, {"extra_args": ["--could-not-care-less"]})


def test_plan_hash_deterministic_across_field_order_and_recipe_file(tmp_path):
    entries = load_roster_recipes([_basic_recipe(tmp_path)])
    recipe = resolve_recipe_entry(entries, model="Qwen-AgentWorld-35B-A3B", variant=None)
    recipe["__reordered__"] = False
    recipe.pop("__reordered__")
    recipe["_recipe_hash"] = hashlib.sha256((tmp_path / "pool-rccs.yaml").read_bytes()).hexdigest()
    spec_a = build_spec_from_recipe(recipe, site=SITE_RCCS, project_id="p", owner_id="u")
    spec_b = build_spec_from_recipe(recipe, site=SITE_RCCS, project_id="p", owner_id="u")
    assert spec_a.plan_hash() == spec_b.plan_hash()
    # hash stability under shuffled json representation
    d = spec_a.to_plan_dict()
    canonical = json.dumps(d, sort_keys=True, default=str)
    assert spec_a.plan_hash() == hashlib.sha256(canonical.encode()).hexdigest()


def test_site_difference_targets_not_model(tmp_path):
    entries = load_roster_recipes([_basic_recipe(tmp_path)])
    recipe = resolve_recipe_entry(entries, model="Qwen-AgentWorld-35B-A3B", variant=None)
    a = build_spec_from_recipe(recipe, site=SITE_RCCS, project_id="p", owner_id="u")
    b = build_spec_from_recipe(recipe, site=SITE_AI4S, project_id="p", owner_id="u")
    assert a.model == b.model and a.engine == b.engine
    assert a.site != b.site
    assert a.plan_hash() != b.plan_hash()


def test_project_and_owner_propagated(tmp_path):
    entries = load_roster_recipes([_basic_recipe(tmp_path)])
    recipe = resolve_recipe_entry(entries, model="Qwen-AgentWorld-35B-A3B", variant=None)
    spec = build_spec_from_recipe(
        recipe,
        site=SITE_RCCS,
        project_id="poster-iso-a",
        owner_id="alice",
    )
    assert spec.model.project_id == "poster-iso-a"
    assert spec.model.owner_id == "alice"


def test_spec_is_frozen_typed_object(tmp_path):
    entries = load_roster_recipes([_basic_recipe(tmp_path)])
    recipe = resolve_recipe_entry(entries, model="Qwen-AgentWorld-35B-A3B", variant=None)
    spec = build_spec_from_recipe(recipe, site=SITE_RCCS, project_id="p", owner_id="u")
    assert isinstance(spec, ResolvedEndpointSpec)
    from dataclasses import FrozenInstanceError

    with pytest.raises(FrozenInstanceError):
        spec.engine = None  # type: ignore[misc]


def test_rivault_job_timeout_promoted_from_entry_level(tmp_path):
    """Legacy overrides: ``RIVAULT_JOB_TIMEOUT`` at entry/defaults level must
    flow into the serving env channel (Bdrt rosters place it next to
    ``timeout``, which the spec previously ignored)."""
    from umrs.orchestrator.serving_spec import build_spec_from_legacy_entry

    top_only = {
        "model": "glm-5.2",
        "path": "/models/glm-5.2",
        "timeout": 172800,
        "RIVAULT_JOB_TIMEOUT": "0",
    }
    spec_a = build_spec_from_legacy_entry(
        dict(top_only, node_count=2, gpu_count=1, partition="spark-double"),
        site=SITE_RCCS,
        project_id="p",
        owner_id="u",
        storage_ref="/models/glm-5.2",
    )
    assert spec_a.env.get("RIVAULT_JOB_TIMEOUT") == "0"

    # env_vars-home must win when both exist.
    from_env = {
        "model": "glm-5.2",
        "path": "/models/glm-5.2",
        "timeout": 172800,
        "RIVAULT_JOB_TIMEOUT": "0",
        "env_vars": {"RIVAULT_JOB_TIMEOUT": "82200"},
        "node_count": 2,
        "gpu_count": 1,
        "partition": "spark-double",
    }
    spec_b = build_spec_from_legacy_entry(
        from_env,
        site=SITE_RCCS,
        project_id="p",
        owner_id="u",
        storage_ref="/models/glm-5.2",
    )
    assert spec_b.env.get("RIVAULT_JOB_TIMEOUT") == "82200"

    # Absent in both homes: stays absent (script default applies).
    bare = {
        "model": "glm-5.2",
        "path": "/models/glm-5.2",
        "timeout": 172800,
        "node_count": 2,
        "gpu_count": 1,
        "partition": "spark-double",
    }
    spec_c = build_spec_from_legacy_entry(
        bare,
        site=SITE_RCCS,
        project_id="p",
        owner_id="u",
        storage_ref="/models/glm-5.2",
    )
    assert "RIVAULT_JOB_TIMEOUT" not in spec_c.env
