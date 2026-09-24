"""rccs curated roster: Qwen3.8-27B NVFP4 on one GB10 (vLLM recipe
dgx_spark_gb10 / nvfp4_nvidia), resolved through the live roster lookup."""

from __future__ import annotations

from pathlib import Path

import pytest
from umrs.orchestrator import provision_roster as pr

POOL = Path(__file__).resolve().parents[1] / "deploy/rccs/pool-rccs.yaml"


@pytest.fixture
def defaults(monkeypatch):
    monkeypatch.setenv(pr.ROSTER_ENV_VAR, str(POOL))
    pr._cache.clear()
    yield pr.resolve_model_provision_defaults("Qwen3.8-27B-NVFP4")
    pr._cache.clear()


def test_resolves_single_gb10_node(defaults):
    assert defaults is not None
    assert (defaults["partition"], defaults["node_count"], defaults["gpu_count"]) == (
        "ng-dgx-m0",
        1,
        1,
    )
    assert defaults["max_model_len"] == 262144


def test_recipe_engine_knobs_ride_the_env(defaults):
    env = defaults["env_vars"]
    assert env["VLLM_DOCKER_IMAGE"] == "docker.io/vllm/vllm-openai:qwen38"
    assert float(env["VLLM_GPU_MEM_UTIL"]) <= 0.78  # under the GB10 partition cap
    assert (env["VLLM_TOOL_CALL_PARSER"], env["VLLM_REASONING_PARSER"]) == ("qwen3_xml", "qwen3")
    assert env["VLLM_SPECULATIVE_CONFIG"] == '{"method":"mtp","num_speculative_tokens":5}'
    assert "--kv-cache-dtype fp8" in env["VLLM_EXTRA_ARGS"]


def test_legacy_plan_builds(monkeypatch):
    from umrs.orchestrator.serving_spec import SiteSpec

    monkeypatch.setenv(pr.ROSTER_ENV_VAR, str(POOL))
    pr._cache.clear()
    plan = pr.resolve_model_serving_spec(
        "Qwen3.8-27B-NVFP4",
        site=SiteSpec(cluster="rccs", backend="slurm-remote", partition="ng-dgx-m0"),
        storage_ref="/lvs0/models/Qwen3.8-27B-NVFP4",
    )
    pr._cache.clear()
    assert plan is not None
    assert plan.site.partition == "ng-dgx-m0"
