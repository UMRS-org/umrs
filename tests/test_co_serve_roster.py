"""ADR-109 D-1/D-2 (V3): the roster's ``co_serve`` key and the sidecar rule.

``svc-vision2`` (SAM3.1, BentoML) is co-served in the ``svc-vision`` job. The
roster declares it under ``co_serve``; only vision co-servings exist, so any
other type is rejected by name. An LLM job starts the ``vision-bentoml``
sidecar only when its entry declares one.
"""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from umrs.orchestrator import co_serve
from umrs.orchestrator.co_serve import CoServe, parse_co_serve, sidecar_profiles
from umrs.orchestrator.endpoint_relay import entry_digest, keep_alive_policies

SAM3 = {
    "name": "sam3.1",
    "model": "SAM3.1",
    "aliases": ["svc-vision2"],
    "model_type": "vision",
    "adapter": "sam3",
}
ENTRY = {
    "name": "qwen3.6-35b-a3b-nvfp4",
    "model": "Qwen3.6-35B-A3B-NVFP4",
    "aliases": ["svc-vision"],
    "service": {"project_id": "default", "user": "mnagaso", "replicas": 1},
    "env_vars": {"RIVAULT_PROFILES": "harness vllm litellm-db"},
}
POOL = Path(__file__).resolve().parents[1] / "deploy/rccs/pool-rccs.yaml"


def _entry(*items) -> dict:
    return {**ENTRY, "co_serve": list(items)}


# ── parsing and validation ──────────────────────────────────────────────


def test_a_vision_co_serving_parses_with_its_aliases_and_adapter():
    specs, errors = parse_co_serve(_entry(SAM3))
    assert errors == []
    assert specs == [
        CoServe("sam3.1", "SAM3.1", "vision", "sam3", ("svc-vision2",), {})
    ]


def test_an_entry_without_co_serve_has_none():
    assert parse_co_serve(ENTRY) == ([], [])
    assert co_serve.co_serves(None) == []


LLM_SIDE = {**SAM3, "name": "side-llm", "model_type": "llm"}


def test_a_non_vision_co_serving_is_rejected_by_name():
    specs, errors = parse_co_serve(_entry(LLM_SIDE, SAM3))
    assert [s.name for s in specs] == ["sam3.1"]
    assert errors == [
        "qwen3.6-35b-a3b-nvfp4: co_serve 'side-llm' has model_type 'llm'; "
        "only vision co-serving is supported"
    ]


def test_a_rejected_co_serving_is_logged_and_skipped(caplog):
    with caplog.at_level("WARNING"):
        assert co_serve.co_serves(_entry(LLM_SIDE)) == []
    assert "side-llm" in caplog.text


@pytest.mark.parametrize(
    ("item", "needle"),
    [
        ({**SAM3, "adapter": ""}, "needs an adapter"),
        ({**SAM3, "model": ""}, "needs a name and a model"),
        ({**SAM3, "aliases": "svc-vision2"}, "aliases must be names"),
        ({**SAM3, "env": ["X=1"]}, "env a mapping"),
        ("sam3.1", "must be a mapping"),
    ],
)
def test_invalid_items_are_rejected(item, needle):
    specs, errors = parse_co_serve(_entry(item))
    assert specs == [] and needle in errors[0]


def test_a_job_runs_one_sidecar():
    second = {**SAM3, "name": "hmae", "model": "HMAE", "adapter": "hmae"}
    specs, errors = parse_co_serve(_entry(SAM3, second, SAM3))
    assert [s.name for s in specs] == ["sam3.1"]
    assert "second vision serving" in errors[0] and "declared twice" in errors[1]


def test_co_serve_must_be_a_list():
    assert parse_co_serve({**ENTRY, "co_serve": SAM3})[1] == [
        "qwen3.6-35b-a3b-nvfp4: co_serve must be a list"
    ]


def test_memory_budget_env_defaults_and_roster_overrides():
    """ADR-109 D-2: VISION_MAX_BATCH=1 and a capped allocator, via co_serve
    env defaults only; vLLM's utilisation is not touched."""
    (spec,) = parse_co_serve(_entry({**SAM3, "env": {"VISION_GPU_MEMORY_BUDGET_GIB": 12}}))[0]
    env = co_serve.job_env([spec])
    assert env["VISION_MAX_BATCH"] == "1"
    assert env["VISION_GPU_MEMORY_BUDGET_GIB"] == "12"
    assert "expandable_segments:True" in env["VISION_PYTORCH_CUDA_ALLOC_CONF"]
    assert env["RIVAULT_VISION_STRICT"] == "0"
    assert not any(key.startswith("VLLM_") for key in env)


def test_job_ids_and_ports_of_a_co_served_row():
    assert co_serve.co_job_id("4242", "sam3.1") == "4242::sam3.1"
    assert co_serve.primary_job_id("4242::sam3.1") == "4242"
    assert co_serve.primary_job_id("4242") == "4242"
    assert co_serve.co_port(11434) == 12434
    metadata = {co_serve.CO_SERVE_KEY: [SAM3]}
    assert co_serve.submit_env(metadata, 11434) == {"BENTOML_VISION_DOCKER_PORT": "12434"}
    assert co_serve.submit_env({}, 11434) == {}


# ── role aliases ────────────────────────────────────────────────────────


def test_the_co_serve_alias_names_the_co_served_model(tmp_path, monkeypatch):
    from umrs.orchestrator.provision_roster import (
        resolve_model_recipe,
        role_alias_models,
    )

    roster = tmp_path / "pool.yaml"
    roster.write_text(yaml.safe_dump({"endpoints": [_entry(SAM3)]}), encoding="utf-8")
    monkeypatch.setenv("UMRS_PROVISION_ROSTER", str(roster))
    assert role_alias_models("svc-vision2") == ("SAM3.1",)
    assert role_alias_models("svc-vision") == ("Qwen3.6-35B-A3B-NVFP4",)
    # SAM3.1 is never provisioned alone from the Qwen recipe.
    assert resolve_model_recipe("SAM3.1") is None


# ── the sidecar rule ────────────────────────────────────────────────────


def test_sidecar_profiles_add_only_for_a_co_served_job():
    assert sidecar_profiles("harness vllm vision-bentoml litellm-db", co_served=False) == (
        "harness vllm litellm-db"
    )
    assert sidecar_profiles("harness vllm litellm-db", co_served=True) == (
        "harness vllm litellm-db vision-bentoml"
    )


def test_launcher_llm_jobs_keep_stripping_the_vision_sidecar(monkeypatch):
    from umrs.server import rivault_launcher

    monkeypatch.setenv("RIVAULT_PROFILES", "harness vllm vision-bentoml")
    assert rivault_launcher._resolve_profiles_for_model({"model_type": "openai://m"}) == (
        "harness",
        "vllm",
    )


def test_launcher_adds_the_sidecar_for_a_vision_co_serve(monkeypatch):
    from umrs.server import rivault_launcher

    monkeypatch.setenv("RIVAULT_PROFILES", "harness vllm")
    cfg = {"model_type": "openai://m", "co_serve": [SAM3]}
    assert rivault_launcher._resolve_profiles_for_model(cfg) == (
        "harness",
        "vllm",
        "vision-bentoml",
    )


def _provision_env(monkeypatch, tmp_path, entry, env_vars):
    from umrs.api.endpoints import _apply_co_serve

    roster = tmp_path / "pool.yaml"
    roster.write_text(yaml.safe_dump({"endpoints": [entry]}), encoding="utf-8")
    monkeypatch.setenv("UMRS_PROVISION_ROSTER", str(roster))
    env = dict(env_vars)
    metadata = {"launch_profiles": ["harness", "vllm"]}
    request = SimpleNamespace(model=entry["model"], variant=None)
    _apply_co_serve(request, env, metadata, is_vision=False)
    return env, metadata


def test_provision_strips_the_sidecar_from_a_plain_llm_job(monkeypatch, tmp_path):
    env, metadata = _provision_env(
        monkeypatch,
        tmp_path,
        ENTRY,
        {"RIVAULT_PROFILES": "harness vllm vision-bentoml litellm-db"},
    )
    assert env == {"RIVAULT_PROFILES": "harness vllm litellm-db"}
    assert co_serve.CO_SERVE_KEY not in metadata


def test_provision_adds_the_sidecar_and_budget_for_a_co_served_job(monkeypatch, tmp_path):
    env, metadata = _provision_env(
        monkeypatch,
        tmp_path,
        _entry(SAM3),
        {"RIVAULT_PROFILES": "harness vllm litellm-db", "VISION_MAX_BATCH": "2"},
    )
    assert env["RIVAULT_EXTRA_PROFILES"] == "vision-bentoml"
    assert env["VISION_MAX_BATCH"] == "2"  # an explicit request value wins
    assert env["VISION_GPU_MEMORY_BUDGET_GIB"] == "16"
    assert metadata["launch_profiles"] == ["harness", "vllm", "vision-bentoml"]
    assert metadata[co_serve.CO_SERVE_KEY] == [
        {**SAM3, "aliases": ["svc-vision2"]}
    ]


# ── the rccs roster (guarded until the V3 live leg) ─────────────────────


def _pool_entries() -> list[dict]:
    return yaml.safe_load(POOL.read_text(encoding="utf-8"))["endpoints"]


def _svc_vision(entries: list[dict]) -> dict:
    return next(e for e in entries if "svc-vision" in (e.get("aliases") or ()))


def test_live_rccs_roster_does_not_start_the_sidecar_yet():
    """Guard: no umrs-vision archive is staged on rccs, so the live svc-vision
    entry carries no co_serve and its profiles keep the sidecar out."""
    entry = _svc_vision(_pool_entries())
    assert "co_serve" not in entry
    assert "vision-bentoml" not in entry["env_vars"]["RIVAULT_PROFILES"]
    (policy,) = keep_alive_policies([entry])
    assert policy.co_serve == ()


def test_the_commented_rccs_co_serve_block_enables_svc_vision2():
    """The block marked for the V3 live leg parses to the ADR-109 D-2 entry,
    and enabling it changes the recipe digest (the controller rolls)."""
    text = POOL.read_text(encoding="utf-8")
    match = re.search(r"^    # co_serve:\n    #   (- \{.*\})$", text, re.MULTILINE)
    assert match, "commented co_serve block missing from pool-rccs.yaml"
    items = yaml.safe_load(match.group(1))
    entry = _svc_vision(_pool_entries())
    enabled = {**entry, "co_serve": items}
    specs, errors = parse_co_serve(enabled)
    assert errors == []
    assert [(s.model, s.aliases, s.adapter) for s in specs] == [
        ("SAM3.1", ("svc-vision2",), "sam3")
    ]
    (policy,) = keep_alive_policies([enabled])
    assert policy.co_serve == ("SAM3.1",)
    assert entry_digest(enabled) != entry_digest(entry)
    assert entry["env_vars"]["VLLM_GPU_MEM_UTIL"] == "0.45"
