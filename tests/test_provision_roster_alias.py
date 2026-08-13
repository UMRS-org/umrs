"""Roster ``aliases:`` regression coverage (2026-08-13).

bdrt brought up the deepseek/qwen pair on jobs whose catalog slug did not match
the recipe's canonical key (registry row ``qwen3.6-35b-a3b`` vs roster entry
``Qwen3.6-35B-A3B-NVFP4``). ``resolve_model_provision_defaults`` missed, so the
provision silently fell back to the default plan (frozen nightly image
+ 4096 ctx + triton MoE ifrastructure class: engine init crash /
silo-cup-or-sim: ``KeyError: 'layers.0.mlp.experts.w2_input_scale'``).
"""

from __future__ import annotations

import importlib.util
import os
import tempfile
from pathlib import Path

_MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "src" / "umrs" / "orchestrator" / "provision_roster.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("_provision_roster_alias", _MODULE_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_ROSTER = """\
pool: test-alias
defaults:
  source: bdrt
  tier: system

endpoints:
  - name: qwen3.6-35b-a3b
    model: Qwen3.6-35B-A3B-NVFP4
    aliases:
      - qwen3.6-35b-a3b
    tier: system
    node_count: 1
    partition: spark
    max_model_len: 262144
    env_vars:
      VLLM_DOCKER_IMAGE: "docker.io/vllm/vllm-openai:latest"
      VLLM_MOE_BACKEND: "marlin"
      VLLM_MAX_MODEL_LEN: "262144"
"""


def _write_roster(tmpdir: str) -> str:
    path = os.path.join(tmpdir, "pool.yaml")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(_ROSTER)
    return path


class TestRosterAliases:
    def test_alias_resolves_recipe_env(self, monkeypatch):
        mod = _load_module()
        with tempfile.TemporaryDirectory() as tmp:
            monkeypatch.setenv(mod.ROSTER_ENV_VAR, _write_roster(tmp))
            mod._cache.clear()
            for name in ("qwen3.6-35b-a3b", "Qwen3.6-35B-A3B-NVFP4"):
                out = mod.resolve_model_provision_defaults(name)
                assert out is not None, f"{name} missed"
                assert out["node_count"] == 1
                assert out["max_model_len"] == 262144
                ev = out["env_vars"]
                assert ev["VLLM_MOE_BACKEND"] == "marlin"
                assert ev["VLLM_MAX_MODEL_LEN"] == "262144"

    def test_alias_does_not_shadow_other_models(self, monkeypatch):
        mod = _load_module()
        with tempfile.TemporaryDirectory() as tmp:
            monkeypatch.setenv(mod.ROSTER_ENV_VAR, _write_roster(tmp))
            mod._cache.clear()
            assert mod.resolve_model_provision_defaults("qwen3.6-35b") is None

    def test_alias_serving_spec_path(self, monkeypatch):
        mod = _load_module()
        with tempfile.TemporaryDirectory() as tmp:
            monkeypatch.setenv(mod.ROSTER_ENV_VAR, _write_roster(tmp))
            mod._cache.clear()
            out = mod.resolve_model_provision_defaults("qwen3.6-35b-a3b", variant=None)
            assert out is not None and out["partition"] == "spark"
