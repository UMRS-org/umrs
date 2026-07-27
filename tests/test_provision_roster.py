"""Roster resolution for single-endpoint provisions.

``provision_roster`` is what stops ``r4 endpoint up --model X`` (and the VS Code
play button) from landing a large multi-node model on a 1-node/1-GPU default.
Its failure mode is silent — Slurm simply holds the undersized job pending — so
the resolution rules are pinned here.

The module is declared a *foundation* module: stdlib + optional PyYAML only, no
imports from ``umrs.api`` / ``umrs.server``. It is therefore loaded straight
from its file rather than through ``umrs.orchestrator``, which keeps these tests
hermetic (no psycopg2 / no live model store) and fails loudly if the module ever
grows a heavyweight dependency.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

_MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "src" / "umrs" / "orchestrator" / "provision_roster.py"
)


def _load_module():
    """Import provision_roster in isolation, asserting it stays stdlib-only."""
    spec = importlib.util.spec_from_file_location("_provision_roster", _MODULE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def roster(tmp_path, monkeypatch):
    """Write a roster file, point the env var at it, return the fresh module."""

    def _write(payload: dict, filename: str = "pool-test.yaml") -> object:
        path = tmp_path / filename
        path.write_text(yaml.safe_dump(payload), encoding="utf-8")
        existing = os.environ.get("UMRS_PROVISION_ROSTER")
        joined = f"{existing}{os.pathsep}{path}" if existing else str(path)
        monkeypatch.setenv("UMRS_PROVISION_ROSTER", joined)
        # Fresh module per call so the mtime cache never leaks across tests.
        return _load_module()

    return _write


_NAMED_ENTRY = {
    "endpoints": [
        {
            "name": "laguna-s-2.1-fp8",
            "model": "laguna-s-2.1-fp8",
            "tier": "system",
            "node_count": 2,
            "partition": "edge-double",
            "max_model_len": 262144,
            "env_vars": {
                "VLLM_DOCKER_IMAGE": "docker.io/vllm/vllm-openai:hy3-arm64-cu130",
                "VLLM_TP": "2",
            },
        }
    ]
}


def test_named_entry_resolves_without_a_variant(roster):
    """A roster whose only entry carries a ``name`` must still answer a plain
    model lookup.

    Regression: ``_build_index`` keyed solely on ``(model, name)``. When every
    entry had a ``name`` — as every bdrt roster entry does — there was no
    ``(model, None)`` key, so ``r4 endpoint up --model laguna-s-2.1-fp8`` missed
    the roster entirely and silently fell back to 1-node/1-GPU defaults,
    discarding the per-deployment vLLM overrides along with the sizing.
    """
    module = roster(_NAMED_ENTRY)

    defaults = module.resolve_model_provision_defaults("laguna-s-2.1-fp8")

    assert defaults is not None, "named entry must be reachable without a variant"
    assert defaults["node_count"] == 2
    assert defaults["partition"] == "edge-double"
    assert defaults["env_vars"]["VLLM_DOCKER_IMAGE"] == "docker.io/vllm/vllm-openai:hy3-arm64-cu130"


def test_explicit_variant_still_resolves(roster):
    """Named variants stay addressable for explicit ``pool up``."""
    module = roster(_NAMED_ENTRY)

    defaults = module.resolve_model_provision_defaults(
        "laguna-s-2.1-fp8", variant="laguna-s-2.1-fp8"
    )

    assert defaults is not None
    assert defaults["node_count"] == 2


def test_first_entry_per_model_is_the_canonical_default(roster):
    """With several variants of one model, the first is the default."""
    module = roster(
        {
            "endpoints": [
                {"name": "big", "model": "m", "node_count": 4, "partition": "p-big"},
                {"name": "small", "model": "m", "node_count": 1, "partition": "p-small"},
            ]
        }
    )

    assert module.resolve_model_provision_defaults("m")["node_count"] == 4
    assert module.resolve_model_provision_defaults("m", variant="small")["node_count"] == 1
    assert module.resolve_model_provision_defaults("m", variant="big")["node_count"] == 4


def test_unknown_model_returns_none(roster):
    """No roster entry means no defaults — never a bogus fallback."""
    module = roster(_NAMED_ENTRY)

    assert module.resolve_model_provision_defaults("not-in-roster") is None


def test_defaults_block_merges_into_entries(roster):
    """Top-level ``defaults`` apply, with entry fields winning."""
    module = roster(
        {
            "defaults": {"tier": "system", "partition": "shared"},
            "endpoints": [{"name": "v", "model": "m", "partition": "override"}],
        }
    )

    defaults = module.resolve_model_provision_defaults("m")

    assert defaults["tier"] == "system"
    assert defaults["partition"] == "override"


def test_missing_roster_file_is_not_fatal(roster, tmp_path, monkeypatch):
    """A stale path must degrade to "no roster", not raise."""
    module = roster(_NAMED_ENTRY)
    monkeypatch.setenv("UMRS_PROVISION_ROSTER", str(tmp_path / "absent.yaml"))

    assert module.resolve_model_provision_defaults("laguna-s-2.1-fp8") is None


def test_unset_env_var_disables_the_roster(roster, monkeypatch):
    """Unset means no roster — historical behaviour for clusters without one."""
    module = roster(_NAMED_ENTRY)
    monkeypatch.delenv("UMRS_PROVISION_ROSTER", raising=False)

    assert module.resolve_model_provision_defaults("laguna-s-2.1-fp8") is None


def test_returned_mapping_is_a_copy(roster):
    """Callers mutate the result; the cached index must not be corrupted."""
    module = roster(_NAMED_ENTRY)

    first = module.resolve_model_provision_defaults("laguna-s-2.1-fp8")
    first["node_count"] = 99

    second = module.resolve_model_provision_defaults("laguna-s-2.1-fp8")
    assert second["node_count"] == 2
