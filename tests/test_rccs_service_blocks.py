"""rccs service tier (ADR-106 plan §10.3, 2026-11-03): the five models that
served once carry ``service`` blocks; svc-chat (DeepSeek, blocked) and Nimble
(keep_alive until M-3) do not."""

from __future__ import annotations

from pathlib import Path

import yaml
from umrs.orchestrator.endpoint_relay import keep_alive_policies

POOL = Path(__file__).resolve().parents[1] / "deploy/rccs/pool-rccs.yaml"


def _policies():
    entries = yaml.safe_load(POOL.read_text(encoding="utf-8"))["endpoints"]
    return {p.model: p for p in keep_alive_policies(entries)}


def test_live_roles_are_kept_up_by_the_controller():
    policies = _policies()
    for model in (
        "Qwen3.8-27B-NVFP4",
        "Qwen3.6-35B-A3B-NVFP4",
        "Qwen3-Embedding-0.6B",
        "Qwen3-Reranker-0.6B",
        "PaddleOCR-VL-1.6",
    ):
        policy = policies[model]
        assert policy.service is True, model
        assert (policy.project_id, policy.user, policy.replicas) == ("default", "mnagaso", 1)


def test_non_chat_roles_probe_models():
    policies = _policies()
    for model in ("Qwen3-Embedding-0.6B", "Qwen3-Reranker-0.6B", "PaddleOCR-VL-1.6"):
        assert policies[model].readiness_probe == "models", model
    assert policies["Qwen3.6-35B-A3B-NVFP4"].readiness_probe in (None, "chat")


def test_blocked_or_relayed_models_have_no_service_block():
    policies = _policies()
    assert "DeepSeek-V4-Flash" not in policies
    nimble = policies.get("Bespoke-Nimble-9B")
    assert nimble is None or nimble.service is False
