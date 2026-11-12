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


FRONT_SITE = POOL.parents[1] / "rccs-front/site.yaml"


def test_single_node_roles_prefer_ng_dgx_s_with_m_as_fallback():
    """Maintainer 2026-11-03: single-node roles on -s always, -m only as a
    fallback. Each entry lists ng-dgx-s first (m3, the step lane, last), and
    the rccs front prefers ng-dgx-s on start ties."""
    entries = yaml.safe_load(POOL.read_text(encoding="utf-8"))["endpoints"]
    single = [
        e
        for e in entries
        if ("service" in e or e["model"] == "Bespoke-Nimble-9B") and e.get("node_count", 1) == 1
    ]
    assert len(single) == 6
    for entry in single:
        candidates = entry["partition"].split(",")
        assert (candidates[0], candidates[-1]) == ("ng-dgx-s", "ng-dgx-m3"), entry["name"]
    scheduler = yaml.safe_load(FRONT_SITE.read_text(encoding="utf-8"))["scheduler"]
    assert scheduler["preferred_partitions"] == "ng-dgx-s"


def test_deepseek_v41_chat_services_on_m_pairs_and_ai_pro6000():
    """Maintainer 2026-11-12: DeepSeek-V4.1-Flash-1017 as the always-on chat
    service on an ng-dgx-m pair (TensorFold recipe) and on both ai-pro6000
    nodes (SGLang, NVFP4). The m3 step lane is never a candidate, and the
    older DeepSeek-V4-Flash no longer answers for svc-chat."""
    entries = {e["name"]: e for e in yaml.safe_load(POOL.read_text(encoding="utf-8"))["endpoints"]}
    policies = _policies()
    tf, sg = entries["deepseek-v4.1-flash-1017"], entries["deepseek-v4.1-flash-1017-nvfp4"]
    tf_env, sg_env = tf["env_vars"], sg["env_vars"]
    facts = {
        "service": (
            policies["DeepSeek-V4.1-Flash-1017"].service,
            policies["DeepSeek-V4.1-Flash-1017-NVFP4"].service,
        ),
        "nodes": (tf["node_count"], sg["node_count"]),
        "tf_partitions": tf["partition"].split(","),
        "sg_partition": sg["partition"],
        "runners": (tf_env["RIVAULT_NATIVE_LLM_RUNNER"], sg_env["RIVAULT_NATIVE_LLM_RUNNER"]),
        "tf_fit": (tf_env["TF_DRAFTS"], tf_env["TF_CONTEXT"], tf_env["TF_MIN_AVAIL_GIB"]),
        "tf_rendezvous": (tf_env["TF_MASTER_IFACE"], tf_env["TF_WORKER_MODE"]),
        "chat_owners": sorted(n for n, e in entries.items() if "svc-chat" in e.get("aliases", [])),
    }
    assert facts == {
        "service": (True, True),
        "nodes": (2, 2),
        "tf_partitions": ["ng-dgx-m2", "ng-dgx-m0", "ng-dgx-m1"],
        "sg_partition": "ai-pro6000",
        "runners": ("tensorfold-dsv41", "sglang-mn"),
        "tf_fit": ("1", "131072", "100"),
        "tf_rendezvous": ("enP7s7", "step"),
        "chat_owners": ["deepseek-v4.1-flash-1017", "deepseek-v4.1-flash-1017-nvfp4"],
    }
