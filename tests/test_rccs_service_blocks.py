"""rccs service tier (ADR-106 plan §10.3, 2026-11-03): the five models that
served once carry ``service`` blocks; svc-chat (DeepSeek, blocked) and Nimble
(keep_alive until M-3) do not."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import yaml
from umrs.orchestrator.endpoint_relay import keep_alive_policies

POOL = Path(__file__).resolve().parents[1] / "deploy/rccs/pool-rccs.yaml"
GIB = 1 << 30

# SGLang 314bce7c + checkpoint 3ae9c0d3, measured by failed job 502594:
# SM120 selects V4 pages (584 B raw, 585 B after page padding); 40 SWA layers
# plus the 41/40 DSpark draft inflation price one SWA token at 23,985 B. One
# ratio-1 and three ratio-2 owner layers price a full token with the same
# inflation. Revalidate every constant against the pinned source if either pin,
# the model shape, speculative settings, page size or KV layout changes.
SWA_BYTES_PER_TOKEN = Fraction(585 * 40 * 41, 40)
FULL_BYTES_PER_TOKEN = Fraction(41, 40) * (585 + 68 + 3 * Fraction(585 + 68, 2))
OBSERVED_AVAILABLE_BYTES = 3675 * GIB // 1000


def _pool_entry(name: str) -> dict:
    entries = yaml.safe_load(POOL.read_text(encoding="utf-8"))["endpoints"]
    return next(entry for entry in entries if entry["name"] == name)


def _swa_cap_tokens(max_running: int) -> int:
    per_request = 128 + 128 * 6 + 256 + 2 * 6
    unaligned = max_running * per_request + 2 * 8192 + 256
    unaligned += 4 * max_running * (128 + 256)
    return (unaligned + 255) // 256 * 256


def _ratio2_state_bytes(max_running: int) -> int:
    # Six draft tokens select an eight-row ratio-2 ring. Three owner layers
    # carry (kv, score) fp32 state with 2*512 values per row and request slot.
    ring_size = 1 << (6 + 1).bit_length()
    state_rows = (max_running + 1) * ring_size + ring_size + 1
    state_rows = (state_rows + 1) // 2 * 2
    return state_rows * (2 * 512) * 4 * 3


def _required_bytes(max_running: int) -> Fraction:
    fixed = _swa_cap_tokens(max_running) * SWA_BYTES_PER_TOKEN
    fixed += _ratio2_state_bytes(max_running)
    return fixed + 131072 * FULL_BYTES_PER_TOKEN


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
    service on ng-dgx-m pairs (TensorFold recipe). ai-pro6000 NVFP4 remains an
    on-demand svc-chat entry (aliases kept) but its keep-alive service block is
    disabled until external prerequisites are cleared. The m3 step lane is never
    a candidate, and the older DeepSeek-V4-Flash no longer answers for svc-chat."""
    entries = {e["name"]: e for e in yaml.safe_load(POOL.read_text(encoding="utf-8"))["endpoints"]}
    policies = _policies()
    tf, sg = entries["deepseek-v4.1-flash-1017"], entries["deepseek-v4.1-flash-1017-nvfp4"]
    tf_env, sg_env = tf["env_vars"], sg["env_vars"]
    facts = {
        "tf_service": policies["DeepSeek-V4.1-Flash-1017"].service,
        "ai_pro6000_kept": "DeepSeek-V4.1-Flash-1017-NVFP4" in policies,
        "nodes": (tf["node_count"], sg["node_count"]),
        "tf_partitions": tf["partition"].split(","),
        "sg_partition": sg["partition"],
        "runners": (tf_env["RIVAULT_NATIVE_LLM_RUNNER"], sg_env["RIVAULT_NATIVE_LLM_RUNNER"]),
        "tf_fit": (
            tf_env["TF_DRAFTS"],
            tf_env["TF_CONTEXT"],
            tf_env["TF_MIN_AVAIL_GIB"],
            tf_env["TF_MEMORY_RESERVE_GIB"],
            tf_env["TF_PRESTART_RECLAIM_GIB"],
            tf_env["TF_MIN_AVAIL_WAIT_S"],
        ),
        "tf_rendezvous": (tf_env["TF_MASTER_IFACE"], tf_env["TF_WORKER_MODE"]),
        "chat_owners": sorted(n for n, e in entries.items() if "svc-chat" in e.get("aliases", [])),
    }
    assert facts == {
        "tf_service": True,
        "ai_pro6000_kept": False,
        "nodes": (2, 2),
        "tf_partitions": ["ng-dgx-m2", "ng-dgx-m0", "ng-dgx-m1"],
        "sg_partition": "ng-dgx-l",
        "runners": ("tensorfold-dsv41", "sglang-mn"),
        "tf_fit": ("1", "131072", "111", "11.5", "100", "300"),
        "tf_rendezvous": ("enP7s7", "step"),
        "chat_owners": ["deepseek-v4.1-flash-1017", "deepseek-v4.1-flash-1017-nvfp4"],
    }


def test_nvfp4_smoke_budget_keeps_all_other_serving_settings():
    entry = _pool_entry("deepseek-v4.1-flash-1017-nvfp4")
    env = entry["env_vars"]
    assert int(env["SG_MAX_RUNNING"]) == 32
    assert int(env["SG_MAX_RUNNING"]) > 0
    assert entry["max_model_len"] == 131072
    assert "service" not in entry
    assert {
        key: env[key]
        for key in (
            "SG_CONTEXT",
            "SG_KV_DTYPE",
            "SG_MEM_FRACTION",
            "SG_ENGRAM_CONFIG",
            "SG_SPEC",
            "SG_SPEC_BLOCK",
            "SG_RUNTIME",
            "SG_IMAGE",
            "SG_TP",
            "SG_GPUS_PER_NODE",
            "SG_MODEL_ROOT",
        )
    } == {
        "SG_CONTEXT": "131072",
        "SG_KV_DTYPE": "fp8_e4m3",
        "SG_MEM_FRACTION": "0.88",
        "SG_ENGRAM_CONFIG": '{"placement":"cpu","replication":"none","lookup":"auto"}',
        "SG_SPEC": "DSPARK",
        "SG_SPEC_BLOCK": "5",
        "SG_RUNTIME": "singularity",
        "SG_IMAGE": "/home/users/masaru.nagaso/umrs_models_hs/images/sglang-314bce7c-arm64.sif",
        "SG_TP": "4",
        "SG_GPUS_PER_NODE": "2",
        "SG_MODEL_ROOT": (
            "/home/users/masaru.nagaso/umrs_models_hs/hub/"
            "models--nvidia--DeepSeek-V4.1-Flash-1017-NVFP4"
        ),
    }
    assert env["SG_MODEL_DIR"] == (
        f"{env['SG_MODEL_ROOT']}/snapshots/3ae9c0d3fbbea5370bd06e679d22044db15883f1"
    )


def test_nvfp4_swa_cap_leaves_one_full_context_at_measured_budget():
    max_running = int(_pool_entry("deepseek-v4.1-flash-1017-nvfp4")["env_vars"]["SG_MAX_RUNNING"])
    assert _swa_cap_tokens(256) == 707840
    assert round(float(_swa_cap_tokens(256) * SWA_BYTES_PER_TOKEN / GIB), 2) == 15.81
    assert _swa_cap_tokens(max_running) == 103168
    assert _swa_cap_tokens(64) == 189440

    # Job 502594 printed 3.68 GB. Conservatively use only 3.675 GiB, include
    # the ratio-2 fixed state, and require room for one complete 131072-token
    # request. This arithmetic is a smoke admission check, never health proof.
    required = _required_bytes(max_running)
    assert required <= OBSERVED_AVAILABLE_BYTES
    fixed_without_full = required - 131072 * FULL_BYTES_PER_TOKEN
    full_tokens = int((OBSERVED_AVAILABLE_BYTES - fixed_without_full) / FULL_BYTES_PER_TOKEN)
    full_tokens = full_tokens // 256 * 256
    assert full_tokens >= 131072
    assert _required_bytes(64) > OBSERVED_AVAILABLE_BYTES


def test_ai_pro6000_deepseek_v41_nvfp4_not_service_kept():
    """Regression: ai-pro6000 NVFP4 stays on-demand until external prereqs
    (GPU reset/podman, receipt/logs) are cleared; m-pair TensorFold remains kept."""
    policies = _policies()
    assert policies["DeepSeek-V4.1-Flash-1017"].service is True
    assert "DeepSeek-V4.1-Flash-1017-NVFP4" not in policies
    entries = {e["name"]: e for e in yaml.safe_load(POOL.read_text(encoding="utf-8"))["endpoints"]}
    sg = entries["deepseek-v4.1-flash-1017-nvfp4"]
    assert "service" not in sg
    assert "svc-chat" in sg.get("aliases", [])
    assert sg["env_vars"]["SG_RUNTIME"] == "singularity"


def test_ng_dgx_l_reaches_cluster_and_hold_sources():
    rccs = POOL.parent
    cluster_line = next(
        line
        for line in (rccs / "cluster.env").read_text().splitlines()
        if line.startswith("UMRS_PARTITIONS=")
    )
    assert "ng-dgx-l" in cluster_line.split('"')[1].split()
    hold = (rccs / "dsv41/pro6000_hold.sbatch").read_text()
    assert "#SBATCH -p ng-dgx-l" in hold.splitlines()


def test_dsv41_report_source_records_current_rollout_without_invented_results():
    """The report does not turn either failed readiness attempt into a result."""
    report = (POOL.parent / "dsv41/dsv41-serving-bench.html").read_text()
    assert "501690" in report
    assert "ng-dgx-m1" in report
    assert "VLLM_API_KEY" in report
    assert "468620" in report
    assert "468760" in report
    assert "no nvfp4 benchmark results" in report.lower()
    assert "468124 holds" not in report
