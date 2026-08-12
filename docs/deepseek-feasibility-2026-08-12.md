# DeepSeek-V4-Flash-DSpark — RCCS feasibility report (read-only, 2026-08-12)

## Verdict

**Schema-valid but NOT deployable today.** The unified ServingPlan resolution
for the reference recipe (`pool-rccs-3models.yaml` entry) passes cleanly, and
the weights are fully staged on Lustre; however the required custom runtime
image and proposer patch are not present on the cluster, and the multi-node
launch path on this site still carries an untested cross-node TP profile.

## Inputs verified

| Item | State | Evidence |
| --- | --- | --- |
| Weights | ✅ staged | `/lvs0/rccs-dmp/masaru.nagaso/umrs_models/DeepSeek-V4-Flash-DSpark` — 315 GB, 98 safetensors shards + `config.json` (`DeepseekV4ForCausalLM`) |
| Catalog name | ⚠ registration required | registered name must equal roster `model:` (`deepseek-v4-flash-dspark`) |
| Custom image | ❌ missing | `vllm-dspark-runtime:mia-raf-pr1` must be built locally on ALL m1 GB10 nodes via the recipe's `build-dspark-vllm-runtime.sh`; not pulled prior to this report |
| Proposer patch | ❌ missing | `dspark_proposer.py` overlay must be BAKED INTO the image (docker.io/vllm/vllm-openai does NOT ship DSpark spec-decode/nvfp4_ds_mla/B12X kernels) |
| Recipe schema | ✅ valid | `ServingPlan` resolution succeeds: nodes=2, util 0.80, max_model_len 1048576, kv `nvfp4_ds_mla` via VLLM_EXTRA_ARGS, 11 dspark-adjacent envs materialized; plan_hash `3e727fa765f86a36` |
| Launch topology | ⚠ untested | Cross-node TP=2 over 2×1-GPU GB10 nodes via Ray — satisfies already-tested single-node lanes, but RoCE isolation limits partitions; the multi-node worker path is exactly the one annotated as site-failing vs (b) below |

## Site constraints (from the stable roster's own notes)

1. GB10 partitions are isolated 2-node partitions — `node_count: 2` is fine
   inside ONE partition (ng-dgx-m1), but larger topologies need multi-job
   SIF-style launches outside `pool up` (GLM-5.2 PP=6 path is explicitly
   annotated "UMRS multi-job support pending").
2. This site's proctrack/cgroup kills non-batch cgroup vehicles within
   ~40–120 s — the Ray HEAD + srun-worker style used historically for DSpark
   is therefore fragile; the Ray head/driver path under the batch step works
   (270373 proves single-node), but worker-node long-term survivability is
   exactly the risk class these notes flag.

## What would close the gap (ordered)

1. Build `vllm-dspark-runtime:mia-raf-pr1` on both ng-dgx-m1 nodes with the
   DSpark overlay baked in (`COPY overlay/vllm/v1/spec_decode/dspark_proposer.py`).
2. Register `deepseek-v4-flash-dspark` into the RCCS catalog.
3. One controlled 2-node TP=2 launch with dspark recipe to measure Ray worker
   survival on this site's cgroup policy before declaring acceptance.

Until (1) exists DO NOT attempt pool up — public `vllm-openai` images will
fail at model-config inspection (`--kv-cache-dtype nvfp4_ds_mla`, DSpark
speculative config, and B12X runtime envs are all recipe-only features).
