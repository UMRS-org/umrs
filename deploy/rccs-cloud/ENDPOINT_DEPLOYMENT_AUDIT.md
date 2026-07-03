# Endpoint Deployment Audit — RCCS Cloud Multi-Node

**Date:** 2026-07-03
**Scope:** Full audit of endpoint provisioning → Slurm → Podman → Ray → vLLM lifecycle
**Target:** Production-quality multi-node LLM endpoint deployment on RCCS Cloud

---

## 1. Executive Summary

The endpoint deployment pipeline has **three architectural layers** that are
partially disconnected:

1. **DeploymentSpec** (Python) — well-designed immutable spec, but
   `EndpointProvisionBackend` doesn't fully forward multi-node parameters.
2. **serve-model.slurm** (Bash) — has accumulated ~15 ad-hoc patches for
   CDI, ENTRYPOINT, Ray install, device mapping, and shell quoting.
   These patches are **only on the remote** — not in the repo.
3. **serve-model.yaml** (Compose) — vLLM command uses shell variable
   expansion that breaks when variables are empty.

**Root causes of multi-node failures (in order of discovery):**

| # | Issue | Root Cause | Status |
|---|-------|-----------|--------|
| 1 | `node_count` not in sbatch | `EndpointProvisionBackend` omits `node_count` from `submit_kwargs` | Patched remote |
| 2 | `VLLM_HOST_MODEL_PATH` empty in Ray head | Variable only in env file heredoc, not a shell var | Patched remote |
| 3 | CDI `nvidia.com/gpu=all` unresolvable | Podman 4.9.3 cannot resolve CDI via `podman run` | Patched remote |
| 4 | vLLM ENTRYPOINT intercepts Ray commands | Image has `ENTRYPOINT ["vllm","serve"]` | Patched remote |
| 5 | Ray not in vLLM Docker image | `docker.io/vllm/vllm-openai` has no `ray` package | Patched remote |
| 6 | Shell var expansion `$1`/`$2`/`$3` | Positional params don't expand through srun/ssh | Patched remote |
| 7 | GPU count check takes ~30s/iter | `pip install ray` runs in a fresh container each poll | Patched remote |
| 8 | `_resolve_model_dir` function corrupted | Multiple sed operations on remote | Patched remote |

**Current state:** Ray head + worker start successfully on 2 nodes.
The compose stack (podman-compose) then starts vLLM with
`--distributed-executor-backend ray`.

---

## 2. Endpoint Registration Audit

### 2.1 DeploymentSpec (Good)

`DeploymentSpec` is a frozen dataclass that captures all deployment parameters:
- Identity (alias, HF ID, model family, served names)
- Topology (TP, PP, EP, GPUs/node, node count)
- Runtime (tool/reasoning parsers, KV cache dtype, eager mode)
- Capacity (max model len, max num seqs, GPU mem util)
- Infrastructure (partition, Slurm account, container image, time limit)
- Health (readiness probe, model, path)

**Verdict:** ✅ Well-structured. `generate_sbatch_env()` produces explicit env vars.

### 2.2 EndpointProvisionBackend (Needs Fix)

**Problem:** `submit_job()` doesn't forward `node_count`, `partition_override`,
or `nodelist_hint` from `job.spec` to `self.orchestrator.submit_job()`.

The local `adapters.py` was patched on the remote but **not committed to the repo**.
This means any fresh checkout will fail multi-node deployment.

**Fix:** Add forwarding of `node_count`, `partition_override`, `nodelist_hint`
in `EndpointProvisionBackend.submit_job()`.

### 2.3 Model Metadata (Adequate)

Model registry entries provide:
- `runtime_config` (model family, parsers, quantization)
- `serving_profile` (TP, PP, GPU count, node count, max model len)
- `storage_ref` (filesystem path to model weights)

**Verdict:** ✅ Sufficient for PP, TP, Ray. No manual edits needed.

---

## 3. Endpoint Provisioning Audit

### 3.1 Current Flow

```
API request (model, gpu_count, node_count, partition, ...)
    ↓
EndpointProvisionBackend.submit_job()
    ↓
build_deployment_profile(roster_defaults, spec_overrides)
    ↓
DeploymentSpec.from_registry(entry, profile, site_config)
    ↓
deployment_spec.generate_sbatch_env()
    ↓
SlurmPodmanBackend.submit_job() → sbatch
```

### 3.2 Issues

1. **`node_count` lost in `EndpointProvisionBackend.submit_job()`**
   - `submit_kwargs` only includes `model_path`, `env`, `gpu_count`, `requesting_user`
   - Missing: `node_count`, `partition_override`, `nodelist_hint`
   - **Fix:** Add these to `submit_kwargs` from `job.spec`

2. **`build_deployment_profile` merges `spec_overrides` into `profile`**
   - But `from_registry` reads `profile.get("gpu_count")` and `profile.get("node_count")`
   - This works but is fragile — the profile dict conflates roster defaults
     with API overrides.

3. **`_apply_advanced_layout_kwargs` is a no-op for `node_count`**
   - The method exists but `EndpointProvisionBackend.submit_job()` doesn't call it
   - **Fix:** Either call it or inline the forwarding

---

## 4. Slurm Integration Audit

### 4.1 sbatch Generation (Good)

`SlurmPodmanBackend.submit_job()` builds sbatch args:
- `--partition` from `partition_override` or auto-select
- `--nodelist` from `nodelist_hint`
- `--nodes` from `node_count` (if > 1)
- `--gres=gpu:N` (unless `site.scheduler.gpus == 0`)
- `--time`, `--cpus-per-task`, `--mem`, `--account`, `--qos`

**Verdict:** ✅ Correct. Handles whole-node allocation (gpus=0) and multi-node.

### 4.2 Node Allocation & Discovery

- Slurm allocates `--nodes=N` across the partition
- `scontrol show hostnames "$SLURM_JOB_NODELIST"` discovers allocated nodes
- Head node = first node; workers = remaining nodes

**Verdict:** ✅ Works correctly.

### 4.3 Environment Propagation

- `serve-model.slurm` receives env vars via `--export` (sbatch)
- Inner `serve-model.slurm` writes `env.${uuid}` with all VLLM_* vars
- `podman-compose` reads `env.${uuid}` via `--env-file`

**Verdict:** ✅ Works, but fragile — shell variable expansion in YAML commands.

---

## 5. Multi-node Orchestration Audit

### 5.1 Current Flow (serve-model.slurm)

```
SLURM allocates 2+ nodes
    ↓
serve-model.slurm detects SLURM_NNODES > 1
    ↓
Ray head started via podman run --entrypoint /bin/sh
    ↓
Ray workers started via srun + podman run
    ↓
GPU count check (python3 ray.init from host)
    ↓
VLLM_DISTRIBUTED_EXECUTOR_BACKEND=ray + RAY_ADDRESS exported
    ↓
podman-compose starts vLLM with --distributed-executor-backend ray
```

### 5.2 Issues

1. **Ray install on every container start** — `pip install -q ray` runs
   in the Ray head and each worker container. This adds ~10-30s per container
   and requires network access.

   **Fix:** Pre-install Ray in a custom vLLM image, or use `pip install`
   only once and cache the wheel.

2. **No deterministic Ray head readiness** — The script uses a TCP port
   check (`echo > /dev/tcp/HEAD_IP/PORT`), which is fast but doesn't
   verify Ray GCS is actually ready.

   **Fix:** Use `python3 -c "import ray; ray.init(address='...')"` from
   the host (Ray 2.55.1 is installed on compute nodes).

3. **No Ray worker failure handling** — If a worker fails to join,
   the script waits 60 iterations × 5s = 5 min then proceeds anyway.

   **Fix:** Check `ray status` output and fail fast if workers don't join.

4. **No Ray cleanup on exit** — The cleanup trap kills Ray head PID and
   worker PIDs, but doesn't run `ray stop` on each node.

   **Fix:** Add `ray stop` to the cleanup trap.

---

## 6. Ray Integration Audit

### 6.1 Ray Head Startup

```bash
podman run --rm --network host --entrypoint /bin/sh \
    --device /dev/nvidia0 --device /dev/nvidiactl --device /dev/nvidia-uvm --device /dev/nvidia-uvm-tools \
    -v /usr/lib/aarch64-linux-gnu:/usr/lib/aarch64-linux-gnu:ro \
    -v "${_vllm_host_model_path}:/models:ro" \
    -e RAY_HEAD_IP="${HEAD_IP}" -e RAY_PORT="${RAY_PORT}" -e VLLM_TP="${VLLM_TP}" \
    "${VLLM_IMAGE}" \
    -c 'pip install -q ray 2>/dev/null; exec python3 -m ray.scripts.scripts start --head --node-ip-address="$RAY_HEAD_IP" --port="$RAY_PORT" --num-gpus="$VLLM_TP" --include-dashboard=false --block'
```

**Issues:**
- `pip install -q ray` runs every time (~10-30s)
- `--device /dev/nvidia0` only maps 1 GPU; should map all GPUs
- No `--shm-size` for Ray object store (warning: using /tmp/ray)
- No Ray temp dir specified

### 6.2 Ray Worker Startup

```bash
srun --nodes=1 --ntasks=1 --nodelist="${wnode}" \
    podman run --rm --network host --entrypoint /bin/sh \
        --device /dev/nvidia0 --device /dev/nvidiactl --device /dev/nvidia-uvm --device /dev/nvidia-uvm-tools \
        -v /usr/lib/aarch64-linux-gnu:/usr/lib/aarch64-linux-gnu:ro \
        -v "${_vllm_host_model_path}:/models:ro" \
        -e RAY_ADDRESS="${RAY_ADDRESS}" -e VLLM_TP="${VLLM_TP}" \
        "${VLLM_IMAGE}" \
        -c 'pip install -q ray 2>/dev/null; exec python3 -m ray.scripts.scripts start --address="$RAY_ADDRESS" --num-gpus="$VLLM_TP" --block'
```

**Issues:**
- Same `pip install` overhead
- Same single-GPU device mapping
- No `--shm-size` for Ray object store

### 6.3 GPU Count Check

```bash
_have_gpus="$(python3 -c "import ray; ray.init(address='${RAY_ADDRESS}'); print(int(ray.cluster_resources().get('GPU',0)))" 2>/dev/null | tail -1)"
```

**Issues:**
- Uses host `python3` which has Ray 2.55.1 installed — ✅ Fast
- But if Ray isn't installed on the host, this fails silently
- No timeout on the `ray.init()` call

---

## 7. Podman Runtime Audit

### 7.1 GPU Exposure

**Current:** `--device /dev/nvidia0 --device /dev/nvidiactl --device /dev/nvidia-uvm --device /dev/nvidia-uvm-tools`

**Issues:**
- Only maps GPU 0, not all GPUs
- `nvidia-caps` directories not mapped
- No `NVIDIA_VISIBLE_DEVICES` env var set

**Fix:** Use `--device nvidia.com/gpu=all` when CDI is available,
fall back to mapping all `/dev/nvidia*` devices.

### 7.2 Container Networking

**Current:** `--network host` for Ray head/worker

**Verdict:** ✅ Correct for multi-node Ray (needs host network for GCS).

### 7.3 Volume Mounts

**Current:**
- `-v /usr/lib/aarch64-linux-gnu:/usr/lib/aarch64-linux-gnu:ro` (NVIDIA libs)
- `-v ${_vllm_host_model_path}:/models:ro` (model weights)

**Issues:**
- Hardcoded `/usr/lib/aarch64-linux-gnu` — not portable across architectures
- No `libnvidia-ml.so` explicitly mounted (it's in the aarch64 dir)

### 7.4 ENTRYPOINT

**Current:** `--entrypoint /bin/sh` to override vLLM's `ENTRYPOINT ["vllm","serve"]`

**Verdict:** ✅ Correct. Without this, `podman run ... /bin/sh -c "..."` becomes
`vllm serve /bin/sh -c "..."`.

---

## 8. vLLM Startup Audit

### 8.1 serve-model.yaml vLLM Command

```yaml
command:
  - "-c"
  - 'exec vllm serve "$$VLLM_MODEL_REF" $$VLLM_EXTRA_ARGS --served-model-name "$$VLLM_SERVED_MODEL_NAME" --api-key "$$VLLM_API_KEY" -tp "$$VLLM_TP" -pp "$$VLLM_PP" $${VLLM_DISTRIBUTED_EXECUTOR_BACKEND:+--distributed-executor-backend $$VLLM_DISTRIBUTED_EXECUTOR_BACKEND} --max-model-len "$${VLLM_MAX_MODEL_LEN:-4096}" --gpu-memory-utilization "$${VLLM_GPU_MEM_UTIL:-0.95}" --enforce-eager --enable-auto-tool-choice --tool-call-parser "$${VLLM_TOOL_CALL_PARSER:-hermes}"'
```

**Issues:**
- Uses `$$` (dollar-dollar) for compose variable expansion — fragile
- `$${VLLM_DISTRIBUTED_EXECUTOR_BACKEND:+...}` only adds the flag when
  the variable is set and non-empty — correct but hard to read
- `--enforce-eager` is hardcoded — should be configurable
- `--enable-auto-tool-choice` is hardcoded — should be configurable
- `--tool-call-parser` defaults to "hermes" — should come from DeploymentSpec

### 8.2 Environment Generation

The `serve-model.slurm` generates `env.${uuid}` with:
- `VLLM_MODEL`, `VLLM_MODEL_REF`, `VLLM_SERVED_MODEL_NAME`
- `VLLM_TP`, `VLLM_PP`, `SLURM_NODES`
- `VLLM_DOCKER_TAG`, `VLLM_GPU_MEM_UTIL`
- `VLLM_TOOL_CALL_PARSER`, `VLLM_REASONING_PARSER`, `VLLM_TRUST_REMOTE_CODE`
- `VLLM_DISTRIBUTED_EXECUTOR_BACKEND` (set to "ray" for multi-node)
- `RAY_ADDRESS` (set to head_ip:6379 for multi-node)

**Verdict:** ✅ Works for the tested models. But the env var names are
inconsistent (some use `VLLM_` prefix, some don't).

---

## 9. Model Path Handling Audit

### 9.1 Current Path Resolution

`serve-model.slurm` resolves model paths in `_vllm_host_model_path`:
1. `VLLM_HOST_MODEL_PATH` env var (if set)
2. `VLLM_MODEL_PATH` env var (if set)
3. HF cache layout: `${VLLM_PATH_TO_HUGGINGFACE}/hub/models--${VLLM_MODEL}/snapshots`

Then resolves through HF cache snapshots via `_resolve_model_dir()`.

### 9.2 Issues

1. **No GGUF-specific path handling** — GGUF models need the directory
   containing `model.gguf` + `config.json` + tokenizer files.
   The current code doesn't handle this case.

2. **No unified path resolution** — The same path resolution logic is
   duplicated between `serve-model.slurm` and `DeploymentSpec`.

3. **Container path inconsistency** — The model is mounted at `/models`
   in Ray head/worker, but at `${VLLM_CONTAINER_MODEL_PATH}` in the
   compose stack. These should be consistent.

---

## 10. Configuration Architecture Audit

### 10.1 Current State

Configuration is scattered across:
- `site.yaml` (site config)
- `pool-rccs-glm5.yaml` (pool roster)
- `serve-model.slurm` (shell variables)
- `serve-model.yaml` (compose env vars)
- `DeploymentSpec` (Python dataclass)
- `EndpointProvisionBackend` (submit_kwargs)

### 10.2 Issues

1. **No single source of truth** — The same parameter (e.g., `VLLM_TP`)
   may be set in 3-4 places with different precedence rules.

2. **No validation across layers** — `DeploymentSpec.validate()` checks
   topology, but `serve-model.slurm` doesn't validate that `VLLM_TP`
   matches `gpus_per_node`.

3. **No documentation of env var contract** — The env vars that
   `serve-model.slurm` expects are not documented anywhere.

---

## 11. Architectural Weaknesses

### 11.1 Duplicated Shell Scripts

`serve-model.slurm` and `serve-model.yaml` are duplicated across:
- `umrs/src/umrs/mini-rivault-llm/` (inner)
- `umrs/deploy/rccs-cloud/` (outer wrapper)

The outer wrapper delegates to the inner script, but both contain
GPU override generation, env file creation, and compose invocation.

### 11.2 No Ray Integration in Compose

Ray head/worker startup is in `serve-model.slurm` (bash), not in
`serve-model.yaml` (compose). This means:
- Ray containers are not managed by podman-compose
- Ray cleanup is manual (kill PIDs)
- Ray health is not checked by compose healthchecks

### 11.3 No Model-Specific Configuration

The pool roster (`pool-rccs-glm5.yaml`) provides per-model env vars,
but these are manually maintained. There's no automatic derivation of:
- TP from model size vs GPU memory
- PP from model size vs aggregate GPU memory
- Tool/reasoning parser from model family

---

## 12. Recommended Refactoring

### 12.1 Unify Endpoint Provisioning

**Goal:** One clean path from API request → sbatch.

```
API request
    ↓
EndpointProvisionBackend.submit_job()
    ↓
DeploymentSpec.from_registry() → validate() → generate_sbatch_env()
    ↓
SlurmPodmanBackend.submit_job() → sbatch with --export env
    ↓
serve-model.slurm reads env vars → starts compose stack
```

**Fix:** Add `node_count`, `partition_override`, `nodelist_hint` forwarding
in `EndpointProvisionBackend.submit_job()`.

### 12.2 Clean Up serve-model.slurm

**Goal:** Remove ad-hoc patches, use structured configuration.

1. Replace hardcoded device paths with a GPU discovery function
2. Replace `pip install ray` with pre-installed Ray or a custom image
3. Use `ray status` for deterministic readiness check
4. Add `ray stop` to cleanup trap
5. Document all env vars the script expects

### 12.3 Clean Up serve-model.yaml

**Goal:** Remove hardcoded vLLM flags, use env var expansion.

1. Remove hardcoded `--enforce-eager`, `--enable-auto-tool-choice`
2. Make `--tool-call-parser` conditional on env var
3. Add `--distributed-executor-backend` from env var
4. Add `--ray-address` from env var

### 12.4 Add Multi-Node Support to Compose

**Goal:** Let podman-compose manage Ray head/worker.

1. Add `ray-head` service to `serve-model.yaml`
2. Add `ray-worker` service (launched via srun)
3. Use compose `depends_on` for vLLM → Ray head
4. Use compose healthcheck for Ray head readiness

### 12.5 Unify Model Path Handling

**Goal:** One path resolution function, used everywhere.

1. Add `resolve_model_path()` to `DeploymentSpec`
2. Generate `VLLM_HOST_MODEL_PATH` and `VLLM_CONTAINER_MODEL_PATH` from spec
3. Remove path resolution from `serve-model.slurm`

### 12.6 Document Env Var Contract

**Goal:** Document all env vars that `serve-model.slurm` expects.

Create `docs/deployment/env-var-contract.md` with:
- Variable name
- Description
- Required/optional
- Default value
- Set by (DeploymentSpec, site config, etc.)

---

## 13. RCCS Cloud Specifics

### 13.1 Node Topology

- `ng-dgx-m0` → nodes 08-09 (GB10, 128 GB GPU)
- `ng-dgx-m1` → nodes 10-11
- `ng-dgx-m2` → nodes 12-13
- `ng-dgx-m3` → nodes 14-15

### 13.2 Slurm Constraints

- `ng-dgx-m*` partitions reject `--gres`/`--gpus` (whole-node allocation)
- `ng-dgx-m*` partitions reject explicit `--mem` requests
- `--time` is accepted (default 4h)

### 13.3 Podman Constraints

- Podman 4.9.3 on compute nodes
- CDI spec at `/etc/cdi/nvidia.yaml` but `podman run --device nvidia.com/gpu=all` FAILS
- Must use direct `/dev/nvidia0` device mapping
- Must mount `/usr/lib/aarch64-linux-gnu` for NVIDIA libraries

### 13.4 Ray Constraints

- Ray 2.55.1 is installed on compute nodes (host Python)
- vLLM Docker image does NOT include Ray
- Must `pip install ray` in container or use custom image

### 13.5 Known Limitations

1. **Single GPU per node** — Each `ng-dgx-m*` node has 1 GPU. Multi-GPU TP
   requires multi-node Ray.
2. **No CDI in podman run** — Must use direct device mapping.
3. **No Ray in vLLM image** — Must install at runtime or build custom image.
4. **No `--shm-size` in Ray containers** — Ray falls back to /tmp/ray.

---

## 14. Verification Plan

For each target model:

| Model | Nodes | TP | PP | Expected | Status |
|-------|-------|----|----|----------|--------|
| GLM-5.2 IQ2 GGUF | 2 | 1 | 2 | 223 GB across 2×128 GB | Ray starts, vLLM pending |
| Ornith-1.0-397B W4A16 | 2 | 1 | 2 | 397B across 2 nodes | Not tested |
| DiffusionGemma-26B A4B | 1 | 1 | 1 | 25B on 1 node | Not tested |
| Agents-A1 | 1 | 1 | 1 | 35B on 1 node | Not tested |
| Qwen3.6-27B-Base | 1 | 1 | 1 | 27B on 1 node | Not tested |

### Verification Steps

1. Submit endpoint via `umrs-client endpoint up`
2. Check `squeue` for Slurm allocation
3. Check `serve-model.slurm` logs for Ray head/worker startup
4. Check `podman-compose` logs for vLLM startup
5. Curl the LiteLLM port with the master key
6. Verify model is served via `/v1/models`
7. Send a chat completion request
8. Check GPU memory usage via `nvidia-smi`

---

## 15. Implementation Phases

### Phase 1 — Stabilization (Immediate)

**Goal:** Make the current pipeline reliable. No refactoring.

1. **Fix `EndpointProvisionBackend.submit_job()`** — forward `node_count`, `partition_override`, `nodelist_hint`
2. **Fix outer `serve-model.slurm` GPU override** — change CDI `nvidia.com/gpu=all` to direct `/dev/nvidia*` device mapping
3. **Fix inner `serve-model.slurm`** — `VLLM_HOST_MODEL_PATH` as shell var, `--entrypoint /bin/sh` for Ray, `pip install ray`, env var expansion
4. **Fix `x_nvidia-override.yaml`** — direct `/dev/nvidia*` instead of CDI
5. **Verify GLM-5.2 IQ2 multi-node** — submit, Ray cluster, vLLM, inference
6. **Verify single-node models** — DiffusionGemma, Agents-A1, Qwen3.6

### Phase 2 — Cleanup (After Phase 1 Stable)

**Goal:** Simplify. Remove duplicated logic.

1. **Remove duplicated shell** — consolidate outer + inner `serve-model.slurm`
2. **Remove duplicated env generation** — `DeploymentSpec.generate_sbatch_env()` is the single source
3. **Unify model path handling** — one `resolve_model_path()` in `DeploymentSpec`
4. **Clean `DeploymentSpec`** — remove fields that belong in `DeploymentPlanner`

### Phase 3 — Architecture (After Phase 2 Clean)

**Goal:** Introduce `DeploymentPlanner` and deployment state machine.

1. **`DeploymentPlanner`** — decides TP, PP, node count, distributed backend, image based on model + quantization + GPU memory
   ```
   Model Registry
       ↓
   DeploymentPlanner (TP/PP/nodes/backend/image)
       ↓
   DeploymentSpec (immutable plan)
       ↓
   everything else
   ```
2. **Deployment state machine** — explicit states with logs, timestamps, restartability, cleanup
   ```
   ALLOCATING → ALLOCATED → RAY_STARTING → RAY_READY
       → VLLM_STARTING → VLLM_READY → REGISTERING → READY
       ↘ FAILED (from any state)
   ```
3. **Move deployment decisions into Python** — Bash becomes declarative
   ```
   Python: DeploymentSpec → JSON config
   Bash: read JSON, execute
   ```

### Phase 4 — Future (Optional)

**Goal:** Extensibility without redesign.

1. Compose-managed Ray (optional — Slurm is the primary orchestrator)
2. DeepEP support
3. TensorRT-LLM support
4. SGLang support
5. LMDeploy support
6. Kubernetes backend

---

## 16. Layer Responsibility Boundaries

Each layer owns one responsibility. Avoid letting multiple layers solve the same problem.

```
UMRS
  ↓  deployment metadata, topology, environment generation
DeploymentSpec
  ↓  immutable deployment plan (TP, PP, nodes, env vars)
Slurm
  ↓  resource allocation, node discovery
serve-model.slurm
  ↓  orchestration (Ray bootstrap, compose invocation)
Ray
  ↓  distributed execution
vLLM
  ↓  inference
```

**Key principle:** Deployment decisions (TP, PP, node count) belong in `DeploymentPlanner` → `DeploymentSpec`, NOT in shell scripts or pool rosters.

---

## 17. Summary

**Phase 1 (Stabilize):**
- Upstream all remote fixes
- Fix CDI → direct device mapping
- Fix Ray startup (entrypoint, pip install, env vars)
- Verify GLM IQ2 + single-node models

**Phase 2 (Cleanup):**
- Consolidate duplicated shell
- Unify env generation via DeploymentSpec
- Single model path resolution

**Phase 3 (Architecture):**
- DeploymentPlanner for TP/PP/nodes/backend
- Deployment state machine
- Move logic from Bash to Python

**Phase 4 (Future):**
- Compose-managed Ray (optional)
- DeepEP, TensorRT-LLM, SGLang, LMDeploy
- Kubernetes backend
