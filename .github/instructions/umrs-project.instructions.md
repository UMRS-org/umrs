---
description: UMRS project knowledge — architecture, deployment, vision models, known bugs, and debugging guides. Load when working on any UMRS file.
applyTo: "**/*.py,**/*.sh,**/*.yaml,**/*.yml,**/Dockerfile*"
---

# UMRS Project Knowledge Base

## Project Overview

UMRS (Unified Model & Resource Server) is a multi-user AI inference platform that:
- Routes embedding / LLM requests through a ZeroMQ message queue
- Runs model stacks on a Slurm-managed compute node via Mini-Rivault
- Exposes vision model inference (SAM3, HMAE) through a BentoML task-queue service
- Uses MinIO (S3-compatible) as the image/result storage layer

---

## Directory Layout

```
src/umrs/                    ← main Python package
  mini-rivault-llm/          ← LLM + vision compose stack (profiles: harness, vllm, vision-bentoml)
    serve-model.yaml         ← main compose file
    serve-model.slurm        ← actual Slurm job script (runs on compute node)
    x_nvidia-override.yaml   ← GPU passthrough for vllm + bentoml-vision
  vision_service/
    service.py               ← BentoML VisionModelService (SAM3 + HMAE tasks)
    Dockerfile               ← builds ghcr.io/riken-rccs/umrs-vision:latest
    requirements-vision.txt  ← Python deps (torch/torchvision installed separately)
  inference/                  ← Embedding & vision model implementations
    hmae/                    ← HMAE model code (hvit.py, engine.py, __init__.py)
    hmae_inference.py        ← HMAE inference helpers (load, preprocess, infer, visualise)
    sam3_segmentation.py     ← SAM3 segmentation helpers
  pipeline_sdk/              ← Pipeline SDK (workflow engine, DAG scheduler)
    base.py                  ← Abstract Pipeline class + PipelineConfig + ComputeBackend
    step.py                  ← Step abstract base, StepContext, StepResult
    workflow.py              ← WorkflowPipeline, ExecutionMode, SourceType, StepSpec
    dag.py                   ← DAGScheduler for topological step ordering
    adapter.py               ← Integration adapters (EmbeddingAdapter, etc.)
    connectors/              ← Backend connectors (UMRS, BentoML, local)
  plugins/                   ← Plugin system (sandboxed execution)
    models.py                ← PluginSpec, PluginSourceType, PluginStatus, PluginResult
    runner.py                ← PluginRunner abstract base
    slurm_runner.py          ← SlurmPluginRunner (Singularity on HPC)
    k8s_runner.py            ← K8sPluginRunner (Kubernetes Jobs)
    result_writer.py         ← PluginResultWriter
    audit.py                 ← PluginAuditLog
    step_adapter.py          ← step_to_plugin_spec() converter
    user_workflow.py         ← UserWorkflow, UserWorkflowRunner
  orchestrator/              ← Endpoint & job orchestration
    endpoint_registry.py     ← Endpoint lifecycle management
    kubernetes_backend.py    ← K8s job orchestration
    slurm_backend.py         ← Slurm job orchestration
    compose_backend.py       ← Docker compose orchestration
    hybrid/                  ← Hybrid orchestration (Slurm + K8s + Compose)
      wrappers/              ← Container runtime wrappers (subpackage)
        base.py              ← RuntimeWrapper ABC, LaunchSpec, MountSpec, ContainerEndpoint
        singularity.py       ← SingularityWrapper (HPC default)
        docker.py            ← DockerWrapper (dev/test)
        kubernetes.py        ← K8sJobWrapper, K8sBootstrapWrapper
      resolvers/             ← Model reference resolvers (subpackage)
        base.py              ← ModelRefResolver ABC, ResolvedModel, ResolutionError
        filesystem.py        ← FilesystemResolver (bind mounts)
        pvc.py               ← PVCResolver (K8s PVCs)
        huggingface.py       ← HuggingFaceResolver (download + cache)
        s3.py                ← S3Resolver (S3 download + cache)
        registry.py          ← ResolverRegistry (dispatcher)
      composer.py            ← HybridOrchestrator
      targets.py             ← ExecutionTarget implementations
  model_registry/            ← Model registry (scope, type, version, serving profiles)
    store/                   ← Store subpackage (split from monolithic store.py)
      errors.py              ← ModelNotFoundError, DuplicateModelError, etc.
      path_validator.py      ← PathValidator, DEFAULT_ALLOWED_PATHS
      acl.py                 ← AccessContext, ACLEnforcer
      crud.py                ← ModelStore (CRUD + policies)
  rag/                       ← RAG profiles, component registries, and search logic
    search.py                ← BM25, vector, hybrid search, reranking, context assembly
    models.py                ← PipelineDefinition, RAGProfile, ChunkerSpec, etc.
    registry.py              ← ChunkerRegistry, EmbedderRegistry, RerankerRegistry
  jobs/                      ← Job system (submit, cancel, retry, audit)
  modality.py                ← Modality/task resolution (text, image, volume, etc.)
src/client_umrs.py           ← CLI client (argparse, ZMQ control channel)
src/server_umrs.py           ← Main server (ZMQ listener, task dispatch)

demos/                       ← Demo scripts (15 directories: 00–14)
  lib/common.sh              ← Shared bash helpers (log, ok, warn, die, banner, preflight)
  sample_data/               ← Shared test data
  00_start_services/         ← Start/stop all services
  01_health_check/           ← Health check (ping)
  02_text_embedding/         ← Text embedding pipeline
  03_sam3_segmentation/      ← SAM3 segmentation (BentoML vision)
  04_hmae_inference/         ← HMAE inference (canonical, BentoML vision)
  05_rag_search/             ← RAG search (vector, hybrid, reranking)
  06_server_maintenance/     ← Server maintenance (recompute, update, check, reload)
  07_model_registry/         ← Model registry operations
  08_user_model/             ← User-scoped model management + custom workflows
  09_defect_analysis/        ← Custom defect analysis workflows with step scripts
  10_hmae_full_pipeline/     ← Full HMAE pipeline (workflow.json + steps/)
  11_job_submission/         ← Job submission & cancellation
  12_pipeline_sdk/           ← Pipeline SDK demo (text_analysis_pipeline.py)
  13_plugins/                ← Plugin system demo (word_count_plugin.py)
  14_opossum_analysis/       ← Opossum motion analysis (canonical)

deploy/rccs-cloud/           ← Production deployment (RCCS Cloud login node)
  demos/                     ← Deployment-level demo wrappers (00–10)
    00_health_check.sh … 10_opossum_analysis.sh
  docker-compose.yml         ← MinIO + PostgreSQL (podman, rootless)
  serve-model.slurm          ← Model serving Slurm wrapper
  ssh_tunnel.sh              ← SSH tunnel management (ports 5555, 5556, 9000, 8081)
  start.sh / stop.sh         ← Service lifecycle

### SSH Connection
- SSH alias: `rccs-cloud` → `login.cloud.r-ccs.riken.jp`
- Login node: `login1.cloud.r-ccs.riken.jp`
- User: `masaru.nagaso`
- Tunnelled ports: 5555 (ZMQ control), 5556 (ZMQ data), 9000 (MinIO/S3), 8081 (RAG service)
```

---

## Compute Node Hardware

- **CPU/GPU**: NVIDIA GB10 (Blackwell), aarch64, CUDA 13.0, Driver 580.95.05
- **CUDA capability**: 12.1 (requires cu130 wheels)
- **OS**: Ubuntu 24.04.3 LTS
- **IP**: 172.19.39.115
- **Container runtime**: rootless podman

---

## Mini-Rivault Stack

### Startup Flow
1. `sbatch` triggers `deploy/rccs-cloud/serve-model.slurm` (wrapper on login node)
2. Wrapper `export`s `RIVAULT_PROFILES` (default: `harness vllm vision-bentoml`), sets host-context overrides (`VLLM_PATH_TO_HUGGINGFACE`, `HMAE_MODEL_PATH`, etc.), reloads the BentoML vision tarball if stale
3. Wrapper calls `exec bash ./serve-model.slurm` (the real Slurm script in `src/umrs/mini-rivault-llm/`)
4. `serve-model.slurm` auto-detects GPU, picks nvidia/rocm override YAML, writes `env.{uuid}`
5. Reads `${RIVAULT_PROFILES:-"harness vllm"}` and runs `podman-compose ... --profile <p> ... up`

### Profiles
- `harness`: nginx + litellm + ll-postgres + coredns + minio + umrs-out-nginx
- `vllm`: vLLM service (LLM inference)

Note: Vision serving (BentoML) runs via the `vision-bentoml` profile in `src/umrs/mini-rivault-llm/`. The vision service source code lives in `src/umrs/vision_service/`.
- `vision-bentoml`: BentoML vision service (SAM3 + HMAE)

### Key env vars (set in wrapper slurm or env.{uuid})
| Variable | Where Set | Purpose |
|---|---|---|
| `RIVAULT_DIR` | host env / slurm | path to rivault/ on compute node |
| `VLLM_PATH_TO_HUGGINGFACE` | wrapper slurm | `${RIVAULT_DIR}/models` (NOT /models which is ai-login internal) |
| `HMAE_MODEL_PATH` | wrapper slurm | `/home/mnagaso/hmae-sample` (checkpoint dir mount source) |
| `HMAE_CONFIG_PATH` | serve-model.yaml | `/models/hmae/infer_hmae_vit_b.yaml` (inside container) |
| `HMAE_CHECKPOINT` | serve-model.yaml | `/models/hmae/latest_checkpoint.pth` (inside container) |
| `RIVAULT_PROFILES` | wrapper slurm (exported) | space-separated list of compose profiles; default `harness vllm vision-bentoml` |
| `VLLM_DOCKER_TAG` | wrapper slurm | `cu130-nightly-aarch64` (Blackwell GPU) |
| `VLLM_GPU_MEM_UTIL` | wrapper slurm | `0.10` (low — always-on LLM takes most GPU) |

### Tarball Auto-Reload
The wrapper (`deploy/rccs-cloud/serve-model.slurm`) checks if `${RIVAULT_DIR}/images/umrs-vision.tar` is newer than the cached podman image. If yes, it runs `podman rmi -f` + `podman load`. Always re-save tarball after rebuilding image:
```bash
rm -f src/umrs/mini-rivault-llm/images/umrs-vision.tar
podman save ghcr.io/riken-rccs/umrs-vision:latest -o src/umrs/mini-rivault-llm/images/umrs-vision.tar
```

### Slurm Output File Location
The wrapper has `#SBATCH --output` pointing to `${JOB_INFO_DIR}/slurm-%j.out` (resolved from the login node). To read the output:
```bash
# On the login node:
cat ~/.config/umrs/tmp/slurm-<N>.out
# Or via the compute node directly:
cat /slurm-<N>.out
```

---

## Model Type Scheme

Vision models use a double-scheme that separates launch infrastructure from wire protocol:

```
rivault://bentoml://http://localhost:{LOCAL_PORT}/vision/infer/submit
```

- `rivault://` — requires Slurm job launch + SSH tunnel (handled by `models_loading.py` → `launch_rivault_model()`)
- `bentoml://` — BentoML async task-queue API (handled inside `launch_rivault_model()`)

The adapter name is specified via the `vision_model_type` config field (e.g. `"sam3"`, `"hmae"`, or any custom adapter name), not derived from the URL path.

### Routing Flow
```
Config:  rivault://bentoml://http://localhost:{LOCAL_PORT}/vision/infer/submit
         vision_model_type: "sam3"          ← adapter registry key
         vision_model_config: { ... }       ← adapter-specific params
              │
  models_loading.py: rivault:// handler strips prefix, sets type="rivault"
              │
  model_cfg["model_type"] = "bentoml://http://localhost:{LOCAL_PORT}/vision/infer/submit"
              │
  server_umrs.py dispatches to launch_rivault_model()
              │
  rivault_launcher.py: model_type.startswith("bentoml://") → sbatch with vision-bentoml profile
              │
  SSH tunnel established, BentoML adapter created via make_bentoml_vision_compute(),
  vision_model_type + vision_model_config passed through to /infer endpoint
              │
  Vision service dispatches to ADAPTER_REGISTRY[vision_model_type].infer()
```

### Adapter Discovery
Adapters are loaded at service startup from directories listed in `VISION_ADAPTERS_DIR` (colon-separated, like `$PATH`):
- `/adapters/system` — built-in adapters (SAM3, HMAE) from `mini-rivault-vision/default_adapters/`
- `/adapters/user` — user-provided custom adapters

Each `.py` file is scanned for a `VisionModelAdapter` subclass and registered under the filename stem.

### RIVAULT_PROFILES Propagation (important)
`RIVAULT_PROFILES` contains spaces, which are mangled when passed as:
```bash
sbatch --export=ALL,RIVAULT_PROFILES="harness vllm vision-bentoml"  # ← BROKEN through podman exec quoting
```
**Working approaches**:
```bash
# 1. Set env in shell before sbatch (works with podman exec bash -c '...')
podman exec ai-login bash -c 'export RIVAULT_PROFILES="harness vllm vision-bentoml" && sbatch --export=ALL ...'

# 2. Rely on the wrapper default (recommended — already in serve-model.slurm)
# The wrapper exports: RIVAULT_PROFILES="${RIVAULT_PROFILES:-harness vllm vision-bentoml}"
podman exec ai-login sbatch /root/.config/umrs/rivault_data/serve-model.slurm
```

---

## Vision Service (BentoML)

### Endpoints
All prefixed with `/vision/` through NGINX at `http://localhost:11434/vision`

The service exposes a single generic `/infer` endpoint that dispatches to adapters based on `model_type`:

| Method | Path | Description |
|---|---|---|
| POST | `/infer/submit` | Submit vision task (model_type + inference_config) |
| GET | `/infer/status?task_id=...` | Poll task status |
| GET | `/infer/get?task_id=...` | Get task result |
| GET | `/healthz` | Health check |

The `model_type` field in the request payload selects the adapter (e.g. `"sam3"`, `"hmae"`, or a custom adapter name). The `inference_config` dict is passed through to the adapter's `infer()` method.

### BentoML Task Status Values
- `in_progress` — running
- `completed` — success (**NOT** `"success"` — this is an old BentoML v1 name)
- `failed` — errored
- `cancelled` — cancelled

### Docker Image
- Image: `ghcr.io/riken-rccs/umrs-vision:latest`
- Built from: `src/umrs/vision_service/Dockerfile`
- Base: `python:3.11-slim`
- PyTorch: `2.10.0+cu130` (from `https://download.pytorch.org/whl/cu130`)
- Size: ~5.7GB (CUDA libs included), tarball ~5.3GB
- CUDA bundled inside PyTorch wheels; host driver mounted via CDI

### GPU Passthrough
GPU override YAMLs: `src/umrs/mini-rivault-llm/x_nvidia-override.yaml` (covers both vllm and bentoml-vision profiles):
```yaml
services:
  vllm:
    devices: [...]
  bentoml-vision:
    devices: [{ driver: cdi, source: nvidia.com/gpu=all }]
    environment:
      NVIDIA_VISIBLE_DEVICES: all
      NVIDIA_DRIVER_CAPABILITIES: compute,utility
```

---

## HMAE Model

### Architecture
- **Type**: HierarchicalMaskedAutoEncoder (`hvit_b` variant)
- **Input**: 1-channel grayscale, `img_size=1024`, `patch_size=32`
- **Quadtree decomposition**: edge-guided variable-size patches (engine.py)
- **fixed_length**: 1024 patches
- **mask_ratio**: 0.5
- **BF16**: enabled on CUDA

### Inference Modes
| Mode | Description | Output |
|---|---|---|
| `reconstruct` | Full encode+decode (no masking) | `*_recon.png` |
| `mae` | Masked reconstruction (random mask) | `*_mae.png` + `mae_loss` |
| `features` | Encoder-only (CLS + patch tokens) | `*_features.pt` |

### Model Files Location
- **Checkpoint**: `/home/mnagaso/hmae-sample/latest_checkpoint.pth` (1.3GB)
- **Config**: `/home/mnagaso/hmae-sample/infer_hmae_vit_b.yaml`
- **Test images**: `/home/mnagaso/hmae-sample/spring8concrete/volume_*.png` (6 files, 8192×8192 grayscale CT)
- These are bind-mounted into container at `/models/hmae/` via `HMAE_MODEL_PATH`

### Known Gotchas
- `latest_checkpoint.pth` was originally `640` permissions — rootless podman sub-UID can't read it.
  **Fix**: `chmod 644 /home/mnagaso/hmae-sample/latest_checkpoint.pth`
- Git LFS tracked `best_model.pth` but was never committed on the current branch — no checkpoint from git
- Config's `checkpoint_path` default (`./checkpoints/latest_checkpoint.pth`) is overridden by service.py from env var `HMAE_CHECKPOINT`

---

## SAM3 Model

### Architecture
- **Type**: SAM2 (facebook/sam2-hiera-small)
- **Task**: Text-prompted zero-shot segmentation
- **Input**: Any image (RGB, any size)
- Model path: `${SAM3_MODEL_PATH}` → bound into container at `/models/sam3`

### Inference Speed (GPU)
- ~46–58s per image at 8192×8192 (was ~300s on CPU)

---

## S3/MinIO

- **Endpoint**: `http://localhost:19000` (from host), `http://minio-host:19000` (from inside containers)
- **Credentials**: `minioadmin` / `minioadmin` (default)
- **SAM3 bucket**: `vision-demo`
- **HMAE bucket**: `hmae-demo`
- Buckets auto-created by the demo scripts via `boto3`

---

## Known Bugs & Fixes

### 1. BentoML polling uses wrong status string
- Bug: Old scripts checked for `"success"` — BentoML returns `"completed"`
- Files: `demos/slurm-dev/scripts/demo_sam3_segmentation.py`, `demo_hmae_inference.py`
- Fix: Accept both `"success"` and `"completed"` in success check

### 2. HMAE task crashed silently (no error details)
- Bug: `hmae_reconstruct()` had no `try/except` — any exception returned BentoML's generic `"An unexpected error has occurred"` with no details
- Fix: Wrapped entire task body in `try/except Exception as exc: return {"error": str(exc), ...}` like `sam3_segment` does

### 3. Checkpoint permissions (640 → 644)
- Bug: `/home/mnagaso/hmae-sample/latest_checkpoint.pth` had `640` (owner-read only)
- Rootless podman containers run as a sub-UID that is not in the `mnagaso` group
- Fix: `chmod 644 /home/mnagaso/hmae-sample/latest_checkpoint.pth`

### 4. rivault.py infinite loop on job failure
- Bug: `get_slurm_node()` and `extract_node_address()` had while loops with no timeout
- If job died before first check, loops ran forever holding `_launch_lock` → deadlock
- Fix: Use `get_slurm_job_state()` + `RuntimeError` if job is gone / timeout

### 5. HuggingFace cache lookup fails with trailing newline
- Bug: `huggingface_hub.snapshot_download` reads `refs/main` with `f.read()` (no `.strip()`)
- Fix: Write ref files without trailing newlines: `printf 'hash' > refs/main`

### 6. `podman save` fails on existing tarball
- Error: `docker-archive doesn't support modifying existing images`
- Fix: Always `rm -f` the old tarball before `podman save`

### 7. `RIVAULT_PROFILES` env var mangled by sbatch `--export` through podman exec
- Passing `sbatch --export=ALL,RIVAULT_PROFILES="harness vllm vision-bentoml"` through `podman exec` quoting layers strips the value at the first space — only `"harness"` reaches the job.
- **Root fix**: The wrapper script (`serve-model.slurm`) now `export`s the default directly:
  `export RIVAULT_PROFILES="${RIVAULT_PROFILES:-harness vllm vision-bentoml}"`
- This means the default always includes `vision-bentoml` regardless of how sbatch was invoked.

### 8. Demo scripts reuse Slurm job without checking vision-bentoml profile
- Bug: If demo 01 (text embedding) started the Rivault job with only `harness vllm` profiles, demos 02 and 03 found the running job and reused it — but `rivault_bentoml-vision_1` was never started, so `/vision/healthz` returned 502 and the demo waited forever.
- Fix (demos/02_sam3_segmentation/run_demo.sh, demos/03_hmae_inference/run_demo.sh):
  Before reusing an existing job, probe `${BENTOML_URL}/healthz`:
  - Healthy → reuse as before
  - Unhealthy → `scancel` the job and resubmit with `vision-bentoml` included

### 9. bentoml:// standalone handler removed from models_loading.py
- The old standalone `bentoml://` check in `models_loading.py` was removed. Vision models must now use `rivault://bentoml://` scheme. The `rivault://` handler strips the prefix correctly and the remainder (`bentoml://http://...`) matches the existing check in `launch_rivault_model()`.

---

## Debugging Guide

### Check BentoML container logs
```bash
# Note: SSH to compute-01 from ai-login may require host key setup
# Instead, read logs directly from the host (compute node is localhost in slurm-dev)
podman logs rivault_bentoml-vision_1 2>&1 | tail -100
```

### Check BentoML image loading / RIVAULT_PROFILES in job output
```bash
# Job output on the login node:
grep 'RIVAULT_PROFILES\|rivault_\|bentoml\|vision\|mini-rivault' ~/.config/umrs/tmp/slurm-<N>.out
```

### Verify vision-bentoml profile is active
```bash
# Check job output for container names — rivault_bentoml-vision_1 should appear
grep '^rivault_\|^exit_' /slurm-<N>.out
# Quick health check
curl -sf http://localhost:11434/vision/healthz && echo OK || echo FAIL
```

### Check Slurm job status
```bash
squeue -u masaru.nagaso
```

### Get HMAE task error details
When a task fails, fetch the result immediately — with `try/except` in service.py it now returns `{"error": "...", ...}`:
```bash
curl -s "http://localhost:11434/vision/hmae_reconstruct/get?task_id=<ID>" | python3 -m json.tool
```

### BentoML health check
```bash
curl -sf "http://localhost:11434/vision/healthz"
```

### Test images on compute node
```
/home/mnagaso/hmae-sample/spring8concrete/volume_{170,180,190,200,210,220}.png
```

---

## Workflow: Rebuild & Redeploy Vision Image

```bash
# 1. Edit src/umrs/vision_service/service.py or models/

# 2. Rebuild image (only COPY layer rebuilds, rest cached ~30s)
cd /home/mnagaso/workspace/umrs
podman build -t ghcr.io/riken-rccs/umrs-vision:latest -f src/umrs/vision_service/Dockerfile .

# 3. Save tarball (auto-detected by wrapper on next job — tarball newer than cached image triggers reload)
rm -f src/umrs/mini-rivault-llm/images/umrs-vision.tar
podman save ghcr.io/riken-rccs/umrs-vision:latest -o src/umrs/mini-rivault-llm/images/umrs-vision.tar

# 4. Submit new job (wrapper defaults to vision-bentoml — no --export needed)
scancel <old_job_id>
sbatch deploy/rccs-cloud/serve-model.slurm
```

---

## Workflow: Run Demos

All demos (except 08, 12, 13) require SSH tunnels + activated venv +
`unset UMRS_SERVER_HOST`. All scripts source `demos/lib/common.sh`
which provides `$CLIENT`, `preflight()`, `wait_tasks()`, helpers.

### Run a single demo
```bash
source deploy/rccs-cloud/.venv/bin/activate
unset UMRS_SERVER_HOST
cd demos/01_health_check && bash run_demo.sh
```

### Demo inventory

| # | Name | Server? | Bucket alias | Notes |
|---|------|---------|--------------|-------|
| 01 | Health Check | yes | — | ping + list_buckets |
| 02 | Text Embedding | yes | user1-textfiles | upload text → embed → export CSV |
| 03 | SAM3 Segmentation | yes | user1-sam3 | upload images → BentoML → download results |
| 04 | HMAE Inference | yes | user1-hmae | upload images → HMAE → download recon |
| 05 | RAG Search | yes | — | needs port 8081 (RAG service) |
| 06 | Server Maintenance | yes | — | list/check/update/recompute/status |
| 07 | Model Registry | yes | — | model list/register/upload/download |
| 08 | User Model | no | — | pytest + inline smoke test (offline) |
| 09 | Defect Analysis | yes | user1-sam3 | SAM3 segmentation reused for defects |
| 10 | HMAE Full Pipeline | yes | user1-hmae | HMAE reconstruction pipeline |
| 11 | Job Submission | yes | — | job submit/cancel with --keep-alive |
| 12 | Pipeline SDK | no | — | local TextAnalysisPipeline |
| 13 | Plugins | no | — | local plugin execution demo |
| 14 | Opossum Analysis | yes* | user1-textfiles | video motion analysis (*local fallback) |

### Demo 03 — SAM3 Segmentation
```bash
cd demos/03_sam3_segmentation
bash run_demo.sh
```

### Demo 04 — HMAE Inference
```bash
cd demos/04_hmae_inference
bash run_demo.sh
```

### Demo 12 — Pipeline SDK (local, no server required)
```bash
cd demos/12_pipeline_sdk
bash run_demo.sh
```

### Demo 13 — Plugin System (local, no server required)
```bash
cd demos/13_plugins
bash run_demo.sh
```

---

## Client CLI (src/client_umrs.py)

### Command Reference
| Subcommand | Category | Purpose |
|---|---|---|
| `ping` | Basic | Check server readiness |
| `upload_file <path> <bucket_alias>` | Data | Upload file to S3 bucket |
| `upload_dir <path> <bucket_alias>` | Data | Upload directory |
| `download_results <alias> <path>` | Data | Download computed results |
| `list_buckets` | Info | List available bucket aliases |
| `status [category]` | Admin | Server summary (embeddings, models, jobs, workflows) |
| `recompute` | Maintenance | Recompute all embeddings |
| `update` | Maintenance | Update embeddings for missing/mismatched entries |
| `check` | Maintenance | Check S3/DB consistency |
| `reload` | Admin | Reload server config (with optional user config upload) |
| `export_embeddings <alias> <path>` | Data | Export embeddings to CSV/JSONL |
| `model register` | Model | Register local model |
| `model list` | Model | List registered models |
| `model download` | Model | Download from HuggingFace (supports `--server-side`) |
| `model upload` | Model | Copy local dir to staging + register |
| `job submit <type>` | Job | Submit job (endpoint_provision, index_build, cleanup) |
| `job cancel <id>` | Job | Cancel running job |
| `upload-workflow <path>` | Workflow | Upload workflow.json + step scripts as zip |
| `workflow list` | Workflow | List registered workflows |

### Global Flags
- `-c <conf>` — Config file path
- `-u <username>` — Override username
- `--output json|text` — Output format
- `-k key:value` — Metadata (repeatable)

---

## Pipeline SDK (src/umrs/pipeline_sdk/)

### Lifecycle

All UMRS pipelines follow a standard 6-phase lifecycle:

```
configure  →  ingest  →  compute  →  collect  →  verify  →  present
```

Users subclass `Pipeline` from `umrs.pipeline_sdk.base` and implement hooks.

### Compute Backends
| Backend | Description |
|---|---|
| `LOCAL` | Direct computation in process |
| `UMRS_EMBEDDING` | Upload data → UMRS server computes embeddings → pgvector |
| `BENTOML_TASK` | Upload data → submit BentoML task → poll → retrieve |

### Workflow Pipelines (declarative)
`WorkflowPipeline` reads a `workflow.json` file and executes it as a Pipeline:
```python
from umrs.pipeline_sdk.workflow import WorkflowPipeline
pipeline = WorkflowPipeline("/path/to/workflow.json")
results = pipeline.run()
```

### Execution Modes
| Mode | Description |
|---|---|
| `IN_PROCESS` | Direct call in server process (dev/default) |
| `SUBPROCESS` | Isolated subprocess per step |
| `SLURM` | One Slurm job per step (HPC GPU) |
| `K8S` | One Kubernetes Job per step (cloud) |

### Source Types
| Type | Description |
|---|---|
| `LOCAL_PATH` | Local filesystem path |
| `S3` | S3 object |
| `GIT` | Git repository at specific commit |
| `UPLOADED` | Server-side directory from client upload |

### DAG Scheduling
Steps declare dependencies via `depends_on` in `workflow.json`. The `DAGScheduler` resolves topological order and handles parallel independent steps.

---

## Plugin System (src/umrs/plugins/)

Plugins enable sandboxed execution of user-provided code on HPC or cloud backends.

### PluginSpec
Each plugin is defined by a `PluginSpec` with:
- **Source**: Git repo (with commit hash), S3 wheel, or local path
- **Entrypoint**: `module:function` format (e.g., `my_plugin.main:run`)
- **Resources**: CPU, memory, GPU count limits
- **Timeout**: Execution timeout in seconds
- **Requirements**: Additional pip dependencies

### Plugin Runners
| Runner | Backend | Sandboxing |
|---|---|---|
| `SlurmPluginRunner` | Slurm HPC | Singularity container |
| `K8sPluginRunner` | Kubernetes | Pod isolation |
| `PluginRunner` | Local | Process isolation (dev) |

### Audit & Results
- `PluginAuditLog` records all plugin executions
- `PluginResultWriter` handles output persistence
- `UserWorkflow` composes multiple plugins into a DAG

---

## Experiences & Modalities

"Experiences" are user-level project scopes. Each experience has:
- An `xp_name` (e.g. "HMAE Inference", "Text Embedding")
- A SQL database (e.g. `hmae_inference_usr1_hmae`)
- One or more **modalities**, each with:
  - S3 buckets: `<user>.<experience>.<alias>` (input) + optional `-output` (results)
  - A database table for embeddings/metadata
  - Model binding (which model processes this modality)

### Modality/Task Resolution (src/umrs/modality.py)
Pure data types: `text`, `text_file`, `image`, `volume`, `movie`, `volume_movie`
Tasks: `embedding`, `segmentation`, `reconstruction`, `mae_reconstruction`, `feature_extraction`, `detection`, `classification`

Legacy compound names map to `(modality, task)` pairs (e.g., `"segmentation"` → `("image", "segmentation")`).

### Example User Config
```json
{
  "username": "user1",
  "experiences": [
    {
      "xp_name": "HMAE Inference",
      "sql_username": "usr1_hmae",
      "modalities": [
        {
          "modality": "volume",
          "task": "mae_reconstruction",
          "s3": { "bucket_alias": "user1-hmae", ... },
          "model": { "model_name": "hmae-pretrained", ... }
        }
      ]
    }
  ]
}
```

---

## RCCS-Cloud Deployment (deploy/rccs-cloud/)

### Architecture
```
Local machine        rccs-cloud (login node)      ng-dgx-s (compute)
umrs-client      ← SSH tunnel →  umrs-server      ← Slurm →  mini-rivault
                                 minio (podman)              vLLM / BentoML
                                 postgres (podman)
                                 /lvs0/rccs-dmp/ (Lustre shared)
```

Key properties:
- Native `umrs-server` process (not containerized) → can invoke Slurm CLI + SSH
- Podman containers for MinIO and PostgreSQL (rootless, Lustre xattr workarounds)
- SSH tunnels for compute node access (ZMQ 5555/5556, MinIO 9000, RAG 8081)
- Model staging root: `~/.config/umrs/models/`

### Performance (GPU)
- SAM3: ~46–58s per 8192×8192 image
- HMAE reconstruct: ~16s per 8192×8192 image (first call loads model ~10s extra)
