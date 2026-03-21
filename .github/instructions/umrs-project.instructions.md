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
  rivault/                   ← Rivault stack (Docker compose + Slurm wrappers)
    serve-model.yaml         ← main compose file (profiles: harness, vllm, vision-bentoml)
    serve-model.slurm        ← actual Slurm job script (runs on compute node)
    x_nvidia-override.yaml   ← GPU passthrough for vllm + bentoml-vision
  vision_service/
    service.py               ← BentoML VisionModelService (SAM3 + HMAE tasks)
    Dockerfile               ← builds ghcr.io/riken-rccs/umrs-vision:latest
    requirements-vision.txt  ← Python deps (torch/torchvision installed separately)
  models/
    hmae/                    ← HMAE model code (hvit.py, engine.py, __init__.py)
    hmae_inference.py        ← HMAE inference helpers (load, preprocess, infer, visualise)
    sam3_segmentation.py     ← SAM3 segmentation helpers
demos/
  slurm-dev/                 ← the real running demo environment
    docker-compose.yaml      ← ai-login + umrs-client + minio containers
    umrs-config/rivault_data/
      serve-model.slurm      ← WRAPPER that sbatch runs; sets host-side env overrides
    scripts/
      demo_sam3_segmentation.py  ← SAM3 BentoML client
      demo_hmae_inference.py     ← HMAE BentoML client (3 modes)
  02_sam3_segmentation/
    run_demo.sh              ← end-to-end SAM3 demo (single image, downloads results)
    vision_models.json       ← SAM3 model config (rivault://bentoml:// scheme)
  03_hmae_inference/
    run_demo.sh              ← end-to-end HMAE demo (single image, downloads results)
    vision_models.json       ← HMAE model config (rivault://bentoml:// scheme)
    test_data/               ← test images (symlinks OK)
    results/                 ← downloaded outputs land here
  04_rag_search/
    run_demo.sh              ← RAG search demo (vector, hybrid, reranking, Q&A)
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
1. `sbatch` triggers `demos/slurm-dev/umrs-config/rivault_data/serve-model.slurm` (wrapper inside ai-login)
2. Wrapper `export`s `RIVAULT_PROFILES` (default: `harness vllm vision-bentoml`), sets host-context overrides (`VLLM_PATH_TO_HUGGINGFACE`, `HMAE_MODEL_PATH`, etc.), reloads the BentoML vision tarball if stale
3. Wrapper calls `exec bash ./serve-model.slurm` (the real Slurm script in `src/umrs/rivault/`)
4. `serve-model.slurm` auto-detects GPU, picks nvidia/rocm override YAML, writes `env.{uuid}`
5. Reads `${RIVAULT_PROFILES:-"harness vllm"}` and runs `podman-compose ... --profile <p> ... up`

### Profiles
- `harness`: nginx + litellm + ll-postgres + coredns + minio + umrs-out-nginx
- `vllm`: vLLM service (LLM inference)
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
The wrapper (`demos/slurm-dev/umrs-config/rivault_data/serve-model.slurm`) checks if `${RIVAULT_DIR}/images/umrs-vision.tar` is newer than the cached podman image. If yes, it runs `podman rmi -f` + `podman load`. Always re-save tarball after rebuilding image:
```bash
rm -f src/umrs/rivault/images/umrs-vision.tar
podman save ghcr.io/riken-rccs/umrs-vision:latest -o src/umrs/rivault/images/umrs-vision.tar
```

### Slurm Output File Location
The wrapper has `#SBATCH --output` pointing to `${JOB_INFO_DIR}/slurm-%j.out` (resolved from ai-login as `/root/.config/umrs/tmp/slurm-<N>.out`). However the **actual** file written on the compute node host goes to the working directory at sbatch-submission time (`//slurm-<N>.out` = `/slurm-<N>.out` on the host). To read the real output:
```bash
# Inside ai-login (via volume mount path):
podman exec ai-login cat /root/.config/umrs/tmp/slurm-<N>.out
# On the host directly:
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
- `/adapters/system` — built-in adapters (SAM3, HMAE) from `rivault/default_adapters/`
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
`src/umrs/rivault/x_nvidia-override.yaml` must include **both** `vllm` and `bentoml-vision`:
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
# Job output goes to /slurm-<N>.out on the HOST (not inside ai-login)
grep 'RIVAULT_PROFILES\|rivault_\|bentoml\|vision\|mini-rivault' /slurm-<N>.out
# The job also writes to ai-login volume mount path (may be stale from previous job with same ID!):
podman exec ai-login cat /root/.config/umrs/tmp/slurm-<N>.out
# ALWAYS check Modify timestamp: stale files from previous sessions have old dates
podman exec ai-login stat /root/.config/umrs/tmp/slurm-<N>.out | grep Modify
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
podman exec ai-login squeue
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
rm -f src/umrs/rivault/images/umrs-vision.tar
podman save ghcr.io/riken-rccs/umrs-vision:latest -o src/umrs/rivault/images/umrs-vision.tar

# 4. Submit new job (wrapper defaults to vision-bentoml — no --export needed)
podman exec ai-login scancel <old_job_id>
podman exec ai-login sbatch /root/.config/umrs/rivault_data/serve-model.slurm
```

---

## Workflow: Run Demos

### Demo 02 — SAM3 Segmentation
```bash
cd demos/02_sam3_segmentation
bash run_demo.sh
# Results in: demos/02_sam3_segmentation/results/
```

### Demo 03 — HMAE Inference
```bash
cd demos/03_hmae_inference
HMAE_MODE=reconstruct bash run_demo.sh   # modes: reconstruct, mae, features, all
# Results in: demos/03_hmae_inference/results/
```

### Performance (GPU)
- SAM3: ~46–58s per 8192×8192 image
- HMAE reconstruct: ~16s per 8192×8192 image (first call loads model ~10s extra)
