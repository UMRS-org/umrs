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
  03_hmae_inference/
    run_demo.sh              ← end-to-end HMAE demo (single image, downloads results)
    test_data/               ← test images (symlinks OK)
    results/                 ← downloaded outputs land here
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
2. Wrapper sets host-context overrides (`VLLM_PATH_TO_HUGGINGFACE`, `HMAE_MODEL_PATH`, etc.)
3. Wrapper calls `exec bash ./serve-model.slurm` (the real Slurm script in `src/umrs/rivault/`)
4. `serve-model.slurm` auto-detects GPU, picks nvidia/rocm override YAML, writes `env.{uuid}`
5. Runs `podman-compose ... --profile harness --profile vllm --profile vision-bentoml up`

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
| `RIVAULT_PROFILES` | wrapper slurm / sbatch | space-separated list of compose profiles to activate |
| `VLLM_DOCKER_TAG` | wrapper slurm | `cu130-nightly-aarch64` (Blackwell GPU) |
| `VLLM_GPU_MEM_UTIL` | wrapper slurm | `0.10` (low — always-on LLM takes most GPU) |

### Tarball Auto-Reload
`serve-model.slurm` checks if `rivault/images/umrs-vision.tar` is newer than the cached podman image.
If yes, it runs `podman rmi -f` + `podman load`. Always re-save tarball after rebuilding image:
```bash
rm -f src/umrs/rivault/images/umrs-vision.tar
podman save ghcr.io/riken-rccs/umrs-vision:latest -o src/umrs/rivault/images/umrs-vision.tar
```

---

## Vision Service (BentoML)

### Endpoints
All prefixed with `/vision/` through NGINX at `http://localhost:11434/vision`

| Method | Path | Description |
|---|---|---|
| POST | `/sam3_segment/submit` | Submit SAM3 task |
| GET | `/sam3_segment/status?task_id=...` | Poll status |
| GET | `/sam3_segment/get?task_id=...` | Get result |
| POST | `/hmae_reconstruct/submit` | Submit HMAE task |
| GET | `/hmae_reconstruct/status?task_id=...` | Poll status |
| GET | `/hmae_reconstruct/get?task_id=...` | Get result |
| GET | `/healthz` | Health check |

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

### 7. `RIVAULT_PROFILES` env var needs exact string for sbatch `--export`
- `sbatch --export=ALL,RIVAULT_PROFILES="harness vllm vision-bentoml"` — space-separated, quoted

---

## Debugging Guide

### Check BentoML container logs
```bash
podman exec ai-login ssh compute 'podman logs rivault_bentoml-vision_1 2>&1 | tail -100'
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

# 3. Save tarball (auto-detected by serve-model.slurm on next job)
rm -f src/umrs/rivault/images/umrs-vision.tar
podman save ghcr.io/riken-rccs/umrs-vision:latest -o src/umrs/rivault/images/umrs-vision.tar

# 4. Submit new job (cancels old one if needed)
podman exec ai-login scancel <old_job_id>
podman exec ai-login sbatch --export=ALL,RIVAULT_PROFILES="harness vllm vision-bentoml" \
  /root/.config/umrs/rivault_data/serve-model.slurm
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
