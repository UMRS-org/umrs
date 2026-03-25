# Demo 14 — HMAE Inference (Jens's Vision Stack)

Uses Jens's original standalone BentoML vision service (`mini-rivault-vision`)
for HMAE inference instead of the UMRS-integrated `vision_service/` stack.

## Differences from Demo 03

| Aspect | Demo 03 (UMRS) | Demo 14 (Jens) |
|--------|---------------|----------------|
| Backend | `vision_service/` generic adapter | `mini-rivault-vision/` standalone |
| BentoML endpoint | `/infer/submit` (JSON) | `/vision_task/submit` (pickle) |
| Env var | `VISION_STACK=umrs` (default) | `VISION_STACK=jens` |
| Slurm script | `serve-model.slurm` | `serve-model-jens.slurm` |
| Symlink | `/opt/rivault` | `/opt/rivault-jens` |

## Quick Start

```bash
# 1. Set vision stack to Jens's implementation
export VISION_STACK=jens

# 2. Start services (same as usual)
cd ../00_start_services && ./start.sh

# 3. Run the HMAE demo
./run_demo.sh
```

## HMAE Inference Modes

The Jens stack supports three modes via the `task` parameter:
- **reconstruct** — Full encode + decode reconstruction
- **mae** — Masked autoencoder reconstruction with anomaly scoring
- **features** — Encoder-only feature extraction (CLS + patch tokens)

## Prerequisites

- Services running (demos/00\_start\_services/)
- HMAE model checkpoint pre-staged on the Rivault node (`HMAE_MODEL_PATH`)
- Jens's `mini-rivault-vision` submodule initialised (`git submodule update --init --recursive`)
