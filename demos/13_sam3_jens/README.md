# Demo 13 — SAM3 Segmentation (Jens's Vision Stack)

Uses Jens's original standalone BentoML vision service (`mini-rivault-vision`)
for SAM3 segmentation instead of the UMRS-integrated `vision_service/` stack.

## Differences from Demo 02

| Aspect | Demo 02 (UMRS) | Demo 13 (Jens) |
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

# 3. Run the SAM3 demo
./run_demo.sh
```

## Prerequisites

- Services running (demos/00\_start\_services/)
- SAM3 model weights pre-staged on the Rivault node
- Jens's `mini-rivault-vision` submodule initialised (`git submodule update --init --recursive`)
