# Demo 15: SAM3 Segmentation (user_script)

Equivalent to [Demo 02](../02_sam3_segmentation/README.md), but uses the
`user_script://` model type with the Pipeline SDK instead of BentoML/Rivault.
The SAM2 model is loaded directly by the Step class — no BentoML service or
Rivault sandbox is required.

## Prerequisites

- Services running (see [Demo 00](../00_start_services/README.md))
- SAM2 model weights downloadable from HuggingFace (`facebook/sam2-hiera-small`)
  or pre-staged on the server node
- Python packages: `torch`, `sam2`, `Pillow`, `numpy`
- UMRS client installed (`pip install .` from `src/`)

## Architecture

```
Client → UMRS Server → user_script Pipeline
              │              └── SAM3SegmentStep (GPU)
              │                   ├── Load SAM2 from HF (lazy, cached)
              │                   ├── Generate automatic masks
              │                   └── Upload overlay PNG + mask JSON
              └── S3 (MinIO)
```

No BentoML, NGINX, or Slurm needed — the model runs directly in the UMRS
server process via the Pipeline SDK.

## Quick Start

```bash
cd demos/15_sam3_user_script
bash run_demo.sh
```

## Configuration

### workflow.json

The `workflow.json` defines the pipeline with a single `SAM3SegmentStep`:

| Field | Description |
|-------|-------------|
| `source.path` | Absolute path to the `scripts/` directory on the server |
| `steps[0].config.model_id` | HuggingFace model ID (default: `facebook/sam2-hiera-small`) |
| `steps[0].config.points_per_side` | Grid density for automatic mask generation (default: 32) |

### vision_models.json

Uses the new modality/task format:

```json
{
  "modalities": [{
    "modality": "image",
    "task": "segmentation",
    "model": {
      "model_type": "user_script:///path/to/workflow.json",
      "return_type": "file"
    }
  }]
}
```

## Outputs

For each input image, the step produces:
- **`segmented/<filename>`** — PNG overlay with coloured mask regions
- **`segmented/<filename>.masks.json`** — JSON with per-mask metadata (area, bbox, IoU, stability score)

## Differences from Demo 02

| | Demo 02 (BentoML) | Demo 15 (user_script) |
|---|---|---|
| Model type | `rivault://bentoml://...` | `user_script:///path/to/workflow.json` |
| Runtime | BentoML service in Rivault sandbox | Pipeline SDK step in UMRS server |
| Infrastructure | Slurm + BentoML + NGINX | None extra |
| Model loading | Pre-staged by admin | Lazy-loaded from HuggingFace |
| Customisation | Change `vision_model_config` | Edit `workflow.json` or `sam3_segment.py` |

## Customising

Edit `scripts/sam3_segment.py` to change the segmentation logic, add
post-processing, or support a different SAM model variant. No UMRS code
changes are needed — just update `workflow.json` if the entrypoint changes.
