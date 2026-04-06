# Demo 10: HMAE Full Pipeline (All 3 Modes)

Runs all three HMAE (Hierarchical Masked Auto-Encoder) inference modes as a
chained Step workflow: full reconstruction, masked reconstruction with
anomaly scoring, and feature extraction.

## Pipeline

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  1. Reconstruct  │     │  2. MAE          │     │  3. Features     │
│  (full decode)   │────▶│  (masked + loss) │────▶│  (encoder-only)  │
└──────────────────┘     └──────────────────┘     └──────────────────┘
     BentoML                  BentoML                  BentoML
  mode=reconstruct           mode=mae               mode=features
  → *_recon.png              → *_mae.png             → *_features.pt
                             → mae_loss              → CLS embeddings
```

| Step | Class | Mode | Outputs |
|------|-------|------|---------|
| 1 | `HMAEReconstructStep` | `reconstruct` | `reconstruct/<image>_recon.json` + visualization PNG |
| 2 | `HMAEMaeStep` | `mae` | `mae/<image>_mae.json` + visualization PNG + anomaly flag |
| 3 | `HMAEFeaturesStep` | `features` | `features/<image>_feat.json` + `.npy` embeddings |

## Prerequisites

- Services running (see [Demo 00](../00_start_services/README.md))
- HMAE model checkpoint pre-staged on the Rivault node at `$HMAE_MODEL_PATH`
- Mini-Rivault with `vision-bentoml` profile active

## Quick Start

```bash
cd demos/10_hmae_full_pipeline
bash run_demo.sh
```

## What This Demo Does

1. Verifies UMRS services are running
2. Uploads test images (X-ray CT slices) to S3
3. The server runs the 3-step Step workflow:
   - **Step 1 (Reconstruct)**: Full encode+decode — visualizes model reconstruction quality
   - **Step 2 (MAE)**: 50% random masking — measures reconstruction loss as anomaly score
   - **Step 3 (Features)**: Encoder-only — extracts CLS token (768-d) for downstream tasks
4. Downloads all results to `results/`

## Configuration

Override defaults via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `BENTOML_URL` | `http://localhost:11434/vision` | BentoML vision service URL |
| `UMRS_USERNAME` | `rivault_user` | UMRS user for bucket access |
| `MAX_IMAGES` | `3` | Max images to process |

### Anomaly Threshold

The MAE step flags images as anomalous when `mae_loss > anomaly_threshold`.
Adjust in `workflow.json`:

```json
{
  "name": "hmae_mae",
  "config": {
    "anomaly_threshold": 0.3
  }
}
```

## Output Structure

```
results/
├── reconstruct/
│   └── <image>_recon.json     ← Full reconstruction metadata
├── mae/
│   └── <image>_mae.json       ← MAE metadata (includes mae_loss, is_anomaly)
└── features/
    ├── <image>_feat.json      ← Feature extraction metadata
    ├── <image>_emb.npy        ← Per-image CLS embedding (768-d)
    └── all_embeddings.npy     ← Stacked N×768 embedding matrix
```

### MAE Result JSON

```json
{
  "source_key": "volume_170.png",
  "task": "mae_reconstruction",
  "mae_loss": 0.1842,
  "anomaly_threshold": 0.3,
  "is_anomaly": false,
  "output_s3_key": "s3://bucket/volume_170_mae.png"
}
```

### Features Result JSON

```json
{
  "source_key": "volume_170.png",
  "task": "feature_extraction",
  "features_s3_key": "s3://bucket/volume_170_features.pt",
  "embedding_key": "features/volume_170_emb.npy",
  "embedding_dim": 768
}
```

## Step Architecture

Each step inherits `Step` and follows the standard interface:

```python
from umrs.pipeline_sdk import Step, StepContext, StepResult

class HMAEReconstructStep(Step):
    name = "hmae_reconstruct"

    def setup(self, config: dict) -> None:
        self.bentoml_url = config["bentoml_url"]

    def run(self, ctx: StepContext) -> StepResult:
        # Process images via BentoML, upload results to S3
        return StepResult(output_keys=[...])
```

Steps share state via `ctx.metadata`:
- Step 1 stores `ctx.metadata["original_keys"]` and `ctx.metadata["original_bucket"]`
- Steps 2 and 3 read these to process the same original images
- If metadata is missing, steps fall back to parsing the previous step's JSON outputs

## Workflow Definition

See [workflow.json](workflow.json) for the full definition. The workflow is a standard
Step pipeline that can be executed in any supported mode:

| File | Mode | Description |
|------|------|-------------|
| `workflow.json` | `in_process` | Direct execution (default) |

## See Also

- [Demo 04: HMAE Inference](../04_hmae_inference/README.md) — single-mode HMAE via umrs-client
- [Demo 08: User-Defined Model](../08_user_model/README.md) — Step framework introduction + custom vision adapters
- [Demo 09: Defect Analysis](../09_defect_analysis/README.md) — multi-step pipeline with SAM3 + HMAE
- [UDM Design Doc](../../doc/design_user_defined_models.md) — Step specification

## Vision Adapter Architecture

All three steps call the BentoML vision service using `model_type: "hmae"`.
This adapter is loaded from `rivault/default_adapters/hmae_adapter.py` via
auto-discovery — the same mechanism users can leverage for their own models.

To replace the HMAE adapter or add custom vision processing:

1. Write a `VisionModelAdapter` subclass (see [Demo 08](../08_user_model/README.md))
2. Place it in `USER_ADAPTERS_DIR` with optional `requirements.txt`
3. Update `workflow.json` to reference your adapter's `model_type`
