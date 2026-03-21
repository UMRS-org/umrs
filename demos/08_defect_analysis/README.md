# Demo 08: Automated Structural Defect Analysis

Multi-step UDM (User-Defined Models) workflow that chains SAM3 segmentation,
HMAE feature extraction, anomaly detection, and unsupervised clustering into a
single reproducible pipeline for materials-science image analysis.

## Pipeline Overview

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐     ┌────────────────────┐
│  1. SAM3        │     │  2. Region       │     │  3. HMAE        │     │  4. Structure      │
│  Segmentation   │────▶│  Extraction      │────▶│  Analysis       │────▶│  Discovery         │
│  (text-prompt)  │     │  (crop patches)  │     │  (feat+anomaly) │     │  (cluster+report)  │
└─────────────────┘     └──────────────────┘     └─────────────────┘     └────────────────────┘
     BentoML                 PIL / S3               BentoML x2            UMAP + KMeans
```

| Step | Class | What it does |
|------|-------|--------------|
| 1 | `SAM3SegmentationStep` | Text-prompted segmentation via BentoML; outputs bounding boxes + overlay |
| 2 | `RegionExtractionStep` | Crops detected regions from the original image, uploads to S3 |
| 3 | `HMAEAnalysisStep`     | Submits each patch for feature extraction **and** anomaly scoring |
| 4 | `StructureDiscoveryStep` | Dimensionality reduction (UMAP/PCA), KMeans clustering, analysis report |

## Prerequisites

- Mini-Rivault stack running with `vision-bentoml` profile (Demo 00)
- SAM3 + HMAE models loaded in BentoML service
- MinIO accessible (default `localhost:19000`)
- UMRS package installed: `pip install -e src/`
- Python packages: `numpy`, `requests`, `Pillow`, `boto3`
- Optional: `umap-learn`, `scikit-learn` (falls back to PCA + KMeans without UMAP)

## Quick Start

```bash
cd demos/06_defect_analysis
bash run_demo.sh
```

The script will:
1. Check service health (BentoML, MinIO)
2. Generate synthetic test images if none found in `test_data/`
3. Upload images to S3
4. Run the 4-step workflow
5. Download results to `results/`
6. Print a summary report

## Configuration

Override defaults via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `BENTOML_URL` | `http://localhost:11434/vision` | BentoML vision service URL |
| `S3_ENDPOINT_URL` | `http://localhost:19000` | MinIO endpoint |
| `S3_ACCESS_KEY` | `minioadmin` | MinIO access key |
| `S3_SECRET_KEY` | `minioadmin` | MinIO secret key |
| `BUCKET` | `defect-analysis` | S3 bucket for I/O |
| `PROMPT` | `defect` | SAM3 text prompt (e.g. "crack", "pore", "void") |
| `ANOMALY_THRESHOLD` | `0.3` | MAE loss threshold for flagging anomalies |
| `N_CLUSTERS` | `4` | Number of KMeans clusters |
| `IMAGE_DIR` | `./test_data` | Directory containing input images |

### Using real CT images

```bash
IMAGE_DIR=/home/mnagaso/hmae-sample/spring8concrete \
PROMPT="crack" \
ANOMALY_THRESHOLD=0.25 \
N_CLUSTERS=6 \
bash run_demo.sh
```

## Workflow Files

| File | Execution Mode | Description |
|------|---------------|-------------|
| `workflow.json` | `in_process` | Direct Python execution (development/testing) |
| `workflow_slurm.json` | `slurm` | Slurm + Singularity batch execution (HPC) |

## Step Details

### Step 1: SAM3 Segmentation (`sam3_segmentation.py`)

- **Input**: Raw images (S3 keys)
- **Output**: `<image>_sam3_meta.json` per image (bounding boxes, scores, overlay path)
- **API**: `POST /infer/submit` (model_type=sam3) → poll → get
- **Config**: `prompt`, `threshold`, `mask_threshold`, `timeout`

### Step 2: Region Extraction (`region_extraction.py`)

- **Input**: SAM3 meta JSONs + original images
- **Output**: Cropped patches as `patches/<image>_patch_<N>.png`
- **Config**: `min_patch_size`, `padding`, `max_patches_per_image`

### Step 3: HMAE Analysis (`hmae_analysis.py`)

Runs **two** BentoML tasks per patch:
- **Features mode** → CLS token (768-d) saved as `.npy`
- **MAE mode** → reconstruction loss for anomaly scoring
- **Output**: `analysis/<patch>_result.json` per patch + `analysis/all_embeddings.npy`
- **Config**: `anomaly_threshold`, `timeout`

### Step 4: Structure Discovery (`structure_discovery.py`)

- **Input**: Embedding matrix + anomaly scores
- **Output**: `analysis/report.json` with cluster assignments, UMAP projection, anomaly flags
- **Config**: `n_clusters`, `anomaly_prefix`

## Output Files

```
results/
├── analysis/
│   ├── report.json              ← Main analysis report
│   ├── all_embeddings.npy       ← N×768 embedding matrix
│   ├── projection_2d.npy        ← N×2 UMAP/PCA coordinates
│   └── <patch>_result.json      ← Per-patch details
├── patches/
│   └── <image>_patch_<N>.png    ← Cropped region images
└── *_sam3_seg.png               ← Segmentation overlays
```

### Report JSON structure

```json
{
  "total_patches": 12,
  "n_clusters": 4,
  "method": "umap+kmeans",
  "anomaly_threshold": 0.3,
  "cluster_summary": {
    "cluster_0": {
      "count": 4,
      "anomaly_count": 0,
      "anomaly_rate": 0.0,
      "mean_anomaly_score": 0.15
    },
    "cluster_2": {
      "count": 2,
      "anomaly_count": 2,
      "anomaly_rate": 1.0,
      "mean_anomaly_score": 0.52
    }
  },
  "patches": [
    {
      "name": "sample_001_patch_0",
      "cluster": 0,
      "anomaly_score": 0.12,
      "is_anomaly": false,
      "umap_x": -1.23,
      "umap_y": 0.45
    }
  ]
}
```

## Architecture: How UDM Orchestrates This

```
                           WorkflowExecutor
                           ┌──────────────────────────────┐
  workflow.json ──────────▶│  ScriptLoader.load_workflow() │
  steps/*.py                │  ScriptLoader.load_steps()   │
                           │                              │
  s3_keys ────────────────▶│  .run(s3, keys, bucket, ...)│
                           │    ├─ step 1: SAM3           │
                           │    │   └─ ctx.upload(meta)    │
                           │    ├─ step 2: RegionExtract   │
                           │    │   └─ ctx.upload(patches) │
                           │    ├─ step 3: HMAEAnalysis    │
                           │    │   └─ ctx.upload(embeds)  │
                           │    └─ step 4: Discovery       │
                           │        └─ report.json         │
                           └──────────────────────────────┘
```

Each step receives a `StepContext` providing:
- `download_bytes(key)` / `upload_bytes(key, data)` — S3 I/O
- `ctx.metadata` — shared state between steps
- Step config dict from `workflow.json`

## Extending the Pipeline

Add a new step by:

1. Create a Python file in `steps/` with a class inheriting `UMRSStep`
2. Implement `setup(config)`, `run(ctx) → StepResult`, `teardown()`
3. Add entry to `workflow.json` `steps` array

Example: add a visualization step after step 4:

```python
class VisualizationStep(UMRSStep):
    name = "visualize"
    def run(self, ctx: StepContext) -> StepResult:
        # Load report.json, generate matplotlib scatter plot
        ...
```

## Standalone Steps

For testing individual steps:

- `hmae_features.py` — Feature extraction only
- `anomaly_detection.py` — Anomaly scoring only

These are drop-in replacements if you want a 5-step pipeline with features
and anomaly detection as separate stages.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "BentoML health check failed" | Start Mini-Rivault: `bash demos/00_start_services/start_all.sh` |
| "SAM3 returned 0 objects" | Try a different `PROMPT` (e.g. "structure", "region", "object") |
| Step 3 times out | Increase `timeout` in workflow.json or set `TIMEOUT=600` |
| No UMAP available | Install: `pip install umap-learn`. PCA is used as fallback. |
| Import error for steps | Ensure UMRS is installed: `pip install -e src/` |
