# Demo 09: Multi-Step Vision Pipeline

End-to-end **3-step** image analysis pipeline demonstrating the UMRS
Step/`workflow.json` pattern.

## Pipeline Overview

```
┌─────────────┐     ┌───────────────┐     ┌──────────────┐
│  1. Segment  │────▶│ 2. Crop Boxes │────▶│ 3. Statistics │
│  (SAM3 GPU)  │     │   (CPU PIL)   │     │  (CPU numpy)  │
└─────────────┘     └───────────────┘     └──────────────┘
    Input images       Per-object patches    Per-patch metrics
```

| Step | Class | Runs on | Description |
|------|-------|---------|-------------|
| 1 | `SegmentStep` | GPU (BentoML) | Submit images to a vision model, get bounding boxes + scores |
| 2 | `CropRegionsStep` | CPU | Crop each bounding box from the original image |
| 3 | `ImageStatsStep` | CPU | Compute RGB mean/std, brightness, contrast, histograms |

## Quick Start

```bash
# From demos/ directory
bash 09_multistep_vision_pipeline/run_demo.sh

# With your own images
bash 09_multistep_vision_pipeline/run_demo.sh /path/to/images
```

## What the Script Does

1. **Pre-flight** — verifies `umrs-client` can reach the server
2. **Upload workflow** — pushes `workflow.json` + `steps/` to the server
3. **Upload images** — sends images to the configured bucket
4. **Wait & download** — polls until the 3-step pipeline completes, then
   downloads all results
5. **Show report** — prints the aggregate analysis report

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PIPELINE_BUCKET` | `user1-pipeline` | S3 bucket alias for this pipeline |
| `MAX_IMAGES` | `3` | Limit how many images to upload |
| `UMRS_CONF` | `~/.config/umrs/conf_umrs.json` | Server configuration file |

## Customising

### Use a different segmentation model

Edit `workflow.json` → `steps[0].config`:

```json
{
  "bentoml_url": "http://your-gpu-host:PORT/vision",
  "prompt": "defect",
  "threshold": 0.3
}
```

### Add a new step

1. Create `steps/my_step.py` implementing `Step`
2. Add an entry to `workflow.json → steps[]`
3. Re-upload the workflow: `umrs-client upload-workflow .`

## Output

```
results/
├── image_001_segment_meta.json   # per-image segmentation metadata
├── patches/
│   ├── image_001_patch_0000.png  # cropped regions
│   └── ...
└── analysis/
    ├── image_001_patch_0000_stats.json
    └── report.json                # aggregate summary
```

## See Also

- **Demo 08** — Step/workflow.json basics
- **Demo 12** — Pipeline SDK (6-phase lifecycle)
- **Demo 15** — Complex DAG with parallel branches
