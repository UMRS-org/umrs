# Demo 09: Structural Defect Analysis (Private)

This private demo bundles a defect-analysis workflow reference that combines
SAM3-style segmentation, region extraction, HMAE-derived analysis, and
structure clustering.

## What is in this directory

- `steps/` contains the step implementations:
  - `sam3_segmentation.py`
  - `region_extraction.py`
  - `hmae_analysis.py`
  - `hmae_features.py`
  - `anomaly_detection.py`
  - `structure_discovery.py`
- `workflow.json` and `workflow_slurm.json` capture the intended multi-step
  workflow shape.
- `run_demo.sh` provides a practical shell demo that reuses the configured
  SAM3 experience to exercise upload, server-side processing, and result
  download.

## Important Current-State Note

The checked-in `run_demo.sh` does **not** automatically execute the full
private multi-step workflow. It currently reuses the public SAM3-style bucket
flow (`user1-sam3` by default) so the data path can still be exercised even if
the full private workflow is not wired into the current site config.

If you want to use `workflow.json` directly, update the `source.path` values to
match your checkout path under `demos_private/09_defect_analysis/steps` before
uploading or registering the workflow.

## Quick Start

```bash
cd demos_private/09_defect_analysis
bash run_demo.sh
```

Useful overrides:

```bash
MAX_IMAGES=4 bash run_demo.sh
IMAGE_DIR=/path/to/real/images bash run_demo.sh
DEFECT_BUCKET=my-segmentation-bucket bash run_demo.sh
```

## Script Defaults

| Setting | Default |
|---|---|
| Bucket alias | `user1-sam3` |
| Image source | `../03_sam3_segmentation/test_data_full` |
| Result dir | `results/` |

## Public Analogue

The public replacement for a generic multi-step image workflow is:

- [../../demos/09_multistep_vision_pipeline/README.md](../../demos/09_multistep_vision_pipeline/README.md)

## Related Files

- [workflow.json](workflow.json)
- [workflow_slurm.json](workflow_slurm.json)
- [steps/](steps/)
