# Demo 09: Structural Defect Analysis (Private)

This private demo bundles a defect-analysis workflow reference that combines
SAM3-style segmentation, region extraction, HMAE-derived analysis, and
structure clustering.

## What is in this directory

- `workflow/` contains the upload-ready in-process workflow bundle:
  - `workflow/workflow.json`
  - `workflow/steps/`
- `workflow_slurm/` contains the upload-ready Slurm-oriented bundle:
  - `workflow_slurm/workflow.json`
  - `workflow_slurm/steps/`
- `workflow/steps/` contains the step implementations:
  - `sam3_segmentation.py`
  - `region_extraction.py`
  - `hmae_analysis.py`
  - `hmae_features.py`
  - `anomaly_detection.py`
  - `structure_discovery.py`
- `README.md (step-by-step commands)` provides a practical shell demo that reuses the configured
  SAM3 experience to exercise upload, server-side processing, and result
  download.

## Important Current-State Note

The checked-in `README.md (step-by-step commands)` does **not** automatically execute the full
private multi-step workflow. In normal user mode it reuses the already
provisioned SAM3 bucket flow (`demo-sam3` by default) so the data path can still
be exercised even when the private `demo-defect` bucket has not been
operator-provisioned. Use `DEFECT_BUCKET=demo-defect` only after an admin reload
has provisioned that private experience server-side.

If you want to upload the full workflow, use `workflow/` or `workflow_slurm/`
directly with `umrs-client upload-workflow ...`. Those bundle directories are
already shaped to match the current CLI contract.

## Quick Start

```bash
cd demos_private/09_defect_analysis
DEFECT_BUCKET=demo-sam3 \
  IMAGE_DIR=../../demos/sample_data/images \
  MAX_IMAGES=1 \
  RESULT_DOWNLOAD_TIMEOUT=1800 \
  SAM3_REQUIRE_RESULTS=1 \
  # Follow the step-by-step commands in README.md
```

Useful overrides:

```bash
MAX_IMAGES=4 # Follow the step-by-step commands in README.md
IMAGE_DIR=/path/to/real/images # Follow the step-by-step commands in README.md
DEFECT_BUCKET=demo-defect # Follow the step-by-step commands in README.md  # only after operator provisioning
```

## Running Manually

```bash
# 1. Verify server connectivity
umrs-client ping

# 2. Verify that the SAM3 model is already visible to the server
umrs-client model list --scope system
umrs-client model list --scope user

# Optional: stage a user-scoped copy when your site exposes a writable model root
umrs-client model download facebook/sam2-hiera-small --name sam3 \
  --scope user --server-side --staging-dir /path/to/site-user-model-root

# Optional: upload the full defect-analysis workflow bundle instead of using
# the simpler SAM3-only shell path.
umrs-client upload-workflow workflow/ --name structural-defect-analysis
# or the Slurm-oriented variant:
# umrs-client upload-workflow workflow_slurm/ --name defect-analysis-slurm

# 3. Upload images for defect analysis
#    Images are dispatched to SAM3 segmentation on the GPU node.
#    The segmentation model detects structural features and defects
#    using text-prompted segmentation.
umrs-client upload_dir ../../demos/sample_data/images demo-sam3

# 4. Download segmentation results
#    Retrieves bounding boxes, masks, scores, and overlay images.
umrs-client download_results demo-sam3 ./results --force-overwrite
```

## Script Defaults

| Setting | Default |
|---|---|
| Bucket alias | `demo-sam3` |
| Image source | `../../tmp/hmae/images` |
| Result dir | `results/` |

This private demo follows the same general-user shape as the public vision
paths: it assumes the SAM3 experience is already available on the site.

## Public Analogue

The public replacement for a generic multi-step image workflow is:

- [../../demos/09_multistep_vision_pipeline/README.md](../../demos/09_multistep_vision_pipeline/README.md)

## Related Files

- [workflow/workflow.json](workflow/workflow.json)
- [workflow_slurm/workflow.json](workflow_slurm/workflow.json)
- [workflow/steps/](workflow/steps/)
