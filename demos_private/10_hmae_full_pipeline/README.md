# Demo 10: HMAE Full Pipeline (Private)

This private demo documents a three-mode HMAE workflow reference:

1. full reconstruction
2. masked reconstruction / anomaly scoring
3. feature extraction

## What is in this directory

- `steps/hmae_reconstruct.py`
- `steps/hmae_mae.py`
- `steps/hmae_features.py`
- `workflow.json`

These files describe the intended full HMAE pipeline.

## Important Current-State Note

The shipped `run_demo.sh` is simpler than the full workflow description above.
It currently uploads a subset of images to the configured HMAE bucket
(`user1-hmae` by default) and downloads whatever the server-side HMAE
experience produces. It does **not** automatically force the site to execute
all three HMAE modes unless your server/workflow setup has been wired for that.

If you want to use `workflow.json` directly, update `source.path` so it points
to your checkout under `demos_private/10_hmae_full_pipeline/steps`.

## Quick Start

```bash
cd demos_private/10_hmae_full_pipeline
bash run_demo.sh
```

Useful overrides:

```bash
MAX_IMAGES=5 bash run_demo.sh
IMAGE_DIR=/path/to/volume_slices bash run_demo.sh
HMAE_BUCKET=my-hmae-bucket bash run_demo.sh
```

## Script Defaults

| Setting | Default |
|---|---|
| Bucket alias | `user1-hmae` |
| Image source | `../03_sam3_segmentation/test_data_full` |
| Result dir | `results/` |

## Public Analogue

The public replacement for a generic multi-output vision workflow is:

- [../../demos/10_vision_multi_mode/README.md](../../demos/10_vision_multi_mode/README.md)

## Related Demos

- Private single-mode HMAE: [../04_hmae_inference/README.md](../04_hmae_inference/README.md)
- Public workflow SDK overview: [../../demos/12_pipeline_sdk/README.md](../../demos/12_pipeline_sdk/README.md)
