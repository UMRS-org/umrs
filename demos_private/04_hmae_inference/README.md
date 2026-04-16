# Demo 04: HMAE Inference (Private)

This private demo exercises an HMAE-backed image reconstruction experience on a
site where the HMAE model checkpoint and BentoML vision runtime are already
available.

Unlike the public [demos/04_vision_inference](../../demos/04_vision_inference/README.md),
this demo assumes a private or site-specific HMAE setup.

## What the current script does

`run_demo.sh` performs a small, practical flow:

1. Runs the standard pre-flight checks.
2. Optionally registers a local HMAE model directory if `HMAE_LOCAL_MODEL_DIR`
   is set.
3. Uploads PNG slice images to the configured HMAE bucket (`user1-hmae` by default).
4. Downloads the reconstruction outputs after server-side processing completes.

## Quick Start

```bash
cd demos_private/04_hmae_inference
bash run_demo.sh
```

To use a custom image directory:

```bash
bash run_demo.sh /path/to/volume_slices
```

To register a local checkpoint directory before running:

```bash
HMAE_LOCAL_MODEL_DIR=/path/to/hmae_checkpoint_dir bash run_demo.sh
```

## Running Manually

```bash
# 1. Verify server connectivity
umrs-client ping

# 2. (Optional) Register a local HMAE model checkpoint
#    Use this when you have pre-downloaded HMAE weights and want to
#    register them as a system-scoped model for the server to use.
umrs-client model register "hmae-pretrained" /path/to/hmae_checkpoint_dir \
    --type embedding --scope system \
    --description "HMAE ViT pretrained checkpoint"

# 3. Upload volume slice images to the HMAE bucket
#    The server will automatically dispatch images to the BentoML
#    vision service running HMAE on GPU via Slurm.
umrs-client upload_dir ../03_sam3_segmentation/test_data_full user1-hmae

# 4. Download reconstruction results
#    Waits for server-side HMAE processing and retrieves output
#    files (reconstructed images, metadata) to the local results/ directory.
umrs-client download_results user1-hmae ./results --force-overwrite
```

## Defaults Used By The Script

| Setting | Default |
|---|---|
| Bucket alias | `user1-hmae` |
| Default image dir | `../03_sam3_segmentation/test_data_full` |
| Result dir | `results/` |

## Notes

- The repo-contained HMAE implementation used by the current branch lives under
  `src/umrs/inference/` and the BentoML adapter path under
  `src/umrs/vision_service/default_adapters/`.
- This README intentionally avoids the older `process-objects` / `wait-output`
  command flow because the current demo script uses the modern upload +
  download-results path.

## Related Demos

- Public generic vision demo: [../../demos/04_vision_inference/README.md](../../demos/04_vision_inference/README.md)
- Private HMAE pipeline reference: [../10_hmae_full_pipeline/README.md](../10_hmae_full_pipeline/README.md)
- Public workflow docs: [../../demos/12_pipeline_sdk/README.md](../../demos/12_pipeline_sdk/README.md)
