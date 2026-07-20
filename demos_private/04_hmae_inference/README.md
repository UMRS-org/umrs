# Demo 04: HMAE Inference (Private)

This private demo exercises an HMAE-backed image reconstruction experience on a
site where the HMAE model checkpoint and BentoML vision runtime are already
available.

Unlike the public [demos/04_vision_inference](../../demos/04_vision_inference/README.md),
this demo assumes a private or site-specific HMAE setup.

## What the current script does

The README performs a small, practical flow:

1. Runs the standard pre-flight checks.
2. Optionally registers a local HMAE model directory if `HMAE_LOCAL_MODEL_DIR`
   is set.
3. Uploads PNG slice images to the configured HMAE bucket (`demo-hmae` by default).
4. Downloads the reconstruction outputs after server-side processing completes.

## Quick Start

```bash
cd demos_private/04_hmae_inference
# See the numbered steps below in this README
```

To use a custom image directory:

```bash
cd demos_private/04_hmae_inference
# See the numbered steps below in this README /path/to/volume_slices
```

To register a local checkpoint directory before running:

```bash
cd demos_private/04_hmae_inference
HMAE_LOCAL_MODEL_DIR=/path/to/hmae_checkpoint_dir
# See the numbered steps below in this README
```

## Running Manually

```bash
# 1. Verify server connectivity
umrs ping

# 2. (Optional) Upload a local HMAE model checkpoint
#    On a local-client / remote-server deployment, a plain `model register`
#    path only works for paths that already exist on the server. Use
#    `model upload` plus an explicit server-visible register path instead.
umrs model upload /path/to/hmae_checkpoint_dir \
  --name "hmae-pretrained" \
  --type vision --scope user \
  --description "HMAE ViT pretrained checkpoint" \
  --register-path /path/to/site-user-model-root/hmae-pretrained \
  --force

# 3. Upload volume slice images to the HMAE bucket
#    The server will automatically dispatch images to the BentoML
#    vision service running HMAE on GPU via Slurm.
umrs data dir-upload ../../demos/03_sam3_segmentation/test_data_full demo-hmae

# 4. Download reconstruction results
#    Waits for server-side HMAE processing and retrieves output
#    files (reconstructed images, metadata) to the local results/ directory.
umrs data download demo-hmae ./results --force-overwrite
```

## Defaults Used By The Script

| Setting | Default |
|---|---|
| Bucket alias | `demo-hmae` |
| Default image dir | `../../demos/03_sam3_segmentation/test_data_full` |
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
