# Demo 10: HMAE Full Pipeline (Private)

This private demo documents a three-mode HMAE workflow reference:

1. full reconstruction
2. masked reconstruction / anomaly scoring
3. feature extraction

## What is in this directory

- `workflow/workflow.json`
- `workflow/steps/hmae_reconstruct.py`
- `workflow/steps/hmae_mae.py`
- `workflow/steps/hmae_features.py`

These files describe the intended full HMAE pipeline.

## Important Current-State Note

The shipped `run_demo.sh` is simpler than the full workflow description above.
It currently uploads a subset of images to the configured HMAE bucket
(`user1-hmae` by default) and downloads whatever the server-side HMAE
experience produces. It does **not** automatically force the site to execute
all three HMAE modes unless your server/workflow setup has been wired for that.

The checked-in workflow bundle is the uploadable form used by the current CLI:
`source.type = "uploaded"` with `path = "steps"`.

## How the Workflow Is Structured

This demo includes an upload-ready workflow bundle under `workflow/`:

- `workflow/workflow.json`
- `workflow/steps/`

`workflow/workflow.json` defines:

- `name`, `version`, `description`: workflow identity and summary
- `source`: where the server should load step code from
- `steps`: ordered list of step classes to run
- `resources`: expected GPU / memory / timeout envelope
- `output`: overall output contract for the workflow

For uploaded package mode, the workflow uses:

```json
"source": {
	"type": "uploaded",
	"path": "steps"
}
```

That tells the server to load the step modules from the uploaded `steps/`
directory inside the workflow bundle.

The workflow runs these steps in order:

| Step | Entrypoint | Purpose |
|---|---|---|
| `hmae_reconstruct` | `hmae_reconstruct:HMAEReconstructStep` | Full-image reconstruction |
| `hmae_mae` | `hmae_mae:HMAEMaeStep` | Masked reconstruction + anomaly scoring |
| `hmae_features` | `hmae_features:HMAEFeaturesStep` | CLS / patch feature extraction |

All three steps call the BentoML HMAE service through the configured
`bentoml_url`.

## How the Steps Connect

The step files are designed to chain together through both S3 outputs and
shared metadata in `StepContext`.

| Step | Reads | Writes | Passes Forward |
|---|---|---|---|
| `workflow/steps/hmae_reconstruct.py` | Original uploaded images | `reconstruct/<basename>_recon.json` | `original_keys`, `original_bucket` in `ctx.metadata` |
| `workflow/steps/hmae_mae.py` | Original keys from metadata, or reconstruct JSON fallback | `mae/<basename>_mae.json` | anomaly score metadata per image |
| `workflow/steps/hmae_features.py` | Original keys from metadata, or MAE JSON fallback | `features/<basename>_feat.json`, `features/<basename>_emb.npy`, `features/all_embeddings.npy` | stacked embedding matrix in `StepResult.embeddings` |

### Step 1: Reconstruction

`workflow/steps/hmae_reconstruct.py` submits each image with
`inference_config={"task": "reconstruction"}`. For each source image it writes
a metadata JSON file that records the original S3 key and the reconstruction
artifact returned by the HMAE service.

### Step 2: MAE Reconstruction

`workflow/steps/hmae_mae.py` runs masked reconstruction with
`inference_config={"task": "mae_reconstruction"}`. It records the returned
`mae_loss`, compares that to `anomaly_threshold`, and marks each image as normal
or anomalous in `mae/<basename>_mae.json`.

### Step 3: Feature Extraction

`workflow/steps/hmae_features.py` runs
`inference_config={"task": "feature_extraction"}`. It downloads the returned
feature artifact, extracts the CLS vector, writes one `.npy` embedding per
image, and also writes `features/all_embeddings.npy` for the full batch.

## How To Work On The Workflow

There are two different ways to use this demo, and they are easy to conflate:

1. `run_demo.sh` is the simple operational path. It uploads images to the
	configured HMAE buckets and downloads whatever the server-side HMAE
	experience produces.
2. `workflow/` is the explicit custom-workflow reference.
	Use this when you want to change how the three HMAE stages are chained,
	what metadata is emitted, or what final artifacts are produced.

If you want to iterate on the workflow itself:

1. Edit one or more files under `workflow/steps/`.
2. Update `workflow/workflow.json` if you need to change step order or config.
3. Upload input data to the bucket wired to that workflow and inspect the JSON,
	PNG, `.pt`, and `.npy` outputs written by each stage.

In practice, use `run_demo.sh` when validating the deployed site behavior, and
use `workflow/` when developing or debugging the
pipeline logic itself.

## Uploading the Workflow Package

If you want the server to run the workflow defined in this directory, use the
upload-ready `workflow/` bundle.

### What must be in the uploaded directory

The directory you pass to `umrs-client upload-workflow` must contain:

- `workflow.json` at the top level
- the referenced Python step files, here under `steps/`

This means the following two commands will fail for different reasons:

```bash
umrs-client upload-workflow ./workflow/workflow.json
# fails because the argument is a file, not a directory

umrs-client upload-workflow ./workflow/steps
# fails because that directory does not contain workflow.json at its top level
```

`upload-workflow` does not accept a single JSON file, and it does not
automatically merge `workflow/workflow.json` with `workflow/steps/` if you pass
them separately. You must point it at one directory that already has both.

For this demo, the uploadable directory layout is now already present as:

```text
workflow/
	workflow.json
	steps/
		hmae_reconstruct.py
		hmae_mae.py
		hmae_features.py
```

### Upload, verify, and run

From `demos_private/10_hmae_full_pipeline`, the exact working flow is:

```bash
# 1. Upload the workflow package
umrs-client upload-workflow ./workflow --name hmae-full-pipeline

# 2. Verify registration
umrs-client --output text workflow list --name hmae-full-pipeline

# 3. Upload input images to the bucket you want the workflow to use
umrs-client upload_dir ../../tmp/spring8-data/spring8concrete user1-hmae

# 4. Trigger the registered workflow against that bucket
umrs-client workflow run hmae-full-pipeline --bucket user1-hmae

# 5. Download results written by the workflow
umrs-client download_results user1-hmae ./results/workflow --force-overwrite
```

### When to use `workflow run`

Use `umrs-client workflow run ...` when you want to execute the registered
workflow definition explicitly. Use `run_demo.sh` when you want to exercise the
site's existing HMAE experience wiring without re-registering a custom
workflow package.

## Quick Start

```bash
cd demos_private/10_hmae_full_pipeline
bash run_demo.sh
```

Useful overrides:

```bash
MAX_IMAGES=5 bash run_demo.sh
IMAGE_DIR=../../tmp/spring8-data/spring8concrete bash run_demo.sh
HMAE_BUCKET=my-hmae-bucket bash run_demo.sh
```

## Running Manually

The full pipeline runs 3 HMAE tasks sequentially, each using a different
bucket alias configured in `user1.json`:

```bash
# 1. Verify server connectivity and reload config
umrs-client ping
umrs-client --admin-token "$UMRS_ADMIN_TOKEN" reload

# 2. Task 1: Reconstruction (bucket: user1-hmae)
#    Full image reconstruction — the model reconstructs the input image
#    and outputs the reconstructed version as PNG + pkl.
umrs-client upload_dir ../../tmp/spring8-data/spring8concrete user1-hmae
umrs-client download_results user1-hmae ./results/reconstruction --force-overwrite

# 3. Task 2: MAE Reconstruction (bucket: user1-hmae-mae)
#    Masked autoencoder reconstruction — the model masks random patches,
#    reconstructs them, and reports the reconstruction loss (anomaly score).
umrs-client upload_dir ../../tmp/spring8-data/spring8concrete user1-hmae-mae
umrs-client download_results user1-hmae-mae ./results/mae_reconstruction --force-overwrite

# 4. Task 3: Feature Extraction (bucket: user1-hmae-features)
#    Extracts CLS and patch-level feature vectors for downstream analysis.
#    Outputs .pt (PyTorch tensor) files plus pkl metadata.
umrs-client upload_dir ../../tmp/spring8-data/spring8concrete user1-hmae-features
umrs-client download_results user1-hmae-features ./results/feature_extraction --force-overwrite
```

> **Note**: Running all 3 tasks sequentially may cause CUDA out-of-memory on
> the third task if the GPU does not release memory between tasks. If this
> happens, restart the BentoML service or run tasks individually.

## Script Defaults

| Setting | Default |
|---|---|
| Bucket alias | `user1-hmae` |
| Image source | `../../tmp/spring8-data/spring8concrete` |
| Result dir | `results/` |

## Public Analogue

The public replacement for a generic multi-output vision workflow is:

- [../../demos/10_vision_multi_mode/README.md](../../demos/10_vision_multi_mode/README.md)

## Related Demos

- Private single-mode HMAE: [../04_hmae_inference/README.md](../04_hmae_inference/README.md)
- Public workflow SDK overview: [../../demos/12_pipeline_sdk/README.md](../../demos/12_pipeline_sdk/README.md)
