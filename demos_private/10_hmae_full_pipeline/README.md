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

The shipped `README.md (step-by-step commands)` is simpler than the full workflow description above.
It uploads Spring-8 concrete PNG slices to the configured `demo_user` HMAE
buckets (`demo-hmae`, `demo-hmae-mae`, and `demo-hmae-features`) and downloads
the server-side HMAE outputs for each task. It does **not** re-register the
workflow package unless you explicitly run the workflow commands below.

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

1. `README.md (step-by-step commands)` is the simple operational path. It uploads images to the
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

In practice, use `README.md (step-by-step commands)` when validating the deployed site behavior, and
use `workflow/` when developing or debugging the
pipeline logic itself.

## Uploading the Workflow Package

If you want the server to run the workflow defined in this directory, use the
upload-ready `workflow/` bundle.

### What must be in the uploaded directory

The directory you pass to `umrs-client workflow upload` must contain:

- `workflow.json` at the top level
- the referenced Python step files, here under `steps/`

This means the following two commands will fail for different reasons:

```bash
umrs-client workflow upload ./workflow/workflow.json
# fails because the argument is a file, not a directory

umrs-client workflow upload ./workflow/steps
# fails because that directory does not contain workflow.json at its top level
```

`workflow upload` does not accept a single JSON file, and it does not
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
umrs-client workflow upload ./workflow --name hmae-full-pipeline

# 2. Verify registration
umrs-client workflow list --output text --name hmae-full-pipeline

# 3. Upload Spring-8 concrete inputs to the bucket you want the workflow to use
umrs-client data dir-upload ../../tmp/hmae/spring8concrete demo-hmae

# 4. Trigger the registered workflow against that bucket
#    (capture the execution_id from the output for steps 5-6)
EXEC_ID=$(umrs-client workflow run hmae-full-pipeline --bucket demo-hmae | grep execution_id | awk '{print $NF}')

# 5. Poll until the workflow completes
umrs-client workflow status "$EXEC_ID"

# 6. Download results written by the workflow
umrs-client workflow results "$EXEC_ID" ./results/workflow --force-overwrite
```

### When to use `workflow run`

Use `umrs-client workflow run ...` when you want to execute the registered
workflow definition explicitly. Use `README.md (step-by-step commands)` when you want to exercise the
site's existing HMAE experience wiring without re-registering a custom
workflow package.

## Quick Start

```bash
cd demos_private/10_hmae_full_pipeline
# Follow the step-by-step commands in the Running Manually section above
```

Useful overrides:

```bash
MAX_IMAGES=5
IMAGE_DIR=../../tmp/hmae/spring8concrete
HMAE_BUCKET=demo-hmae
```

## Running Manually

Each HMAE task can be run individually via its experience bucket, or all three
in sequence via the `hmae-full-pipeline` workflow. Inputs are uploaded with
`umrs-client data dir-upload` (legacy S3 transfer); outputs are fetched with
`umrs-client data download` on the output bucket or `umrs-client workflow results`
for workflow executions.

### Individual experience runs

```bash
# 1. Verify server connectivity
umrs-client ping

# Task 1: Reconstruction (bucket: demo-hmae)
#    Full image reconstruction — PNG + pkl output.
umrs-client data dir-upload ../../tmp/hmae/spring8concrete demo-hmae

# Task 2: MAE Reconstruction (bucket: demo-hmae-mae)
#    Masked reconstruction with anomaly scoring.
umrs-client data dir-upload ../../tmp/hmae/spring8concrete demo-hmae-mae

# Task 3: Feature Extraction (bucket: demo-hmae-features)
#    CLS and patch-level feature vectors.
umrs-client data dir-upload ../../tmp/hmae/spring8concrete demo-hmae-features
```

Outputs are written back to the respective S3 buckets. Download them with:

```bash
# Download results from each experience bucket
umrs-client data download demo-hmae ./results/reconstruction --force-overwrite
umrs-client data download demo-hmae-mae ./results/mae_reconstruction --force-overwrite
umrs-client data download demo-hmae-features ./results/feature_extraction --force-overwrite
```

> **Note**: The individual experience runs use the site's existing experience
> wiring (no workflow registration needed). For the full chained pipeline,
> see the "Uploading the Workflow Package" section above.

### Full workflow run

```bash
# Run the registered 3-step workflow against the reconstruction bucket
EXEC_ID=$(umrs-client workflow run hmae-full-pipeline --bucket demo-hmae | grep execution_id | awk '{print $NF}')

# Wait until it completes
umrs-client workflow status "$EXEC_ID"

# Fetch all results
umrs-client workflow results "$EXEC_ID" ./results/workflow --force-overwrite
```

This private demo assumes the HMAE task buckets are already configured by the
site. It does not treat admin reload as part of the normal user flow.

> **Note**: Running all 3 tasks sequentially may cause CUDA out-of-memory on
> the third task if the GPU does not release memory between tasks. If this
> happens, restart the BentoML service or run tasks individually.

## Script Defaults

| Setting | Default |
|---|---|
| Bucket alias | `demo-hmae` |
| Image source | `../../tmp/hmae/spring8concrete` |
| Result dir | `results/` |

## Public Analogue

The public replacement for a generic multi-output vision workflow is:

- [../../demos/10_vision_multi_mode/README.md](../../demos/10_vision_multi_mode/README.md)

## Related Demos

- Private single-mode HMAE: [../04_hmae_inference/README.md](../04_hmae_inference/README.md)
- Public workflow SDK overview: [../../demos/12_pipeline_sdk/README.md](../../demos/12_pipeline_sdk/README.md)
