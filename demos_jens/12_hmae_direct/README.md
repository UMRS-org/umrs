# Demo 12: HMAE (Hierarchical Masked Auto-Encoder)

Upload images through UMRS → Slurm dispatches `internal_genAI` on GPU → HMAE runs
inference in one or more modes → results downloaded as pickle files.

Three inference modes:

| Mode | Task | Output |
|------|------|--------|
| `mae` | Masked reconstruction | Prediction tensor + mask + loss |
| `reconstruct` | Full reconstruction (no masking) | Prediction tensor |
| `features` | Encoder feature extraction | Feature tensor (768-dim) |

---

## Prerequisites

- UMRS server running with `VISION_STACK=jens`
- SSH tunnel open (if running remotely against RCCS Cloud)
- `umrs-client` installed and on PATH
- `internal_genAI/bento/hmaeinfer/checkpoints/latest_checkpoint.pth` (pre-staged on compute node)
- **Your user must be enrolled on the server.** If you haven't been enrolled yet,
  ask an admin to run:
  ```bash
  umrs-client admin user add <your_user_config.json> --admin-token $UMRS_ADMIN_TOKEN
  ```
  Without enrollment, the server will reject your config upload and no MinIO
  buckets will be created (you'll see "No bucket" errors on `upload_dir`).

---

## Step-by-Step Demo

### 1. Open an SSH tunnel (skip if already open)

In a **dedicated terminal**, start the tunnel and leave it running:

```bash
cd deploy/ansible
./bootstrap.sh --profile rccs-cloud --tunnel
```

Open a **new terminal** for the remaining steps.

### 2. Go to the demo directory

```bash
cd umrs/demos_jens/12_hmae_direct
```

### 3. Verify the server is reachable

```bash
umrs-client ping
```

Expected output:

```
  ● Server ready  |  profile=127.0.0.1-direct  |  127.0.0.1:5555
```

### 4. Register the experience into your user config

The file `experience.json` in this directory defines all three HMAE modes.
Review it:

```bash
cat experience.json
```

Merge it into your active user config:

```bash
umrs-client config add-experience experience.json --update
```

This adds (or replaces) the `"Jens HMAE Direct"` experience in your user config
file at `~/.config/umrs/users/<your_user>.json`.

### 5. Reload so the server provisions the MinIO buckets

```bash
umrs-client reload
```

Then check that provisioning succeeded:

```bash
umrs-client status
```

### 6. Register the model in the model registry (optional)

The HMAE checkpoint is pre-staged on the compute node. To make the model
visible in `umrs-client model list` and the monitor Models tab, register it:

```bash
umrs-client model register hmae \
  /path/to/internal_genAI/bento/hmaeinfer/checkpoints \
  --type vision \
  --description "HMAE hierarchical masked auto-encoder"
```

Replace `/path/to/internal_genAI/bento/hmaeinfer/checkpoints` with the actual
path on the server where `latest_checkpoint.pth` resides.

If you have the checkpoint locally and want to copy it to the server staging
area **and** register in one step:

```bash
umrs-client model upload ./hmaeinfer/checkpoints \
  --name hmae \
  --type vision \
  --description "HMAE hierarchical masked auto-encoder" \
  --register-path /path/on/server/models/hmae
```

> **Note**: `--register-path` is needed when client and server have different
> filesystems (e.g. Mac → HPC). It tells the server which path to record.

Verify:

```bash
umrs-client model list
```

### 7. Stage input images

Create a staging directory and symlink images into it:

```bash
RUN_ID=$(date +%Y%m%d-%H%M%S)
UPLOAD_DIR=".upload_staging/${RUN_ID}"
mkdir -p "${UPLOAD_DIR}"
```

Link one sample image. Use Spring-8 concrete CT images if available:

```bash
# Option A: Spring-8 concrete CT slices (1024x1024 grayscale, preferred)
ln -sf "$(cd ../internal_genAI/bento/hmaeinfer/spring8concrete && pwd)/volume_170.png" "${UPLOAD_DIR}/"
```

```bash
# Option B: General sample images (fallback)
ln -sf "$(cd ../../demos/sample_data/images && pwd)/Cat_1.jpg" "${UPLOAD_DIR}/"
```

Verify:

```bash
ls -la "${UPLOAD_DIR}/"
```

### 8. Upload & run — MAE mode (masked reconstruction)

```bash
umrs-client upload_dir "${UPLOAD_DIR}" "jens-hmae-mae/${RUN_ID}"
```

### 9. Upload & run — Reconstruct mode (full reconstruction)

```bash
umrs-client upload_dir "${UPLOAD_DIR}" "jens-hmae-reconstruct/${RUN_ID}"
```

### 10. Upload & run — Features mode (encoder feature extraction)

```bash
umrs-client upload_dir "${UPLOAD_DIR}" "jens-hmae-features/${RUN_ID}"
```

> **Note**: Each upload dispatches the vision stack for that particular mode.
> You can run just one mode if you prefer — steps 8, 9, 10 are independent.

### 11. Wait and download results

Each Slurm job builds/starts the BentoML container, loads the HMAE checkpoint,
runs inference, and uploads results back to S3. This can take several minutes on
first run or ~1–2 minutes on subsequent runs.

Download results for each mode:

```bash
mkdir -p "results/${RUN_ID}/mae/artifacts"
umrs-client download_results "jens-hmae-mae/${RUN_ID}" "results/${RUN_ID}/mae/artifacts" --force-overwrite
```

```bash
mkdir -p "results/${RUN_ID}/reconstruct/artifacts"
umrs-client download_results "jens-hmae-reconstruct/${RUN_ID}" "results/${RUN_ID}/reconstruct/artifacts" --force-overwrite
```

```bash
mkdir -p "results/${RUN_ID}/features/artifacts"
umrs-client download_results "jens-hmae-features/${RUN_ID}" "results/${RUN_ID}/features/artifacts" --force-overwrite
```

> **Tip**: If no results appear yet, wait and retry the `download_results`
> command — the Slurm job may still be running.

### 12. Inspect results

List all downloaded files:

```bash
ls -la results/${RUN_ID}/*/artifacts/
```

Inspect a MAE result:

```bash
python3 -c "
import pickle, sys
with open(sys.argv[1], 'rb') as f:
    data = pickle.load(f)
print('status:', data.get('status'))
print('keys:  ', list(data.keys()))
if 'prediction' in data:
    print('prediction shape:', data['prediction'].shape)
if 'mask' in data:
    print('mask shape:', data['mask'].shape)
    print('loss:', data.get('loss'))
" results/${RUN_ID}/mae/artifacts/*.pkl
```

Inspect a features result:

```bash
python3 -c "
import pickle, sys
with open(sys.argv[1], 'rb') as f:
    data = pickle.load(f)
print('status:', data.get('status'))
print('keys:  ', list(data.keys()))
if 'features' in data:
    print('features shape:', data['features'].shape)
" results/${RUN_ID}/features/artifacts/*.pkl
```

---

## Using your own images

Replace step 7 with:

```bash
IMAGE_DIR=/path/to/your/images
RUN_ID=$(date +%Y%m%d-%H%M%S)
UPLOAD_DIR=".upload_staging/${RUN_ID}"
mkdir -p "${UPLOAD_DIR}"
find "${IMAGE_DIR}" -maxdepth 1 -type f \( -name '*.png' -o -name '*.jpg' -o -name '*.jpeg' \) \
  | head -n 3 \
  | while read -r f; do ln -sf "$(cd "$(dirname "$f")" && pwd)/$(basename "$f")" "${UPLOAD_DIR}/"; done
```

Then continue from step 8.

## Running the automated script (alternative)

Instead of the manual steps above, you can run everything in one go:

```bash
# Run all three modes
HMAE_MODE=all \
  MAX_IMAGES=1 \
  RESULT_DOWNLOAD_TIMEOUT=1500 \
  # Follow the step-by-step commands in README.md
```

```bash
# Run only MAE mode
HMAE_MODE=mae \
  MAX_IMAGES=1 \
  RESULT_DOWNLOAD_TIMEOUT=1500 \
  # Follow the step-by-step commands in README.md
```

---

## Environment Variable Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `HMAE_MODE` | `all` | `all`, `mae`, `reconstruct`, or `features` |
| `MAX_IMAGES` | `1` | Max images to process |
| `IMAGE_DIR` | *(auto-detect)* | Source images; prefers `spring8concrete/` if present |
| `RUN_ID` | `YYYYMMDD-HHMMSS` | Unique run identifier |
| `RESULT_DOWNLOAD_TIMEOUT` | `20` | Seconds to wait for results |
| `HMAE_REQUIRE_RESULTS` | `0` | Set to `1` to fail if no results |
| `UMRS_PROFILE` | *(none)* | Connection profile (optional) |
