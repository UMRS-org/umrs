# Demo 11: SAM3 Text-Prompt Segmentation

Upload images through UMRS → Slurm dispatches `internal_genAI` on GPU → SAM3 segments objects matching a text prompt → results downloaded as pickle files.

---

## Prerequisites

- UMRS server running with `VISION_STACK=jens`
- SSH tunnel open (if running remotely against RCCS Cloud)
- `umrs` installed and on PATH
- `internal_genAI/bento/sam3infer/` contains `bpe.gz` and `sam3.pt` (pre-staged on compute node)
- **Your user must be enrolled on the server.** If you haven't been enrolled yet,
  ask an admin to run:
  ```bash
  umrs admin user add <your_user_config.json> --admin-token $UMRS_ADMIN_TOKEN
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
cd umrs/demos_jens/11_sam3_direct
```

### 3. Verify the server is reachable

```bash
umrs ping
```

Expected output:

```
  ● Server ready  |  profile=127.0.0.1-direct  |  127.0.0.1:5555
```

### 4. Register the experience into your user config

The file `experience.json` in this directory describes the SAM3 segmentation task.
Edit `vision_model_config.prompt` inside it if you want a different prompt (default: `"cat"`):

```bash
cat experience.json
```

Merge it into your active user config:

```bash
umrs experience add experience.json
```

This adds (or replaces) the `"Jens SAM3 Direct"` experience in your user config
file at `~/.config/umrs/users/<your_user>.json`.

### 5. Reload so the server provisions the MinIO bucket

```bash
umrs reload
```

Then check that provisioning succeeded:

```bash
umrs status
```

### 6. Register the model in the model registry (optional)

The SAM3 model weights are pre-staged on the compute node. To make the model
visible in `umrs model list` and the monitor Models tab, register it:

```bash
umrs model register sam3 \
  /path/to/internal_genAI/bento/sam3infer \
  --type vision \
  --description "SAM3 text-prompt segmentation"
```

Replace `/path/to/internal_genAI/bento/sam3infer` with the actual path on the
server where `sam3.pt` and `bpe.gz` reside.

If you have the model weights locally and want to copy them to the server
staging area **and** register in one step:

```bash
umrs model upload ./sam3infer \
  --name sam3 \
  --type vision \
  --description "SAM3 text-prompt segmentation" \
  --register-path /path/on/server/models/sam3
```

> **Note**: `--register-path` is needed when client and server have different
> filesystems (e.g. Mac → HPC). It tells the server which path to record.

Verify:

```bash
umrs model list
```

### 7. Stage input images

Create a staging directory and symlink images into it:

```bash
RUN_ID=$(date +%Y%m%d-%H%M%S)
UPLOAD_DIR=".upload_staging/${RUN_ID}"
mkdir -p "${UPLOAD_DIR}"
```

Link one sample image (or replace the path with your own images):

```bash
ln -sf "$(cd ../../demos/sample_data/images && pwd)/Cat_1.jpg" "${UPLOAD_DIR}/"
```

Verify:

```bash
ls -la "${UPLOAD_DIR}/"
```

### 8. Upload images to UMRS

```bash
umrs data dir-upload "${UPLOAD_DIR}" "jens-sam3-direct/${RUN_ID}"
```

This uploads the image(s) to MinIO bucket `jens-sam3-direct/<RUN_ID>/` and tells
the UMRS server to dispatch the `internal_genAI` vision stack via Slurm.

### 9. Wait and download results

The Slurm job builds/starts the BentoML container, loads SAM3, runs inference,
and uploads results back to S3. This can take several minutes on first run
(container image build) or ~1–2 minutes on subsequent runs.

```bash
mkdir -p "results/${RUN_ID}/artifacts"
umrs data download "jens-sam3-direct/${RUN_ID}" "results/${RUN_ID}/artifacts" --force-overwrite
```

> **Tip**: If no results appear yet, wait and retry the `download_results`
> command — the Slurm job may still be running.

### 10. Inspect results

List downloaded files:

```bash
ls -la "results/${RUN_ID}/artifacts/"
```

Inspect a pickle result:

```bash
python3 -c "
import pickle, sys
with open(sys.argv[1], 'rb') as f:
    data = pickle.load(f)
print('status:', data.get('status'))
print('keys:  ', list(data.keys()))
if 'scores' in data:
    print('scores:', data['scores'])
    print('boxes: ', data['boxes'])
" results/${RUN_ID}/artifacts/*.pkl
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

## Changing the text prompt

Edit `experience.json` — change `"prompt"` under `vision_model_config`:

```json
"vision_model_config": { "prompt": "dog" }
```

Then re-register:

```bash
umrs config add-experience experience.json --update
```

## Running the automated script (alternative)

Instead of the manual steps above, you can run everything in one go:

```bash
SAM3_PROMPT=cat \
  MAX_IMAGES=1 \
  RESULT_DOWNLOAD_TIMEOUT=1500 \
  # See the numbered steps above in this README
```

---

## Environment Variable Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `SAM3_PROMPT` | `cat` | Text prompt for segmentation |
| `MAX_IMAGES` | `1` | Max images to process |
| `IMAGE_DIR` | `../../demos/sample_data/images` | Source image directory |
| `SAM3_BUCKET` | `jens-sam3-direct` | MinIO bucket alias |
| `RUN_ID` | `YYYYMMDD-HHMMSS` | Unique run identifier |
| `RESULT_DOWNLOAD_TIMEOUT` | `20` | Seconds to wait for results |
| `UMRS_PROFILE` | *(none)* | Connection profile (optional) |
