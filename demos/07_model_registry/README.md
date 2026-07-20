# Demo 07 — Model Registration & Staging

This demo shows how to manage models in UMRS: register, list, and download
models — including user's own trained models.

## Prerequisites

- UMRS server running and reachable
- `umrs` available (see [demos/README.md](../README.md#prerequisites))
- User onboarded (see [Demo 00](../00_onboarding/))
- `huggingface_hub` installed (for `model download` only)

## Commands

### 1. Download models from HuggingFace

By default, the download runs on the **client machine**, then registers that
path with the server. This only works when the client and server share the
same filesystem view (for example, the client is the login node itself or the
model directory lives on a shared mount).

```bash
# Client-side download (only when the server can also see the downloaded path):
umrs model download intfloat/e5-base-v2 --scope user

# Custom staging directory (only if your site exposes a writable, allowed root):
umrs model download intfloat/e5-base-v2 --scope user --staging-dir /path/to/site-model-root

# Server-side download (recommended for Mac-local clients talking to a remote server):
umrs model download intfloat/e5-base-v2 --scope user --server-side \
  --staging-dir /path/to/site-user-model-root
```

### 2. Register a model that's already on disk

For **air-gapped machines** or **user's own trained models**:

```bash
# System-wide model (admin only):
umrs model register e5-base-v2 /shared/models/e5-base-v2 \
    --type embedding --scope system --description "Multilingual embedding model"

# User's own model (only visible to the user):
umrs model register my-finetuned-mae /home/user/models/mae-finetuned \
    --type vision --scope user --description "Fine-tuned MAE for defect detection"

# Project-scoped model (visible to project members):
umrs model register project-vit /shared/projects/spring8/models/vit \
    --type vision --scope project --description "ViT trained on Spring-8 data"
```

### 3. List registered models

```bash
# List all models:
umrs model list

# Filter by scope:
umrs model list --scope user
umrs model list --scope system
```

## Manual Equivalent to the Demo Script

```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$(git rev-parse --show-toplevel)/$(test -d "$(git rev-parse --show-toplevel)/umrs/demos" && echo umrs)/demos/07_model_registry"

CONF="${UMRS_USER_CONF:-$HOME/.config/umrs/users/user1.json}"
MODEL_SCOPE="${MODEL_SCOPE:-user}"
MODEL_TYPE="${MODEL_TYPE:-embedding}"
umrs config use --user "$CONF"

# 0. Verify connectivity.
umrs ping

# 1. List the current catalog.
umrs model list || true

# 2. Optionally register a model from a server-visible path.
if [[ -n "${UMRS_SHARED_MODEL_PATH:-}" ]]; then
  MODEL_NAME="$(basename "$UMRS_SHARED_MODEL_PATH")"
  umrs model register "$MODEL_NAME" "$UMRS_SHARED_MODEL_PATH" \
    --type "$MODEL_TYPE" --scope "$MODEL_SCOPE" \
    --description "Auto-registered from shared models directory"
fi

# 3. Optionally upload a tiny test model.
if [[ "${DEMO_LOCAL_MODEL_UPLOAD:-0}" = "1" ]]; then
  rm -rf .test_model
  mkdir -p .test_model
  printf '{"model_type":"test","framework":"demo"}\n' > .test_model/config.json
  printf 'dummy weights\n' > .test_model/model.bin
  umrs model upload .test_model \
    --name test-upload-model \
    --type embedding \
    --scope user \
    --description "Demo model uploaded via model upload" \
    --force
fi

# 4. Verify the catalog and scope filters.
umrs model list
umrs model list --scope system || true
umrs model list --scope user || true

# 5. Optional server-side download.
if [[ "${DEMO_DOWNLOAD:-0}" = "1" && -n "${MODEL_STAGING_DIR:-}" ]]; then
  umrs model download intfloat/e5-base-v2 \
    --scope "$MODEL_SCOPE" \
    --server-side \
    --staging-dir "$MODEL_STAGING_DIR"
fi
```

## Model Scopes

| Scope | Who can see | Typical path | Use case |
|-------|------------|--------------|----------|
| `system` | All users | Site-managed shared root | Pre-trained models (e5, SAM3, ...) |
| `project` | Project members | `/shared/projects/<id>/models/` | Team-trained models |
| `user` | Owner only | `/home/<user>/models/` | Personal experiments |

## How It Works

- **`model download`** (default: client-side):
  1. Downloads from HuggingFace on the client machine
  2. Sends `MODEL_REGISTER` to the server to record the path
  3. Therefore it only succeeds when that path is also visible on the server
- **`model download --server-side`**:
  1. Sends `MODEL_DOWNLOAD` to the server
  2. Server downloads in a **background thread** (non-blocking)
  3. Auto-registers when download completes
- **`model register`** — records any local path in the model catalog
  (validates the path exists on the server's filesystem)
- **`model list`** — returns the catalog, optionally filtered by scope

The model catalog is stored at `~/.config/umrs/model_catalog.json`.
For production with PostgreSQL, the full `model_registry/store.py`
(ModelStore + ACL + PathValidator) can be used instead.

## User's Own Model Workflow

1. **Train** your model anywhere (local machine, Jupyter, Slurm job)
2. **Copy** the weights to a path visible to the UMRS server.
   On Mac → HPC workflows, this usually means staging the directory on the
   server first, then using `model register`, or using
   `model download --server-side` with a site-approved staging root.
3. **Register**: `umrs model register my-model /path/to/weights --scope user`
4. **Use** in your config JSON by referencing the registered name

## Air-gapped / Restricted Network Support

If no machine has internet access:

1. Download on an external machine with internet
2. Copy to the shared filesystem (scp, rsync, USB)
3. Register: `umrs model register <name> /path/to/model`

## Run Demo

```bash
# See the "Running Manually" section above for step-by-step commands
umrs model list --scope system
umrs model list --scope user
```
