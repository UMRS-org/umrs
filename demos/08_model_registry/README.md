# Demo 08 — Model Registration & Staging

This demo shows how to manage models in UMRS: register local models,
list available models, and download models from HuggingFace.

## Prerequisites

- UMRS server running (see `demos/00_start_services/`)
- `umrs-client` available (installed via `pip install -e src/`)
- `huggingface_hub` installed (for `model download` only)

## Commands

### 1. Download models from HuggingFace

The download runs **on the client machine** (login node), then registers
the path with the server. This works even if the compute nodes have no
internet access.

```bash
# Download + register in one step:
umrs-client model download intfloat/e5-base-v2

# Custom staging directory (e.g. shared filesystem):
umrs-client model download intfloat/e5-base-v2 --staging-dir /shared/models

# Or use the existing download script (pre-downloads to rivault/models/):
../slurm-dev/scripts/download_rivault_models.sh intfloat/e5-base-v2
```

### 2. Register a model that's already on disk

For **air-gapped machines** (no internet at all): copy model files manually
(USB, scp from another host, etc.), then register:

```bash
umrs-client model register e5-base-v2 /shared/models/e5-base-v2 \
    --type embedding --scope system --description "Multilingual embedding model"
```

### 3. List registered models

```bash
umrs-client model list
```

## How It Works

- **`model download`** runs entirely on the client side:
  1. Downloads from HuggingFace using `huggingface_hub.snapshot_download()`
     on the login node (which typically has internet access)
  2. Sends `MODEL_REGISTER` to the server to record the path
- **`model register`** sends `MODEL_REGISTER` over ZeroMQ — the server
  validates the path exists and adds it to `model_catalog.json`
- **`model list`** sends `MODEL_LIST` — the server returns the catalog
- The model catalog is stored as a JSON file under the UMRS config directory
  (`~/.config/umrs/model_catalog.json`).
- For production deployments with PostgreSQL, the catalog can be backed by
  `model_registry/store.py` (ModelStore) instead.

## Air-gapped / Restricted Network Support

If the login node also has no internet, models can be staged manually:

1. Download on an external machine with internet access
2. Copy to the shared filesystem (scp, rsync, USB)
3. Register: `umrs-client model register <name> /path/to/model`

## Run Demo

```bash
./run_demo.sh
```
