# Demo 08 — Model Registration & Staging

This demo shows how to manage models in UMRS: register local models,
list available models, and download models from HuggingFace.

## Prerequisites

- UMRS server running (see `demos/00_start_services/`)
- `umrs-client` available (installed via `pip install -e src/`)

## Commands

### 1. Download models from HuggingFace

```bash
# Uses the existing download script (pre-downloads to shared dir):
../slurm-dev/scripts/download_rivault_models.sh intfloat/e5-base-v2

# Or via umrs-client (server downloads and registers):
python3 ../../src/client_umrs.py model download intfloat/e5-base-v2
```

### 2. Register a local model

```bash
python3 ../../src/client_umrs.py model register e5-base-v2 /path/to/models/e5-base-v2 \
    --type embedding --scope system --description "Multilingual embedding model"
```

### 3. List registered models

```bash
python3 ../../src/client_umrs.py model list
```

## How It Works

- **Client** sends `MODEL_REGISTER`, `MODEL_LIST`, or `MODEL_DOWNLOAD` commands
  over ZeroMQ to the UMRS server.
- **Server** handles the command via `model_registry_handler.py`:
  - `MODEL_REGISTER` — validates the path and adds to `model_catalog.json`
  - `MODEL_LIST` — reads and returns the catalog
  - `MODEL_DOWNLOAD` — downloads from HuggingFace using `snapshot_download`,
    then registers
- The model catalog is stored as a JSON file under the UMRS config directory
  (`~/.config/umrs/model_catalog.json`).
- For production deployments with PostgreSQL, the catalog can be backed by
  `model_registry/store.py` (ModelStore) instead.

## Run Demo

```bash
./run_demo.sh
```
