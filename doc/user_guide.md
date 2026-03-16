# UMRS User Guide

Getting started with UMRS as an end-user: install the client, upload data,
compute embeddings, and use vision models.

## 1. Install the Client

The UMRS client is a Python package that provides the `umrs-client` CLI.

```bash
# Clone the repository (or get a copy of src/)
cd umrs/src

# Install with pip (in a virtual environment)
python3 -m venv ~/.venv/umrs
source ~/.venv/umrs/bin/activate   # or: . ~/.venv/umrs/bin/activate
pip install .

# Or with uv
uv pip install .
```

Verify:
```bash
umrs-client --help
```

## 2. Get Your Configuration

Your admin provides two files:

1. **User config** (`conf_umrs.json`) — defines your username, experiments,
   modalities, and bucket aliases.
2. **S3 credentials** (`.s3cfg`) — access key for the object store.

Place them where your config references them (typically `~/.s3cfg_<username>`).

Example user config:
```json
{
  "username": "alice",
  "fullname": "Alice Researcher",
  "s3_configuration_file": "~/.s3cfg_alice",
  "experiences": [
    {
      "xp_name": "TextAnalysis",
      "sql_username": "alice",
      "modalities": [
        {
          "modality_name": "text",
          "s3": { "bucket_alias": "alice-text" },
          "model": {
            "model_name": "intfloat/e5-base-v2",
            "vector_size": "768"
          },
          "sql": {
            "store_embeddings": "yes",
            "table_name": "text"
          }
        }
      ]
    }
  ]
}
```

## 3. Connect to the Server

```bash
# Check if the UMRS server is reachable
umrs-client -c conf_umrs.json --server-ip <SERVER_IP> ping
```

Expected output:
```json
{"status": "ok", "server_ready": true}
```

If your server is on a remote machine, use SSH port forwarding:
```bash
ssh -L 5555:localhost:5555 login-node
```

## 4. List Your Buckets

```bash
umrs-client -c conf_umrs.json list_buckets
```

This shows the bucket aliases defined in your config, mapped to S3 paths.

## 5. Upload Data

### Text Files

```bash
# Upload a single text file
umrs-client -c conf_umrs.json upload_file myfile.txt alice-text

# Upload all files in a directory
umrs-client -c conf_umrs.json upload_dir ./texts/ alice-text
```

### Images

For image modalities (segmentation, tomography):
```bash
umrs-client -c conf_umrs.json upload_file photo.jpg alice-images
umrs-client -c conf_umrs.json upload_dir ./images/ alice-images
```

Upload automatically triggers embedding computation on the server.

### Subdirectories

Use `/` in the bucket alias to organise data:
```bash
umrs-client -c conf_umrs.json upload_dir ./batch1/ alice-text/experiment-a
```

## 6. Check Processing Status

```bash
# All summaries
umrs-client -c conf_umrs.json status all

# Specific target
umrs-client -c conf_umrs.json status process_objects

# Human-readable output
umrs-client -c conf_umrs.json --output text status all
```

Wait until all tasks show `"Done"` before exporting results.

## 7. Export Embeddings

Once processing is complete, export computed embeddings:

```bash
# CSV format
umrs-client -c conf_umrs.json export_embeddings alice-text ./embeddings.csv

# JSONL format
umrs-client -c conf_umrs.json export_embeddings alice-text ./embeddings.jsonl --format jsonl
```

Database connection defaults to environment variables (`DB_HOST`, `DB_PORT`,
`POSTGRES_USER`, `POSTGRES_PASSWORD`) or can be overridden:
```bash
umrs-client -c conf_umrs.json export_embeddings alice-text ./out.csv \
    --db-host localhost --db-port 15432 --db-user umrs --db-password umrs_dev
```

## 8. Download Results

If your modality produces output files (e.g., segmentation masks):

```bash
umrs-client -c conf_umrs.json download_results alice-images ./results/ \
    --force-overwrite
```

This polls the server and downloads results as they become available.

## 9. Using Vision Models

Vision models (SAM3 segmentation, HMAE feature extraction) run inside
Mini-Rivault for security isolation. Your admin configures the vision profile;
from the user side the workflow is the same as any modality.

### SAM3 Image Segmentation

1. Your config includes an image modality with `model_type: "bentoml://..."`:
   ```json
   {
     "modality_name": "image",
     "s3": { "bucket_alias": "alice-images" },
     "model": {
       "model_name": "facebook/sam3",
       "vector_size": "256",
       "model_type": "bentoml://http://localhost:{LOCAL_PORT}/vision/submit"
     }
   }
   ```

2. Upload images — the server routes them through a Mini-Rivault instance with
   the `vision-bentoml` profile:
   ```bash
   umrs-client -c conf_umrs.json upload_dir ./images/ alice-images
   ```

3. Check status and download results as usual.

### HMAE Feature Extraction

Same workflow — your config specifies the HMAE model:
```json
{
  "modality_name": "tomography",
  "model": {
    "model_name": "HMAE ViT-B",
    "vector_size": "768",
    "model_type": "bentoml://http://localhost:{LOCAL_PORT}/vision/submit"
  }
}
```

## 10. Server Maintenance Commands

These tell the server to reprocess data (useful after model updates):

```bash
# Recompute all embeddings from scratch
umrs-client -c conf_umrs.json recompute

# Update only missing/changed embeddings
umrs-client -c conf_umrs.json update

# Check S3 ↔ database consistency
umrs-client -c conf_umrs.json check

# Reload server config (e.g., after admin adds new users)
umrs-client -c conf_umrs.json reload
```

## 11. Adding a Custom Processing Pipeline

To add a new model or modality:

1. **Pre-stage model weights** on the shared filesystem (coordinate with admin)
2. **Add a modality entry** in your user config JSON
3. **Register the model** in `vision_models.json` (for vision) or set
   `model_type` to an appropriate scheme (`openai://`, `bentoml://`)
4. **Reload** the server: `umrs-client -c conf_umrs.json reload`

For custom vision models, the admin adds them to the Mini-Rivault
`bentoml-vision` service and pre-stages weights. All vision inference is
sandboxed inside Mini-Rivault — no direct GPU access from the client.

## Quick Reference

| Command | Description |
|---------|-------------|
| `umrs-client ping` | Check server readiness |
| `umrs-client list_buckets` | Show available bucket aliases |
| `umrs-client upload_file <file> <bucket>` | Upload single file |
| `umrs-client upload_dir <dir> <bucket>` | Upload directory |
| `umrs-client status [target]` | Check processing status |
| `umrs-client download_results <bucket> <path>` | Download output files |
| `umrs-client export_embeddings <bucket> <path>` | Export embeddings to CSV/JSONL |
| `umrs-client recompute` | Recompute all embeddings |
| `umrs-client update` | Update missing embeddings |
| `umrs-client check` | S3 ↔ DB consistency check |
| `umrs-client reload` | Reload server configuration |

All commands accept `-c <config>` (config file), `--server-ip <ip>`, `--output json|text`, and `-v` (verbose).

## See Also

- [Admin Guide](admin_guide.md) — deployment and administration
- [UMRS Overview](umrs_overview.md) — architecture details
- [Demo: Text Embedding](../demos/01_text_embedding/README.md)
- [Demo: SAM3 Segmentation](../demos/02_sam3_segmentation/README.md)
- [Demo: HMAE Inference](../demos/03_hmae_inference/README.md)
