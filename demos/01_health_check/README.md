# Demo 01: Health Check

Quick connectivity check for the UMRS server.

## Prerequisites

- Services running (see [Demo 00](../00_start_services/README.md))
- UMRS client installed (`pip install .` from `src/`)
- User config JSON from your admin

## Quick Start

```bash
cd demos/01_health_check
bash run_demo.sh
```

## What This Demo Does

1. **Ping** the UMRS server to verify it is reachable
2. **List buckets** to confirm the user's S3 configuration
3. **Server status** to confirm backend services are healthy

## Running Manually

```bash
# Ping the server
python ../../src/client_umrs.py --conf ../../docker/conf_umrs.json ping

# List configured buckets
python ../../src/client_umrs.py --conf ../../docker/conf_umrs.json list_buckets
```

## Expected Output

```
Server is alive — UMRS is reachable.
Bucket list:
  bucket_alias_1  →  s3://bucket-name-1
```

If the ping fails, check that:
- The UMRS server is running (`demos/00_start_services/start_all.sh`)
- The config JSON has the correct `server_host` and `server_port`

## Next Steps

- [Demo 02: Text Embedding](../02_text_embedding/README.md) — upload text files and compute embeddings
- [Demo 03: SAM3 Segmentation](../03_sam3_segmentation/README.md) — image segmentation via SAM3
