# UMRS Demo Guide — Step by Step

This guide walks you through every UMRS demo from start to finish.
Each section explains what the demo does, how to run it manually (command
by command), and what to expect.

The demos work with **any deployment** — Docker Compose (local dev),
rccs-cloud, hpbd-atom, or any other setup — as long as the UMRS server is
reachable from your client machine.

> **Automated run:** To run all core demos automatically, use
> `bash demos/run_all.sh` (see [run_all.sh](../demos/run_all.sh)).

---

## Table of Contents

1. [Setup](#setup)
2. [Demo 01 — Health Check](#demo-01--health-check)
3. [Demo 02 — Text Embedding](#demo-02--text-embedding)
4. [Demo 03 — SAM3 Segmentation](#demo-03--sam3-segmentation)
5. [Demo 04 — Vision Inference](#demo-04--vision-inference)
6. [Demo 05 — RAG Search](#demo-05--rag-search)
7. [Demo 06 — Server Maintenance](#demo-06--server-maintenance)
8. [Demo 07 — Model Registry](#demo-07--model-registry)
9. [Demo 08 — User-Defined Models](#demo-08--user-defined-models)
10. [Demo 09 — Multi-Step Vision Pipeline](#demo-09--multi-step-vision-pipeline)
11. [Demo 10 — Multi-Mode Vision Pipeline](#demo-10--multi-mode-vision-pipeline)
12. [Demo 11 — Job Submission](#demo-11--job-submission)
13. [Demo 12 — Pipeline SDK](#demo-12--pipeline-sdk)
14. [Demo 13 — Plugins](#demo-13--plugins)
15. [Demo 14 — Video Analysis](#demo-14--video-analysis)
16. [Demo 15 — Complex DAG](#demo-15--complex-dag)
17. [Demo 16 — Custom Workflow](#demo-16--custom-workflow)
18. [Demo 17 — Multi-User Isolation](#demo-17--multi-user-isolation)
19. [Demo 18 — Model Lifecycle](#demo-18--model-lifecycle)
20. [Demo 19 — RAG Advanced](#demo-19--rag-advanced)
21. [Demo 20 — Durable Jobs](#demo-20--durable-jobs)
22. [Demo 21 — Remote Plugins](#demo-21--remote-plugins)
23. [Demo 22 — Monitor TUI](#demo-22--monitor-tui)
24. [Environment Variables Reference](#environment-variables-reference)
25. [Troubleshooting](#troubleshooting)

---

## Setup

### 1. Install the UMRS client

```bash
cd src
pip install -e .
which umrs-client          # should print a path
```

Or use a site-specific venv if your admin provides one.

### 2. Ensure the server is running

The server can be running anywhere — locally via Docker Compose, on an HPC
login node, or in the cloud. You just need network access to it.

**Local (Docker Compose):**
```bash
cd docker
docker compose up -d
```

**Remote (any deployment):**
Your admin will provide instructions. The server needs to expose these ports:

| Port | Service | Required? |
|------|---------|-----------|
| 5555 | ZMQ control channel | Yes |
| 5556 | ZMQ data channel | Yes |
| 9000 | MinIO S3 | Yes |
| 8081 | RAG service | Only for RAG demos (05, 19) |

### 3. Set up SSH tunnels (remote servers only)

If the server is on a remote machine, tunnel the required ports:

```bash
ssh -N \
    -L 5555:localhost:5555 \
    -L 5556:localhost:5556 \
    -L 9000:localhost:9000 \
    -L 8081:localhost:8081 \
    <your-server-host>
```

Site-specific tunnel scripts may exist at `deploy/<site>/ssh_tunnel.sh`.

### 4. Set up the server configuration

Make sure you have a server config file. The default location is:
```
~/.config/umrs/conf_umrs.json
```

Site-specific `setup_client.sh` scripts create this. If running locally
with Docker Compose, copy the example:
```bash
mkdir -p ~/.config/umrs/users
cp docker/user_files/user1.json ~/.config/umrs/users/user1.json
```

### 5. Verify connectivity

```bash
umrs-client -c ~/.config/umrs/conf_umrs.json ping
# Expected: server responds with "ready"
```

---

## Demo 01 — Health Check

**Purpose:** Verify that all UMRS services are reachable.

**Requires server:** Yes  
**Time:** < 30 seconds

### Automated
```bash
cd demos/01_health_check
bash run_demo.sh
```

### Manual Steps

```bash
# 1. Check port connectivity
nc -z localhost 5555 && echo "ZMQ control OK" || echo "ZMQ FAILED"
nc -z localhost 5556 && echo "ZMQ data OK"    || echo "ZMQ data not bound (OK)"
curl -sf http://localhost:9000/minio/health/live && echo "MinIO OK" || echo "MinIO FAILED"

# 2. Ping the UMRS server
umrs-client -c ~/.config/umrs/conf_umrs.json ping

# 3. Check RAG service (optional)
curl -sf http://localhost:8081/docs && echo "RAG OK" || echo "RAG not available"

# 4. List configured buckets
umrs-client -c ~/.config/umrs/conf_umrs.json list_buckets
```

### Expected output
```
✓ ZMQ control  (localhost:5555)
✓ MinIO S3     (localhost:9000)
Server is ready.
```

### Next step → Demo 02

---

## Demo 02 — Text Embedding

**Purpose:** Upload text files, compute embeddings server-side, and export
the resulting vectors to CSV.

**Requires server:** Yes  
**Time:** 1–5 minutes (depends on model and number of files)

### Automated
```bash
cd demos/02_text_embedding
bash run_demo.sh
```

### Manual Steps

```bash
# 1. Ensure the embedding model is available
umrs-client -c ~/.config/umrs/conf_umrs.json \
    model download intfloat/e5-base-v2 --scope system --server-side

# 2. Upload sample text files
umrs-client -c ~/.config/umrs/conf_umrs.json \
    upload_dir ../sample_data/texts user1-textfiles

# 3. Wait for processing to finish
umrs-client -c ~/.config/umrs/conf_umrs.json status all
# Repeat until status shows "completed" or "done"

# 4. Export embeddings to CSV
umrs-client -c ~/.config/umrs/conf_umrs.json \
    export_embeddings user1-textfiles results/embeddings.csv --format csv
```

### What happens behind the scenes

1. Text files are uploaded to MinIO (S3) in the configured bucket
2. Server detects new files, chunks them, and sends them to the embedding model
3. Resulting vectors are stored in pgvector (PostgreSQL)
4. `export_embeddings` reads from pgvector and writes to CSV

### Next step → Demo 03

---

## Demo 03 — SAM3 Segmentation

**Purpose:** Upload images and run SAM3 segmentation on HPC via BentoML.

**Requires server:** Yes (with vision service on a GPU node)  
**Time:** 3–10 minutes (Slurm job launch + model loading + inference)

### Automated
```bash
cd demos/03_sam3_segmentation
bash run_demo.sh
```

### Manual Steps

```bash
# 1. Pre-flight
umrs-client -c ~/.config/umrs/conf_umrs.json ping

# 2. Reload config (ensures SAM3 experience is registered)
umrs-client -c ~/.config/umrs/conf_umrs.json reload

# 3. Download the SAM3 model weights
umrs-client -c ~/.config/umrs/conf_umrs.json \
    model download facebook/sam2-hiera-small --name sam3 --scope system --server-side

# 4. Upload images
umrs-client -c ~/.config/umrs/conf_umrs.json \
    upload_dir test_data_full user1-sam3

# 5. Wait for results and download
umrs-client -c ~/.config/umrs/conf_umrs.json \
    download_results user1-sam3 results/ --force-overwrite

# 6. Check results
ls -lh results/
```

### Force recompute
```bash
FORCE_RECOMPUTE=1 bash run_demo.sh
```

### Next step → Demo 04

---

## Demo 04 — Vision Inference

**Purpose:** Upload images and run a vision model (SAM3 or any registered model)
via BentoML on HPC. Demonstrates the generic vision inference interface.

**Requires server:** Yes (with vision service on a GPU node)  
**Time:** 3–10 minutes

### Automated
```bash
cd demos/04_vision_inference
bash run_demo.sh
```

### Manual Steps

```bash
# 1. Pre-flight + reload config
umrs-client -c ~/.config/umrs/conf_umrs.json ping
umrs-client -c ~/.config/umrs/conf_umrs.json reload

# 2. Upload images
umrs-client -c ~/.config/umrs/conf_umrs.json \
    upload_dir ../sample_data/images user1-sam3

# 3. Download results
umrs-client -c ~/.config/umrs/conf_umrs.json \
    download_results user1-sam3 results/ --force-overwrite
```

### Behind the scenes

- By default, Demo 04 runs SAM3 segmentation (same model as Demo 03)
- To use a different model: register it via Demo 07 (`model register`),
  then set `VISION_BUCKET` to the matching bucket alias

### Next step → Demo 05

---

## Demo 05 — RAG Search

**Purpose:** Semantic search over your embedded text data using vector search,
hybrid search (BM25 + vector with RRF), cross-encoder reranking, and Q&A.

**Requires server:** Yes + RAG service (port 8081)  
**Prerequisite:** Run Demo 02 first to have embeddings  
**Time:** < 1 minute

### Automated
```bash
cd demos/05_rag_search
bash run_demo.sh
```

### Manual Steps

```bash
# 1. Verify RAG service is available
curl -sf http://localhost:8081/docs && echo "RAG OK"

# 2. Vector search
curl -s -X POST http://localhost:8081/rag/search \
    -H 'Content-Type: application/json' \
    -d '{"query": "What numerical methods are used for seismic simulation?", "top_k": 3}'

# 3. Hybrid search (BM25 + vector)
curl -s -X POST http://localhost:8081/rag/search \
    -H 'Content-Type: application/json' \
    -d '{"query": "GPU acceleration for HPC", "top_k": 3, "search_method": "hybrid"}'

# 4. Search with reranking
curl -s -X POST http://localhost:8081/rag/search \
    -H 'Content-Type: application/json' \
    -d '{"query": "What is RiVault?", "top_k": 3, "enable_reranking": true}'

# 5. Q&A (requires LLM backend)
curl -s -X POST http://localhost:8081/rag/qa \
    -H 'Content-Type: application/json' \
    -d '{"query": "How does containerization help reproducibility on HPC clusters?", "top_k": 5}'
```

### Next step → Demo 06

---

## Demo 06 — Server Maintenance

**Purpose:** Data integrity operations — list buckets, check S3/DB consistency,
update missing embeddings, recompute all embeddings.

**Requires server:** Yes  
**Time:** 1–3 minutes

### Automated
```bash
cd demos/06_server_maintenance
bash run_demo.sh
```

### Manual Steps

```bash
# 1. List all buckets
umrs-client -c ~/.config/umrs/conf_umrs.json list_buckets

# 2. Check S3 ↔ DB consistency
umrs-client -c ~/.config/umrs/conf_umrs.json check

# 3. Update: fix missing/mismatched embeddings
umrs-client -c ~/.config/umrs/conf_umrs.json update

# 4. (Optional) Full recompute
DEMO_RECOMPUTE=1 umrs-client -c ~/.config/umrs/conf_umrs.json recompute
```

### Next step → Demo 07

---

## Demo 07 — Model Registry

**Purpose:** Register models from the shared filesystem, upload models from
your local machine, and manage the model registry.

**Requires server:** Yes  
**Time:** 1–2 minutes

### Automated
```bash
cd demos/07_model_registry
bash run_demo.sh
```

### Manual Steps

```bash
# 1. List currently registered models
umrs-client -c ~/.config/umrs/conf_umrs.json model list

# 2. Register a model already on the server filesystem
umrs-client -c ~/.config/umrs/conf_umrs.json \
    model register "my-model" /path/on/server \
    --type embedding --scope system --description "My model"

# 3. Upload a model from your local machine
umrs-client -c ~/.config/umrs/conf_umrs.json \
    model upload /local/path/to/model \
    --name "my-model" --type embedding --scope user \
    --description "Uploaded model" --force

# 4. (Optional) Download from HuggingFace
umrs-client -c ~/.config/umrs/conf_umrs.json \
    model download intfloat/e5-base-v2 --scope system --server-side

# 5. Filter by scope
umrs-client -c ~/.config/umrs/conf_umrs.json model list --scope system
umrs-client -c ~/.config/umrs/conf_umrs.json model list --scope user
```

### Next step → Demo 08

---

## Demo 08 — User-Defined Models

**Purpose:** Create custom processing steps using the Pipeline SDK without
modifying UMRS server code. Runs locally (no server needed).

**Requires server:** No (runs tests locally)  
**Time:** < 1 minute

### Automated
```bash
cd demos/08_user_model
bash run_demo.sh
```

### Manual Steps

```bash
# 1. Run the unit tests
cd demos/08_user_model
PYTHONPATH=../../src:$PYTHONPATH python3 -m pytest -v

# 2. Examine the example step
cat example_scripts/text_stats.py

# 3. Examine the workflow definition
cat workflow.json
```

### Key concepts
- A **Step** is a Python class that processes objects (files) from S3
- A **Workflow** chains multiple steps with a JSON definition
- Steps run locally or remotely (Slurm, K8s) without changing code

### Next step → Demo 09

---

## Demo 09 — Multi-Step Vision Pipeline

**Purpose:** 3-step chained pipeline: segmentation → crop regions → image
statistics.  Demonstrates the UMRS Step/workflow.json pattern.

**Requires server:** Yes (with vision service)  
**Time:** 5–15 minutes

### Automated
```bash
cd demos/09_multistep_vision_pipeline
bash run_demo.sh
```

### Manual Steps

```bash
# 1. Upload workflow
umrs-client -c ~/.config/umrs/conf_umrs.json \
    upload-workflow .

# 2. Reload config
umrs-client -c ~/.config/umrs/conf_umrs.json reload

# 3. Upload images
umrs-client -c ~/.config/umrs/conf_umrs.json \
    upload_dir ../sample_data/images user1-pipeline

# 4. Download results
umrs-client -c ~/.config/umrs/conf_umrs.json \
    download_results user1-pipeline results/
```

### Pipeline steps

1. **SegmentStep** — detect objects via BentoML vision model (GPU)
2. **CropRegionsStep** — extract bounding-box patches (CPU)
3. **ImageStatsStep** — per-patch RGB statistics + aggregate report (CPU)

### Next step → Demo 10

---

## Demo 10 — Multi-Mode Vision Pipeline

**Purpose:** Run 3 analysis modes in **parallel** on the same images:
segmentation, feature extraction, and thumbnails.

**Requires server:** Yes (with vision service)  
**Time:** 5–15 minutes

### Automated
```bash
cd demos/10_vision_multi_mode
bash run_demo.sh
```

### How parallel execution works

The `workflow.json` declares all three branches as `parallel`, so they
receive the same input keys and run concurrently.  GPU branches share
the BentoML service; the thumbnail branch runs on CPU.

### Next step → Demo 11

---

## Demo 11 — Job Submission

**Purpose:** Explicit job submit/cancel, endpoint warm-up, explicit cleanup,
retry policies.

**Requires server:** Yes  
**Time:** 1–3 minutes

### Automated
```bash
cd demos/11_job_submission
bash run_demo.sh
```

### Manual Steps

```bash
# 1. Submit a job
umrs-client -c ~/.config/umrs/conf_umrs.json \
    job submit endpoint_provision \
    --spec '{"model": "sam3"}' \
    --project demo

# 2. Check job status
umrs-client -c ~/.config/umrs/conf_umrs.json status all

# 3. Cancel a running job
umrs-client -c ~/.config/umrs/conf_umrs.json \
    job cancel <job_id>
```

`UMRS_SERVICE_LIFECYCLE` still applies to upload-triggered processing in the
server, but the durable endpoint path shown here is validated as explicit
submit, reuse, and cancel rather than `--keep-alive`-driven lifecycle.

### Next step → Demo 12

---

## Demo 12 — Pipeline SDK

**Purpose:** Demonstrate the 6-phase pipeline lifecycle using the LOCAL
backend. No server needed.

**Phases:** configure → ingest → compute → collect → verify → present

**Requires server:** No (LOCAL backend)  
**Time:** < 1 minute

### Automated
```bash
cd demos/12_pipeline_sdk
bash run_demo.sh
```

### Manual Steps

```bash
cd demos/12_pipeline_sdk

# Run the example pipeline against sample text data
python3 text_analysis_pipeline.py --input-dir ../sample_data/texts

# To use the UMRS backend instead of LOCAL:
# 1. Edit text_analysis_pipeline.py:
#    ComputeBackend.LOCAL  →  ComputeBackend.UMRS_EMBEDDING
# 2. Re-run the script
```

### Next step → Demo 13

---

## Demo 13 — Plugins

**Purpose:** Sandboxed plugin execution — validate, run, and audit plugins
on Slurm/K8s backends.

**Requires server:** No (simulated locally)  
**Time:** < 1 minute

### Automated
```bash
cd demos/13_plugins
bash run_demo.sh
```

### Manual Steps

```bash
cd demos/13_plugins
python3 run_plugin_demo.py
```

### Next step → Demo 14

---

## Demo 14 — Video Analysis

**Purpose:** Detect motion events in video files using OpenCV frame
differencing.  No external model or GPU required — runs entirely on CPU.

**Requires server:** Yes  
**Time:** 2–10 minutes

### Automated
```bash
cd demos/14_video_analysis

# Generate synthetic test videos + analyse
bash run_demo.sh

# Your own videos
bash run_demo.sh /path/to/your/videos
```

### Manual Steps

```bash
# 1. Generate test videos (optional)
python generate_test_videos.py -o sample_videos/ --count 4

# 2. Upload workflow
umrs-client -c ~/.config/umrs/conf_umrs.json \
    upload-workflow video_workflow/

# 3. Upload videos
umrs-client -c ~/.config/umrs/conf_umrs.json \
    upload_dir sample_videos/ user1-video

# 4. Download results
umrs-client -c ~/.config/umrs/conf_umrs.json \
    download_results user1-video results/
```

### Next step → Demo 15

---

## Demo 15 — Complex DAG

**Purpose:** Multi-branch DAG pipeline with parallel execution and merging.
Runs locally.

**Requires server:** No  
**Time:** < 1 minute

### Automated
```bash
cd demos/15_complex_dag
bash run_demo.sh
```

### Next step → Demo 16

---

## Demo 16 — Custom Workflow

**Purpose:** Upload a complete workflow directory to the server and execute it.

**Requires server:** Yes  
**Time:** 1–3 minutes

### Automated
```bash
cd demos/16_custom_workflow
bash run_demo.sh
```

### Manual Steps

```bash
# 1. Upload your workflow
umrs-client -c ~/.config/umrs/conf_umrs.json \
    upload-workflow ./my_workflow --name my-workflow

# 2. List workflows
umrs-client -c ~/.config/umrs/conf_umrs.json workflow list
```

### Next step → Demo 17

---

## Demo 17 — Multi-User Isolation

**Purpose:** Prove that two users (user1, user2) have isolated S3 buckets
and databases.

**Requires server:** Yes  
**Requires:** Two user configs (`user1.json`, `user2.json`)  
**Time:** 1–2 minutes

### Automated
```bash
cd demos/17_multi_user
bash run_demo.sh
```

### Manual Steps

```bash
# User 1: upload some data
umrs-client -c ~/.config/umrs/conf_umrs.json \
    upload_file myfile.txt user1-textfiles

# User 2: try to list User 1's buckets (should fail or show empty)
umrs-client -c ~/.config/umrs/users/user2.json list_buckets
```

### Next step → Demo 18

---

## Demo 18 — Model Lifecycle

**Purpose:** Full model lifecycle: register → list → use → deprecate.

**Requires server:** Yes  
**Time:** 1–2 minutes

### Automated
```bash
cd demos/18_model_lifecycle
bash run_demo.sh
```

### Next step → Demo 19

---

## Demo 19 — RAG Advanced

**Purpose:** Side-by-side comparison of all search methods: vector, BM25,
hybrid (RRF), and reranked.

**Requires server:** Yes + RAG service (port 8081)  
**Prerequisite:** Run Demo 02 first  
**Time:** < 1 minute

### Automated
```bash
cd demos/19_rag_advanced
bash run_demo.sh
```

### Next step → Demo 20

---

## Demo 20 — Durable Jobs

**Purpose:** Job submit, poll, idempotent resubmit, cancel — demonstrates
the durable job system.

**Requires server:** Yes  
**Time:** 1–2 minutes

### Automated
```bash
cd demos/20_durable_jobs
bash run_demo.sh
```

### Next step → Demo 21

---

## Demo 21 — Remote Plugins

**Purpose:** Define plugin specs, validate them, execute, and view audit logs.

**Requires server:** No (simulated locally)  
**Time:** < 1 minute

### Automated
```bash
cd demos/21_remote_plugins
bash run_demo.sh
```

### Next step → Demo 22

---

## Demo 22 — Monitor TUI

**Purpose:** Live-refreshing terminal dashboard showing job status with
keyboard navigation.

**Requires server:** Yes  
**Time:** Interactive (Ctrl+C to exit)

### Automated
```bash
cd demos/22_monitor
bash run_demo.sh
```

### Manual
```bash
umrs-client -c ~/.config/umrs/conf_umrs.json monitor
```

---

## Environment Variables Reference

These variables can be set before running any demo to override defaults.

### Connection

| Variable | Default | Description |
|----------|---------|-------------|
| `UMRS_CONF` | `~/.config/umrs/conf_umrs.json` | Server config JSON path |
| `UMRS_ZMQ_HOST` | `localhost` | ZMQ server host |
| `UMRS_ZMQ_PORT` | `5555` | ZMQ control port |
| `UMRS_ZMQ_DATA_PORT` | `5556` | ZMQ data port |
| `UMRS_MINIO_HOST` | `localhost` | MinIO host |
| `UMRS_MINIO_PORT` | `9000` | MinIO port |
| `RAG_URL` | `http://localhost:8081` | RAG service URL |

### Data & Buckets

| Variable | Default | Used by |
|----------|---------|---------|
| `TEXT_BUCKET` | `user1-textfiles` | Demo 02 |
| `SAM3_BUCKET` | `user1-sam3` | Demo 03 |
| `VISION_BUCKET` | `user1-sam3` | Demo 04 |
| `PIPELINE_BUCKET` | `user1-pipeline` | Demo 09 |
| `MULTI_BUCKET` | `user1-multi` | Demo 10 |
| `VIDEO_BUCKET` | `user1-video` | Demo 14 |
| `USER1_BUCKET` | `user1-textfiles` | Demo 17 |
| `USER2_BUCKET` | `user2-textfiles` | Demo 17 |

### Models & Processing

| Variable | Default | Used by |
|----------|---------|---------|
| `EMBEDDING_MODEL` | `intfloat/e5-base-v2` | Demo 02 |
| `SAM3_MODEL_REPO` | `facebook/sam2-hiera-small` | Demo 03 |
| `FORCE_RECOMPUTE` | *(unset)* | Demo 03, 04 |
| `MAX_IMAGES` | `3` | Demo 03, 04, 09, 10 |
| `MAX_VIDEOS` | `6` | Demo 14 |
| `DEMO_DOWNLOAD` | `0` | Demo 07 |
| `DEMO_RECOMPUTE` | `0` | Demo 06 |

---

## Troubleshooting

### "Server not reachable"

1. Verify the server is running on the remote host
2. Check SSH tunnels: `nc -z localhost 5555`
3. Check MinIO: `curl -sf http://localhost:9000/minio/health/live`
4. Run Demo 01 for detailed diagnostics

### "umrs-client not found"

```bash
pip install -e src/
# or activate your site venv:
source deploy/<site>/.venv/bin/activate
```

### "No buckets configured"

Either the user config is missing experiences, or the server hasn't been
reloaded since the config was updated:
```bash
umrs-client -c ~/.config/umrs/conf_umrs.json reload
```

### Embeddings not computed (0 rows exported)

The embedding model may not be running. Check:
```bash
umrs-client -c ~/.config/umrs/conf_umrs.json status all
```
If the model needs to be launched, the server will do so automatically on
the next upload. On HPC this means a Slurm job is submitted.

### Vision demos time out

SAM3 requires GPU nodes. On HPC, the first run triggers a Slurm
job which may queue for a while. Check Slurm:
```bash
squeue -u $USER
```

### RAG service not available

The RAG service is optional. It requires `UMRS_RAG_ENABLE=true` (or `1`) in
the deployment environment before the site stack is started or restarted.
Check with your admin if you do not control the deployment.

---

## Recommended Learning Path

```
Beginner
  Demo 01  Health Check
  Demo 02  Text Embedding
  Demo 06  Server Maintenance

Intermediate
  Demo 05  RAG Search
  Demo 07  Model Registry
  Demo 08  User-Defined Models
  Demo 12  Pipeline SDK

Advanced (requires GPU / HPC)
  Demo 03  SAM3 Segmentation
  Demo 04  Vision Inference
  Demo 09  Multi-Step Vision Pipeline
  Demo 10  Multi-Mode Vision Pipeline

Infrastructure
  Demo 11  Job Submission
  Demo 15  Complex DAG
  Demo 16  Custom Workflow
  Demo 17  Multi-User Isolation
  Demo 18  Model Lifecycle
  Demo 20  Durable Jobs

Expert
  Demo 13  Plugins
  Demo 14  Video Analysis
  Demo 19  RAG Advanced
  Demo 21  Remote Plugins
  Demo 22  Monitor TUI
```
