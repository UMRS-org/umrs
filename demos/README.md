# UMRS Demos

Step-by-step demos ordered from basic usage to advanced features.
Each demo has a `run_demo.sh` that can be executed standalone.

These demos work with **any deployment** (Docker Compose, rccs-cloud, hpbd-atom,
or any other setup) as long as the UMRS server is reachable from your client
machine.

> For a full walkthrough of every demo, see **[doc/demo_guide.md](../doc/demo_guide.md)**.

## Prerequisites

All demos (except 08, 12, 13 which run locally) require:

1. **UMRS server running** on any supported deployment:
   - Docker Compose (local dev): `docker compose up` from `docker/`
   - HPC/cloud: follow the deployment guide for your site

2. **Connection selected** for the deployment you want to use.

  Check the active connection profile with:
  ```bash
  umrs-client --output text profile show
  ```

  If the profile reports `tunnel_mode=true`, start the matching SSH tunnel
  before running demos. Site-specific tunnel scripts are available under
  `deploy/<site>/ssh_tunnel.sh`.

3. **Python venv** with `umrs-client` installed.

   This project uses [uv](https://docs.astral.sh/uv/) as the standard package manager.
   Install it once if you haven't already (`brew install uv` on macOS, or
   `curl -LsSf https://astral.sh/uv/install.sh | sh`).

  **Option A — site-specific setup script (recommended):**
   ```bash
   cd deploy/<site>/
  ./setup_client.sh            # installs user config, connection profile, and umrs package
  umrs-client --output text profile show
   ```

   **Option B — manual:**
   ```bash
   uv venv .venv
   source .venv/bin/activate    # Windows: .venv\Scripts\activate
   uv pip install -e src/       # install umrs package
   which umrs-client            # verify it's in PATH
   ```

4. **User config** at `~/.config/umrs/users/user1.json` and, optionally, a
  **connection profile** under `~/.config/umrs/connections/`. Site-specific
  `setup_client.sh` scripts create both for you.

  To import an additional site profile manually:
  ```bash
  umrs-client profile import deploy/<site>/client/connections/<site>.json
  umrs-client profile use <site>
  ```

All demos source `lib/common.sh` which provides the `$CLIENT` command,
`preflight()`, `wait_tasks()`, `banner()`, `log()`, `ok()`, `warn()`, `die()`.

## Getting Started

| # | Demo | What you'll learn |
|---|------|-------------------|
| 01 | [Health Check](01_health_check/) | Ping server, verify connectivity |

## Basic Usage

Core data flow: upload data → compute embeddings → search/export.

| # | Demo | What you'll learn |
|---|------|-------------------|
| 02 | [Text Embedding](02_text_embedding/) | Upload text files, compute embeddings, export to CSV |
| 03 | [SAM3 Segmentation](03_sam3_segmentation/) | Image segmentation via SAM3 + BentoML on HPC |
| 04 | [Vision Inference](04_vision_inference/) | Generic vision model inference via BentoML (SAM3 or any registered model) |
| 05 | [RAG Search](05_rag_search/) | Semantic search: vector, hybrid BM25, reranking, Q&A |

## Operations & Maintenance

Manage models and maintain data integrity.

| # | Demo | What you'll learn |
|---|------|-------------------|
| 06 | [Server Maintenance](06_server_maintenance/) | `list_buckets`, `check`, `update`, `recompute` — data integrity |
| 07 | [Model Registry](07_model_registry/) | Register, list, download models (HuggingFace + local + user) |

## Advanced Workflows

Multi-step pipelines, user-defined models, and complex analysis.

| # | Demo | What you'll learn |
|---|------|-------------------|
| 08 | [User-Defined Models](08_user_model/) | Custom model workflows via Step + `workflow.json` |
| 09 | [Multi-Step Vision Pipeline](09_multistep_vision_pipeline/) | 3-step pipeline: segment → crop regions → image statistics |
| 10 | [Multi-Mode Vision](10_vision_multi_mode/) | Parallel branches: segmentation + features + thumbnails |

## Infrastructure

Job management, pipeline SDK, and plugin system for HPC/cloud.

| # | Demo | What you'll learn |
|---|------|-------------------|
| 11 | [Job Submission](11_job_submission/) | Explicit job submit/cancel, endpoint warm-up, explicit cleanup |
| 12 | [Pipeline SDK](12_pipeline_sdk/) | 6-phase pipeline lifecycle (configure→ingest→compute→collect→verify→present) |
| 13 | [Plugins](13_plugins/) | Sandboxed plugin execution on Slurm/K8s |

## Analysis Pipelines

Domain-specific analysis workflows running on UMRS.

| # | Demo | What you'll learn |
|---|------|-------------------|
| 14 | [Video Analysis](14_video_analysis/) | Video motion detection via OpenCV frame differencing (CPU only) |

## Advanced Features

Complex DAG pipelines, multi-user isolation, durable jobs, and extensibility.

| # | Demo | What you'll learn |
|---|------|-------------------|
| 15 | [Complex DAG](15_complex_dag/) | Multi-branch DAG pipeline with parallel execution and merging |
| 16 | [Custom Workflow](16_custom_workflow/) | Upload and execute server-side custom workflows |
| 17 | [Multi-User Isolation](17_multi_user/) | Bucket isolation between users, cross-user access denied |
| 18 | [Model Lifecycle](18_model_lifecycle/) | Full model lifecycle: register → list → use → deprecate |
| 19 | [RAG Advanced](19_rag_advanced/) | Compare vector, BM25, hybrid (RRF), and reranked search |
| 20 | [Durable Jobs](20_durable_jobs/) | Job submit, poll, idempotent resubmit, cancel |
| 21 | [Remote Plugins](21_remote_plugins/) | Plugin spec definition, validation, execution, audit log |
| 22 | [Monitor (TUI)](22_monitor/) | Live-refreshing jobs dashboard with keyboard navigation |

## Quick Reference

### Which demo should I start with?

```
New to UMRS?
  → Demo 01 (health check) then Demo 02 (text embedding)

Working with images?
  → Demo 03 (SAM3) or Demo 04 (vision inference)

Want to search over embeddings?
  → Demo 05 (RAG search) — needs Demo 02 data first

Something seems wrong with my data?
  → Demo 06 (check + update)

Need a custom model or pipeline?
  → Demo 08 (Step) or Demo 09 (multi-step vision pipeline)

Managing models on HPC?
  → Demo 07 (model registry)

Need to pre-warm GPU nodes?
  → Demo 11 (explicit endpoint warm-up and cancel)

Analyzing video data?
  → Demo 14 (video motion detection)

Want parallel DAG pipelines?
  → Demo 15 (complex DAG with merging)

Multi-tenant isolation?
  → Demo 17 (multi-user bucket isolation)

Comparing search strategies?
  → Demo 19 (RAG advanced: vector vs BM25 vs hybrid)

Need durable job management?
  → Demo 20 (submit, poll, retry, cancel)

Want a live TUI dashboard?
  → Demo 22 (monitor: jobs table, status filters, detail views)
```

### CLI Commands × Demo Coverage

| Command | Demo | Description |
|---------|------|-------------|
| `ping` | 01 | Check server connectivity |
| `upload_file` | 02 | Upload a single file |
| `upload_dir` | 02, 03, 04, 09, 10, 14 | Upload a directory |
| `download_results` | 03, 04, 09, 10, 14 | Download computed outputs |
| `export_embeddings` | 02 | Export embeddings to CSV/JSONL |
| `list_buckets` | 06 | Show configured S3 buckets |
| `status` | 02, 11 | Query server task status |
| `check` | 06 | Verify S3 ↔ DB consistency |
| `update` | 06 | Fix missing/mismatched embeddings |
| `recompute` | 03, 04, 06 | Force full re-processing |
| `reload` | 03, 04 | Reload server config |
| `model register` | 07 | Register a local model |
| `model list` | 07 | List registered models |
| `model upload` | 07 | Upload model to staging root |
| `model download` | 07 | Download from HuggingFace |
| `job submit` | 11 | Submit endpoint/index/cleanup job |
| `job cancel` | 11 | Cancel a running job |
| `monitor` | 22 | Live-refreshing jobs dashboard (TUI) |
| `upload-workflow` | 14 | Upload a UDM workflow to server |

### Environment Variables

| Variable | Used by | Description |
|----------|---------|-------------|
| `UMRS_PROFILE` | all | Named connection profile to use for the demo commands |
| `UMRS_USER_CONF` | all | Path to the user config JSON (default: `~/.config/umrs/users/user1.json`) |
| `UMRS_CONF` | all | Explicit config override used by the demo helper (legacy/manual override) |
| `FORCE_RECOMPUTE` | 03 | Set to `1` to trigger full recompute after upload |
| `MAX_IMAGES` | 03, 09, 10 | Limit number of images to upload (default: 2-3) |
| `HMAE_LOCAL_MODEL_DIR` | 04 | Path to HMAE checkpoint dir for model registration |
| `RAG_URL` | 05 | RAG service URL (default: `http://localhost:8081`) |
| `DEMO_RECOMPUTE` | 06 | Set to `1` to run full recompute step |
| `DEMO_DOWNLOAD` | 07 | Set to `1` to download model from HuggingFace |
| `GEN_TEST_VIDEOS` | 14 | Set to `1` to generate synthetic test videos |
| `VIDEO_BUCKET` | 14 | Bucket alias for video uploads (default: `user1-textfiles`) |
