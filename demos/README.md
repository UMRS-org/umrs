# UMRS Demos

Step-by-step demos ordered from basic usage to advanced features.
Each demo has a `run_demo.sh` that can be executed standalone.

These demos work with **any deployment** (Docker Compose, rccs-cloud, hpbd-atom,
or any other setup) as long as the UMRS server is reachable from your client
machine.

> This file is the canonical walkthrough for the public demo suite.

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

4. **User config and connection profile** — create both with the interactive
  init subcommands:

  ```bash
  umrs-client profile init           # creates ~/.config/umrs/connections/<name>.json
  umrs-client config init            # creates ~/.config/umrs/users/<username>.json
  ```

  Or use a site-specific setup script which does both for you:
  ```bash
  cd deploy/<site>/
  ./setup_client.sh
  ```

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
| 00 | [User Onboarding](00_onboarding/) | Push user config → server provisions buckets + DB automatically |
| 01 | [Health Check](01_health_check/) | Ping server, verify connectivity |

> **Note:** All server-backed demos call the shared `onboard()` helper which
> runs `reload` + `status reload` + `list_buckets`.  This is idempotent — the
> server skips provisioning when the config hasn't changed.  Demos 08, 12, and
> 13 run locally and do not require onboarding.

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
| 23 | [Auth Enforcement](23_auth_enforcement/) | Admin token management, role-based command authorization |
| 24 | [Thread Pool](24_thread_pool/) | Bounded thread pool metrics, backpressure indicator |

## Platform APIs

Low-level REST APIs that power the platform internals.

| # | Demo | What you'll learn |
|---|------|-------------------|
| 25 | [Endpoint Provisioning](25_endpoint_provisioning/) | REST API for endpoint lifecycle, warm pool reuse, multi-model |
| 26 | [MCP Servers](26_mcp_servers/) | Register/unregister MCP servers, list tools, invoke tools |
| 27 | [Pipeline CRUD](27_pipeline_crud/) | Pipeline definitions, RAG profiles, component registry |
| 28 | [Egress Audit](28_egress_audit/) | Query and log outbound network events from Mini-RiVault |
| 29 | [LLM Completions](29_llm_completions/) | OpenAI-compatible chat/text completions and embeddings |
| 30 | [File Management](30_file_management/) | Upload, download (presigned URLs), bulk upload, delete |

## Quick Reference

### Which demo should I start with?

```
New to UMRS?
  → Demo 00 (onboarding), then Demo 01 (health check), then Demo 02 (text embedding)

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

Need admin-token / RBAC validation?
  → Demo 23 (auth enforcement)

Need to inspect thread-pool capacity or backpressure?
  → Demo 24 (thread pool metrics)

Need to manage endpoints via REST API?
  → Demo 25 (endpoint provisioning)

Want to register/invoke MCP servers?
  → Demo 26 (MCP server management)

Working with pipeline definitions or RAG profiles?
  → Demo 27 (pipeline CRUD)

Need to audit outbound network traffic?
  → Demo 28 (egress audit)

Want to call the OpenAI-compatible LLM API directly?
  → Demo 29 (LLM completions)

Need to manage files with presigned URLs?
  → Demo 30 (file management)
```

### CLI Commands × Demo Coverage

| Command | Demo | Description |
|---------|------|-------------|
| `ping` | 00, 01 | Check server connectivity |
| `reload` | 00, 03, 04, 09, 10, 14, 16 | Push config/workflow changes to the server |
| `upload_file` | 17, 18, 24 | Upload a single file |
| `upload_dir` | 02, 03, 04, 09, 10, 14 | Upload a directory |
| `download_results` | 03, 04, 09, 10, 14 | Download computed outputs |
| `export_embeddings` | 02 | Export embeddings to CSV/JSONL |
| `list_buckets` | 00, 01, 06, 17 | Show configured S3 buckets |
| `status` | 00, 02, 06, 20, 24 | Query task or pool status |
| `check` | 06 | Verify S3 ↔ DB consistency |
| `update` | 06 | Fix missing/mismatched embeddings |
| `recompute` | 03, 04, 06 | Force full re-processing |
| `model register` | 07 | Register a local model |
| `model list` | 07 | List registered models |
| `model upload` | 07 | Upload model to staging root |
| `model download` | 02, 03, 07, 18 | Stage a model from HuggingFace |
| `job submit` | 11, 20, 22 | Submit endpoint/index/cleanup jobs |
| `job cancel` | 11, 20 | Cancel a running job |
| `monitor` | 22 | Live-refreshing jobs dashboard (TUI) |
| `upload-workflow` | 09, 10, 14, 16 | Upload a workflow bundle to the server |
| `admin token` | 23 | Generate, list, and revoke admin tokens |

### Environment Variables

| Variable | Used by | Description |
|----------|---------|-------------|
| `UMRS_PROFILE` | all | Named connection profile to use for the demo commands |
| `UMRS_USER_CONF` | all | Path to the user config JSON (default: `~/.config/umrs/users/user1.json`) |
| `UMRS_CONF` | all | Explicit config override used by the demo helper (legacy/manual override) |
| `UMRS_ADMIN_TOKEN` | 06, 09, 10, 14, 16, 23 | Admin token for reload and maintenance on auth-enforced deployments |
| `TEXT_BUCKET` | 02, 18, 24 | Bucket alias for text uploads |
| `SAM3_BUCKET` | 03 | Bucket alias for SAM3 segmentation uploads |
| `VISION_BUCKET` | 04 | Bucket alias for generic vision inference |
| `PIPELINE_BUCKET` | 09 | Bucket alias for the multi-step vision pipeline |
| `MULTI_BUCKET` | 10 | Bucket alias for the multi-mode vision workflow |
| `FORCE_RECOMPUTE` | 03, 04 | Set to `1` to trigger full recompute after upload |
| `MAX_IMAGES` | 03, 04, 09, 10 | Limit number of images to upload |
| `MODEL_STAGING_DIR` | 02, 07, 18 | Writable server-side staging directory for model downloads |
| `DEMO_LOCAL_MODEL_UPLOAD` | 07, 18 | Enable the optional local model upload steps |
| `RAG_URL` | 05 | RAG service URL (default: `http://localhost:8081`) |
| `DEMO_RECOMPUTE` | 06 | Set to `1` to run full recompute step |
| `DEMO_DOWNLOAD` | 07 | Set to `1` to download model from HuggingFace |
| `VIDEO_BUCKET` | 14 | Bucket alias for video uploads (default: `user1-video`) |
| `MAX_VIDEOS` | 14 | Max number of videos to upload |
| `USER1_BUCKET` | 17 | Bucket alias for user1 in the isolation demo |
| `USER2_BUCKET` | 17 | Bucket alias for user2 in the isolation demo |
| `DEMO_LOCAL_ONLY` | 23 | Run only local token management without server-side auth checks |
| `DEMO_SUBMIT_TASK` | 24 | Submit a lightweight upload before re-querying pool metrics |
