# UMRS Demos

Step-by-step demos ordered from basic usage to advanced features.
Each demo's `README.md` contains step-by-step commands inlined under
"Quick Start" and "Running Manually" sections — no separate script files exist.

These demos work with **any deployment** (Docker Compose, rccs-cloud, hpbd-atom,
or any other setup) as long as the UMRS server is reachable from your client
machine.

> This file is the compact index for the public demo suite. For a full
> operator/user walkthrough, see [USER_MANUAL.md](USER_MANUAL.md).

## Documentation Map

| Document | Purpose |
|----------|---------|
| [README.md](README.md) | Quick setup, demo index, and command coverage |
| [USER_MANUAL.md](USER_MANUAL.md) | End-to-end user workflow, experience lifecycle, recovery, and troubleshooting |
| [00_onboarding/README.md](00_onboarding/README.md) | First-run onboarding and provisioning details |
| [99_cleanup/README.md](99_cleanup/README.md) | Teardown and cleanup flow |

## Prerequisites

All demos (except 08, 12, 13, 15, 21 which run locally) require:

1. **UMRS server running** on any supported deployment:
   - Docker Compose (local dev): `docker compose up` from `docker/`
   - HPC/cloud: follow the deployment guide for your site

2. **Connection selected** for the deployment you want to use.

  Check the active connection profile with:
  ```bash
  umrs --output text profile show
  ```

  If the profile reports `tunnel_mode=true`, start the matching SSH tunnel
  before running demos. Site-specific tunnel scripts are available under
  `deploy/<site>/ssh_tunnel.sh`.

3. **Python venv** with `umrs` installed.

   This project uses [uv](https://docs.astral.sh/uv/) as the standard package manager.
   Install it once if you haven't already (`brew install uv` on macOS, or
   `curl -LsSf https://astral.sh/uv/install.sh | sh`).

  **Option A — site-specific setup script (recommended):**
   ```bash
   cd deploy/<site>/
  ./setup_client.sh            # installs user config, connection profile, and umrs package
  umrs --output text profile show
   ```

   **Option B — manual:**
   ```bash
   uv venv .venv
   source .venv/bin/activate    # Windows: .venv\Scripts\activate
   uv pip install -e src/       # install umrs package
   which umrs            # verify it's in PATH
   ```

4. **User config and connection profile** — create both with the interactive
  init subcommands:

  ```bash
  umrs profile init           # creates ~/.config/umrs/connections/<name>.json
  umrs config init            # creates ~/.config/umrs/users/<username>.json
  ```

  Or use a site-specific setup script which does both for you:
  ```bash
  cd deploy/<site>/
  ./setup_client.sh
  ```

  To import an additional site profile manually:
  ```bash
  umrs profile import deploy/<site>/client/connections/<site>.json
  umrs profile use <site>
  ```

All demos source `lib/common.sh` which provides the `$CLIENT` command,
profile-derived connection defaults, `preflight()`, `wait_tasks()`, and
logging helpers. Maintenance commands (`reload`, `check`, `update`,
`recompute`) are routed through admin-aware helpers and require
`UMRS_ADMIN_TOKEN` on auth-enforced deployments. Normal user demos avoid those
admin-only actions by default and assume the site/operator has already prepared
the server-side configuration.

## Getting Started

| # | Demo | What you'll learn |
|---|------|-------------------|
| 00 | [User Onboarding](00_onboarding/) | Push user config → server provisions buckets + DB automatically |
| 01 | [Health Check](01_health_check/) | Ping server, verify connectivity |

> **Note:** The normal user-facing demos assume the site/operator has already
> prepared the server-side configuration. Admin-token flows are demonstrated
> separately in the maintenance and auth demos.

## Experience Lifecycle

Most workload demos add an experience fragment to your user JSON, then reload so
UMRS provisions the declared MinIO buckets and Postgres/pgvector database.

```bash
# Add or update a workload experience in the active user config
umrs experience add path/to/experience.json --update

# Provision declared buckets, IAM policy, Postgres role/database, and tables
umrs reload

# Remove an experience from the user JSON only; data is retained
umrs experience remove <xp_name>

# Recover a missing experience JSON from existing MinIO/S3 buckets and
# optional Postgres/pgvector status
umrs experience recover [xp_name] \
  --username <user> \
  --s3cfg ~/.config/umrs/s3cfg_<user> \
  --output-path recovered-experience.json
```

Removing an experience does **not** delete MinIO buckets or pgvector data.
Reload simply drops the experience from the active in-memory registry. If the
same experience is re-added later, existing buckets and database tables can be
reused. Use explicit purge tooling only when the data should be destroyed.
## Bucket Alias Naming Convention

All `experience.json` files use bucket aliases following the pattern:

```
demo-<feature>
```

| Experience | Bucket Alias(s) |
|------------|----------------|
| Text Embedding (02) | `demo-texts`, `demo-textfiles` |
| SAM3 Segmentation (03) | `demo-sam3` |
| Vision Inference (04) | `demo-images` |
| Vision Pipeline (09) | `demo-pipeline` |
| Vision Multi Mode (10) | `demo-vision-multi` |
| Video Analysis (14) | `demo-video` |
| Custom Workflow (16) | `demo-custom-text` |

RAG demos (05, 19) operate on `demo-textfiles` (the text-file modality from
Demo 02) and do not declare their own bucket alias in an experience fragment.

Use descriptive, lowercase, hyphen-separated names. Avoid generic aliases like
`my-bucket` or `user1-textfiles` — they obscure ownership and make cleanup harder.
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
| 06 | [Server Maintenance](06_server_maintenance/) | `list_buckets`, admin-only `check`, `update`, `recompute` — data integrity |
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
| 25 | [Endpoint Provisioning](25_endpoint_provisioning/) | REST API for endpoint lifecycle, warm pool reuse, MCP/RAG mounts, EgressGuard intent |
| 26 | [MCP Servers](26_mcp_servers/) | Register/unregister MCP servers, list tools, validate mounted custom MCP manifests |
| 27 | [Pipeline CRUD](27_pipeline_crud/) | Pipeline definitions, RAG profiles, component registry |
| 28 | [Egress Audit](28_egress_audit/) | Query/log outbound network events and inspect endpoint egress allowlist intent |
| 29 | [LLM Completions](29_llm_completions/) | OpenAI-compatible chat/text completions and embeddings |
| 30 | [File Management](30_file_management/) | Upload, download (presigned URLs), bulk upload, delete |

## Client Management & Utilities

Management, lifecycle, and utility commands not covered by workload demos.

| # | Demo | What you'll learn |
|---|------|-------------------|
| 31 | [Client Management](31_client_management/) | `info`, `completion`, `profile list/show`, `config list/show`, `experience add/list`, `job list`, `endpoint up/list/status/stop`, `workflow list/run`, `model delete`, `purge_remote_buckets` |

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
  → Demo 06 (check + update; operator/admin workflow)

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
  → Demo 25 (endpoint provisioning with MCP/RAG/egress mounts)

Want to register/invoke MCP servers?
  → Demo 26 (MCP server management and mounted custom MCP validation)

Working with pipeline definitions or RAG profiles?
  → Demo 27 (pipeline CRUD)

Need to audit outbound network traffic?
  → Demo 28 (egress audit)

Want to call the OpenAI-compatible LLM API directly?
  → Demo 29 (LLM completions)

Need to manage files with presigned URLs?
  → Demo 30 (file management)

Need to see connection info, list profiles, or manage endpoints?
  → Demo 31 (client management & utility commands)
```

### CLI Commands × Demo Coverage

| Command | Demo | Description |
|---------|------|-------------|
| `ping` | 00, 01, 31 | Check server connectivity |
| `info` | 31 | Show connection setup, transport modes, auth, S3 status |
| `status` | 00, 02, 06, 20, 24 | Query task or pool status |
| `reload` | 00, 06, 23 | Push config/workflow changes to the server |
| `upload_file` | 17, 18, 24 | Upload a single file |
| `upload_dir` | 02, 03, 04, 09, 10, 14 | Upload a directory |
| `download_results` | 03, 04, 09, 10, 14 | Download computed outputs |
| `export_embeddings` | 02 | Export embeddings to CSV/JSONL |
| `list_buckets` | 00, 01, 06, 17 | Show configured S3 buckets |
| `purge_remote_buckets` | 31 | Delete extra remote S3 buckets not in user config |
| `recompute` | 06 | Force full re-processing |
| `update` | 06 | Fix missing/mismatched embeddings |
| `check` | 06 | Verify S3 ↔ DB consistency |
| `model register` | 07 | Register a local model |
| `model list` | 07 | List registered models |
| `model upload` | 07 | Upload model to staging root |
| `model download` | 02, 03, 07, 18 | Stage a model from HuggingFace |
| `model delete` | 31 | Remove a registered model from the catalog |
| `job submit` | 11, 20, 22 | Submit endpoint/index/cleanup jobs |
| `job cancel` | 11, 20 | Cancel a running job |
| `job status` | 20 | Show a specific durable job |
| `job list` | 31 | List durable jobs with optional filters |
| `endpoint up` | 25, 31 | Provision a reusable endpoint |
| `endpoint list` | 25, 31 | List endpoint provision jobs |
| `endpoint status` | 25, 31 | Show endpoint provision job details |
| `endpoint stop` | 25, 31 | Stop an endpoint provision job |
| `workflow list` | 31 | List registered workflows |
| `workflow run` | 16, 31 | Trigger execution of a registered workflow |
| `workflow upload` | 09, 10, 14, 16 | Upload a workflow bundle to the server |
| `workflow status` | 14, 31 | Show execution status for a workflow run |
| `workflow results` | 14, 31 | Fetch results for a completed workflow run |
| `workflow cancel` | 14, 31 | Cancel a running workflow execution |
| `workflow errors` | 14, 31 | Show error logs for a failed workflow execution |
| `workflow visualize` | 09, 14, 15 | Render the workflow DAG as ASCII art |
| `workflow validate-sandbox` | 14 | Validate a workflow sandbox (Dockerfile + workflow.json) |
| `monitor` | 22 | Live-refreshing jobs dashboard (TUI) |
| `config init` | 00, 31 | Create a new user config JSON |
| `config list` | 31 | List discovered config candidates |
| `config show` | 31 | Show resolved config stack for server, user, and S3 |
| `config use` | 06, 07, 17 | Persist the current server/user/s3 config selections |
| `experience add` | 00, 02, 03, 04, 09, 10, 14, 16, 31 | Add experience fragment(s) from a JSON file |
| `experience list` | 31 | List experiences in the current user config |
| `experience remove` | 00, 99 | Remove an experience from the user JSON while retaining MinIO/Postgres data |
| `experience recover` | 00, 99 | Reconstruct an experience JSON fragment from MinIO/S3 and optional pgvector status |
| `profile init` | 00, 31 | Create a new connection profile |
| `profile list` | 31 | List available connection profiles |
| `profile show` | 31 | Show effective connection resolved from profile/env/flags |
| `profile use` | 31 | Set the current default connection profile |
| `profile import` | 00, 31 | Import a connection profile JSON file |
| `login` | 00, 31 | Authenticate and configure connection in one step |
| `completion` | 31 | Generate shell completion scripts |
| `admin token` | 23 | Generate, list, and revoke admin tokens |

### Environment Variables

| Variable | Used by | Description |
|----------|---------|-------------|
| `UMRS_PROFILE` | all | Named connection profile to use for the demo commands |
| `UMRS_USER_CONF` | all | Path to the user config JSON (default: `~/.config/umrs/users/user1.json`) |
| `UMRS_CONF` | all | Explicit config override used by the demo helper (legacy/manual override) |
| `UMRS_ADMIN_TOKEN` | 06, 23 | Admin token for operator/maintenance flows on auth-enforced deployments |
| `TEXT_BUCKET` | 02, 18, 24 | Bucket alias for text uploads |
| `SAM3_BUCKET` | 03 | Bucket alias for SAM3 segmentation uploads |
| `VISION_BUCKET` | 04 | Bucket alias for generic vision inference |
| `PIPELINE_BUCKET` | 09 | Bucket alias for the multi-step vision pipeline |
| `MULTI_BUCKET` | 10 | Bucket alias for the multi-mode vision workflow |
| `FORCE_RECOMPUTE` | 03, 04 | Operator-only maintenance; ignored in normal user mode unless `DEMO_ADMIN_ACTIONS=1` |
| `MAX_IMAGES` | 03, 04, 09, 10 | Limit number of images to upload |
| `MODEL_STAGING_DIR` | 02, 07, 18 | Writable server-side staging directory for model downloads |
| `DEMO_LOCAL_MODEL_UPLOAD` | 07, 18 | Enable the optional local model upload steps |
| `RAG_URL` | 05 | RAG service URL (profile-derived, default: `http://127.0.0.1:8081`) |
| `MINI_RIVAULT_MOUNT_DEMO_ROOT` | 05, 19, 25, 26, 28 | Root for generated demo MCP/project/RAG mount assets |
| `DEMO_MCP_SOURCE` | 25, 26 | Override source path for `/mcp-servers/demo-tools` |
| `DEMO_PROJECT_DATA_SOURCE` | 05, 25 | Override source path for `/project` |
| `DEMO_RAG_INDEX_SOURCE` | 05, 19, 25 | Override source path for `/rag-index` |
| `DEMO_EGRESS_DOMAINS` | 25, 28 | Comma-separated EgressGuard allowlist domains |
| `DEMO_RECOMPUTE` | 06 | Set to `1` to run full recompute step |
| `DEMO_DOWNLOAD` | 07 | Set to `1` to download model from HuggingFace |
| `VIDEO_BUCKET` | 14 | Bucket alias for video uploads (default: `user1-video`) |
| `MAX_VIDEOS` | 14 | Max number of videos to upload |
| `USER1_BUCKET` | 17 | Bucket alias for user1 in the isolation demo |
| `USER2_BUCKET` | 17 | Bucket alias for user2 in the isolation demo |
| `DEMO_LOCAL_ONLY` | 23 | Run only local token management without server-side auth checks |
| `DEMO_SUBMIT_TASK` | 24 | Submit a lightweight upload before re-querying pool metrics |
