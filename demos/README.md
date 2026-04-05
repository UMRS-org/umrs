# UMRS Demos

Step-by-step demos ordered from basic usage to advanced features.
Each demo has a `run_demo.sh` that can be executed standalone.

## Prerequisites

All demos (except 08, 12, 13 which run locally) require:

1. **SSH tunnels** to the UMRS server (ports 5555, 5556, 9000; 8081 for RAG):
   ```bash
   bash deploy/rccs-cloud/ssh_tunnel.sh
   ```
2. **Python venv** with `umrs-client` installed:
   ```bash
   source deploy/rccs-cloud/.venv/bin/activate
   unset UMRS_SERVER_HOST   # clear Docker Compose env leak
   ```
3. **User config** at `~/.config/umrs/users/user1.json` with configured experiences.

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
| 04 | [HMAE Inference](04_hmae_inference/) | HMAE reconstruction/feature extraction (engine: `tmp/hmae-infer/`) |
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
| 09 | [Defect Analysis](09_defect_analysis/) | Multi-step pipeline: SAM3 → crop → HMAE → clustering |
| 10 | [HMAE Full Pipeline](10_hmae_full_pipeline/) | Full HMAE pipeline: reconstruct + MAE + features in parallel |

## Infrastructure

Job management, pipeline SDK, and plugin system for HPC/cloud.

| # | Demo | What you'll learn |
|---|------|-------------------|
| 11 | [Job Submission](11_job_submission/) | Explicit job submit/cancel, `--keep-alive`, service lifecycle |
| 12 | [Pipeline SDK](12_pipeline_sdk/) | 6-phase pipeline lifecycle (configure→ingest→compute→collect→verify→present) |
| 13 | [Plugins](13_plugins/) | Sandboxed plugin execution on Slurm/K8s |

## Analysis Pipelines

Domain-specific analysis workflows running on UMRS.

| # | Demo | What you'll learn |
|---|------|-------------------|
| 14 | [Opossum Analysis](14_opossum_analysis/) | Video motion analysis for opossum mating detection (engine: `tmp/opossum/`) |

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

## Quick Reference

### Which demo should I start with?

```
New to UMRS?
  → Demo 01 (health check) then Demo 02 (text embedding)

Working with images?
  → Demo 03 (SAM3) or Demo 04 (HMAE)

Want to search over embeddings?
  → Demo 05 (RAG search) — needs Demo 02 data first

Something seems wrong with my data?
  → Demo 06 (check + update)

Need a custom model or pipeline?
  → Demo 08 (Step) or Demo 09 (multi-step defect analysis)

Managing models on HPC?
  → Demo 07 (model registry)

Need to pre-warm GPU nodes?
  → Demo 11 (job submission with --keep-alive)

Analyzing video data?
  → Demo 14 (opossum motion detection)

Want parallel DAG pipelines?
  → Demo 15 (complex DAG with merging)

Multi-tenant isolation?
  → Demo 17 (multi-user bucket isolation)

Comparing search strategies?
  → Demo 19 (RAG advanced: vector vs BM25 vs hybrid)

Need durable job management?
  → Demo 20 (submit, poll, retry, cancel)
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
| `upload-workflow` | 14 | Upload a UDM workflow to server |

### Environment Variables

| Variable | Used by | Description |
|----------|---------|-------------|
| `UMRS_USER_CONF` | all | Path to user config JSON (default: `~/.config/umrs/users/user1.json`) |
| `FORCE_RECOMPUTE` | 03 | Set to `1` to trigger full recompute after upload |
| `MAX_IMAGES` | 03, 09, 10 | Limit number of images to upload (default: 2-3) |
| `HMAE_LOCAL_MODEL_DIR` | 04 | Path to HMAE checkpoint dir for model registration |
| `RAG_URL` | 05 | RAG service URL (default: `http://localhost:8081`) |
| `DEMO_RECOMPUTE` | 06 | Set to `1` to run full recompute step |
| `DEMO_DOWNLOAD` | 07 | Set to `1` to download model from HuggingFace |
| `GEN_TEST_VIDEOS` | 14 | Set to `1` to generate synthetic test videos |
| `VIDEO_BUCKET` | 14 | Bucket alias for video uploads (default: `user1-textfiles`) |
