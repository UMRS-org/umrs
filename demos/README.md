# UMRS Demos

Step-by-step demos ordered from basic usage to advanced features.
Each demo has a `README.md` with full documentation and a `run_demo.sh`
that can be executed standalone.

## Getting Started

| # | Demo | What you'll learn |
|---|------|-------------------|
| 00 | [Start Services](00_start_services/) | Start/stop UMRS services, health checks (admin prerequisite) |

## Basic Usage

Core data flow: upload data → compute embeddings → search/export.

| # | Demo | What you'll learn |
|---|------|-------------------|
| 01 | [Text Embedding](01_text_embedding/) | Upload text files, compute embeddings, export to CSV |
| 02 | [SAM3 Segmentation](02_sam3_segmentation/) | Image segmentation via SAM3 + BentoML on Mini-Rivault |
| 03 | [HMAE Inference](03_hmae_inference/) | HMAE feature extraction / reconstruction (3 modes) |
| 04 | [RAG Search](04_rag_search/) | Semantic search: vector, hybrid BM25, reranking, Q&A |

## Operations & Maintenance

Manage models and maintain data integrity.

| # | Demo | What you'll learn |
|---|------|-------------------|
| 05 | [Server Maintenance](05_server_maintenance/) | `list_buckets`, `check`, `update`, `recompute` — data integrity |
| 06 | [Model Registry](06_model_registry/) | Register, list, download models (HuggingFace + local + user) |

## Advanced Workflows

Multi-step pipelines, user-defined models, and complex analysis.

| # | Demo | What you'll learn |
|---|------|-------------------|
| 07 | [User-Defined Models](07_user_model/) | Custom model workflows via Step + `workflow.json` |
| 08 | [Defect Analysis](08_defect_analysis/) | Multi-step pipeline: SAM3 → crop → HMAE → clustering |
| 09 | [HMAE Full Pipeline](09_hmae_full/) | Full HMAE pipeline: reconstruct + MAE + features in parallel |

## Infrastructure

Job management, pipeline SDK, and plugin system for HPC/cloud.

| # | Demo | What you'll learn |
|---|------|-------------------|
| 10 | [Job Submission](10_job_submission/) | Explicit job submit/cancel, `--keep-alive`, service lifecycle |
| 11 | [Pipeline SDK](11_pipeline_sdk/) | 6-phase pipeline lifecycle (configure→ingest→compute→collect→verify→present) |
| 12 | [Plugins](12_plugins/) | Sandboxed plugin execution on Slurm/K8s |

## Alternative Vision Backends

Use Jens's standalone `mini-rivault-vision` stack instead of the UMRS generic adapter.
Set `VISION_STACK=jens` in the environment.

| # | Demo | What you'll learn |
|---|------|-------------------|
| 13 | [SAM3 (Jens)](13_sam3_jens/) | SAM3 segmentation via Jens's `/vision_task` endpoint |
| 14 | [HMAE (Jens)](14_hmae_jens/) | HMAE inference via Jens's `/vision_task` endpoint |

## Quick Reference

### Which demo should I start with?

```
New to UMRS?
  → Demo 00 (start services) then Demo 01 (text embedding)

Working with images?
  → Demo 02 (SAM3) or Demo 03 (HMAE)

Want to search over embeddings?
  → Demo 04 (RAG search) — needs Demo 01 data first

Something seems wrong with my data?
  → Demo 05 (check + update)

Need a custom model or pipeline?
  → Demo 07 (Step) or Demo 08 (multi-step defect analysis)

Managing models on HPC?
  → Demo 06 (model registry)

Need to pre-warm GPU nodes?
  → Demo 10 (job submission with --keep-alive)
```

### CLI Commands × Demo Coverage

| Command | Demo | Description |
|---------|------|-------------|
| `ping` | 01 | Check server connectivity |
| `upload_file` | 01 | Upload a single file |
| `upload_dir` | 01, 02, 03 | Upload a directory |
| `download_results` | 02, 03 | Download computed outputs |
| `export_embeddings` | 01 | Export embeddings to CSV/JSONL |
| `list_buckets` | 05 | Show configured S3 buckets |
| `status` | 01, 10 | Query server task status |
| `check` | 05 | Verify S3 ↔ DB consistency |
| `update` | 05 | Fix missing/mismatched embeddings |
| `recompute` | 05 | Force full re-embedding |
| `reload` | 02, 03 | Reload server config |
| `model register` | 06 | Register a local model |
| `model list` | 06 | List registered models |
| `model download` | 06 | Download from HuggingFace |
| `job submit` | 10 | Submit endpoint/index/cleanup job |
| `job cancel` | 10 | Cancel a running job |
