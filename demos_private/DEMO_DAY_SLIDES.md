# UMRS Demo Day — Presentation Outline

## Slide 1: Title Slide

```
UMRS — Unified Multimodal Retrieval System
Demo Day — Production Platform Walkthrough

Presented by: [Your Name]
Date: [Date]
```

## Slide 2: What is UMRS?

```
UMRS is the production platform for:
• Multimodal data processing (text, vision, embeddings)
• AI-powered inference pipelines (LLMs, vision models, RAG)
• HPC-integrated GPU acceleration (Slurm + containers)
• Secure, audited, multi-tenant operations

Key Features:
✓ Upload data → AI processes it → Download results
✓ Model registry with version control
✓ RAG (Retrieval-Augmented Generation) pipelines
✓ Workflow system for complex multi-step processing
✓ Complete audit trail and security
```

## Slide 3: Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│  Client Layer                                      │
│  • umrs CLI (upload, download, model, etc.)  │
│  • r4 platform SDK (RAG, workflows, MCP)           │
└──────────────────┬──────────────────────────────────┘
                   │ HTTP / ZMQ
┌──────────────────▼──────────────────────────────────┐
│  UMRS Server                                        │
│  • Job orchestration & durable tracking             │
│  • Container lifecycle management                   │
│  • S3/MinIO object storage                          │
│  • PostgreSQL + pgvector (RAG)                      │
└──────┬───────────────┬─────────────────────────────┘
       │               │
  ┌────▼────┐    ┌─────▼─────┐
  │ S3/MinIO│    │ PostgreSQL│
  │Storage  │    │ + pgvector│
  └─────────┘    │ (RAG)     │
                 └───────────┘
                 
       ┌───────────────────────┐
       │  GPU Compute (Slurm)  │
       │  • vLLM / LiteLLM     │
       │  • Vision Models      │
       │  • HPC Containers     │
       └───────────────────────┘
```

## Slide 4: UMRS Client CLI — Getting Started

```bash
# Install
pip install -e /path/to/umrs/src/

# Configure connection profile
umrs --profile rccs-cloud ping

# Check server health
umrs info

# Authenticate (if required)
umrs login --api-key <JWT_TOKEN>

# Show current profile
umrs profile show
```

## Slide 5: Demo 1 — Text Embeddings & RAG

```bash
# ─── Step 1: Upload text data ───
# Create sample text files
mkdir -p /tmp/rag-texts
echo "RAG stands for Retrieval-Augmented Generation." > /tmp/rag-texts/doc1.txt
echo "UMRS processes multimodal data including text, images, and embeddings." > /tmp/rag-texts/doc2.txt

# Upload to text bucket (auto-fallback to available bucket)
umrs data dir-upload /tmp/rag-texts mnagaso3-text

# Verify upload
umrs data list mnagaso3-text

# ─── Step 2: Check models ───
# List available models
umrs model list --scope system

# ─── Step 3: RAG Ingestion ───
# Create RAG pipeline (requires admin access)
r4 rag ingest create --project default --file <pipeline-config.json>

# List RAG pipelines
r4 rag ingest list --project default

# ─── Step 4: Download results ───
umrs data download mnagaso3-text /tmp/rag-results --force-overwrite
```

## Slide 6: Demo 2 — SAM3.1 Image Segmentation

```bash
# ─── Prerequisites ───
# SAM3 experience must be configured by admin
# Bucket: demo-sam3 or demo-defect

# ─── Step 1: Upload images ───
# Use sample images from the repo
cd /home/mnagaso/workspace/RCCS/ai4s-platform/umrs

# Upload directory of images
umrs data dir-upload demos/sample_data/images demo-sam3

# ─── Step 2: Monitor processing ───
# Check job status
umrs status all

# ─── Step 3: Download segmentation results ───
# SAM3 returns: bounding boxes, masks, scores, overlay images
umrs data download demo-sam3 /tmp/segmentation-results --force-overwrite

# ─── Step 4: View results ───
ls -la /tmp/segmentation-results/
# Expected: overlay images with segmentation masks, JSON metadata
```

## Slide 7: Demo 3 — HMAE Inference

```bash
# ─── Prerequisites ───
# HMAE model checkpoint must be staged on server
# Bucket: user1-hmae (admin must provision)

# ─── Step 1: Register HMAE model (admin) ───
umrs model upload /path/to/hmae_checkpoint \
  --name "hmae-pretrained" \
  --type vision \
  --scope user \
  --register-path /site-models/hmae-pretrained

# ─── Step 2: Upload volume slices ───
umrs data dir-upload \
  demos/03_sam3_segmentation/test_data_full \
  user1-hmae

# ─── Step 3: Wait for reconstruction ───
# HMAE processes images and creates reconstructions
# This may take several minutes on Slurm

# ─── Step 4: Download reconstructions ───
umrs data download user1-hmae /tmp/hmae-results --force-overwrite

# ─── Step 5: View reconstruction outputs ───
ls -la /tmp/hmae-results/
# Expected: reconstructed images, metadata JSON files
```

## Slide 8: Advanced Features

```bash
# ─── Upload Workflows ───
# Upload a custom workflow bundle
umrs workflow upload \
  demos_private/09_defect_analysis/workflow/ \
  --name "defect-analysis"

# ─── Visualize Workflow DAG ───
umrs workflow visualize defect-analysis

# ─── Run Workflow ───
umrs workflow run defect-analysis --bucket demo-sam3

# ─── Check Workflow Status ───
umrs workflow status <execution-id>

# ─── Auto-Trigger (if configured) ───
# When auto_trigger=true in workflow.json, 
# uploading data automatically triggers the workflow
umrs data dir-upload /path/to/data <bucket>
# Workflow runs automatically → download results
umrs data download <bucket> /tmp/results
```

## Slide 9: Doctor Mode — System Health

```bash
# ─── Run Full Health Check ───
umrs doctor

# ─── Check Specific Components ───
umrs doctor server        # Server connectivity
umrs doctor auth          # Authentication status
umrs doctor storage       # S3/MinIO connectivity
umrs doctor models        # Model registry
umrs doctor rag           # RAG service
umrs doctor workflows     # Workflow system
umrs doctor jobs          # Job system
umrs doctor demos         # Demo readiness

# ─── Example Output ───
# ✓ Server: Server is reachable
# ✓ Storage: Connected, 1 buckets
# ✓ Models: 35 system models (5 embeddings)
# ✓ RAG: RAG service available
# ⚠ RAG: No pipelines → "Run: r4 rag ingest create --project default"
```

## Slide 10: Security & Governance

```
UMRS Security Features:

✓ JWT-based authentication
✓ S3 presigned URLs (time-limited, unguessable)
✓ Bucket alias system (isolated data per user/project)
✓ Model scope (system / project / user)
✓ RAG content scanning (PII/credentials)
✓ EgressGuard for MCP network access
✓ Complete audit trail

Security Invariants:
• No model downloads inside inference containers
• No direct file uploads to vLLM (presigned URLs only)
• HTTP-only internal network (no SSL between components)
• RAG/MCP run in sandboxed containers
```

## Slide 11: Demo Day — Live Demo Plan

```
Phase 1: Basic Operations (15 min)
  1. Connect to server: `umrs ping`
  2. Upload text data: `umrs data dir-upload ...`
  3. List uploaded files
  4. Download results

Phase 2: Vision Inference (20 min)
  1. Upload images: `umrs data dir-upload ... demo-sam3`
  2. Wait for processing (check with `umrs status`)
  3. Download segmentation overlays
  4. Show segmentation masks on images

Phase 3: HMAE Pipeline (20 min)
  1. Show HMAE model registration
  2. Upload volume slices
  3. Wait for reconstruction
  4. Show reconstruction outputs

Phase 4: Q&A (15 min)
  Show doctor mode, security features, advanced workflows
```

## Slide 12: Resources & Next Steps

```
Documentation:
• demos/README.md — Public demo index
• demos/USER_MANUAL.md — User workflow guide
• demos/DEMO_RUNNER.md — Demo runner guide
• umrs/src/umrs/ — Source code
• docs/ — Architecture documentation

Getting Help:
• Check doctor: `umrs doctor`
• View server logs (admin)
• Contact platform team

Next Steps:
• Deploy private demo bucket aliases
• Stage HMAE/SAM3 model checkpoints
• Configure RAG pipelines
• Add user access provisioning
```
