# UMRS Demo Day — Command Reference

## Quick Reference Card

### Authentication & Connection
```bash
# Check server connectivity
umrs-client ping

# Show connection info
umrs-client info

# List available profiles
umrs-client profile list

# Show current profile
umrs-client profile show
```

### Bucket Operations
```bash
# List bucket aliases
umrs-client bucket list

# List files in a bucket
umrs-client data list <bucket-alias>

# Purge a bucket (CAUTION!)
umrs-client bucket purge <bucket-alias>
```

### Model Registry
```bash
# List system models
umrs-client model list --scope system

# List user models
umrs-client model list --scope user

# Register a model (path must exist on server)
umrs-client model register <name> <path> --type <embedding|vision|text> --scope <system|project|user>

# Upload model files + register
umrs-client model upload <dir> --name <name> --type <type> --scope <scope>

# Download model (server-side)
umrs-client model download <hf-model-id> --name <local-name> --scope user
```

### Data Upload/Download
```bash
# Upload directory
umrs-client data dir-upload <local-dir> <bucket-alias>

# Upload single file
umrs-client data file-upload <file> <bucket-alias>

# Download results
umrs-client data download <bucket-alias> <local-dir> [--force-overwrite]

# List uploaded files
umrs-client data list <bucket-alias>
```

### Workflow Management
```bash
# Upload workflow bundle
umrs-client workflow upload <workflow-dir> --name <workflow-name>

# List workflows
umrs-client workflow list

# Visualize workflow DAG
umrs-client workflow visualize <workflow-name>

# Run workflow
umrs-client workflow run <workflow-name> --bucket <bucket-alias>

# Check workflow status
umrs-client workflow status <execution-id>

# Check workflow errors
umrs-client workflow errors <execution-id>
```

### Status & Monitoring
```bash
# Check job status
umrs-client status all

# Check specific job
umrs-client job status <job-id>

# View health report
umrs-client doctor

# Doctor subcommands
umrs-client doctor server    # Server connectivity
umrs-client doctor storage   # S3/MinIO
umrs-client doctor models    # Models
umrs-client doctor rag       # RAG service
umrs-client doctor workflows # Workflow system
```

---

## Demo 1: Text Embeddings & RAG

### Setup
```bash
# 1. Create sample text files
mkdir -p /tmp/umrs-rag-demo
echo "Retrieval-Augmented Generation (RAG) is a technique that enhances language model outputs by combining retrieval with generation." > /tmp/umrs-rag-demo/doc1.txt
echo "UMRS is a unified multimodal retrieval system for processing, storing, and retrieving diverse data types." > /tmp/umrs-rag-demo/doc2.txt
echo "The platform supports text, images, embeddings, and custom data modalities with automatic pipeline orchestration." > /tmp/umrs-rag-demo/doc3.txt
```

### Upload
```bash
# 2. Upload text files to bucket
umrs-client data dir-upload /tmp/umrs-rag-demo mnagaso3-text
```

### Verify
```bash
# 3. Check upload status
umrs-client data list mnagaso3-text

# 4. List available models (for RAG embedding)
umrs-client model list --scope system
```

### RAG Pipeline
```bash
# 5. Create RAG ingestion pipeline (admin access)
r4 rag ingest create --project default --file /path/to/pipeline-config.json

# 6. List pipelines
r4 rag ingest list --project default
```

### Download Results
```bash
# 7. Download processed results
umrs-client data download mnagaso3-text /tmp/umrs-rag-results --force-overwrite
```

---

## Demo 2: SAM3.1 Image Segmentation

### Prerequisites
- SAM3 experience must be provisioned by admin
- Bucket `demo-sam3` or `demo-defect` must exist
- Model `facebook/sam2-hiera-small` must be staged

### Setup
```bash
# 1. Verify server connectivity
umrs-client ping

# 2. Check available models
umrs-client model list --scope system
# Look for sam3, vision, or segmentation models

# 3. List bucket aliases
umrs-client bucket list
# Should show demo-sam3 or demo-defect bucket
```

### Upload Images
```bash
# 4. Upload images for segmentation
umrs-client data dir-upload \
  /home/mnagaso/workspace/RCCS/ai4s-platform/umrs/demos/sample_data/images \
  demo-sam3
```

### Monitor Processing
```bash
# 5. Check job status
umrs-client status all

# Wait until processing completes (Slurm job)
# Check with: umrs-client status all
```

### Download Results
```bash
# 6. Download segmentation results
umrs-client data download demo-sam3 /tmp/segmentation-results --force-overwrite

# 7. View results
ls -la /tmp/segmentation-results/
# Expected: overlay images with segmentation masks, JSON metadata
```

### Workflow-Based Defect Analysis
```bash
# Upload custom workflow
umrs-client workflow upload \
  /home/mnagaso/workspace/RCCS/ai4s-platform/umrs/demos_private/09_defect_analysis/workflow/ \
  --name "defect-analysis"

# Visualize workflow
umrs-client workflow visualize defect-analysis

# Run workflow
umrs-client workflow run defect-analysis --bucket demo-sam3
```

---

## Demo 3: HMAE Inference

### Prerequisites
- HMAE model checkpoint staged on server
- Bucket `user1-hmae` provisioned by admin
- HMAE experience configured

### Setup
```bash
# 1. Verify connectivity
umrs-client ping

# 2. Check HMAE model registration
umrs-client model list --scope user
# Look for hmae-pretrained or similar

# 3. List bucket aliases
umrs-client bucket list
# Should show user1-hmae bucket
```

### Register HMAE Model (if needed)
```bash
# 4. Upload HMAE checkpoint to server (admin only)
umrs-client model upload \
  /path/to/hmae_checkpoint_dir \
  --name "hmae-pretrained" \
  --type vision \
  --scope user \
  --register-path /path/to/site-models/hmae-pretrained \
  --force
```

### Upload Volume Slices
```bash
# 5. Upload volume slice images
umrs-client data dir-upload \
  /home/mnagaso/workspace/RCCS/ai4s-platform/umrs/demos/03_sam3_segmentation/test_data_full \
  user1-hmae
```

### Monitor Processing
```bash
# 6. Check processing status
umrs-client status all

# HMAE reconstruction may take several minutes on GPU
# Wait for Slurm job to complete
```

### Download Reconstruction
```bash
# 7. Download reconstruction outputs
umrs-client data download user1-hmae /tmp/hmae-reconstructions --force-overwrite

# 8. View outputs
ls -la /tmp/hmae-reconstructions/
# Expected: reconstructed images, metadata JSON files
```

---

## Demo Day: Live Demo Script

### Quick Start (5 minutes)
```bash
# Show server is up
umrs-client ping

# Show current setup
umrs-client info
```

### Demo 1: Upload & Download (10 minutes)
```bash
# Create test data
mkdir -p /tmp/demo-upload
for i in $(seq 1 5); do
  echo "Document $i: UMRS is a unified multimodal retrieval system." > /tmp/demo-upload/doc$i.txt
done

# Upload
umrs-client data dir-upload /tmp/demo-upload mnagaso3-text

# Verify upload
umrs-client data list mnagaso3-text

# Download
umrs-client data download mnagaso3-text /tmp/demo-download --force-overwrite

# Show downloaded files
ls -la /tmp/demo-download/
```

### Demo 2: Vision Inference (15 minutes)
```bash
# Check available models
umrs-client model list --scope system | head -10

# Upload images (if demo-sam3 available)
umrs-client data dir-upload \
  demos/sample_data/images demo-sam3 2>/dev/null || \
  echo "demo-sam3 bucket not available — showing upload concept only"

# Show upload concept
echo "In production:"
echo "  umrs-client data dir-upload /path/to/images demo-sam3"
echo ""
echo "SAM3 processes each image on GPU via Slurm, returning:"
echo "  - Bounding boxes"
echo "  - Segmentation masks"
echo "  - Confidence scores"
echo "  - Overlay images"
```

### Demo 3: HMAE Pipeline (15 minutes)
```bash
# Show HMAE model
umrs-client model list --scope user | grep -i hmae || echo "HMAE model not staged on server"

# Show upload concept
echo "HMAE Reconstruction Pipeline:"
echo "  1. Upload volume slices → user1-hmae bucket"
echo "  2. Server dispatches to HMAE model on GPU"
echo "  3. HMAE reconstructs each slice"
echo "  4. Download reconstruction outputs"
echo ""
echo "umrs-client data dir-upload /path/to/slices user1-hmae"
```

### Q&A Prep: Doctor Mode (5 minutes)
```bash
# Show system health
umrs-client doctor

# Show specific components
umrs-client doctor server
umrs-client doctor models
umrs-client doctor storage
```

---

## Troubleshooting Commands

### Common Issues
```bash
# "Server unreachable"
umrs-client ping
# Check network/proxy settings

# "Bucket alias not found"
umrs-client bucket list
# Use available alias or ask admin to provision

# "No models available"
umrs-client model list --scope system
# Ask admin to stage models

# "No RAG pipelines"
r4 rag ingest list --project default
# Ask admin to create RAG pipeline

# "Workflow not found"
umrs-client workflow list
# Upload workflow: umrs-client workflow upload <dir> --name <name>
```

### Health Check
```bash
# Comprehensive system check
umrs-client doctor

# Check individual components
umrs-client doctor server
umrs-client doctor auth
umrs-client doctor storage
umrs-client doctor models
umrs-client doctor rag
umrs-client doctor workflows
umrs-client doctor jobs
```

---

## Key Concepts Cheat Sheet

### Bucket Aliases
- Each bucket represents a data modality (text, vision, etc.)
- Created via experience definitions in site.yaml
- User config references bucket aliases for uploads/downloads

### Model Types
- `embedding` — Vector embeddings for text/RAG
- `vision` — Image models (SAM3, HMAE, etc.)
- `text` — Language models (LLMs)

### Model Scopes
- `system` — Available to all users (admin-managed)
- `project` — Project team access
- `user` — Personal to one user

### Workflow Types
- `uploaded` — Workflow code uploaded with job
- `server` — Server-side workflow definitions
- `slurm` — Slurm-optimized workflow execution

### Data Modalities
- `text` — Text documents, files
- `image` — Images (PNG, JPG, etc.)
- `movie` — Video files
- `custom` — Custom modality definitions
