# Slurm Development Environment — E2E Demo

A Podman Compose setup for running a real Slurm cluster on a single workstation.
UMRS loads an embedding model running in Rivault through the OpenAI API.
Rivault is launched as podman containers on the Slurm compute node (podman-in-podman).
An always-on LLM service (vLLM + LiteLLM + OpenWebUI) is available for RAG chat.

## Architecture

```
                    slurm-net (172.28.0.0/16)
  ┌─────────────────────────────────────────────────────────────────┐
  │                                                                 │
  │  ┌─────────────┐         ┌────────────────────────────────────┐ │
  │  │  ai-login    │  sbatch │  compute-01 (privileged)          │ │
  │  │  slurmctld   │───────>│  slurmd                            │ │
  │  │  UMRS server │        │  ┌──────────────────────────────┐  │ │
  │  │              │ :11434 │  │ podman (Rivault stack)       │  │ │
  │  │  rivault://  │<───────│  │ NGINX → LiteLLM → vLLM      │  │ │
  │  │  openai API  │  SSH   │  │ ll-postgres, mcpo, mcp       │  │ │
  │  └─────────────┘ tunnel  │  └──────────────────────────────┘  │ │
  │         │                └────────────────────────────────────┘ │
  │  ┌──────┴──────┐  ┌──────┐  ┌───────┐                         │
  │  │ postgres     │  │ minio│  │ munge │                         │
  │  │ pgvector     │  │ S3   │  │ auth  │                         │
  │  └─────────────┘  └──────┘  └───────┘                         │
  │                                                                 │
  │  ┌──────────────────────────────────────────┐  (profile: llm)  │
  │  │  Always-On LLM                           │                  │
  │  │  llm-vllm → llm-litellm → OpenWebUI     │                  │
  │  │  Port 14000 (API)  Port 13000 (WebUI)    │                  │
  │  └──────────────────────────────────────────┘                  │
  └─────────────────────────────────────────────────────────────────┘
```

## Prerequisites

- `podman` and `podman-compose` (`pip install podman-compose`)
- 8GB+ RAM available
- NVIDIA GPU + nvidia-container-toolkit for GPU support
- Model weights at `MODEL_PATH` (default: `./rivault/models`)

## Quick Start — Full E2E Demo

```bash
cd demos/slurm-dev

# Run the complete demo (uploads data, embeds, searches, starts LLM + OpenWebUI)
bash scripts/run_e2e_demo.sh
```

## Manual Step-by-Step

```bash
cd demos/slurm-dev

# 1. Start core Slurm cluster
podman-compose up -d

# 2. Check Slurm status
podman exec ai-login sinfo

# 3. Upload test data to MinIO
podman exec umrs-client bash /opt/scripts/setup_demo_data.sh

# 4. Start always-on LLM (vLLM + LiteLLM + OpenWebUI)
#    NOTE: Stop any Slurm embedding jobs first (single GPU)
podman exec ai-login scancel --all
#podman-compose --profile llm up -d
podman-compose -f "$COMPOSE_FILE" -f rivault/demo-llm.yaml -f rivault/x_nvidia-override.yaml --profile llm up -d
# 5. Open the chat UI
#    http://localhost:13000
```

## Services

### Core Services (always started)

| Service | Description | Port |
|---------|-------------|------|
| ai-login | Slurm controller + UMRS server | 6817, 5555 (ZMQ) |
| compute-01 | Compute node (runs Rivault via podman) | SSH: 2222 |
| munge | MUNGE authentication | - |
| postgres | pgvector database | 15432 |
| minio | S3-compatible storage | 19000 (API), 19001 (Console) |
| umrs-client | Test client for UMRS API | - |

### Always-On LLM Services (profile: `llm`)

| Service | Description | Port |
|---------|-------------|------|
| llm-vllm | vLLM inference engine (GPU) | 8000 (internal) |
| llm-litellm | LiteLLM OpenAI-compatible proxy | 14000 |
| llm-postgres | Ephemeral LiteLLM state store | - |
| open-webui | Chat interface | 13000 |

Start with: `podman-compose --profile llm up -d`

## GPU Sharing

The NVIDIA GB10 does not support MIG. The Slurm-launched Rivault embedding
model and the always-on chat LLM both need the GPU. Run them sequentially:

1. **Embedding phase**: Slurm job runs vLLM for embeddings on the GPU
2. **Cancel Slurm job**: `podman exec ai-login scancel --all`
3. **Chat phase**: Start always-on LLM: `podman-compose --profile llm up -d`

## Demo Scripts

| Script | Description | Run from |
|--------|-------------|----------|
| `scripts/run_e2e_demo.sh` | Complete E2E orchestration | Host |
| `scripts/setup_demo_data.sh` | Create MinIO buckets, upload test data | ai-login |
| `scripts/demo_text_rag.py` | Text RAG: upload, embed, search, QA | umrs-client |
| `scripts/demo_image_rag.py` | Image RAG: CLIP embeddings + SAM3 | umrs-client |

## Test Data

- **Text**: 5 science documents in `test_data/text/` (seismic waves, ML geophysics, HPC, RiVault, numerical methods)
- **Images**: Cat, Dog, Eagle images from `data/img/` (mounted into containers)

## Configuration

### Key Config Files

| File | Purpose |
|------|---------|
| `conf_umrs.json` | UMRS server config (points to user config) |
| `user_files/user_rivault.json` | User config: text (Rivault embedding) + image (CLIP) modalities |
| `llm-litellm-config.yaml` | LiteLLM config for always-on LLM |
| `.env` | All environment variables |
| `s3cfg` | MinIO S3 credentials |

### Model Path

Set `MODEL_PATH` in `.env` to your local model weights directory.
Models are mounted: host → compute-01:/models → vLLM container.

For the always-on LLM, set `LLM_MODEL_PATH` in `.env`.

## Slurm Commands

```bash
podman exec ai-login sinfo              # Show nodes
podman exec ai-login squeue             # Show job queue
podman exec ai-login sbatch /root/.config/umrs/rivault_data/serve-model.slurm  # Submit Rivault job
podman exec ai-login scancel <job_id>   # Cancel a job
podman exec ai-login cat /var/run/rivault/rivault-job-<job_id>.json  # Job info
```

## Troubleshooting

### "Node is down"
```bash
podman exec ai-login scontrol update nodename=compute-01 state=resume
```

### "Munge authentication failed"
```bash
podman-compose down -v
podman-compose up -d
```

### Rivault containers not starting
```bash
podman exec compute-01 podman ps -a
podman exec compute-01 podman logs <container_id>
```

### UMRS server not running
```bash
podman exec ai-login cat /var/log/umrs-server.log
```

### vLLM not loading model
```bash
podman logs llm-vllm
# Ensure MODEL_PATH / LLM_MODEL_PATH points to valid model weights
```

## File Structure

```
slurm-dev/
├── docker-compose.yaml        # Main compose file (core + LLM profile)
├── conf_umrs.json             # UMRS server configuration
├── llm-litellm-config.yaml    # LiteLLM config for always-on LLM
├── slurm.conf                 # Slurm cluster configuration
├── gres.conf                  # GPU resource configuration
├── mini-rivault-job.sh        # Slurm job: launches Rivault stack
├── .env                       # Environment variables
├── s3cfg                      # MinIO S3 credentials
├── Dockerfile.ai-login        # Controller + UMRS container
├── Dockerfile.compute         # Compute node (podman-in-podman)
├── Dockerfile.munge           # Munge authentication
├── Dockerfile.umrs-client     # Test client container
├── entrypoint-ai-login.sh     # Controller entrypoint
├── entrypoint-compute.sh      # Compute entrypoint
├── entrypoint-munge.sh        # Munge entrypoint
├── user_files/
│   └── user_rivault.json      # User config (text + image modalities)
├── test_data/
│   └── text/                  # Science text documents for RAG
├── scripts/
│   ├── run_e2e_demo.sh        # Full E2E demo orchestration
│   ├── setup_demo_data.sh     # MinIO bucket/data setup
│   ├── demo_text_rag.py       # Text RAG demo
│   └── demo_image_rag.py      # Image RAG demo (CLIP + SAM3)
└── rivault/                   # Rivault inner stack (podman-in-podman)
    ├── serve-model.slurm      # Launch script
    ├── serve-model.yaml       # Rivault compose (NGINX, LiteLLM, vLLM, MCP)
    ├── demo-llm.yaml          # Rivault LLM compose (recommended)
    ├── x_nvidia-override.yaml  # NVIDIA runtime/device overrides
    ├── litellm_fs/            # LiteLLM config
    ├── mcpo_fs/               # MCP server config
    ├── nginx_fs/              # NGINX proxy config
    └── models/                # Model weights cache
```
