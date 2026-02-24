# Slurm Development Environment — E2E Demo

A Podman Compose setup for running a real Slurm cluster on a single workstation.
UMRS loads an embedding model running in Rivault through the OpenAI API.
Rivault is launched as podman containers on the Slurm compute node (podman-in-podman).
An always-on LLM service (vLLM + LiteLLM + OpenWebUI) runs as a separate compose
project for RAG chat and interactive use.

## Architecture

```
                    slurm-net (172.28.0.0/16)
  ┌─────────────────────────────────────────────────────────────────┐
  │                                                                 │
  │  ┌─────────────┐         ┌────────────────────────────────────┐ │
  │  │  ai-login    │  sbatch │  compute-01 (privileged)          │ │
  │  │  slurmctld   │───────>│  slurmd                            │ │
  │  │  UMRS server │        │  ┌──────────────────────────────┐  │ │
  │  │  RAG service │ :11434 │  │ podman (Rivault stack)       │  │ │
  │  │  rivault://  │<───────│  │ NGINX → LiteLLM → vLLM      │  │ │
  │  │  openai API  │  SSH   │  │ ll-postgres, mcpo, mcp       │  │ │
  │  └─────────────┘ tunnel  │  └──────────────────────────────┘  │ │
  │         │                └────────────────────────────────────┘ │
  │  ┌──────┴──────┐  ┌──────┐  ┌───────┐  ┌────────────┐        │
  │  │ postgres     │  │ minio│  │ munge │  │ umrs-client│        │
  │  │ pgvector     │  │ S3   │  │ auth  │  │            │        │
  │  └─────────────┘  └──────┘  └───────┘  └────────────┘        │
  └─────────────────────────────────────────────────────────────────┘

           demo-llm-net (192.168.5.0/24) — separate compose project
  ┌──────────────────────────────────────────┐
  │  vLLM → LiteLLM → OpenWebUI             │
  │  NGINX, ll-postgres, mcpo, mcp-time      │
  │  Port 14000 (API)  Port 13000 (WebUI)    │
  └──────────────────────────────────────────┘
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

# 4. Stop Slurm embedding jobs (free GPU for chat LLM)
podman exec ai-login scancel --all

# 5. Start always-on LLM (separate compose project)
podman network create --internal --disable-dns --subnet 192.168.5.0/24 demo-llm-net
cd rivault
podman-compose --env-file demo-llm.env \
    -f demo-llm.yaml -f x_nvidia-override.yaml up -d

# 6. Open the chat UI
#    http://localhost:13000
```

## Services

### Core Services (docker-compose.yaml)

| Service | Description | Port |
|---------|-------------|------|
| ai-login | Slurm controller + UMRS server + RAG service | 6817, 5555 (ZMQ), 8081 (RAG) |
| compute-01 | Compute node (runs Rivault via podman) | SSH: 2222 |
| munge | MUNGE authentication | - |
| postgres | pgvector database | 15432 |
| minio | S3-compatible storage | 19000 (API), 19001 (Console) |
| umrs-client | Test client for UMRS API | - |

### Always-On LLM (rivault/demo-llm.yaml)

| Service | Description | Port |
|---------|-------------|------|
| vllm | vLLM inference engine (GPU) | - |
| litellm | LiteLLM OpenAI-compatible proxy | - |
| nginx | NGINX reverse proxy | 14000 |
| ll-postgres | Ephemeral LiteLLM state store | - |
| mcpo / mcp-time | MCP tool servers | - |
| open-webui | Chat interface | 13000 |

Start with:
```bash
cd rivault
podman-compose --env-file demo-llm.env \
    -f demo-llm.yaml -f x_nvidia-override.yaml up -d
```

## GPU Sharing

The NVIDIA GB10 does not support MIG. The Slurm-launched Rivault embedding
model and the always-on chat LLM both need the GPU. Run them sequentially:

1. **Embedding phase**: Slurm job runs vLLM for embeddings on the GPU
2. **Cancel Slurm job**: `podman exec ai-login scancel --all`
3. **Chat phase**: Start the always-on LLM compose project

## Demo Scripts

| Script | Description | Run from |
|--------|-------------|----------|
| `scripts/run_e2e_demo.sh` | Complete E2E orchestration | Host |
| `scripts/setup_demo_data.sh` | Create MinIO buckets, upload test data | ai-login or umrs-client |
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
| `rivault/demo-llm.env` | Environment vars for the always-on LLM stack |
| `rivault/demo-llm.yaml` | Compose file for always-on LLM + OpenWebUI |
| `.env` | Environment variables for the Slurm stack |
| `s3cfg` | MinIO S3 credentials |

### Model Path

Set `MODEL_PATH` in `.env` to your local model weights directory.
Models are mounted: host → compute-01:/models → vLLM container.

For the always-on LLM, set `VLLM_PATH_TO_HUGGINGFACE` in `rivault/demo-llm.env`.

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
podman logs ai-login
```

### RAG service not running
```bash
podman exec ai-login cat /var/log/umrs-rag.log
```

### vLLM not loading model
```bash
# Check logs of the vllm container in the demo-llm project
podman logs rivault_vllm_1
# Ensure VLLM_PATH_TO_HUGGINGFACE in rivault/demo-llm.env points to valid model weights
```

## File Structure

```
slurm-dev/
├── docker-compose.yaml        # Main compose file (Slurm + UMRS)
├── conf_umrs.json             # UMRS server configuration
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
└── rivault/                   # Rivault / LLM stack files
    ├── serve-model.slurm      # Launch script (Slurm batch)
    ├── serve-model.yaml       # Rivault compose (embedding model)
    ├── demo-llm.yaml          # Always-on LLM compose (chat + OpenWebUI)
    ├── demo-llm.env           # Environment for demo-llm.yaml
    ├── x_nvidia-override.yaml # NVIDIA runtime/device overrides
    ├── litellm_fs/            # LiteLLM config
    ├── mcpo_fs/               # MCP server config
    ├── nginx_fs/              # NGINX proxy config
    └── models/                # Model weights cache
```
