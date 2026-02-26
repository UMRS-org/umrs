# Slurm Development Environment — E2E Demo

A Podman Compose setup for running a real Slurm cluster on a single workstation.
UMRS loads an embedding model running in Rivault through the OpenAI API.
The control plane (slurmctld, UMRS, postgres, minio) runs in containers.
The compute node (slurmd) runs directly on the host for native GPU access.
Rivault is launched by Slurm jobs using the host's native podman — no nested
containers needed.
An always-on LLM service (vLLM + LiteLLM + OpenWebUI) runs as a separate compose
project for RAG chat and interactive use.

## Architecture

```
  Host (bare metal, GPU, native podman)
  ├── slurmd   (native process, started by scripts/start-host-slurmd.sh)
  ├── munged   (native process, shares key with container munge)
  │
  └── podman-compose (control plane only)
       │
       │    slurm-net (172.28.0.0/16)
       ├── ai-login     slurmctld + UMRS server + RAG service
       ├── munge        MUNGE key generator
       ├── postgres      pgvector database
       ├── minio         S3-compatible storage
       └── umrs-client   test client

  When Slurm job runs on host:
  └── podman compose (Rivault stack, native GPU via CDI)
       ├── vLLM, LiteLLM, NGINX
       ├── ll-postgres, mcpo, mcp-time
       └── ...

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
- `slurmd` and `munge` installed on the host (`sudo apt-get install -y slurmd munge libmunge-dev`)
- Model weights at `MODEL_PATH` (default: `./rivault/models`)

## Quick Start — Full E2E Demo

```bash
cd demos/slurm-dev

# 1. Start container control plane
podman-compose up -d

# 2. Start host slurmd (in another terminal)
sudo bash scripts/start-host-slurmd.sh

# 3. Run the complete demo
bash scripts/run_e2e_demo.sh
```

## Manual Step-by-Step

```bash
cd demos/slurm-dev

# 1. Start container control plane
podman-compose up -d

# 2. Start host slurmd (another terminal)
sudo bash scripts/start-host-slurmd.sh

# 3. Check Slurm status
podman exec ai-login sinfo -N

# 4. Upload test data to MinIO
podman exec umrs-client bash /opt/scripts/setup_demo_data.sh

# 5. Stop Slurm embedding jobs (free GPU for chat LLM)
podman exec ai-login scancel --all

# 6. Start always-on LLM (separate compose project)
podman network create --internal --disable-dns --subnet 192.168.5.0/24 demo-llm-net
cd rivault
podman-compose --env-file demo-llm.env \
    -f demo-llm.yaml -f x_nvidia-override.yaml up -d

# 7. Open the chat UI
#    http://localhost:13000
```

## Services

### Core Services (docker-compose.yaml)

| Service | Description | Port |
|---------|-------------|------|
| ai-login | Slurm controller + UMRS server + RAG service | 6817, 5555 (ZMQ), 8081 (RAG) |
| munge | MUNGE authentication | - |
| postgres | pgvector database | 15432 |
| minio | S3-compatible storage | 19000 (API), 19001 (Console) |
| umrs-client | Test client for UMRS API | - |

### Host Services (started via scripts/start-host-slurmd.sh)

| Service | Description |
|---------|-------------|
| slurmd | Slurm compute daemon (registers as compute-01) |
| munged | MUNGE authentication (shares key with container munge) |

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
Models are mounted into the ai-login container as `/shared/models`.
When Rivault runs on the host, `VLLM_PATH_TO_HUGGINGFACE` in `serve-model.slurm`
points to the model directory directly.

For the always-on LLM, set `VLLM_PATH_TO_HUGGINGFACE` in `rivault/demo-llm.env`.

## Slurm Commands

```bash
podman exec ai-login sinfo -N           # Show nodes
podman exec ai-login squeue             # Show job queue
podman exec ai-login sbatch /root/.config/umrs/rivault_data/serve-model.slurm  # Submit Rivault job
podman exec ai-login scancel <job_id>   # Cancel a job
cat job-info/rivault-job-<job_id>.json   # Job info (on host)
```

## Troubleshooting

### "Node is down"
```bash
podman exec ai-login scontrol update nodename=compute-01 state=resume
# Also check that host slurmd is still running:
pgrep -x slurmd || echo "slurmd not running — restart with: sudo bash scripts/start-host-slurmd.sh"
```

### "Munge authentication failed"
```bash
# Restart containers (regenerates munge key)
podman-compose down -v
podman-compose up -d
# Then restart host slurmd to pick up the new key
sudo bash scripts/start-host-slurmd.sh
```

### Rivault containers not starting
```bash
# Rivault runs on the host natively, check with podman directly
podman ps -a
podman logs <container_id>
```

### Host slurmd not registering
```bash
# Check host slurmd logs
cat /var/log/slurm/slurmd.log
# Ensure slurmctld port 6817 is exposed and reachable
curl -s localhost:6817 || echo "port 6817 not open — is ai-login running?"
# Resume the node from slurmctld
podman exec ai-login scontrol update nodename=compute-01 state=resume
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
├── docker-compose.yaml        # Main compose file (control plane)
├── conf_umrs.json             # UMRS server configuration
├── slurm.conf                 # Slurm cluster configuration
├── gres.conf                  # GPU resource configuration
├── mini-rivault-job.sh        # Slurm job: launches Rivault stack on host
├── .env                       # Environment variables
├── s3cfg                      # MinIO S3 credentials
├── Dockerfile.ai-login        # Controller + UMRS container
├── Dockerfile.munge           # Munge authentication
├── Dockerfile.umrs-client     # Test client container
├── entrypoint-ai-login.sh     # Controller entrypoint
├── entrypoint-munge.sh        # Munge entrypoint
├── munge-key/                 # Shared munge key (container → host)
├── job-info/                  # Shared job info (host → container)
├── user_files/
│   └── user_rivault.json      # User config (text + image modalities)
├── test_data/
│   └── text/                  # Science text documents for RAG
├── scripts/
│   ├── start-host-slurmd.sh   # Start slurmd + munged on host
│   ├── run_e2e_demo.sh        # Full E2E demo orchestration
│   ├── setup_demo_data.sh     # MinIO bucket/data setup
│   ├── demo_text_rag.py       # Text RAG demo
│   └── demo_image_rag.py      # Image RAG demo (CLIP + SAM3)
└── rivault/                   # Rivault / LLM stack files
    ├── serve-model.slurm      # Launch script (Slurm batch)
    ├── serve-model.yaml       # Rivault compose (embedding model)
    ├── demo-llm.yaml          # Always-on LLM compose (chat + OpenWebUI)
    ├── demo-llm.env           # Environment for demo-llm.yaml
    ├── x_nvidia-override.yaml # NVIDIA CDI GPU overrides
    ├── litellm_fs/            # LiteLLM config
    ├── mcpo_fs/               # MCP server config
    ├── nginx_fs/              # NGINX proxy config
    └── models/                # Model weights cache
```
