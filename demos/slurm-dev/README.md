# Slurm Development Environment

A Docker/Podman Compose setup for running a real Slurm cluster on a single workstation.
UMRS loads an embedding model running in Rivault through the OpenAI API.
Rivault is launched as podman containers on the Slurm compute node (podman-in-podman).

## Architecture

```
                    slurm-net (172.28.0.0/16)
  ┌─────────────────────────────────────────────────────────────┐
  │                                                             │
  │  ┌─────────────┐         ┌────────────────────────────────┐ │
  │  │  ai-login    │  sbatch │  compute-01 (privileged)      │ │
  │  │  slurmctld   │───────>│  slurmd                        │ │
  │  │  UMRS server │        │  ┌──────────────────────────┐  │ │
  │  │              │ :11434 │  │ podman (Rivault stack)   │  │ │
  │  │  rivault://  │<───────│  │ NGINX→LiteLLM→vLLM      │  │ │
  │  │  openai API  │  SSH   │  │ ll-postgres, mcpo, mcp   │  │ │
  │  └─────────────┘ tunnel  │  └──────────────────────────┘  │ │
  │         │                └────────────────────────────────┘ │
  │  ┌──────┴──────┐  ┌──────┐  ┌───────┐                     │
  │  │ postgres     │  │ minio│  │ munge │                     │
  │  │ pgvector     │  │ S3   │  │ auth  │                     │
  │  └─────────────┘  └──────┘  └───────┘                     │
  └─────────────────────────────────────────────────────────────┘
```

## Prerequisites

- Docker/Podman and Docker Compose / podman-compose
- 8GB+ RAM available
- (Optional) NVIDIA GPU + Container Toolkit for GPU support

## Quick Start

```bash
# 1. Configure environment
cp env.template .env
# Edit .env to set MODEL_PATH to your local models directory

# 2. Start the cluster (podman-compose)
podman-compose up -d

# 3. Launch Rivault on Slurm + start UMRS (submit Slurm job)
podman exec ai-login sbatch /etc/slurm/mini-rivault-job.sh

# 4. Check status
podman-compose ps
```

## Components

| Service | Description | Port |
|---------|-------------|------|
| ai-login | Slurm controller + UMRS server | 6817 |
| compute-01 | Compute node (runs Rivault via podman) | SSH: 2222, LiteLLM: 11434 |
| munge | Authentication | - |
| postgres | pgvector database | 15432 |
| minio | S3 storage | 19000, 19001 |
| umrs-client | Lightweight client/test runner for UMRS API | - |

## Shell Commands

```bash
podman-compose up -d      # Start Slurm cluster
podman exec ai-login sbatch /etc/slurm/mini-rivault-job.sh         # Launch Rivault + UMRS via Slurm
podman-compose ps          # Show cluster status
podman exec ai-login sbatch /etc/slurm/test-job.sh                # Submit a test job
podman-compose down -v     # Stop cluster and remove volumes
podman-compose -f demos/slurm-dev/docker-compose.yaml build umrs-client  # Build the umrs-client image
podman-compose -f demos/slurm-dev/docker-compose.yaml up -d umrs-client # Start only the umrs-client service
podman logs --follow umrs-client                                              # Tail umrs-client logs
```

## Direct Slurm Commands

```bash
# Show nodes
podman exec ai-login sinfo

# Show job queue
podman exec ai-login squeue

# Submit Rivault job manually
podman exec ai-login sbatch /etc/slurm/mini-rivault-job.sh

# Cancel a job
podman exec ai-login scancel <job_id>

# View job output
podman exec ai-login cat /var/run/rivault/rivault-job-<job_id>.json

# Check UMRS logs
podman exec ai-login tail -f /var/log/umrs.log

# Check Rivault endpoint from compute node
podman exec compute-01 curl -s http://localhost:11434/v1/models
```

## Configuration

### UMRS Config Files

The following config files are mounted into ai-login at `~/.config/umrs/`:

- `conf_umrs_rivault_openai.json` - Server config (from `src/umrs_cfg_example/`)
- `users/rivault_open_ai.json` - Rivault user config with `rivault://openai://` model type
- `users/openai_api.json` - Direct OpenAI user config

### Model Path

Set `MODEL_PATH` in `.env` to point to your local HuggingFace model cache.
Models are mounted: host → compute-01:/models → vLLM container.

### Adding GPU Support

1. Edit `slurm.conf` to uncomment GRES lines:
   ```
   GresTypes=gpu
   NodeName=compute-01 Gres=gpu:1
   ```

2. Edit `docker-compose.yaml` to add GPU resources to compute-01

## Troubleshooting

### "Node is down"

```bash
docker exec ai-login scontrol update nodename=compute-01 state=resume
```

### "Munge authentication failed"

```bash
podman-compose down -v
podman-compose up -d
```

### Rivault containers not starting

Check podman logs inside compute-01:
```bash
podman exec compute-01 podman ps -a
podman exec compute-01 podman logs <container_id>
```

### UMRS can't connect to model

Check SSH tunnel and Rivault health:
```bash
podman exec ai-login tail -f /var/log/umrs.log
podman exec compute-01 curl -s http://localhost:11434/v1/models
```

## File Structure

```
slurm-dev/
├── docker-compose.yaml      # Main compose file
├── slurm.conf               # Slurm configuration
├── mini-rivault-job.sh      # Slurm job: launches Rivault via podman-compose
├── s3cfg                    # MinIO S3 config for UMRS
├── env.template             # Environment template
├── (setup.sh removed)       # Use podman-compose commands instead
├── Dockerfile.munge         # Munge authentication container
├── Dockerfile.ai-login      # Controller + UMRS + SSH container
├── Dockerfile.compute       # Compute node (podman-in-podman) container
├── entrypoint-munge.sh      # Munge entrypoint
├── entrypoint-ai-login.sh   # Controller entrypoint
└── entrypoint-compute.sh    # Compute entrypoint
```

## Quick Local Demo (UMRS server + client)

Use this when the Slurm cluster and Rivault have been launched with `./setup.sh start` and `./setup.sh rivault`.

1. Copy example server config into your UMRS config dir (or pass it with `-c`):

```bash
mkdir -p ~/.config/umrs/users
cp src/umrs_cfg_example/umrs/conf_umrs_rivault_openai.json ~/.config/umrs/conf_umrs.json
cp demos/slurm-dev/user_files/user_rivault.json ~/.config/umrs/users/
cp docker/s3cfg_user1 ~/.s3cfg_user1
```

2. Start the UMRS server locally (it will connect to the running cluster services):

```bash
python3 src/server_umrs.py -c ~/.config/umrs/conf_umrs.json
```

3. In another terminal, upload a small test file using the client (this will push the file to MinIO and notify the server):

```bash
# create a tiny test file
echo "hello rivault" > /tmp/rivault_test.txt

# use the client config (the client default is ./conf_umrs.json; pass -c to point to your config)
python3 src/client_umrs.py upload_file /tmp/rivault_test.txt bucket-vLLM rivault_user -c ~/.config/umrs/users/user_rivault.json --server-ip 127.0.0.1
```

4. Watch the server logs for the Slurm submission and Rivault launch steps. The server will submit a Slurm job, wait for the compute node, create an SSH tunnel and then call the OpenAI-compatible endpoint to compute embeddings.

```bash
# in the server terminal (or if running inside ai-login container)
tail -f /var/log/umrs.log
```

Notes:
- If ports (MinIO:19000, Slurm:6817, LiteLLM:11434) are in use locally, stop the conflicting containers/processes or adjust port mappings in `docker-compose.yaml` before starting the cluster.
- If Rivault is not reachable, confirm the Slurm job ran on `compute-01` and that `podman` services inside the compute node are up.

