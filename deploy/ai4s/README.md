# UMRS AI4S Supercomputer Deployment

## Overview

This deployment profile targets the RIKEN AI for Science Supercomputer
(400× GB200 NVL4 nodes) from a front-end server (`umrs-front`).

## Architecture

```
┌──────────┐          ┌──────────────────────────────────────┐
│  Client  │──SSH────▶ │  Front-End Server (umrs-front)       │
│  (r4 CLI)│          │  user: mnagaso                       │
└──────────┘          │  ┌──────────┐  ┌──────────────────┐  │
                      │  │ Gateway  │  │  UMRS Server     │  │
                      │  │  :8082   │  │  :8002           │  │
                      │  └──────────┘  └────────┬─────────┘  │
                      │  ┌──────────┐           │             │
                      │  │ Postgres │     SSH ControlMaster   │
                      │  │  :5433   │           │             │
                      │  └──────────┘           │             │
                      │  ┌──────────┐           │             │
                      │  │  MinIO   │           │             │
                      │  │  :9002   │◀──────────┼── reverse   │
                      │  └──────────┘    tunnel │             │
                      └──────────────────────────┼─────────────┘
                                                 │
                              SSH (ea0022@134.160.189.24)
                                   R-CCS network
                                                 │
                      ┌──────────────────────────▼─────────────┐
                      │  AI4S Login Node                        │
                      │  user: ea0022                           │
                      │  - sbatch/squeue/scancel via SSH        │
                      │  - Reverse tunnel: 9000 → front MinIO  │
                      │  - NO persistent services               │
                      │  - Lustre shared filesystem             │
                      └──────────────────────────┬─────────────┘
                                                 │
                                           sbatch/Slurm
                                                 │
                      ┌──────────────────────────▼─────────────┐
                      │  AI4S Compute (GB200 NVL4)             │
                      │  - 2× Grace CPU (aarch64)              │
                      │  - 4× B200 GPU (173.2 GiB each)       │
                      │  - Mini-RiVault via Apptainer/Singularity│
                      │  - Models from Lustre                   │
                      │  - NVLink-C2C + NVLink inter-GPU       │
                      └────────────────────────────────────────┘
```

## Hardware Specs (per compute node)

| Component | Specification |
|-----------|--------------|
| CPU | 2× NVIDIA Grace (aarch64) |
| GPU | 4× NVIDIA B200 |
| GPU Memory | 173.2 GiB per GPU (692.8 GiB total) |
| CPU Memory | 960 GiB |
| Interconnect | InfiniBand XDR 800 Gbps |
| Local NVMe | 7.68 TB |

## Partitions

| Partition | Nodes | GPUs/node | Time Limit | GPU Memory |
|-----------|-------|-----------|------------|------------|
| 1n1gpu | 1 | 1 | 96 hours | 173.2 GiB |
| 1n2gpu | 1 | 2 | 96 hours | 346.4 GiB |
| 1n4gpu | 1 | 4 | 96 hours | 692.8 GiB |
| 2n4gpu | 2 | 4 | 96 hours | 1385.6 GiB |
| 4n4gpu | 4 | 4 | 96 hours | 2771.2 GiB |
| 4n4gpu-p | 4 | 4 | unlimited | 2771.2 GiB |

## User Accounts

| Host | Username | Purpose |
|------|----------|---------|
| umrs-front | mnagaso | Runs persistent services |
| rccs-cloud | masaru.nagaso | RCCS login (may be ProxyJump) |
| AI4S | ea0022 | Slurm job submission |

## Ansible Deployment (Recommended)

The `umrs-front` Ansible profile automates bring-up of the AI4S UMRS stack:

```bash
cd deploy/ansible

# Deploy AI4S backend only
./bootstrap.sh --profile umrs-front-ai4s

# Deploy both RCCS Cloud + AI4S
./bootstrap.sh --profile umrs-front

# Dry run
./bootstrap.sh --profile umrs-front-ai4s --check

# Stop UMRS servers
./bootstrap.sh --profile umrs-front --teardown
```

See [deploy/ansible/README.md](../../../deploy/ansible/README.md) for full
options.

## Manual Setup

### 1. SSH Key Configuration

Generate a dedicated key for the AI4S account:
```bash
ssh-keygen -t ed25519 -f ~/.ssh/ai4s_ed25519 -C "umrs-ai4s"
```

Copy to AI4S login:
```bash
ssh-copy-id -i ~/.ssh/ai4s_ed25519 ea0022@134.160.189.24
```

Configure `~/.ssh/config`:
```
Host ai4s
  HostName 134.160.189.24
  User ea0022
  IdentityFile ~/.ssh/ai4s_ed25519
  ControlMaster auto
  ControlPath /tmp/umrs-ssh-ai4s-%p
  ControlPersist 10m
  # If direct access requires going through rccs-cloud:
  # ProxyJump masaru.nagaso@rccs-cloud
```

### 2. Environment Configuration

```bash
cp env.example .env
# Edit .env with real passwords and keys
```

### 3. Start Services

```bash
chmod +x start.sh stop.sh
./start.sh
```

### 4. Verify

```bash
curl http://localhost:8002/health       # UMRS (AI4S)
curl http://localhost:8082/health       # Gateway (AI4S)
```

## Container Runtime: Apptainer/Singularity

AI4S uses Apptainer (successor to Singularity) for HPC containers.
The Mini-RiVault SIF image must be pre-staged on the Lustre filesystem.

Key differences from Podman (rccs-cloud):
- No rootless daemon; containers run as user processes
- `--nv` flag for GPU passthrough (no CDI/device mapping)
- Bind mounts with `--bind` instead of `-v`
- No compose; single `apptainer run` command in sbatch script

Example sbatch with Apptainer:
```bash
#!/bin/bash
#SBATCH --partition=1n1gpu
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00

apptainer run --nv \
  --bind /lustre/models:/models:ro \
  --bind /lustre/scratch:/scratch \
  /lustre/images/mini-rivault.sif \
  --model /models/Qwen3-32B --port 8001
```

## Coexistence with rccs-front

Both profiles can run simultaneously on the same umrs-front:

| Service | rccs-front port | ai4s port |
|---------|----------------|-----------|
| UMRS API | 8000 | 8002 |
| Gateway | 8080 | 8082 |
| PostgreSQL | 5432 | 5433 |
| MinIO (S3) | 9000 | 9002 |
| MinIO Console | 9001 | 9003 |
| Tunnel pool | 12000-12999 | 13000-13999 |

## Troubleshooting

- **SSH fails to AI4S**: Verify `134.160.189.24` is reachable from umrs-front (R-CCS internal network)
- **"Permission denied"**: Verify key is authorized for `ea0022` on AI4S
- **ProxyJump needed?**: If umrs-front cannot directly reach `134.160.189.24`, add `ProxyJump masaru.nagaso@rccs-cloud` to SSH config
- **Singularity image not found**: Stage the SIF on Lustre before submitting jobs
- **GPU not accessible**: Ensure `--nv` flag is in sbatch script and partition has GPU allocation
