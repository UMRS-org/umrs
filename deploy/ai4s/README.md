# UMRS AI4S Supercomputer Deployment

*Validated: 2026-05-19 — Gateway-backed Slurm smoke e2e passing*

## Overview

This deployment profile targets the RIKEN AI for Science Supercomputer
(400× GB200 NVL4 nodes) from a front-end server (`umrs-front`).

The profile is **validated at smoke level**: UMRS on `umrs-front` submits real
Slurm jobs to AI4S, opens an SSH tunnel back to the allocated compute node, and
the client-facing Gateway (`:18082`) serves chat requests through that tunnel.
Full production requires staging a vLLM Apptainer SIF image and model weights
on AI4S Lustre (see §Production Readiness below).

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

# Full ready check (gateway + sidecar + umrs)
curl http://localhost:18082/ready
```

## AI4S-Specific Slurm Requirements

AI4S Slurm operates in **configless DNS-SRV mode**. Non-interactive SSH
sessions (e.g. the UMRS SSH ControlMaster) do not source `.bashrc`, so
`SLURM_CONF_SERVER` is not set automatically.

UMRS injects this via `extra.ssh_remote_env` in `site.yaml`:

```yaml
extra:
  ssh_remote_env:
    SLURM_CONF_SERVER: "sv08,sv09"
```

Do **not** remove this setting or Slurm commands (`sbatch`, `squeue`, `scancel`)
will fail silently in non-login sessions.

### Memory Requests

AI4S rejects `--mem=0` with `MaxMemPerLimit` because the scheduler interprets
`0` as "all node memory" (up to 1562.50 G) and the account QoS blocks it.
`site.yaml` must specify a bounded value:

```yaml
scheduler:
  memory: "64G"   # explicit bounded request required on AI4S
```

Use a realistic value for the model being served. The smoke wrapper uses 64 G;
production Qwen3-32B on a full B200 node would use 256–512 G.

### sbatch GPU Syntax

Both of the following syntaxes are accepted on AI4S `1n1gpu`:

```bash
--gpus-per-node=1   # official AI4S docs style
--gres=gpu:1        # also accepted
```

UMRS uses `--gres=gpu:N` for compatibility with all cluster flavours.

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
#SBATCH --mem=256G

apptainer run --nv \
  --bind /work/hps0/home/ea0022/umrs_models:/models:ro \
  --bind /work/hps0/home/ea0022/scratch:/scratch \
  /work/hps0/home/ea0022/images/mini-rivault-aarch64.sif \
  --model /models/Qwen3-32B --port 8001
```

## Smoke Wrapper (Pre-production Validation)

Before the full Apptainer SIF and production model weights are staged, UMRS
can validate the complete client → Gateway → UMRS → Slurm → compute path
using the lightweight Python HTTP smoke wrapper:

```
umrs/deploy/ai4s/serve-openai-smoke.slurm
```

This wrapper:
- Allocates a real Slurm node on `1n1gpu`
- Writes `rivault-job-<JOBID>.info` to `JOB_INFO_DIR` so UMRS can discover the
  compute node and open a local tunnel
- Serves a minimal OpenAI-compatible HTTP server (health, models, chat)
- Responds to chat completions with `"gateway hpc ready"` (no GPU required)
- Watches for `stop-<JOBID>.request` in `RIVAULT_CONTROL_DIR` and exits cleanly

### Register the smoke model

```bash
R4_GATEWAY_URL=http://127.0.0.1:18082 \
  r4 model register ai4s-smoke \
  --path /work/hps0/home/ea0022/umrs_models/ai4s-smoke \
  --scope system --type llm --backend vllm \
  --service-kind openai --max-model-len 128 --gpu-mem-util 0.1 --tp 1 \
  --no-detect
```

### Run the smoke e2e test

```bash
R4_GATEWAY_URL=http://127.0.0.1:18082 \
  R4_API_KEY=$R4_API_KEY \
  R4_PROJECT_ID=$R4_PROJECT_ID \
  SLURM_TEST_MODEL_NAME=ai4s-smoke \
  HPC_TEST_PROVISION_TIMEOUT=600 \
  python -m pytest tests/test_hpc_gateway_e2e.py::TestGatewayHPCE2E::test_gateway_provision_chat_and_teardown -q -s
```

Expected: `1 passed` in ~30 s (Slurm queue permitting). Job output confirms
node, port, and clean stop-request handling.

## Coexistence with rccs-front

Both profiles run simultaneously on the same `umrs-front` host. The dual-server
stack is managed by `umrs/deploy/umrs-front/start.sh`.

| Service | rccs-front | ai4s |
|---------|-----------|------|
| UMRS HTTP | `0.0.0.0:8000` | `0.0.0.0:8002` |
| UMRS ZMQ | `127.0.0.1:5555` | `127.0.0.1:5557` |
| Gateway (via tunnel) | `127.0.0.1:18080` | `127.0.0.1:18082` |
| Sidecar | `127.0.0.1:8083` | `127.0.0.1:8084` |
| PostgreSQL | `127.0.0.1:5432` (shared native) | — |
| SeaweedFS (S3) | `${S3_BIND:-0.0.0.0}:9000` (shared) | `0.0.0.0:9002` |
| Endpoint tunnel pool | `12000–12999` | `13000–13999` |

PostgreSQL is **native** on `umrs-front` (not a Podman container). `start.sh`
detects this and skips Podman Postgres launch if port 5432 is already open.

## Production Readiness

The following are required before running a real vLLM workload on AI4S:

- [ ] Build `mini-rivault-aarch64.sif` (aarch64 target for Grace/B200)
- [ ] Stage SIF to `/work/hps0/home/ea0022/images/`
- [ ] Stage model weights to `/work/hps0/home/ea0022/umrs_models/<model-name>/`
- [ ] Update `site.yaml` `runtime.rivault_dir` to the staged directory
- [ ] Update `site.yaml` `scheduler.memory` to match the model (256G for 32B)
- [ ] Register the production model via `r4 model register`
- [ ] Set `LITELLM_MASTER_KEY` in `.env` and the sbatch script consistently
- [ ] Verify `TRANSFORMERS_OFFLINE=1` is exported inside the container

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `squeue`/`sbatch` silently fails | `SLURM_CONF_SERVER` not set | Ensure `extra.ssh_remote_env.SLURM_CONF_SERVER: "sv08,sv09"` in `site.yaml` |
| Job cancelled immediately, `MaxMemPerLimit` | `--mem=0` sent to scheduler | Set `scheduler.memory: "64G"` (or higher) in `site.yaml` |
| SSH fails to AI4S | Host/key not configured on `umrs-front` | Verify `~/.ssh/config` `Host AI4S` entry with `rccs_key` identity |
| Raw IP `134.160.189.24` rejected from `umrs-front` | Key config tied to SSH alias | Use the `AI4S` alias, not raw IP, in `extra.ssh_target` |
| ProxyJump needed | `umrs-front` cannot directly reach `134.160.189.24` | Add `ProxyJump masaru.nagaso@rccs-cloud` to `~/.ssh/config` `Host AI4S` |
| Singularity image not found | SIF not staged | Stage the SIF to Lustre first |
| GPU not accessible | Missing `--nv` flag | Ensure `--nv` is in the sbatch script |
| UMRS sees `pending` host forever | Job-info file not written | Check sbatch stdout for wrapper errors; verify `JOB_INFO_DIR` writable |
