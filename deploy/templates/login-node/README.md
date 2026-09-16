# Login-Node Deployment Template

Deploy the UMRS service **directly on an HPC login node**. This is the simplest
topology: the UMRS server process runs natively on the login node so it can call
the Slurm CLI (`sbatch`, `squeue`, `scancel`) directly. Only the stateful infra
(Postgres + MinIO) runs in containers.

```
  client ──HTTP/ZMQ──> UMRS server (native, login node) ──sbatch──> GPU compute node
                          │
                          ├─ Postgres (podman container, 127.0.0.1:5432)
                          └─ MinIO    (podman container, 127.0.0.1:9000)
```

Distilled from the production `../rccs-cloud/` harness. Strip-and-adapt: every
`<...>` placeholder and empty secret must be filled in for your site.

## Files

| File | Purpose |
|------|---------|
| `docker-compose.yaml` | Postgres + MinIO infra containers (rootless Podman). |
| `site.yaml` | UMRS `SiteConfig` — scheduler/runtime/storage for **your** cluster. |
| `env.example` | Secrets + transport tuning. Copy to `.env` and edit. |
| `start.sh` | Bring up infra, then launch the native UMRS server. |
| `stop.sh` | Stop the native server and infra. |
| `pool.example.yaml` | Optional provision roster (default endpoint sizing). |

## Prerequisites

- Shell + Slurm access on the login node (`sbatch`/`squeue` work).
- Rootless Podman with compose support (`podman compose version`).
- A Python venv with UMRS installed (`pip install -e umrs/` or your site's wheel).
- Model weights pre-staged on a shared filesystem the compute nodes can read.

## Deploy

```bash
cp env.example .env
$EDITOR .env          # set POSTGRES_PASSWORD, S3_ROOT_USER/PASSWORD, etc.
$EDITOR site.yaml     # set partition, account, storage paths for your cluster

./start.sh            # infra containers + native UMRS server
./stop.sh             # tear everything down
```

## What to customise in `site.yaml`

- `scheduler.partition` / `scheduler.account` — your Slurm partition + charge account.
- `scheduler.kind: slurm-podman` — local Slurm + rootless Podman on compute nodes.
- `storage.model_cache_dir` — shared (Lustre/NFS) path visible to compute nodes.
- `partition_profiles.<name>` — one block per partition describing the GPU
  hardware (arch, vendor, VRAM, OCI runtime). The launcher looks up the profile
  for the assigned partition instead of probing at job start.

## Networking notes

- ZMQ binds to `127.0.0.1` when `UMRS_TUNNEL_MODE=true` — clients reach the
  server over SSH port-forwarding. Set it `false` only where ZMQ ports are
  routable and you enable CURVE auth (`UMRS_ZMQ_AUTH_TYPE=curve`).
- Compute nodes reach MinIO via an SSH tunnel to `login:9000`; keep
  `S3_PRESIGN_ENDPOINT=http://localhost:9000` so presigned URLs resolve through
  it.
