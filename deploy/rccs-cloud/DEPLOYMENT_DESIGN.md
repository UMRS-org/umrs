# UMRS Deployment Design — Principles, Rules, and Site Conditions

This document captures the architectural rationale behind the rccs-cloud UMRS
deployment.  It is intended as a reference for maintainers and as a template
when deploying UMRS to similar HPC environments.

---

## 1. UMRS Design Principles

These principles are intrinsic to UMRS and must be respected by any deployment.

### 1.1 ZeroMQ message-based control plane

UMRS uses ZeroMQ REQ-REP (port 5555) for control and PUSH-PULL (ports 5556+)
for data transfer.  The server is the sole broker — there is no external
message queue.  Clients and the server must be able to reach each other on
these ports (directly or via SSH tunnel).

### 1.2 S3 presigned-URL data plane

All large data (images, model weights, results) is exchanged via S3 presigned
URLs.  The server generates presigned URLs that the **client** and **compute
nodes** both use.  This means:

- The `S3_PRESIGN_ENDPOINT` in the URL must be reachable by every actor
  (client, server, compute node).
- Where direct network access is impossible, SSH tunnels to the MinIO host
  must be established before the presigned URL is consumed.

### 1.3 Per-user isolation

Each UMRS user gets:
- Dedicated MinIO IAM credentials (auto-generated, scoped to their buckets).
- Dedicated PostgreSQL database.
- Dedicated S3 buckets with IAM policies preventing cross-user access.

The admin `~/.s3cfg` is used only by the server process to create/manage users.
Per-user `~/.s3cfg_user*` files are generated at startup and must not be shared.

### 1.4 Slurm-based GPU job orchestration

`rivault_launcher.py` manages GPU workloads by calling Slurm CLI commands
(`sbatch`, `squeue`, `scancel`) and creating SSH tunnels (`ssh -N -L`) via
Python `subprocess`.  These are **host-level binaries** that must be available
to the Python process running `server_umrs.py`.

### 1.5 GPU-first compute model

All heavy inference — embeddings, vision models (SAM3, HMAE), LLMs, and
user-defined custom models — is dispatched to **GPU compute nodes** via Slurm.
The login-node server acts purely as an orchestrator: it receives requests,
submits Slurm jobs, establishes SSH tunnels, and relays results.  It does NOT
run inference locally by default.

CPU-only local inference (e.g. sentence-transformers on the login node) exists
as an **optional fallback** for debugging or when no GPU nodes are available.
To use the fallback, change `model_type` in the user config from
`rivault://openai://...` back to `TransformerEncoder` or `SentenceTransformer`.

### 1.6 Model-type scheme routing

Vision models use a composable double-scheme URL:
```
rivault://bentoml://http://localhost:{PORT}/vision/infer/submit
```
- `rivault://` → triggers Slurm job launch + SSH tunnel.
- `bentoml://` → selects the BentoML async task-queue adapter.

This scheme is parsed at runtime — the deployment must not interfere with
localhost port mappings between the server and any SSH tunnels it creates.

### 1.7 Plugin / adapter discovery

Vision adapters are discovered from directories in `VISION_ADAPTERS_DIR` at
service startup.  The BentoML vision service runs **on the compute node**
(inside a container), not on the login node.

### 1.8 Custom model extensibility

Users can add their own models through three pathways, all of which run on
GPU compute nodes by default:

| Scheme | Use case | How it works |
|---|---|---|
| `rivault://openai://http://...` | Embedding / LLM models served via an OpenAI-compatible API | Server submits a Slurm job that starts a vLLM container, creates an SSH tunnel, and queries the `/v1/embeddings` or `/v1/completions` endpoint. |
| `rivault://bentoml://http://...` | Vision or task-queue models with adapter-based dispatch | Server submits a Slurm job that starts a BentoML container.  The `vision_model_type` field selects the adapter (e.g. `"sam3"`, `"hmae"`, or a custom name). |
| `user_script://path/to/script.py` | Arbitrary server-side pipeline (advanced) | Server executes a user-provided Python script on the login node.  The script can itself submit Slurm jobs or call external services. |

BentoML adapter auto-discovery: place a `.py` file containing a
`VisionModelAdapter` subclass in the user adapters directory
(`VISION_ADAPTERS_DIR`).  The file stem becomes the adapter name.

---

## 2. Strict Deployment Rules

These rules must be followed regardless of the target site.

### Rule 1 — The server process MUST have access to Slurm CLI

`server_umrs.py` (via `rivault_launcher.py`) invokes:
- `sbatch` — submit GPU jobs
- `squeue` — poll job status
- `scancel` — cancel running jobs
- `ssh -N -L` — create tunnels to compute nodes

**Consequence**: The server cannot run inside a generic container image
(e.g. `python:3.10-slim`) unless Slurm libraries and SSH client are
bind-mounted or installed.  The recommended approach is to run the server
**natively** on the login node.

### Rule 2 — The server process MUST be able to SSH to compute nodes

SSH tunnels forward ports from compute nodes to the login node (for MinIO
access) and vice-versa.  The server calls `ssh -N -L ...` as subprocesses.
**Passwordless SSH** from the login node to compute nodes (and vice-versa)
is required.

### Rule 3 — S3_PRESIGN_ENDPOINT must be resolvable by all actors

The presigned URL base must be reachable by:
1. The server itself (localhost).
2. Compute nodes (via SSH tunnel → `localhost:9000`).
3. The client (via SSH tunnel or direct).

If any actor cannot reach the endpoint, uploads/downloads will fail silently
or with cryptic HTTP errors.

### Rule 4 — Model cache must be on shared storage

Embedding models and GPU model checkpoints are stored in `$MODEL_DIR`.
This directory must be readable from **both** the login node and all compute
nodes.  On HPC systems, this is typically a parallel filesystem (Lustre, GPFS).

### Rule 5 — conf_umrs.json paths are relative to the config file

`admin_s3_configuration_file`, `user_files_dir`, and each entry in `users`
are resolved relative to the location of `conf_umrs.json`, not the working
directory of the server process.  For site deployments, keep using
`./user_files/...` in the config file and declare `"user_files_dir": "./user_files"`
so uploaded user JSONs land in the same canonical directory.

### Rule 6 — Do not containerize the server on Slurm-managed nodes

Wrapping the server in a container on a Slurm login node adds complexity
without benefit:
- Slurm CLI must be bind-mounted (version-sensitive).
- SSH agent/sockets must be forwarded.
- `~/.ssh/` must be mounted for tunnel creation.
- The container may conflict with the host's `sbatch` version.

Run the server natively; containerize only infrastructure services that do
not need host integration (PostgreSQL, MinIO).

### Rule 7 — Stale per-user s3cfg files must be cleaned on restart

The server generates `~/.s3cfg_user*` files on startup, pointing at the
current MinIO endpoint.  Stale files from a previous deployment (e.g. with
`minio:9000` as the hostname inside a container) will cause S3 operations to
fail.  `start.sh` must remove these before launching the server.

### Rule 8 — Environment variables drive site-specific config

All site-specific paths and credentials must be passed via environment
variables (read from `.env`), not hard-coded.  Key variables:

| Variable | Purpose |
|---|---|
| `DB_HOST` | PostgreSQL hostname (default: `127.0.0.1`) |
| `DB_SUPERUSER` / `DB_SUPERUSER_PASSWORD` | PostgreSQL admin credentials |
| `S3_PRESIGN_ENDPOINT` | MinIO URL reachable by all actors |
| `SENTENCE_TRANSFORMERS_HOME` | Embedding model cache dir |
| `HF_HOME` | Hugging Face cache dir |
| `UMRS_MODEL_STAGING_ROOT` | Model staging area on shared storage |
| `RIVAULT_SLURM_SCRIPT` | Path to site-specific Slurm wrapper |

---

## 3. Site-Specific Conditions — rccs-cloud

These conditions are specific to the RIKEN rccs-cloud HPC cluster.

### 3.1 Network topology

- **Login node**: `login1.cloud.r-ccs.riken.jp` (SSH alias: `rccs-cloud`).
- **Compute nodes**: `dgx-s`, `dgx-m0`–`dgx-m3`.  Not directly routable from
  external networks.
- **No external port exposure**: All external access is via SSH tunnels from
  the user's local machine.
- **Compute ↔ login**: Compute nodes can SSH to the login node (same Lustre
  home dir, shared `~/.ssh/authorized_keys`).  This is used for the MinIO
  SSH tunnel in `serve-model.slurm`.

### 3.2 Filesystem

- **Home directories**: On Lustre (`/lvs0/dne1/rccs-dmp/`).  Does NOT support
  `xattr` — Podman overlay storage fails with
  `lsetxattr: operation not supported`.
- **Podman graphroot**: Must be redirected to a local filesystem
  (e.g. `/var/tmp/$USER/containers/storage`) via `~/.config/containers/storage.conf`.
- **Model cache**: `/lvs0/rccs-dmp/mnagaso/umrs_models` — accessible from both
  login and compute nodes.

### 3.3 Container runtime

- **Rootless podman** (no Docker daemon).
- `podman compose` (via `podman-compose` or `podman compose` v5+).
- Images that require `xattr` during build/pull must be stored on a local FS
  (see 3.2).

### 3.4 Compute nodes (GPU)

- **Primary inference target**: All model types (embeddings, vision, LLMs,
  custom models) run here by default.  The login node does NOT run inference
  in production.
- **GPU**: NVIDIA Blackwell GB10, aarch64 architecture.
- **CUDA**: 13.0, driver 580.95.05.
- **PyTorch wheels**: Must use `cu130` + `aarch64` builds.
- **Container runtime**: Rootless podman with NVIDIA CDI for GPU passthrough.
- **No internet access**: All images must be pre-pulled or loaded from tarballs
  on Lustre.

### 3.5 Slurm

- Partitions: `ng-dgx-s`, `ng-dgx-m0`, `ng-dgx-m1`, `ng-dgx-m2`, `ng-dgx-m3`.
- The login node has `sbatch`, `squeue`, `scancel`, `sinfo` available.
- Compute nodes also have Slurm client tools (for nested jobs, if needed).

### 3.6 Security model

- MinIO listens on `127.0.0.1:9000` only (not exposed to cluster network).
- Compute nodes access MinIO via an SSH reverse tunnel
  (`localhost:9000 → login:9000`), established by `serve-model.slurm`.
- PostgreSQL listens on `127.0.0.1:5432` only.
- All external client access is via SSH tunnels (see `ssh_tunnel.sh`).
- No TLS between internal services (all localhost traffic).

### 3.7 Hybrid architecture summary

```
┌─────────────────────────────────────────────────────┐
│  rccs-cloud login node                              │
│                                                     │
│  ┌──────────────┐  ┌──────────────┐                 │
│  │  PostgreSQL   │  │    MinIO     │  ← containers  │
│  │  (container)  │  │  (container) │                 │
│  └──────┬───────┘  └──────┬───────┘                 │
│         │ :5432            │ :9000                   │
│         └────────┬─────────┘                        │
│                  │ localhost                         │
│         ┌────────┴─────────┐                        │
│         │  umrs-server     │  ← NATIVE process      │
│         │  (Python venv)   │    (has sbatch, ssh)    │
│         └────────┬─────────┘                        │
│                  │ sbatch / ssh                      │
└──────────────────┼──────────────────────────────────┘
                   │ Slurm
┌──────────────────┼──────────────────────────────────┐
│  Compute node    │                                  │
│         ┌────────┴─────────┐                        │
│         │  mini-rivault    │  ← podman containers   │
│         │  (GPU workloads) │    ALL inference here:  │
│         │                  │    embeddings, vision,  │
│         │                  │    LLMs, custom models  │
│         └──────────────────┘                        │
│         SSH tunnel → login:9000 (MinIO)             │
└─────────────────────────────────────────────────────┘
```

---

## 4. File Inventory

| File | Purpose |
|---|---|
| `docker-compose.yml` | Infrastructure containers (postgres + minio only) |
| `setup_server.sh` | One-time: create venv, install deps, symlink slurm script |
| `start.sh` | Start containers + native server |
| `stop.sh` | Stop native server + containers, cancel Slurm jobs |
| `.env` / `env.example` | Site-specific environment variables |
| `conf_umrs.json` | Server config (user list, admin s3cfg path) |
| `user_files/user1.json` | Per-user experiment definitions |
| `serve-model.slurm` | Site-specific Slurm wrapper (sets env overrides) |
| `ssh_tunnel.sh` | Client-side SSH tunnels to login node |
| `health_check_server.sh` | Server-side health checks |
| `setup_client.sh` | Client-side setup (configs + pip install) |
| `README.md` | Step-by-step deployment guide |
| `DEPLOYMENT_DESIGN.md` | This document |
