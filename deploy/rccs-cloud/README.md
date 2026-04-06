# UMRS Deployment Guide — rccs-cloud

This guide walks through deploying UMRS on the rccs-cloud login node using
a **hybrid architecture**: infrastructure services (PostgreSQL, MinIO) run in
rootless podman containers, while `umrs-server` runs **natively** so it can
invoke Slurm CLI tools (`sbatch`, `squeue`, `scancel`) and create SSH tunnels
to compute nodes.

---

## Quick Deploy (already set up)

If the server environment is already configured (Steps 0–1 done previously),
these are the only commands needed to deploy or redeploy after a code change:

```bash
# ── On your local machine ────────────────────────────────────────────────
# 1. Commit & push
cd /path/to/umrs
git push origin feature/user-defined-workflows

# ── On rccs-cloud (via SSH) ──────────────────────────────────────────────
ssh rccs-cloud
cd /lvs0/dne1/rccs-dmp/mnagaso/umrs

# 2. Pull latest code
git pull origin feature/user-defined-workflows

# 3. Update the Python package
source deploy/rccs-cloud/.venv/bin/activate
pip install -e src/ --quiet

# 4. Stop existing services (data preserved)
cd deploy/rccs-cloud
./stop.sh

# 5. Start everything (containers + server)
./start.sh

# 6. Verify (wait ~15 s for server init)
./health_check_server.sh
```

To restart **only** the server (e.g. after editing Python files), keeping
containers running:

```bash
./stop.sh --server
./start.sh --server
```

---

## Architecture Overview

```
Local machine (optional)          rccs-cloud (login node)          ng-dgx-s / ng-dgx-m* (compute)
────────────────────              ───────────────────────          ──────────────────────────
umrs-client  ←── SSH tunnel ───►  umrs-server  (NATIVE)  ←── Slurm ───►  mini-rivault
rag client   ←── SSH tunnel ───►  rag-service  (NATIVE)                  vLLM / BentoML + LiteLLM
                                  minio     (podman container)  ◄── SSH tunnel ──┘
                                  postgres  (podman container)
                                  /lvs0/rccs-dmp/  (Lustre — shared)
```

**Why native?**  `rivault_launcher.py` calls `sbatch`, `squeue`, `scancel`,
and `ssh -N -L` via `subprocess`.  These binaries are available on the login
node but not inside a `python:3.10-slim` container.  Running the server natively
avoids bind-mounting Slurm libs and SSH sockets, which is fragile and
non-portable.

Ports tunneled over SSH (for local Mac access):

| Port | Service |
|------|---------|
| 5555 | ZMQ control plane |
| 5556 | ZMQ data plane (add more ports for concurrent users) |
| 9000 | MinIO S3 API (optional; local admin/debug only) |
| 8081 | RAG service |

For Slurm jobs, MinIO is accessed via an SSH tunnel (`localhost:9000`) created
automatically by `serve-model.slurm`. See [Step 0b](#step-0b--enable-compute--login-ssh-required-for-slurm-jobs).

---

## Prerequisites

- SSH access: `ssh rccs-cloud` works without a password (key configured).
- The repo is cloned on rccs-cloud (e.g. at `/lvs0/rccs-dmp/mnagaso/umrs`).
- `podman` is available on rccs-cloud.
- `podman-compose` is installed (`pip install --user podman-compose`). Ensure
  `~/.local/bin` is in your `PATH` (add `export PATH="$HOME/.local/bin:$PATH"`
  to `~/.bashrc` if needed).
- **Podman storage on a local filesystem** — see [Step 0a](#step-0a--configure-podman-storage-required) below.
- **Passwordless SSH from compute nodes to login node** — see [Step 0b](#step-0b--enable-compute--login-ssh-required-for-slurm-jobs) below.
- Lustre at `/lvs0/rccs-dmp/` is mounted on both login node and compute nodes.
- Python ≥ 3.10 locally (or a venv with the umrs package).

---

## Step 0a — Configure Podman storage (required)

rccs-cloud home directories are on Lustre, which does not support extended
attributes (`xattr`).  Podman's default overlay storage will fail with
`lsetxattr: operation not supported` when pulling or building images.

Redirect Podman storage to a **local** filesystem (e.g. `/tmp`):

```bash
mkdir -p ~/.config/containers
cat > ~/.config/containers/storage.conf << 'EOF'
[storage]
driver = "overlay"
graphroot = "/tmp/${USER}/containers/storage"
runroot  = "/tmp/${USER}/containers/run"
EOF
```

Then reset any existing (broken) storage state:

```bash
podman system reset --force
```

Verify the new graphroot:

```bash
podman info --format '{{.Store.GraphRoot}}'
# Should print: /tmp/<your-username>/containers/storage
```

> **Note**: `/tmp` is local to the login node and is cleared on reboot.
> Container images will need to be re-pulled after a reboot.  If a persistent
> local volume is available (e.g. `/local/$USER`), use that path instead.

---

## Step 0b — Enable compute → login SSH (required for Slurm jobs)

Slurm compute nodes (dgx-s, dgx-m*) cannot reach the login node’s MinIO
service directly (port 9000 is not routable).  The mini-rivault Slurm script
opens an **SSH tunnel** from the compute node back to the login node to
forward MinIO traffic.

For this tunnel to work, **passwordless SSH from compute nodes to the login
node** must be enabled.  Since all nodes share the same home directory on
Lustre, a one-time setup is sufficient:

```bash
# On rccs-cloud (login node):
# Generate a key pair if you don't already have one:
[ -f ~/.ssh/id_rsa ] || ssh-keygen -t rsa -N '' -f ~/.ssh/id_rsa

# Authorize your own key for loopback/compute-to-login SSH:
cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
```

Verify it works:

```bash
ssh -o BatchMode=yes -o StrictHostKeyChecking=no localhost echo "self-ssh OK"
```

You should see `self-ssh OK` with no password prompt.  This also enables SSH
from any compute node back to the login node since they share `~/.ssh/`.

---

## Step 1 — Configure the server

### 1a. Edit `.env`

All deployment files are in `deploy/rccs-cloud/` inside the repo.

```bash
# On rccs-cloud:
cd /lvs0/rccs-dmp/mnagaso/umrs/deploy/rccs-cloud
cp env.example .env
nano .env          # or vim / any editor
```

Key values to adjust:

| Variable | Default | Notes |
|---|---|---|
| `POSTGRES_PASSWORD` | `umrs_rccs` | Change in production |
| `MINIO_ROOT_PASSWORD` | `minioadmin` | Change in production |
| `S3_PRESIGN_ENDPOINT` | `http://localhost:9000` | Compute nodes access MinIO via an SSH tunnel (see below).  The Slurm script opens `localhost:9000` → login node MinIO automatically. |
| `MODEL_DIR` | `/lvs0/rccs-dmp/mnagaso/umrs_models` | Lustre path, shared between login and compute nodes |
| `MINIO_BIND` | `127.0.0.1` | MinIO listens on localhost only.  Compute nodes use the SSH tunnel. |
| `RIVAULT_SLURM_SCRIPT` | `./serve-model.slurm` | Path to the site-specific Slurm wrapper |

### 1b. Edit user config

`user_files/user1.json` defines the experiment modalities (text, image, etc.).
Edit the `fullname` field and bucket aliases to match your setup.
No change is required to use the defaults.

### 1c. Edit server config

`conf_umrs.json` lists the user files.  Paths are **relative to the
`deploy/rccs-cloud/` directory** (e.g. `./user_files/user1.json`).
To add a second user, append another entry to the `users` array and create
a corresponding `user_files/user2.json`.

### 1d. One-time server setup

Run the setup script once to create the Python virtual environment, install
dependencies, and prepare the Slurm script symlink:

```bash
./setup_server.sh
```

This creates `.venv/` under `deploy/rccs-cloud/`, installs CPU-only PyTorch,
the `umrs` package (editable), and symlinks `serve-model.slurm` into
`~/.config/umrs/rivault_data/`.

---

## Step 2 — Start the server

```bash
# On rccs-cloud, inside deploy/rccs-cloud:
./start.sh
```

The script will:

1. Auto-create `.env` from `env.example` if not present (prompts you to edit it).
2. Create the Lustre model cache directories.
3. Start postgres and minio containers via `podman compose up -d`.
4. Wait for postgres and minio to become healthy.
5. Launch `server_umrs.py` **natively** (outside any container) with the correct
   environment variables, writing output to `logs/server.log`.

Check that everything is up:

```bash
./health_check_server.sh
tail -f logs/server.log
```

To restart only the server (e.g. after a code change), keeping containers running:

```bash
./start.sh --server
```

The first launch will:
- Download embedding model weights to `$MODEL_DIR/sentence_transformers/` (first run only).
- Auto-create MinIO buckets and PostgreSQL databases for each configured user.

---

## Step 2.5 — Verify MinIO reachability from Slurm jobs via SSH tunnel

Compute nodes (ng-dgx-s, ng-dgx-m*) cannot reach the login node's MinIO directly
(port 9000 is not routable).  The mini-rivault Slurm script
(`src/umrs/mini-rivault-llm/serve-model.slurm`) automatically opens an SSH
tunnel that forwards `localhost:9000` on the compute node to `127.0.0.1:9000`
on the login node.

> **Prerequisite**: [Step 0b](#step-0b--enable-compute--login-ssh-required-for-slurm-jobs)
> must be completed — passwordless SSH from compute nodes to the login node
> is required for the tunnel.

To verify the tunnel works, submit a quick test job (replace `dgx-s` with any
target partition):

```bash
# On rccs-cloud:
sbatch -p ng-dgx-s --time=00:05:00 --wrap='
  ssh -fN -L 9000:127.0.0.1:9000 '"$HOSTNAME"' -o StrictHostKeyChecking=no -o ExitOnForwardFailure=yes
  sleep 2
  curl -sS -m 5 -I http://localhost:9000/minio/health/live || exit 42
'
```

Check the output once the job completes:

```bash
cat ~/slurm-<jobid>.out
```

Interpretation:
- `HTTP/1.1 200 OK`: tunnel works, MinIO is reachable.
- `Permission denied (publickey...)`: SSH key not in `authorized_keys` — redo
  [Step 0b](#step-0b--enable-compute--login-ssh-required-for-slurm-jobs).
- `Connection refused`: tunnel opened but MinIO is not running on the login node.
- `port forwarding failed`: network issue between compute and login nodes.

To test **all** partitions at once:

```bash
for part in ng-dgx-s ng-dgx-m0 ng-dgx-m1 ng-dgx-m2 ng-dgx-m3; do
  sbatch -p "$part" --time=00:05:00 -J "test-$part" --wrap='
    ssh -fN -L 9000:127.0.0.1:9000 '"$HOSTNAME"' -o StrictHostKeyChecking=no -o ExitOnForwardFailure=yes
    sleep 2
    curl -sS -m 5 -I http://localhost:9000/minio/health/live || exit 42
  '
done
```

---

## Step 3 — Set up the client (local Mac)

From your local machine, inside this directory:

```bash
cd deploy/rccs-cloud
./setup_client.sh
```

This copies the client configs to `~/.config/umrs/` and runs `pip install -e ../../src`.

> **Note**: `setup_client.sh` installs the `umrs` package in editable mode from the
> local `src/` directory.  If you prefer a separate venv, activate it first.

Verify the install:

```bash
umrs-client --help
```

---

## Step 4 — Open SSH tunnels (optional, local client access)

Keep this running in a terminal (or run it in the background):

```bash
./ssh_tunnel.sh             # foreground — Ctrl-C to stop
./ssh_tunnel.sh &           # background
```

To target a non-default host:

```bash
./ssh_tunnel.sh my-host     # defaults to "rccs-cloud"
```

> Add extra `-L` lines to `ssh_tunnel.sh` for additional ZMQ data ports if you
> run multiple concurrent users (e.g. `-L 5557:localhost:5557`).

Note: these tunnels are for your local tools only. Slurm compute nodes do not use
your local tunnels when uploading via presigned URLs.

---

## Step 5 — Run the demos

With tunnels open (if running from local Mac), run any demo from `demos/`:

```bash
# On rccs-cloud (or local with SSH tunnels):
cd deploy/rccs-cloud/demos
source ../.venv/bin/activate

# Quick health check
bash 00_health_check.sh

# Full core demo suite (health → text → SAM3 → HMAE → user model → pipeline → registry)
bash run_all_core.sh
```

Or use `umrs-client` directly:

```bash
# Push text files and retrieve embeddings
umrs-client -c ~/.config/umrs/users/user1.json upload_dir ./my_texts --modality text

# List available buckets
umrs-client -c ~/.config/umrs/users/user1.json list_buckets
```

See [Deployment Verification](#deployment-verification) for the full demo list
and latest test results.

---

## Step 6 — GPU inference on compute nodes

All inference (embeddings, vision, LLMs, custom models) runs on GPU compute
nodes by default.  The `user_files/user1.json` config specifies:

- **Embeddings** (text, image): `model_type: "rivault://openai://http://localhost:8001/v1"` — served via vLLM on a compute node.
- **Vision** (SAM3, HMAE): `model_type: "rivault://bentoml://http://localhost:{LOCAL_PORT}/vision/infer/submit"` — served via BentoML on a compute node.

The server auto-launches Slurm jobs and SSH tunnels when a model is first
requested — no manual GPU setup is needed.

### Supported model types

| Prefix | Description | Example |
|--------|-------------|---------|
| `rivault://openai://…` | Embedding model served by vLLM | `rivault://openai://http://localhost:8001/v1` |
| `rivault://bentoml://…` | Vision model (SAM3/HMAE) served by BentoML | `rivault://bentoml://http://localhost:{LOCAL_PORT}/vision/infer/submit` |
| `user_script://…` | Pipeline SDK workflow (single step) | `user_script://stats:TextStatsStep` |
| `user_workflow://…` | Pipeline SDK workflow (multi-step DAG) | `user_workflow://my_workflow:Main` |
| `pipeline://…` | Named pipeline step (deferred execution) | `pipeline://opossum_analyze:OpossumAnalyzeStep` |
| `TransformerEncoder` | CPU-local sentence-transformers | — |
| `SentenceTransformer` | CPU-local CLIP/SBERT | — |
| `HMAE` | CPU-local HMAE (very slow) | — |
| `SAM3` | CPU-local SAM3 (very slow) | — |

### Prerequisites

1. Mini-rivault stack configured (see `demos/slurm-dev/`).
2. `serve-model.slurm` wrapper linked (done by `setup_server.sh`).
3. Passwordless SSH between login and compute nodes ([Step 0b](#step-0b--enable-compute--login-ssh-required-for-slurm-jobs)).

### CPU-only fallback (optional)

For debugging or when no GPU nodes are available, you can switch individual
modalities to CPU-local inference by changing `model_type` in the user config:

```json
{
  "model_type": "TransformerEncoder"
}
```

or `"SentenceTransformer"` for CLIP-based image embeddings.  This runs
sentence-transformers directly on the login node (CPU-only PyTorch).  No
Slurm job is needed, but performance will be significantly slower.

---

## Day-to-day operations

```bash
# Check status
./health_check_server.sh

# Tail server logs
tail -f logs/server.log

# Container status
podman compose ps

# Reload server config (without restart) — via umrs-client
umrs-client reload

# Restart just the server (containers stay up)
./start.sh --server

# Stop everything (data preserved in volumes)
./stop.sh

# Stop + wipe all persistent data (volumes)
./stop.sh --clean
```

---

## Sharing MinIO S3 with another rccs-cloud user

A single MinIO instance can serve multiple OS-level users.  The recommended
approach is **per-OS-user access via MinIO IAM policies**.

### Option A — Add a second UMRS user (recommended)

UMRS handles multi-user S3 access automatically.  Each user defined in
`conf_umrs.json` gets:
- Their own MinIO IAM credentials (username + password, auto-generated).
- Their own buckets with read/write policies scoped to those buckets only.
- Their own PostgreSQL database.

**Steps:**

1. Create `user_files/user2.json` with the second user's experiments:

```json
{
  "username": "user2",
  "fullname": "Other User",
  "s3_configuration_file": "~/.s3cfg_user2",
  "experiences": [...]
}
```

2. Add it to `conf_umrs.json`:

```json
{
  "admin_s3_configuration_file": "~/.s3cfg",
  "users": [
    "./user_files/user1.json",
    "./user_files/user2.json"
  ]
}
```

3. Send the RELOAD command to provision the new user without restarting:

```bash
umrs-client reload
```

The server will create the MinIO IAM user, generate `~/.s3cfg_user2` in the
home directory, and create all required buckets.

5. Retrieve the generated S3 config for user 2:

```bash
cat ~/.s3cfg_user2
```

Copy that file to the second user's machine at `~/.config/umrs/s3cfg` (or
whatever `s3_configuration_file` path their `user2.json` specifies).

### Option B — Grant another OS user direct MinIO access

If the second OS user wants to access MinIO independently (e.g. to upload from
their own scripts), create credentials for them via the MinIO admin client (`mc`):

```bash
# Install mc if not available (one-time):
# curl -sL https://dl.min.io/client/mc/release/linux-arm64/mc -o ~/.local/bin/mc && chmod +x ~/.local/bin/mc

# Alias for the local MinIO instance
mc alias set local http://127.0.0.1:9000 minioadmin minioadmin

# Create a new MinIO user
mc admin user add local user2 <password>

# Create a policy that allows read/write on their bucket(s) only
mc admin policy create local user2-policy /dev/stdin <<EOF
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Action": ["s3:*"],
    "Resource": [
      "arn:aws:s3:::user2-*",
      "arn:aws:s3:::user2-*/*"
    ]
  }]
}
EOF

mc admin policy attach local user2-policy --user user2
```

Then give `user2` an `s3cfg` file pointing at `localhost:9000` (via the SSH tunnel)
with their new credentials.

> ⚠️ **Security note**: MinIO is bound on localhost only (`MINIO_BIND=127.0.0.1`).
> Compute nodes access MinIO through an SSH tunnel set up by the Slurm script.
> If you need to expose MinIO directly to other hosts, change `MINIO_BIND` to
> `0.0.0.0` in `.env` and ensure firewall rules restrict port 9000 to trusted
> hosts.

---

## Troubleshooting

### `lsetxattr: operation not supported` when pulling/building images

Podman's overlay storage does not work on Lustre (network filesystem).  Configure
Podman to use a local filesystem — see [Step 0a](#step-0a--configure-podman-storage-required).

After updating `storage.conf`, run `podman system reset --force` and retry.

### `docker-compose` / `podman-compose` not found

`podman compose` requires an external compose provider.  Install `podman-compose`:

```bash
pip install --user podman-compose
export PATH="$HOME/.local/bin:$PATH"   # add to ~/.bashrc
```

### Containers not starting

```bash
podman compose logs postgres
podman compose logs minio
```

### Server not starting / crashing

```bash
tail -50 logs/server.log
```

Common issues:
- **Missing venv**: Run `./setup_server.sh` first.
- **Port already in use**: Another server instance may be running. Check `cat server.pid`.
- **Import errors**: Activate the venv (`source .venv/bin/activate`) and check `pip list`.

### MinIO unreachable from compute node

Compute nodes access MinIO via an SSH tunnel (`localhost:9000`), not directly.

1. Ensure [Step 0b](#step-0b--enable-compute--login-ssh-required-for-slurm-jobs) is done
   (your public key is in `~/.ssh/authorized_keys`).
2. Test self-SSH on the login node: `ssh -o BatchMode=yes localhost echo ok`
3. Run the verification in [Step 2.5](#step-25--verify-minio-reachability-from-slurm-jobs-via-ssh-tunnel).
4. Check that `S3_PRESIGN_ENDPOINT=http://localhost:9000` in `.env`.

### Server fails to connect to postgres

Wait 10–15 s after `podman compose up` — postgres needs time to initialize on first run.
The server retries on startup.

### umrs-client cannot connect

Make sure `ssh_tunnel.sh` is running.  Test the tunnel:

```bash
nc -zv localhost 5555   # should succeed
```

### Model download hangs

The embedding model (~440 MB for `intfloat/e5-base-v2`) is downloaded on first use
to `$MODEL_DIR`.  Progress can be watched in server logs:

```bash
tail -f logs/server.log | grep -i download
```

---

## Deployment Verification

After deploying, run the demo suite to verify everything works end-to-end.
The demos are in `deploy/rccs-cloud/demos/`.

### Available demos

| # | Demo | What it tests | GPU needed |
|---|------|---------------|------------|
| 00 | Health Check | ZMQ, MinIO, server ping | No |
| 01 | Text Embedding | Upload text → generate embeddings → export CSV | No (vLLM optional) |
| 02 | RAG Search | Hybrid search over embeddings | No (needs `UMRS_RAG_ENABLE=1`) |
| 03 | SAM3 Segmentation | Upload images → Slurm GPU → download segmentations | Yes |
| 04 | HMAE Inference | Upload images → Slurm GPU → download reconstructions | Yes |
| 05 | User Model | Run user-defined TextStatsStep on S3 objects | No |
| 06 | Pipeline SDK | Multi-step pipeline workflow | No |
| 09 | Model Registry | Register / list / upload models | No |

### Run all core demos

```bash
cd deploy/rccs-cloud/demos
source ../.venv/bin/activate
bash run_all_core.sh
```

Or run individually:

```bash
bash 00_health_check.sh
bash 03_sam3_segmentation.sh   # ~2-5 min (Slurm job)
bash 04_hmae_inference.sh      # ~1-3 min (Slurm job)
```

### Latest test results (2026-04-06)

| Demo | Result | Duration |
|------|--------|----------|
| 00 Health Check | 3/3 passed | instant |
| 01 Text Embedding | 20 texts uploaded + embedded | ~5 s |
| 03 SAM3 Segmentation | 6 images processed | 1m 43s |
| 04 HMAE Inference | 6 images reconstructed | 47s |
| 05 User Model | 10 files analysed | ~2 s |
| 09 Model Registry | 3 models listed | instant |
| Server health check | 11/11 passed | — |
