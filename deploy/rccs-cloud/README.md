# UMRS Deployment Guide — rccs-cloud

This guide walks through deploying UMRS with the server on the rccs-cloud login node.

Recommended mode: run MinIO directly on rccs-cloud and let Slurm compute nodes
access it over the cluster network. Use SSH tunnels only for local client access.

---

## Architecture Overview

```
Local machine (optional)          rccs-cloud (login node)          dgx-s / dgx-m* (compute)
────────────────────              ───────────────────────          ──────────────────────────
umrs-client  ←── SSH tunnel ───►  umrs-server (podman)  ←── Slurm ───►  mini-rivault
rag client   ←── SSH tunnel ───►  rag-service (podman)                  vLLM + LiteLLM
                                  minio (podman)  ◄────────── HTTP/S3 ─────────┘
                                  postgres/pgvector (podman)
                                  /lvs0/rccs-dmp/  (Lustre — shared)
```

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

### 1b. Edit user config

`user_files/user1.json` defines the experiment modalities (text, image, etc.).
Edit the `fullname` field and bucket aliases to match your setup.
No change is required to use the defaults.

### 1c. Edit server config

`conf_umrs.json` lists the user files.  To add a second user, append another
entry to the `users` array and create a corresponding `user_files/user2.json`.

---

## Step 2 — Start the server

```bash
# On rccs-cloud, inside deploy/rccs-cloud:
./start.sh
```

The script will:

1. Auto-create `.env` from `env.example` if not present (prompts you to edit it).
2. Create the Lustre model cache directories.
3. Run `podman compose up -d` (postgres, minio, umrs-server).

Check that everything is up:

```bash
podman compose ps
podman compose logs -f umrs-server
```

The first launch will:
- Pull/build container images (~1–2 min over Lustre-cached docker layers).
- Download embedding model weights to `$MODEL_DIR/sentence_transformers/` (first run only).
- Auto-create MinIO buckets and PostgreSQL databases for each configured user.

---

## Step 2.5 — Verify MinIO reachability from Slurm jobs via SSH tunnel

Compute nodes (dgx-s, dgx-m*) cannot reach the login node’s MinIO directly
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
sbatch -p dgx-s --time=00:05:00 --wrap='
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
for part in dgx-s dgx-m0 dgx-m1 dgx-m2 dgx-m3; do
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

With tunnels open (if running from local Mac), run any demo from `demos/` using
the standard umrs-client:

```bash
# Example: push a text file and retrieve its embedding
umrs-client -c ~/.config/umrs/conf_umrs.json upload_dir ./my_texts --modality text

# RAG search (requires rag service, see below)
umrs-client -c ~/.config/umrs/conf_umrs.json rag_query "What is the main topic?"
```

Or use the demo scripts:

```bash
cd demos/01_text_embedding
./run_demo.sh
```

---

## Step 6 — GPU embeddings via mini-rivault (optional)

Text and image embeddings run locally in the umrs-server container by default
(using CPU).  For GPU-accelerated embeddings on dgx nodes via mini-rivault:

1. Follow the mini-rivault setup in `demos/slurm-dev/` (uses vLLM + LiteLLM).
2. Update `conf_umrs.json` to point your model at the OpenAI-compatible endpoint:

```json
{
  "model_name": "intfloat/e5-base-v2",
  "model_type": "openai://http://localhost:PORT/v1"
}
```

The server will auto-launch a Slurm job via `rivault_launcher.py` when the model
is first requested.

---

## Day-to-day operations

```bash
# Check status
podman compose ps

# Tail server logs
podman compose logs -f umrs-server

# Reload server config (without restart) — via umrs-client
umrs-client reload

# Stop services (data preserved in volumes)
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
    "/root/user_files/user1.json",
    "/root/user_files/user2.json"
  ]
}
```

3. Mount the new file in `docker-compose.yml` — add under `umrs-server` volumes:

```yaml
- ./user_files:/root/user_files:ro
```

(Already present — just add the new JSON file into the `user_files/` directory.)

4. Send the RELOAD command to provision the new user without restarting:

```bash
umrs-client reload
```

The server will create the MinIO IAM user, generate `~/.s3cfg_user2` inside the
container, and create all required buckets.

5. Retrieve the generated S3 config for user 2:

```bash
podman exec umrs-server cat /root/.s3cfg_user2
```

Copy that file to the second user's machine at `~/.config/umrs/s3cfg` (or
whatever `s3_configuration_file` path their `user2.json` specifies).

### Option B — Grant another OS user direct MinIO access

If the second OS user wants to access MinIO independently (e.g. to upload from
their own scripts), create credentials for them via the MinIO admin client (`mc`):

```bash
# From inside the umrs-server container (mc is installed)
podman exec -it umrs-server bash

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
podman compose logs umrs-server
```

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
podman compose logs -f umrs-server | grep -i download
```
