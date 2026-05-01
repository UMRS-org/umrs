# UMRS Deployment Guide — rccs-cloud

This guide walks through deploying UMRS on the rccs-cloud login node using
a **hybrid architecture**: infrastructure services (PostgreSQL, MinIO) run in
rootless podman containers, while `umrs-server` runs **natively** so it can
invoke Slurm CLI tools (`sbatch`, `squeue`, `scancel`) and create SSH tunnels
to compute nodes.

> Admin-only guide. Normal users of the shared deployment should use the [R4 Basic User Quick Start](../../../docs/r4/quickstart.md) instead. They should not need `setup_server.sh`, `setup_client.sh`, `ssh_tunnel.sh`, Slurm settings, or storage credentials.

---

## Quick Deploy (already set up)

If the server environment is already configured (Steps 0–1 done previously),
these are the only commands needed to deploy or redeploy after a code change:

```bash
# ── On your local machine ────────────────────────────────────────────────
# 1. Commit & push
cd /path/to/ai4s-platform
git push origin ai4s-platform

# ── On rccs-cloud (via SSH) ──────────────────────────────────────────────
ssh rccs-cloud
cd /lvs0/dne1/rccs-dmp/mnagaso/ai4s-platform

# 2. Pull latest code
git pull origin ai4s-platform

# 3. Update the Python package and stop the old server
source umrs/deploy/rccs-cloud/.venv/bin/activate
uv pip install -e umrs/src/ --quiet
cd umrs/deploy/rccs-cloud
./stop.sh

# 4. Start everything (containers + server)
./start.sh

# 5. Verify (wait ~15 s for server init)
./health_check_server.sh
```

If `UMRS_RAG_ENABLE=true` is set in `.env`, `./start.sh` also launches the
native `rag-service` on port `8081`, and `./stop.sh --server` stops it
alongside `umrs-server`.

To restart **only** the server (e.g. after editing Python files), keeping
containers running:

```bash
./stop.sh --server
./start.sh --server
```

`./stop.sh --server` leaves postgres, minio, and any active Slurm-backed
endpoint jobs running. Use plain `./stop.sh` when you want a full stack stop.

To validate the harness itself after wrapper, `site.yaml`, or orchestration
changes, run the site-local validator:

```bash
./validate_harness.sh
```

That submits the real `serve-model.slurm` wrapper through `slurm-podman` for
the `text`, `vision`, and `full` profile sets, proves the required endpoints,
and confirms cleanup by cancelling each Slurm job afterward, escalating to
`KILL` when a graceful `TERM` does not finish promptly. Use
`./validate_harness.sh --case text` when you only need one matrix entry.

To validate the proven reuse and restart path after orchestration or wrapper
changes, run:

```bash
./validate_harness.sh --skip-matrix --lifecycle-case full
```

Expected proof is one initial `full` launch, reuse of that same Slurm job,
cleanup of that job, and a fresh second job after cleanup, with no remaining
`umrs-rivault` entries in `squeue` at the end.

---

## Architecture Overview

```
Local machine (optional)          rccs-cloud (login node)          ng-dgx-m* (compute)
────────────────────              ───────────────────────          ──────────────────────────
umrs-client  ←── SSH tunnel ───►  umrs-server  (NATIVE)  ←── Slurm ───►  mini-rivault
rag client   ←── SSH tunnel ───►  rag-service  (NATIVE, optional)        vLLM / BentoML + LiteLLM
                                  minio     (podman container)  ◄── SSH tunnel ──┘
                                  postgres  (podman container)
                                  /lvs0/rccs-dmp/  (Lustre — shared)
```

**Why native?**  `rivault_launcher.py` calls `sbatch`, `squeue`, `scancel`,
and `ssh -N -L` via `subprocess`.  These binaries are available on the login
node but not inside a `python:3.10-slim` container.  Running the server natively
avoids bind-mounting Slurm libs and SSH sockets, which is fragile and
non-portable.

## Canonical Config Contract

For the rccs-cloud deployment, configuration ownership is split on purpose:

- `deploy/rccs-cloud/site.yaml` is the canonical source for non-secret
  site defaults: scheduler settings, runtime paths, profile defaults,
  model-serving ports, and shared storage paths.
  On rccs-cloud, `scheduler.kind` should stay `slurm-podman` so durable
  endpoint jobs use the wrapper-backed Slurm/Podman launch path.
  Its `database.registry_dsn` also keeps the endpoint registry persistent
  across native server restarts, so legacy upload-triggered RiVault launches
  can reattach to already-running Slurm jobs instead of submitting duplicates.
  The active `ng-dgx-m*` partitions also require `scheduler.memory` to stay
  empty today because Slurm rejects explicit `--mem` requests there.
  They likewise require `scheduler.gpus: 0` because those partitions reject
  explicit GPU request flags even though the nodes themselves are GPU hosts.
- `deploy/rccs-cloud/.env` is for secrets and login-node-local overrides
  such as passwords, MinIO binding, and optional explicit paths.
- `deploy/rccs-cloud/serve-model.slurm` should remain a launch wrapper.
  It should consume exported values, not become a second undocumented source
  of scheduler, profile, or path defaults.

`./setup_server.sh` links both the wrapper and `site.yaml` into
`~/.config/umrs/`, and `./start.sh` exports `UMRS_SITE_CONFIG` explicitly so
the native server does not depend on the current working directory.

Ports tunneled over SSH (for local Mac access):

| Port | Service |
|------|---------|
| 5555 | ZMQ control plane |
| 5556 | ZMQ data plane (add more ports for concurrent users) |
| 9000 | MinIO S3 API (optional; local admin/debug only) |
| 8081 | RAG service (optional, enabled with `UMRS_RAG_ENABLE=true`) |

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
cd /lvs0/dne1/rccs-dmp/mnagaso/ai4s-platform/umrs/deploy/rccs-cloud
cp env.example .env
nano .env          # or vim / any editor
```

Common values to adjust:

| Variable | Default | Notes |
|---|---|---|
| `POSTGRES_PASSWORD` | unset | Must be set in `.env` before startup |
| `MINIO_ROOT_PASSWORD` | unset | Must be set in `.env` before startup |
| `S3_PRESIGN_ENDPOINT` | `http://localhost:9000` | Compute nodes access MinIO via an SSH tunnel (see below).  The Slurm script opens `localhost:9000` → login node MinIO automatically. |
| `VISION_S3_PRESIGN_ENDPOINT` | `http://minio-host:19000` | BentoML vision runs inside a compute-node container, so presigned URLs for vision tasks must target the host-gateway alias rather than `localhost`. |
| `VISION_REQUIRE_PRESIGNED_URLS` | `true` | Disables direct S3 fallback in the BentoML vision service so the container no longer needs S3 secrets. |
| `MINIO_BIND` | `127.0.0.1` | MinIO listens on localhost only.  Compute nodes use the SSH tunnel. |
| `UMRS_SITE_CONFIG` | `~/.config/umrs/site.yaml` | Optional override for the canonical site config path |
| `RIVAULT_SLURM_SCRIPT` | `./serve-model.slurm` | Optional override for the site-specific Slurm wrapper; the legacy `~/.config/umrs/rivault_data/serve-model.slurm` path is only a fallback |
| `MODEL_DIR` | unset by default | Optional local override for the shared model root; otherwise `HF_HOME` comes from `site.yaml` `storage.model_cache_dir` |
| `LITELLM_MASTER_KEY` | unset by default | Required for shared deployments; generate an operator-held bearer-style secret |

Do not duplicate scheduler partitions, `RIVAULT_PROFILES`, LiteLLM port, or
model-serving paths in `.env` unless you are intentionally overriding
`site.yaml`.  Those values are part of the site contract, not per-host secrets.

### 1e. Credentials, API keys, and control boundaries

This deployment uses three different credential classes. Keep them separate.

| Credential | Where it lives | Who should control it | How to generate / rotate |
|---|---|---|---|
| PostgreSQL password | `.env` as `POSTGRES_PASSWORD` | RCCS operators on the login node | `openssl rand -base64 24` and restart `./start.sh` |
| MinIO root password | `.env` as `MINIO_ROOT_PASSWORD` | RCCS operators on the login node | `openssl rand -base64 24` and restart `./start.sh` |
| LiteLLM service key | `.env` or operator environment as `LITELLM_MASTER_KEY` | RCCS operators only | Generate a random bearer-style secret, for example `printf 'sk-ll-%s\n' "$(openssl rand -hex 24)"` |
| UMRS admin tokens | `~/.config/umrs/admin_tokens.json` (hashed at rest) | RCCS operators only | Generate with `umrs-server admin token generate <name> --description '<purpose>'` |

**Important:** UMRS admin tokens are shown only once at creation time. The file
`~/.config/umrs/admin_tokens.json` stores only token hashes, so rotation means
generating a new token and revoking the old one rather than recovering the old
value later.

Example operator commands:

```bash
# Generate a new UMRS admin token (stored hashed in ~/.config/umrs/admin_tokens.json)
source .venv/bin/activate
umrs-server admin token generate rccs-ops --description "RCCS deployment operator"

# List token metadata without revealing token values
umrs-server admin token list

# Revoke a token by name
umrs-server admin token revoke rccs-ops
```

If you also deploy the Platform Gateway for user-facing access, treat its
`JWT_SECRET` as a server-side signing secret only. Do not hand `JWT_SECRET` to
end users. User-facing access should come from externally issued OIDC tokens or
deployment-specific bearer tokens, while `GATEWAY_LITELLM_SERVICE_KEY` remains
strictly service-to-service between Gateway and LiteLLM.

### 1b. Edit user config

`user_files/user1.json` defines the experiment modalities (text, image, etc.).
Edit the `fullname` field and bucket aliases to match your setup.
No change is required to use the defaults.

### 1c. Edit server config

`conf_umrs.json` lists the user files and declares `"user_files_dir": "./user_files"`.
Paths are resolved **relative to the config file location**
(e.g. `./user_files/user1.json`), not the shell's current working directory.
The default rccs-cloud bootstrap intentionally loads only `user1.json` so
initial provisioning stays minimal.
To add a second user later, append another entry to the `users` array and
create a corresponding `user_files/user2.json`.

### 1d. One-time server setup

Run the setup script once to create the Python virtual environment, install
dependencies, and prepare the canonical rccs-cloud config links:

```bash
./setup_server.sh
```

This creates `.venv/` under `deploy/rccs-cloud/`, installs CPU-only PyTorch,
the `umrs` package (editable), and symlinks both `serve-model.slurm` and
`site.yaml` into `~/.config/umrs/`.

---

## Step 2 — Start the server

```bash
# On rccs-cloud, inside deploy/rccs-cloud:
./start.sh
```

The script will:

1. Auto-create `.env` from `env.example` if not present (prompts you to edit it).
2. Resolve `UMRS_SITE_CONFIG` from `~/.config/umrs/site.yaml` (or fallback to the
  checked-in `deploy/rccs-cloud/site.yaml`).
3. Create the shared model root and HF cache directories from `site.yaml` (or `MODEL_DIR` if you explicitly override the shared root).
4. Start postgres and minio containers via `podman compose up -d`.
5. Wait for postgres and minio to become healthy.
6. Launch `server_umrs.py` **natively** (outside any container) with the correct
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

If the native server is already running, use `./stop.sh --server` first.

The first launch will:
- Download embedding model weights into the configured HF cache (`site.yaml` `storage.model_cache_dir`, or `${MODEL_DIR}/hf` when `MODEL_DIR` is explicitly overridden).
- Auto-create MinIO buckets and PostgreSQL databases for each configured user.

---

## Step 2.5 — Verify MinIO reachability from Slurm jobs via SSH tunnel

Compute nodes (currently `ng-dgx-m0` to `ng-dgx-m3`) cannot reach the login node's MinIO directly
(port 9000 is not routable).  The mini-rivault Slurm script
(`src/umrs/mini-rivault-llm/serve-model.slurm`) automatically opens an SSH
tunnel that forwards `localhost:9000` on the compute node to `127.0.0.1:9000`
on the login node.

> **Prerequisite**: [Step 0b](#step-0b--enable-compute--login-ssh-required-for-slurm-jobs)
> must be completed — passwordless SSH from compute nodes to the login node
> is required for the tunnel.

Current site defaults advertise `ng-dgx-m0,ng-dgx-m1,ng-dgx-m2,ng-dgx-m3` as
candidate partitions while `ng-dgx-s` is temporarily unavailable. The
Slurm-Podman backend selects one active partition at submission time. To
verify the tunnel works manually, submit a quick test job against one active
partition:

```bash
# On rccs-cloud:
sbatch -p ng-dgx-m0 --time=00:05:00 --wrap='
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
for part in ng-dgx-m0 ng-dgx-m1 ng-dgx-m2 ng-dgx-m3; do
  sbatch -p "$part" --time=00:05:00 -J "test-$part" --wrap='
    ssh -fN -L 9000:127.0.0.1:9000 '"$HOSTNAME"' -o StrictHostKeyChecking=no -o ExitOnForwardFailure=yes
    sleep 2
    curl -sS -m 5 -I http://localhost:9000/minio/health/live || exit 42
  '
done
```

---

## Step 3 — Legacy direct UMRS client testing (admins only)

This section is for operator verification and troubleshooting only.

For the standard shared-user path, send users to the [R4 Basic User Quick Start](../../../docs/r4/quickstart.md) and keep them on the shared Gateway-facing `r4` flow.

From your local machine, inside this directory:

```bash
cd deploy/rccs-cloud
./setup_client.sh
```

This copies the client configs to `~/.config/umrs/` and runs `uv pip install -e ../../src`,
installing the UMRS client package into the venv at `deploy/rccs-cloud/.venv/`.

> **Note**: `setup_client.sh` requires [uv](https://docs.astral.sh/uv/) (`brew install uv`
> on macOS, or `curl -LsSf https://astral.sh/uv/install.sh | sh`).
> After the script finishes, **activate the venv** before running any demos:
> ```bash
> source deploy/rccs-cloud/.venv/bin/activate
> ```

Verify the install:

```bash
umrs-client --help
```

---

## Step 4 — Open SSH tunnels for local admin tools (admins only)

Keep this running in a terminal (or run it in the background):

```bash
./ssh_tunnel.sh             # foreground — Ctrl-C to stop
./ssh_tunnel.sh &           # background
```

To target a non-default host:

```bash
./ssh_tunnel.sh my-host     # defaults to "rccs-cloud"
```

This tunnel forwards the Platform Gateway on `localhost:8080` in addition to
the direct UMRS ZMQ, MinIO, and RAG ports, so local `r4` commands can use the
same session.

> Add extra `-L` lines to `ssh_tunnel.sh` for additional ZMQ data ports if you
> run multiple concurrent users (e.g. `-L 5557:localhost:5557`).

Note: these tunnels are for your local tools only. Slurm compute nodes do not use
your local tunnels when uploading via presigned URLs.

---

## Step 5 — Run UMRS demos for admin verification

With tunnels open (if running from local Mac), run any demo from the
canonical `demos/` directory at the repo root:

```bash
# Activate the client venv
source deploy/rccs-cloud/.venv/bin/activate

# Quick health check
cd demos/01_health_check
bash run_demo.sh

# Text embedding
cd ../02_text_embedding
bash run_demo.sh
```

See [demos/README.md](../../demos/README.md) for the full list of demos and
[demos/README.md](../../demos/README.md) for the canonical demo walkthrough.

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

1. Mini-RiVault runtime assets configured (see `src/umrs/mini-rivault-llm/` and this deploy directory's `site.yaml`).
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

# Stop + wipe only postgres state (keeps MinIO objects)
./stop.sh --postgres-clean

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
  "user_files_dir": "./user_files",
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
mc alias set local http://127.0.0.1:9000 "$MINIO_ROOT_USER" "$MINIO_ROOT_PASSWORD"

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
5. For BentoML vision tasks, check that `VISION_S3_PRESIGN_ENDPOINT=http://minio-host:19000`
  and `VISION_REQUIRE_PRESIGNED_URLS=true` in `.env`.

In the strict vision path the BentoML container uploads via presigned PUT URLs,
so it no longer receives MinIO access keys or secret keys.

### Server fails to connect to postgres

Wait 10–15 s after `podman compose up` — postgres needs time to initialize on first run.
The server retries on startup.

If startup fails with a catalog error such as `could not open file "global/2964"`, the
persistent `pg_data` volume is usually corrupted or incomplete. Confirm with:

```bash
podman logs --tail 40 umrs-postgres
```

To reinitialize only postgres and keep MinIO data, run:

```bash
./stop.sh --postgres-clean
./start.sh
```

If you intentionally want to wipe both postgres and MinIO, run:

```bash
./stop.sh --clean
./start.sh
```

`--postgres-clean` removes only the postgres volume. `--clean` is fully destructive:
it removes both the postgres and minio volumes managed by this RCCS harness.

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

After deploying, verify everything works end-to-end using the canonical demos
at the repo root. See [demos/README.md](../../demos/README.md) for instructions.

```bash
source deploy/rccs-cloud/.venv/bin/activate

# Health check
cd demos/01_health_check && bash run_demo.sh && cd ..

# Text embedding (no GPU needed)
cd 02_text_embedding && bash run_demo.sh && cd ..

# SAM3 segmentation (requires Slurm + GPU)
cd 03_sam3_segmentation && bash run_demo.sh && cd ..
```

### Latest test results (2026-04-06)

| Demo | Result | Duration |
|------|--------|----------|
| 01 Health Check | 3/3 passed | instant |
| 02 Text Embedding | 20 texts uploaded + embedded | ~5 s |
| 03 SAM3 Segmentation | 6 images processed | 1m 43s |
| 04 Vision Inference | 6 images reconstructed | 47s |
| 08 User Model | 10 files analysed | ~2 s |
| 07 Model Registry | 3 models listed | instant |
| Server health check | 11/11 passed | — |
