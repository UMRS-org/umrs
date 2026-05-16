# Demo 00: User Onboarding & Initialization

Creates the `demo_user` account, pushes it to the server to trigger
provisioning, and verifies the new buckets are visible.

See [../README.md](../README.md) for shared setup, connection profiles, and
SSH tunnel guidance.

## What it does

1. **Create user config** — generates `~/.config/umrs/users/demo_user.json`
   (idempotent, skipped if already present)
2. **Show active profile** — confirms the connection profile is set
3. **Pre-flight check** — verifies `umrs-client` can reach the server
4. **Onboard** — calls `reload` to trigger server-side provisioning:
   - MinIO IAM user + per-bucket access policy
   - (No experiences yet — subsequent demos add their own)
5. **Confirm** — shows provisioned status and bucket list

## Usage

```bash
cd "$(git rev-parse --show-toplevel)/umrs/demos/00_onboarding"
bash run_demo.sh

# Use a different username
DEMO_USERNAME=other_user bash run_demo.sh
```

## Prerequisites

- `umrs-client` installed and in PATH
- UMRS server running and reachable
- Admin has enrolled `demo_user` server-side (see Step 1 below)

---

## Step 1 (Admin): Enroll the User

Run once on the server node before the user can onboard.

```bash
# Start infrastructure (postgres + minio + umrs-server)
cd deploy/rccs-cloud
./start.sh

# Bootstrap the admin CLI (first time only — token shown once)
umrs-server admin bootstrap --admin-user admin --json

# Enroll demo_user
umrs-server admin user add demo_user \
    --fullname "Demo User" \
    --role user
```

This writes two files under `user_files_dir` (default `./user_files/`):

| File | Purpose |
|------|---------|
| `user_files/demo_user.json` | Admin record (role, quota, projects) — stays on the server |
| `user_files/demo_user.client.json` | Client config template — hand this to the user |

**On a local / single-machine setup** (admin and user are the same person),
copy the template directly to the client config directory:

```bash
# Path will vary — check your umrs-server config for user_files_dir
cp ./user_files/demo_user.client.json \
   ~/.config/umrs/users/demo_user.client.json
```

Then in Step 2, pass it as `--template`:

```bash
umrs-client config init \
    --username demo_user \
    --template ~/.config/umrs/users/demo_user.client.json
```

**On a remote server**, transfer the file to the user's machine with `scp` or
any secure channel, then run the same `config init --template` command above.

If you skip the template (or don't have one), `config init --username demo_user --fullname "Demo User"` creates a minimal config from scratch — experiences get added later by each demo.

### Manage enrolled users

```bash
umrs-server admin user list
umrs-server admin user show demo_user
umrs-server admin user disable demo_user
umrs-server admin user delete demo_user --purge -y
```

---

## Step 2 (User): Client-Side Setup

```bash
# Authenticate and create a connection profile in one step
umrs-client login --host <server-host> --api-key <token-from-admin>

# Verify connectivity
umrs-client ping

# Trigger provisioning (idempotent)
umrs-client reload

# Confirm buckets are visible
umrs-client list_buckets
```

### Profile options

```bash
# Direct mode (local development)
umrs-client profile init --name local --mode direct

# SSH tunnel mode (remote HPC cluster)
umrs-client profile init --name rccs --mode ssh_tunnel --ssh-host rccs-cloud
```

### Config options

```bash
# Create config from scratch
umrs-client config init --username demo_user --fullname "Demo User"

# Create from admin-supplied template
umrs-client config init --username demo_user --template demo_user.client.json

# Register an existing config as active
umrs-client config use --user ~/.config/umrs/users/demo_user.json
```

---

## What `reload` Provisions

All steps are **idempotent** — re-running with an unchanged config is a no-op.

| Resource | What is created |
|----------|-----------------|
| MinIO IAM user | One per username; random password written to `~/.config/umrs/s3cfg_demo_user` |
| S3 buckets | One per modality (`demo_user/<xp_name>/<alias>`); output bucket where declared |
| IAM policies | Per-bucket `allow-<bucket>` policy + base `list-only` policy attached to user |
| Postgres role | One per `sql_username` in the user config's `experiences` array |
| Postgres database | One per experience with `pgvector` extension enabled |
| Embedding table | One per modality with columns matching the model's vector size |

---

## Expected Result

After onboarding, `list_buckets` shows the aliases from the user config.
Run any numbered demo (02, 03, 04 …) — each adds its own experience.
Run `99_cleanup` to tear everything down when finished.
