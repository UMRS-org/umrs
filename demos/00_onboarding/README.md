# Demo 00: User Onboarding & Initialization

First-run onboarding for a UMRS user. This demo creates a user config (if
needed), pushes it to the server to trigger provisioning, and verifies that
the new buckets are visible afterwards.

See [../README.md](../README.md) for shared setup, connection profiles, and
SSH tunnel guidance.

> **Using the R4 Gateway instead?** If your deployment fronts UMRS with the
> R4 Platform Gateway (OIDC login, `r4` CLI, no direct ZMQ/SSH tunnel), use
> [docs/r4/quickstart.md](../../../docs/r4/quickstart.md) for the user flow
> and
> [docs/r4/admin-guide/README.md → UMRS User Enrollment](../../../docs/r4/admin-guide/README.md#umrs-user-enrollment-required-before-first-use)
> for the admin enrollment step. The provisioning described here (MinIO IAM,
> Postgres role, per-experience buckets) is the same — the admin now creates
> the server-side user record with `umrs-server admin user add <username> ...`,
> and the later admin-authorized `umrs-client reload` still triggers provisioning. End
> users never run `umrs-client profile init` or open an SSH tunnel themselves.

> **Operator-assisted note:** This onboarding flow provisions server-side state.
> On operator-managed deployments, treat provisioning as an operator/site step.
> The normal user demos assume that onboarding has already been completed.

## Quick Start

```bash
cd "$(git rev-parse --show-toplevel)/umrs/demos/00_onboarding"
bash run_demo.sh
```

---

## Step 1 (Admin): Server-Side Setup

The server admin must complete these steps **once** before any user can
onboard. See the [deployment guide](../../deploy/rccs-cloud/README.md) for
full details.

### 1a. Start Infrastructure Services

PostgreSQL (with pgvector) and MinIO run as containers managed by the
deployment's `docker-compose.yml` (or `podman compose` on HPC):

```bash
# On the server host:
cd deploy/rccs-cloud
./start.sh          # starts postgres + minio containers, then umrs-server
```

Key environment variables (set in `.env`):

| Variable | Default | Purpose |
|----------|---------|---------|
| `POSTGRES_USER` | `umrs` | Postgres superuser name |
| `POSTGRES_PASSWORD` | `umrs_rccs` | Postgres superuser password |
| `MINIO_ROOT_USER` | `minioadmin` | MinIO admin access key |
| `MINIO_ROOT_PASSWORD` | `minioadmin` | MinIO admin secret key |

### 1b. Admin S3 Configuration

The server needs an admin-level S3 config file (default `~/.config/umrs/s3cfg`) to
perform IAM operations. This file is referenced by
`admin_s3_configuration_file` in the server config:

```ini
# ~/.config/umrs/s3cfg
[default]
host_base = 127.0.0.1:9000
host_bucket = 127.0.0.1:9000
bucket_location = us-east-1
use_https = False
access_key = minioadmin
secret_key = minioadmin
```

### 1c. Server Config (`conf_umrs.json`)

The server config declares the admin S3 config path, the directory for user
files, and the initial list of users:

```json
{
  "admin_s3_configuration_file": "~/.config/umrs/s3cfg",
  "user_files_dir": "./user_files",
  "users": [
    "./user_files/user1.json"
  ]
}
```

- `user_files_dir` — where new user configs are persisted when received via
  `reload`. Paths are resolved relative to the config file location.
- `users` — initial user list. Seed it with JSON files under `user_files_dir`,
  and use `umrs-server admin user add ...` for ongoing enrollment instead of
  hand-editing this array.

### 1d. Bootstrap the Admin CLI (first time only)

```bash
umrs-server admin bootstrap --admin-user admin --json
```

Save the printed bootstrap token if you need admin/API access later; the raw
token is shown only once.

### 1e. Enroll the User

```bash
umrs-server admin user add alice \
    --fullname "Alice Smith" \
    --role user
```

Add `--email`, `--slurm-account`, and repeated `--project` flags as needed.
This will:
- Write the **admin record** to `<user_files_dir>/alice.json` (role,
  projects, quota — for server-side authorization).
- Write a **client config template** to
  `<user_files_dir>/alice.client.json` (experiences, modalities, buckets
  — for workload provisioning). Send this file to the user.
- Trigger a best-effort daemon reload when `umrs-server` is already running.

The client template path can be overridden with `--client-template <path>`
or skipped entirely with `--no-client-template`.

### 1f. Manage Enrolled Users

```bash
umrs-server admin user list
umrs-server admin user show alice
umrs-server admin user disable alice
umrs-server admin user delete alice --purge -y
```

---

## Step 2 (User): Client-Side Setup

After the admin completes Step 1, the user configures their local client.

### Preferred: `umrs-client login` (single command)

```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT/umrs/demos/00_onboarding"

# 0. Authenticate and create a connection profile in one step.
umrs-client login --host <server-host> --api-key <token-from-admin>

# 1. Show the active connection profile and user.
umrs-client profile show || true

# 2. Verify client → server connectivity.
umrs-client ping

# 3. Trigger onboarding (idempotent).
#    On local/dev or non-role-enforced deployments this may work directly.
#    On operator-managed sites, ask the site/operator to complete provisioning.
umrs-client reload

# 4. Optional: show an overall server summary.
umrs-client status all

# 5. Confirm that bucket aliases are visible.
umrs-client list_buckets
```

### Legacy: Separate `profile init` + `config init`

The old two-step flow still works:

```bash
# 0a. Create a connection profile (interactive — prompts for profile name).
umrs-client profile init

# 0b. Import the client config template the admin sent.
umrs-client config init \
    --template alice.client.json \
    --username alice
# Or generate your own config from scratch:
umrs-client config init --username alice --fullname "Alice Smith"
```

### `profile init` Options (legacy)

```bash
# Direct mode (local development).
umrs-client profile init --name local --mode direct

# SSH tunnel mode (remote HPC cluster).
umrs-client profile init --name rccs --mode ssh_tunnel --ssh-host rccs-cloud

# Custom ports and service URLs.
umrs-client profile init --name custom --host 10.0.0.5 \
    --control-port 6000 --data-port-start 6001 --data-port-end 6008 \
    --minio-url http://10.0.0.5:9000

# Overwrite an existing profile.
umrs-client profile init --name rccs --mode ssh_tunnel --ssh-host rccs-cloud --force
```

### Selecting an Existing Profile

If you already have a profile JSON, import and select it:

```bash
umrs-client profile import /path/to/profile.json
umrs-client profile use my-profile
```

### `config init` Options (legacy)

```bash
# Non-interactive — pass username and full name directly.
umrs-client config init --username alice --fullname "Alice Smith"

# Use an existing JSON as a template (copies structure, replaces identity).
umrs-client config init --username bob --fullname "Bob Jones" \
    --template ~/.config/umrs/users/alice.json

# Write to a custom path instead of the default location.
umrs-client config init --username carol --fullname "Carol Lee" \
    --output-path ./carol.json

# Overwrite an existing config file.
umrs-client config init --username alice --fullname "Alice Smith" --force
```

### Selecting an Existing Config

If you already have a user JSON, register it once:

```bash
umrs-client config use --user ~/.config/umrs/users/user1.json
```

The path above is a JSON file, not a shell command. Open it in an editor if you
want to modify it; do not paste the raw path alone into your shell prompt.

After that, no `-c` flag is needed on any `umrs-client` command.

### Config Update (Operator-Assisted)

After enrollment, update user config using the mechanism supported by your
deployment. On local/dev or non-role-enforced deployments:
`umrs-client reload -c updated_user.json`.
On operator-managed sites, ask the operator/site to apply the updated config.
The server will accept the update only if the username already exists in the
server's user registry.

---

## What `reload` Provisions

When a client calls `reload`, the server performs the following provisioning
steps automatically. All steps are **idempotent** — re-running `reload` with
an unchanged config is a no-op.

### MinIO (S3 Object Storage)

1. **IAM user creation** — the server creates a MinIO user for the username
   declared in the user config. A random strong password is generated and
   written to a per-user S3 config file on the server
   (`~/.config/umrs/s3cfg_<username>`). If the file already exists, the existing
   credentials are reused.
2. **Bucket creation** — one S3 bucket per modality is created, named
   `<username>/<xp_name>/<bucket_alias>`. Modalities that declare
   `need_output_bucket` get a second output bucket.
3. **IAM policy attachment** — a per-bucket policy (`allow-<bucket>`) is
   created and attached to the user, granting `GetObject`, `PutObject`,
   `DeleteObject`, and `ListBucket`. A base `list-only` policy is also
   attached so the user can enumerate their own buckets.

### PostgreSQL (Embedding & Metadata Storage)

1. **Database role** — a Postgres role is created for each `sql_username`
   declared in the user config's `experiences` array (e.g. `usr1_text`).
2. **Database** — a database is created per experience, named
   `<sql_username>_<xp_name>` (e.g. `usr1_text_text_embedding`). The
   `pgvector` extension is enabled in each database.
3. **Embedding table** — a table is created per modality with columns
   matching the model's vector size and the metadata fields declared in
   the user config's `sql.meta_data` array.

### Data Flow

```
Admin                                Server
─────                                ──────
umrs-server admin user add alice ──► create user_files/alice.json       (admin record)
                                     create user_files/alice.client.json (client template)
                                     └─ best-effort daemon reload (SIGHUP)

Admin sends alice.client.json ──────► User

User                                 Server
────                                 ──────
umrs-client config init
  --template alice.client.json
  └─ creates ~/.config/umrs/users/alice.json

reload ───── sends user JSON ──────► validate/update enrolled user config

                                     create_s3_sql_users()
                                       ├─ MinIO: create IAM user + password
                                       ├─ MinIO: create buckets
                                       └─ Postgres: CREATE USER

                                     configure_user()
                                       ├─ Postgres: CREATE DATABASE + pgvector
                                       ├─ Postgres: CREATE TABLE (embeddings)
                                       └─ register bucket entries in memory
```

---

## Expected Result

After onboarding, `list_buckets` should show the aliases configured in the
user JSON. At that point you can continue with Demo 01 or Demo 02.
