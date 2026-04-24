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
> Postgres role, per-experience buckets) is the same — it's still triggered
> by `umrs-client admin user add <user.json>` on the server, but end users
> never run `umrs-client profile init` or open an SSH tunnel themselves.

## Quick Start

```bash
cd "$(git rev-parse --show-toplevel)/umrs/demos/00_onboarding"
bash run_demo.sh
```

## Manual Equivalent To `run_demo.sh`

```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT/umrs/demos/00_onboarding"

# 0a. Create a connection profile (interactive — prompts for profile name).
#     Skip this step if you already have a profile set up.
umrs-client profile init

# 0b. Create a user config (interactive — prompts for username and full name).
#     Writes to ~/.config/umrs/users/<username>.json and sets it as current.
#     Skip this step if you already have a config file.
umrs-client config init

# 1. Show the active connection profile and user.
umrs-client profile show || true

# 2. Verify client → server connectivity.
umrs-client ping

# 3. Trigger onboarding (idempotent).
umrs-client reload

# 4. Optional: show an overall server summary.
umrs-client status all

# 5. Confirm that bucket aliases are visible.
umrs-client list_buckets
```

### `profile init` Options

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

### `config init` Options

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

## What `reload` Provisions

When a client calls `reload`, the server performs the following provisioning
steps automatically. All steps are **idempotent** — re-running `reload` with
an unchanged config is a no-op.

### MinIO (S3 Object Storage)

1. **IAM user creation** — the server creates a MinIO user for the username
   declared in the user config. A random strong password is generated and
   written to a per-user S3 config file on the server
   (`~/.s3cfg_<username>`). If the file already exists, the existing
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

### User Account Registration

User enrollment is controlled by the server administrator. A regular client
**cannot** self-register; it can only update its own config after it has been
enrolled.

#### Admin Enrollment Flow

1. The admin generates an admin token (once):
   ```bash
   umrs-client admin token generate my-admin --description "admin token"
   # Save the printed token — it cannot be retrieved later.
   export UMRS_ADMIN_TOKEN=<token>
   ```

2. The admin prepares a user config JSON (or asks the user to run
   `umrs-client config init` and send the resulting file).

3. The admin enrolls the user on the server:
   ```bash
   umrs-client admin user add /path/to/new_user.json
   ```
   This will:
   - Validate the JSON against the client schema.
   - Write the file to `<user_files_dir>/<username>.json` on the server.
   - Update `conf_umrs.json`'s `users` array.
   - Auto-reload the server to provision MinIO IAM, Postgres roles, and
     buckets.

4. The user can now connect and run commands normally. Subsequent config
   updates (via `maintenance reload -c user.json`) are allowed because the
   user is already enrolled.

#### Listing / Removing Users

```bash
umrs-client admin user list       # show enrolled users from server status
umrs-client admin user remove bob # not yet implemented (placeholder)
```

#### Config Update (Non-Admin)

After enrollment, a user can update their own config by running
`maintenance reload -c updated_user.json`. The server will accept the update
only if the username already exists in the server's user registry.

### Data Flow

```
Client                               Server
──────                               ──────
config init
  └─ creates ~/.config/umrs/
       users/<username>.json

reload ──── sends user JSON ───────► persist user config
                                       └─ user_files/<username>.json
                                       └─ update conf_umrs.json

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

## Server-Side Prerequisites (Admin)

The provisioning above requires infrastructure that the server admin sets up
**once** before any users can onboard. See the
[deployment guide](../../deploy/rccs-cloud/README.md) for full details.

### 1. Start Infrastructure Services

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

### 2. Admin S3 Configuration

The server needs an admin-level S3 config file (default `~/.s3cfg`) to
perform IAM operations. This file is referenced by
`admin_s3_configuration_file` in the server config:

```ini
# ~/.s3cfg
[default]
host_base = 127.0.0.1:9000
host_bucket = 127.0.0.1:9000
bucket_location = us-east-1
use_https = False
access_key = minioadmin
secret_key = minioadmin
```

### 3. Server Config (`conf_umrs.json`)

The server config declares the admin S3 config path, the directory for user
files, and the initial list of users:

```json
{
  "admin_s3_configuration_file": "~/.s3cfg",
  "user_files_dir": "./user_files",
  "users": [
    "./user_files/user1.json"
  ]
}
```

- `user_files_dir` — where new user configs are persisted when received via
  `reload`. Paths are resolved relative to the config file location.
- `users` — initial user list. New users are appended automatically when
  they run `reload` from the client.

### 4. Adding Users

New users must be enrolled by an admin via the `admin user add` command.
See the **Admin Enrollment Flow** section earlier in this document for
the full procedure.

Manual placement of a user JSON in `user_files/` and adding its path to
`conf_umrs.json` is supported but not recommended for production use.

Re-running the same sequence is safe. If the config has not changed, the
server treats the reload as a no-op.

## Expected Result

After onboarding, `list_buckets` should show the aliases configured in the
user JSON. At that point you can continue with Demo 01 or Demo 02.
