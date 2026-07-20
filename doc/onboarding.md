# UMRS Onboarding Process

This document describes the complete onboarding lifecycle for UMRS — from
initial server bootstrap through user provisioning and first successful
operation.

## Overview

UMRS onboarding has six phases:

| Phase | Actor | Purpose |
|-------|-------|---------|
| 1. Bootstrap Auth | Admin (server-side) | Create admin user + initial token |
| 2. Enroll Identity | Admin | Register new user in the server |
| 3. Define Provisioning | Admin/User | Write or generate user config (experiences, modalities) |
| 4. Persist Server Config | Daemon | Store user record, hot-reload |
| 5. Materialize Resources | Daemon | Create MinIO IAM user, buckets, Postgres roles/tables |
| 6. Verify | User/Admin | Confirm connectivity and resource access |

## Deployment Paths

UMRS supports two onboarding styles depending on how users interact with the
platform:

### A. Direct UMRS Client (ZMQ/SSH Tunnel)

Users run `umrs` directly against the server. Suitable for:
- Developer/local deployments
- HPC clusters where SSH tunnels are natural
- Single-user or small-team setups

### B. R4 Platform Gateway (Recommended for Shared Deployments)

Users interact via the `r4` CLI and Gateway. The admin performs server-side
enrollment; end users never touch `umrs` directly.

See [R4 Quick Start](../../docs/r4/quickstart.md) and
[R4 Admin Guide](../../docs/r4/admin-guide/README.md) for the Gateway path.
For umrs-front HTTPS Gateway with OIDC, also see
[umrs-front OIDC Deployment](../../docs/UMRS_FRONT_OIDC_DEPLOYMENT.md). OIDC
authentication must still be paired with a UMRS user record, Slurm account, and
per-user SSH enrollment before endpoint provisioning can run on HPC.

---

## Phase 1: Bootstrap Auth

Run once on the server after initial deployment:

```bash
umrs-server admin bootstrap --admin-user admin --json
```

This creates:
- An admin user record in `<user_files_dir>/admin.json`
- A plaintext admin token (shown once — save it)

The admin token authorizes subsequent user management operations.

## Phase 2: Enroll Identity

The admin registers a new user on the server. Two onboarding paths are
supported depending on whether the user authenticates via institutional OIDC
(e.g. RIKEN Okta) or admin-issued JWT tokens.

### Admin CLI auto-discovery

The `umrs-server admin` command auto-discovers `conf_umrs.json` using the same
chain as the daemon: **CWD → `$UMRS_CONFIG` → `~/.config/umrs`**. Run admin
commands from the deploy directory and no `--config` flag is needed:

```bash
cd /path/to/deploy/rccs-cloud   # conf_umrs.json lives here
umrs-server admin user add alice ...
```

New users are also automatically appended to `conf_umrs.json`'s `users` list.

### Path A: OIDC Onboarding (Institutional IdP)

When users log in via OIDC, the IdP issues a JWT whose `sub` claim is an
opaque identifier (e.g. RIKEN ID `241723`). This value does not satisfy the
local username regex `^[a-z][a-z0-9._-]{1,62}$`, so the admin must create a
mapping from the OIDC subject to the user's login-node OS account.

**Information to collect:**

| Input | Example | Purpose |
|-------|---------|---------|
| OIDC subject (RIKEN ID) | `241723` | Maps IdP `sub` → UMRS user |
| Login-node OS account | `masaru.nagaso` | UMRS username, SSH delegation target |
| Slurm allocation account | `rccs-dmp` | `#SBATCH --account` for jobs |

**Registration:**

```bash
# From the deploy directory — conf_umrs.json is discovered automatically
umrs-server admin user add masaru.nagaso \
    --fullname "Masaru Nagaso" \
    --email masaru.nagaso@riken.jp \
    --role user \
    --project default \
    --oidc-subject 241723 \
    --slurm-account rccs-dmp --site rccs-front
```

When the user logs in via OIDC, the Gateway sidecar passes `X-User-ID: 241723`.
UMRS `resolve_user("241723")` looks up the user record with
`oidc_subject: "241723"`, resolving to `masaru.nagaso`.

The OIDC subject can also be set or changed after user creation:

```bash
umrs-server admin user set-oidc-subject masaru.nagaso 241723
```

### Path B: JWT-Only Onboarding (No IdP)

When OIDC is not available, the admin issues JWT tokens directly. The JWT `sub`
is the UMRS username, so no OIDC subject mapping is needed.

**Information to collect:**

| Input | Example | Purpose |
|-------|---------|---------|
| Login-node OS account | `alice` | UMRS username, JWT `sub` claim |
| Slurm allocation account | `rccs-dmp` | `#SBATCH --account` for jobs |

**Registration with inline onboarding token:**

```bash
# From the deploy directory — conf_umrs.json is discovered automatically
umrs-server admin user add alice \
    --fullname "Alice Smith" \
    --role user \
    --email alice@example.com \
    --project default \
    --slurm-account rccs-dmp --site rccs-front \
    --issue-token --token-days 90
```

The printed JWT is given to the user, who runs:

```bash
r4 auth login --gateway-url <GATEWAY_URL> --api-key <TOKEN>
```

Or generate additional tokens later:

```bash
umrs-server admin token generate-jwt --user alice --project default --days 90
```

### Common Post-Registration Steps

For multi-cluster deployments, scope the Slurm account to a specific site:

```bash
umrs-server admin user set-slurm-account alice research-gpu --site rccs-front
umrs-server admin user set-slurm-account alice gpu-group --site ai4s
```

What this produces:

| File | Purpose |
|------|---------|
| `<user_files_dir>/alice.json` | **Admin record** — role, projects, quota (server auth) |
| `<user_files_dir>/alice.client.json` | **Client config template** — experiences, modalities, buckets (send to user) |

The admin record is stored on the server for authorization. The client config
template is a ready-to-use starting point that the admin can send directly to
the new user — no manual JSON editing required.

By default the client template is written next to the admin record. Override
the path with `--client-template <path>` or skip it entirely with
`--no-client-template`.

The daemon also receives `SIGHUP` for a best-effort hot-reload.

### SSH Delegation Enrollment (HPC Deployments)

After user registration, the UMRS service public key must be installed in the
user's login-node account so UMRS can submit Slurm jobs on their behalf.

Recommended (user runs on the login node):

```bash
r4 onboard
```

The command shows the service key fingerprint and asks before appending a
forced-command line to `~/.ssh/authorized_keys`.

If `r4 onboard` is not available, the admin can provide the forced-command
line for the user to add manually to their `~/.ssh/authorized_keys`.

### Listing / Inspecting Users

```bash
umrs-server admin user list
umrs-server admin user show alice
```

### Regenerating a Client Template

If the client config template needs to be regenerated independently:

```bash
umrs-server admin user generate-template alice \
    --fullname "Alice Smith" \
    -o alice.client.json
```

Or print it to stdout without writing a file:

```bash
umrs-server admin user generate-template alice --json
```

## Phase 3: Define Provisioning (User Config)

A user config (the **client config**) describes what resources should be
provisioned. It contains:
- `username` / `fullname` — identity
- `s3_configuration_file` — path to S3 credentials (generated during provisioning)
- `experiences[]` — logical experiment scopes
  - `modalities[]` — per-modality storage and model configuration

**This is different from the admin record** (`user_files/alice.json`).
The admin record is for server-side authorization. The client config is for
workload provisioning (buckets, databases, embedding pipelines).

### Recommended: `login` creates it automatically

`umrs login` now handles both profile creation *and* user config
creation in one step. If `~/.config/umrs/users/<username>.json` does not
exist, a default config is generated automatically:

```bash
umrs login --host <server-host> --api-key <token-from-admin>
```

If a config file already exists at that path, login asks:

```
  User config already exists: ~/.config/umrs/users/alice.json
  Overwrite with a fresh default config? [y/N]
```

Press Enter (or `n`) to keep the existing config. Type `y` to regenerate from
the default template.

### Adding experiences after login

The auto-generated config contains a single default text-embedding experience.
Add workload-specific experiences with:

```bash
# Import an experience fragment (adds/replaces by xp_name)
umrs experience add path/to/experience.json --update

# Then push to the server and provision resources
umrs reload
```

### Alternative: Generate config explicitly

```bash
# Interactive
umrs config init

# Non-interactive
umrs config init --username alice --fullname "Alice Smith"

# From admin-supplied template (legacy flow, still works)
umrs config init --username alice --template alice.client.json
```

This writes to `~/.config/umrs/users/alice.json`.

### Example Config

```json
{
  "username": "alice",
  "fullname": "Alice Smith",
  "s3_configuration_file": "~/.config/umrs/s3cfg_alice",
  "experiences": [
    {
      "xp_name": "TextAnalysis",
      "sql_username": "alice",
      "modalities": [
        {
          "modality_name": "text",
          "s3": { "bucket_alias": "alice-text" },
          "model": {
            "model_name": "intfloat/e5-base-v2",
            "vector_size": "768"
          },
          "sql": {
            "store_embeddings": "yes",
            "table_name": "text"
          }
        }
      ]
    }
  ]
}
```

### S3 Credentials Path

The default path for S3 credentials is:

```
~/.config/umrs/s3cfg_<username>
```

This follows XDG conventions and avoids polluting `$HOME` with dotfiles.
Legacy paths (`~/.s3cfg_<username>`) are still discovered for backward
compatibility.

## Phase 4: Persist Server Config

### Direct Client Path

The user (or admin) sends the config to the running server:

```bash
umrs reload
```

### R4 Gateway Path

On operator-managed deployments, the admin enrolls the user via
`umrs-server admin user add` and provisioning is triggered server-side.
End users never run `reload` themselves.

## Phase 5: Materialize Resources

When the server receives a user config (via `reload` or admin enrollment), it
provisions the following resources **idempotently**:

### MinIO (S3 Object Storage)

1. **IAM user** — creates a MinIO user for the declared `username`. A random
   strong password is generated and written to the S3 config file at
   `~/.config/umrs/s3cfg_<username>`. If the file already exists, existing
   credentials are reused.
2. **Buckets** — one bucket per modality, named
   `<username>/<xp_name>/<bucket_alias>`. Modalities with
   `need_output_bucket` get a second output bucket.
3. **IAM policies** — per-bucket policies are created and attached
   (`GetObject`, `PutObject`, `DeleteObject`, `ListBucket`). A base
   `list-only` policy enables bucket enumeration.

### PostgreSQL (Embedding & Metadata Storage)

1. **Role** — a Postgres role per `sql_username`.
2. **Database** — a database per experience with `pgvector` enabled.
3. **Embedding table** — per modality, with columns matching the model's
   vector size and declared metadata fields.

### S3 Credentials Delivery

The S3 config file is written to the path declared in the user's
`s3_configuration_file` field. On shared-filesystem (NFS) deployments (typical
for HPC), this file is immediately accessible to the user since the server and
user share the same filesystem. The parent directory (`~/.config/umrs/`) is
created automatically if it does not exist.

**Security note:** S3 credentials are never transmitted over the ZMQ control
channel. They rely on NFS visibility or out-of-band delivery by the admin.

## Phase 6: Verify

### Direct Client Verification

```bash
# Connectivity
umrs ping

# Bucket visibility
umrs bucket list

# Overall status
umrs status all
```

### R4 Gateway Verification

```bash
r4 health
r4 auth status
r4 model list --limit 5
```

---

## Connection Profiles

A **connection profile** tells `umrs` where the server is and how to
reach it. It must exist before any `umrs` command can contact the
server.

### Creating a profile (recommended)

`umrs login` creates the profile and authenticates in a single step.
It also auto-creates the user config (see Phase 3), so for most users this is
the only setup command needed:

```bash
# Direct ZMQ (local or same-host)
umrs login --host <server-host> --api-key <token>

# SSH tunnel mode (remote HPC — forwards ZMQ port over SSH)
umrs login --host <server-host> --mode ssh_tunnel --api-key <token>

# Gateway / R4 mode (HTTPS)
umrs login --gateway-url https://<gateway-host> --api-key <token>

# With an explicit profile name (default is "<host>-<mode>")
umrs login --host rccs-cloud --mode ssh_tunnel \
    --profile-name rccs --api-key <token>
```

Profile files are saved to `~/.config/umrs/connections/<name>.json` and the
JWT to `~/.config/umrs/connections/<name>.token`. The profile is automatically
set as the active one.

### Selecting a profile for subsequent commands

```bash
# Show currently active profile
umrs profile show

# List all profiles
umrs profile list

# Switch to a different profile
umrs profile use rccs

# Override per-command with --profile
umrs --profile rccs list_buckets
```

### SSH tunnel setup (HPC)

For SSH tunnel mode, open the tunnel before running any commands:

```bash
# Tunnel ZMQ control port (5555) and data ports (5556-5563)
ssh -N -L 5555:localhost:5555 \
       -L 5556:localhost:5556 \
       -L 5557:localhost:5557 \
       rccs-cloud &

# Then login
umrs login --host 127.0.0.1 --mode ssh_tunnel --api-key <token>
```

Or use the bootstrap script if provided by your deployment:

```bash
./bootstrap.sh --profile rccs-cloud --tunnel
```

### Creating a profile manually (advanced / legacy)

```bash
# Direct mode (local development)
umrs profile init --name local --mode direct

# SSH tunnel mode
umrs profile init --name rccs --mode ssh_tunnel --ssh-host rccs-cloud

# Import an existing profile JSON
umrs profile import /path/to/profile.json
umrs profile use my-profile
```

---

## Shared-Platform S3 vs User-Owned S3

| Aspect | Shared Platform S3 | User-Owned S3 |
|--------|-------------------|---------------|
| MinIO instance | Admin-managed, part of platform | User provides endpoint |
| IAM provisioning | Automatic via `reload` | Manual — user provides credentials |
| `s3_configuration_file` | Auto-generated by server | User writes manually |
| Discovery | `~/.config/umrs/s3cfg_*` | Any path declared in user config |
| Use case | Default for most deployments | Advanced / multi-cloud setups |

---

## Backward Compatibility

### Legacy s3cfg Paths

Users with existing configs referencing `~/.s3cfg_<username>` do not need to
migrate. The auto-discovery system searches both:
- `~/.config/umrs/s3cfg*` (new default, XDG-compliant)
- `~/.s3cfg*` (legacy fallback)

### Existing User Files

Existing `user_files/*.json` with old paths continue to work as-is. The
`s3_configuration_file` field is treated as an opaque path — the server writes
to whatever is declared. Only new configs generated by `umrs config init`
use the updated default.

---

## Full Onboarding Sequence Diagram

```
Admin (server node)            Server/Daemon                     User (client)
  │                                │                              │
  │─── bootstrap ─────────────────▶│                              │
  │◀── admin token ────────────────│                              │
  │                                │                              │
  │─── cd deploy/ && user add ────▶│  (auto-discovers conf_umrs)  │
  │    alice.json written +        │──── SIGHUP ──▶ hot-reload    │
  │    registered in conf_umrs.json│                              │
  │─── token generate-jwt alice ──▶│                              │
  │◀── JWT ────────────────────────│                              │
  │──────────────────────────── give JWT to user ────────────────▶│
  │                                │                              │
  │                                │           ◀── login ─────────│
  │                                │    (creates profile +        │
  │                                │     user config locally)     │
  │                                │                              │
  │                                │   ◀── config import xp.json ─│  (optional)
  │                                │                              │
  │                                │           ◀── reload ────────│
  │                                │─── create MinIO IAM ────▶    │
  │                                │─── create buckets ──────▶    │
  │                                │─── create PG role ──────▶    │
  │                                │─── write s3cfg ─────────▶ (NFS)
  │                                │                              │
  │                                │           ◀── ping ──────────│
  │                                │           ◀── list_buckets ──│
  │                                │──── [bucket list] ──────────▶│
```

---

## Related Documentation

- [Demo 00: Onboarding](../demos/00_onboarding/README.md) — runnable demo script
- [User Guide](user_guide.md) — end-user CLI usage after onboarding
- [Admin Guide](admin_guide.md) — server administration
- [R4 Quick Start](../../docs/r4/quickstart.md) — Gateway-mediated onboarding
- [R4 Admin Guide](../../docs/r4/admin-guide/README.md) — platform-level user management
- [ZMQ API Contract](zmq_api_contract.md) — transport protocol details
- [Server/Client Deployment](../../docs/SERVER_CLIENT_DEPLOYMENT.md) — deployment topology
