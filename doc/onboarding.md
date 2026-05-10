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

Users run `umrs-client` directly against the server. Suitable for:
- Developer/local deployments
- HPC clusters where SSH tunnels are natural
- Single-user or small-team setups

### B. R4 Platform Gateway (Recommended for Shared Deployments)

Users interact via the `r4` CLI and Gateway. The admin performs server-side
enrollment; end users never touch `umrs-client` directly.

See [R4 Quick Start](../../docs/r4/quickstart.md) and
[R4 Admin Guide](../../docs/r4/admin-guide/README.md) for the Gateway path.

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

The admin registers a new user on the server:

```bash
umrs-server admin user add alice \
    --fullname "Alice Smith" \
    --role user \
    --email alice@example.com \
    --slurm-account research-gpu \
    --project team-alpha
```

This writes `<user_files_dir>/alice.json` on the server and sends `SIGHUP` to
the running daemon for hot-reload.

### Listing / Inspecting Users

```bash
umrs-server admin user list
umrs-server admin user show alice
```

## Phase 3: Define Provisioning (User Config)

A user config describes what resources should be provisioned. It contains:
- `username` / `fullname` — identity
- `s3_configuration_file` — path to S3 credentials (generated during provisioning)
- `experiences[]` — logical experiment scopes
  - `modalities[]` — per-modality storage and model configuration

### Generate Via CLI (Recommended)

```bash
# Interactive
umrs-client config init

# Non-interactive
umrs-client config init --username alice --fullname "Alice Smith"
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
umrs-client reload
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
umrs-client ping

# Bucket visibility
umrs-client list_buckets

# Overall status
umrs-client status all
```

### R4 Gateway Verification

```bash
r4 health
r4 auth status
r4 model list --limit 5
```

---

## Connection Profiles

Before `umrs-client` commands work, a connection profile must exist. Profiles
define how the client reaches the server.

```bash
# Direct mode (local development)
umrs-client profile init --name local --mode direct

# SSH tunnel mode (remote HPC cluster)
umrs-client profile init --name rccs --mode ssh_tunnel --ssh-host rccs-cloud

# Import existing profile
umrs-client profile import /path/to/profile.json
umrs-client profile use my-profile
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
to whatever is declared. Only new configs generated by `umrs-client config init`
use the updated default.

---

## Full Onboarding Sequence Diagram

```
Admin                          Server/Daemon                     User
  │                                │                              │
  │─── bootstrap ─────────────────▶│                              │
  │◀── admin token ────────────────│                              │
  │                                │                              │
  │─── user add alice ────────────▶│                              │
  │    (writes alice.json)         │──── SIGHUP ──▶ hot-reload    │
  │                                │                              │
  │                                │                              │
  │   ┌─────────────────────────── │ ◀── config init ─────────────│
  │   │ (direct path only)         │                              │
  │   └─────────────────────────── │ ◀── reload ──────────────────│
  │                                │                              │
  │                                │─── create MinIO IAM ────▶    │
  │                                │─── create buckets ──────▶    │
  │                                │─── create PG role ──────▶    │
  │                                │─── write s3cfg ─────────▶ (NFS)
  │                                │                              │
  │                                │ ◀── ping ────────────────────│
  │                                │ ◀── list_buckets ────────────│
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
