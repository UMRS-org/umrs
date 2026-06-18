# Demo 06 — Server Maintenance

Data integrity and maintenance operations — discover, audit, and repair
the consistency between S3 objects and the pgvector embedding database.

This is intentionally an operator/admin workflow, not a normal end-user demo.

## Prerequisites

- UMRS server running and reachable
- `umrs-client` available
- At least one user configured with data already uploaded (run `demos/02_text_embedding/` first)
- On auth-enforced deployments, an admin token exported as `UMRS_ADMIN_TOKEN`
  or passed via `--admin-token` (see `demos/23_auth_enforcement/`)

> **Note**: `check`, `update`, `recompute`, and `reload` are maintenance
> commands. On deployments with role enforcement enabled they require admin
> role even when you only target your own data.

## Commands

### 1. Discover configured buckets

See what S3 buckets are configured for your user:

```bash
umrs-client list_buckets
umrs-client list_buckets alice    # specific user
```

### 2. Check S3 ↔ DB consistency

Verify that every object in S3 has a matching embedding in pgvector,
and that checksums match:

```bash
umrs-client --admin-token "$UMRS_ADMIN_TOKEN" check
```

Output shows per-bucket: matches, mismatches, not-found, errors.

### 3. Update missing/mismatched entries

Re-embed only the objects that are missing or have stale embeddings
(e.g., the file was re-uploaded with different content):

```bash
umrs-client --admin-token "$UMRS_ADMIN_TOKEN" update
```

This is **incremental** — it skips objects whose embeddings already match.

### 4. Recompute all embeddings

Force a full re-embedding of every object in every bucket. Use this after
upgrading the embedding model:

```bash
umrs-client --admin-token "$UMRS_ADMIN_TOKEN" recompute
```

**Warning**: This overwrites all existing embeddings. For large datasets,
it will take a long time and consume GPU resources.

### 5. Monitor progress

All maintenance operations run server-side as threads. Check the status:

```bash
umrs-client status all
umrs-client status recompute
umrs-client status update
umrs-client status check
```

## Manual Equivalent To `README.md (step-by-step commands)`

```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT/umrs/demos/06_server_maintenance"

CONF="${UMRS_USER_CONF:-$HOME/.config/umrs/users/user1.json}"
umrs-client config use --user "$CONF"

# 0. Verify client → server connectivity.
umrs-client ping

# 1. Show configured buckets.
umrs-client list_buckets || true

# 2. Run a consistency check.
umrs-client --admin-token "$UMRS_ADMIN_TOKEN" check
sleep 2
umrs-client status check || true

# 3. Update only missing or stale embeddings.
umrs-client --admin-token "$UMRS_ADMIN_TOKEN" update
sleep 2
umrs-client status update || true

# 4. Optional full recompute.
umrs-client --admin-token "$UMRS_ADMIN_TOKEN" recompute
sleep 5
umrs-client status recompute || true
```

If your deployment does not enforce admin tokens, omit `--admin-token`.
That relaxation is deployment-specific; the conceptual role of this demo is
still operator/admin maintenance.

## When to Use What

```
Upload finished, not sure if everything processed?
  └─ umrs-client --admin-token "$UMRS_ADMIN_TOKEN" check
       └─ All consistent? → Done
       └─ Mismatches found?
            └─ umrs-client --admin-token "$UMRS_ADMIN_TOKEN" update   (fix only broken entries)

Changed the embedding model in user config?
  └─ umrs-client --admin-token "$UMRS_ADMIN_TOKEN" reload       (pick up new model)
  └─ umrs-client --admin-token "$UMRS_ADMIN_TOKEN" recompute    (re-embed everything)

First time, don't know what's configured?
  └─ umrs-client list_buckets      (discover your buckets)
```

## How It Works

- `list_buckets` — reads the server's BucketsCfg and filters by username
- `check` — iterates S3 objects, compares checksums with pgvector rows
- `update` — runs `check`, then processes only non-matching objects
- `recompute` — processes all S3 objects unconditionally (full overwrite)
- All three server-side operations (`check`, `update`, `recompute`) send
  a ZMQ command and run as background threads on the server

## Run Demo

```bash
./README.md (step-by-step commands)
```
