# Demo 05 — Server Maintenance

Data integrity and maintenance operations — discover, audit, and repair
the consistency between S3 objects and the pgvector embedding database.

## Prerequisites

- UMRS server running (see `demos/00_start_services/`)
- `umrs-client` available
- At least one user configured with data already uploaded (run `demos/01_text_embedding/` first)

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
umrs-client check
```

Output shows per-bucket: matches, mismatches, not-found, errors.

### 3. Update missing/mismatched entries

Re-embed only the objects that are missing or have stale embeddings
(e.g., the file was re-uploaded with different content):

```bash
umrs-client update
```

This is **incremental** — it skips objects whose embeddings already match.

### 4. Recompute all embeddings

Force a full re-embedding of every object in every bucket. Use this after
upgrading the embedding model:

```bash
umrs-client recompute
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

## When to Use What

```
Upload finished, not sure if everything processed?
  └─ umrs-client check
       └─ All consistent? → Done
       └─ Mismatches found?
            └─ umrs-client update   (fix only broken entries)

Changed the embedding model in user config?
  └─ umrs-client reload            (pick up new model)
  └─ umrs-client recompute         (re-embed everything)

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
./run_demo.sh
```
