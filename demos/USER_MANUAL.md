# UMRS Demo User Manual

This manual is the user-facing path through the UMRS demo suite. It explains how
to prepare a client, add demo experiences, reload provisioning, recover missing
JSON config from MinIO/Postgres state, and clean up safely.

For the compact demo index, see [README.md](README.md).

## 1. Mental Model

UMRS demos use three coordinated resources:

| Layer | User-visible handle | Backing resource |
|-------|---------------------|------------------|
| User JSON | `~/.config/umrs/users/<user>.json` | Declares experiences, modalities, bucket aliases, SQL tables, and model settings |
| MinIO/S3 | bucket alias, for example `jens-sam3-direct` | Physical buckets named from `<user>.<xp_name>.<bucket_alias>` |
| Postgres/pgvector | `sql_username` + `xp_name` | Database named from the experience and SQL user, with one table per modality |

`umrs-client reload` reads the active user JSON on the server and provisions any
missing resources. Reload is idempotent: existing buckets and tables are reused.

## 2. First-Time Setup

Start with the onboarding demo:

```bash
cd umrs/demos/00_onboarding
# Follow the step-by-step commands in README.md
```

For manual setup:

```bash
umrs-client profile show
umrs-client ping
umrs-client config show
```

If the deployment uses SSH tunneling, start the site tunnel before running demo
commands.

## 3. Add A Demo Experience

Workload demos usually ship an `experience.json` file. Add it to the active user
config, then reload:

```bash
cd umrs/demos/03_sam3_segmentation
umrs-client config add-experience experience.json --update
umrs-client reload
umrs-client list_buckets
```

What reload creates:

| Resource | Behavior |
|----------|----------|
| MinIO IAM user | Created once per UMRS username |
| MinIO buckets | Created for each declared modality and optional output bucket |
| Postgres role | Created for each `sql_username` |
| Postgres database | Created for each experience and enabled with pgvector |
| Embedding tables | Created or validated against the declared model/vector shape |

## 4. Run A Demo

Each demo can be run directly:

```bash
cd umrs/demos/02_text_embedding
# Follow the step-by-step commands in README.md
```

For manual file workflows, the common shape is:

```bash
umrs-client upload_dir ./input <bucket-alias>/<run-id>
umrs-client status
umrs-client download_results <bucket-alias>/<run-id> ./results --force-overwrite
```

Use [README.md](README.md) to choose a numbered demo by task.

## 5. Remove An Experience Safely

Use `experience remove` when you want the user JSON to stop declaring an
experience:

```bash
umrs-client experience remove <xp_name>
umrs-client reload
```

Important behavior:

| Resource | After remove + reload |
|----------|-----------------------|
| User JSON | Experience stanza is removed |
| In-memory UMRS registry | Experience buckets disappear from active routing |
| MinIO buckets | Retained |
| Postgres database/tables | Retained |
| pgvector embeddings | Retained |

This is intentional. It prevents accidental data loss and lets you re-add the
same experience later without re-uploading or re-indexing.

## 6. Recover Experience JSON From Existing Data

If an experience was removed from the user JSON but its MinIO/Postgres data still
exists, recover a JSON fragment:

```bash
umrs-client experience recover <xp_name> \
  --username <user> \
  --s3cfg ~/.config/umrs/s3cfg_<user> \
  --output-path recovered-experience.json
```

Then review the generated file. MinIO bucket names can prove the username,
experience name, bucket alias, and output-bucket presence. Postgres inspection can
infer database, SQL user, table names, metadata columns, row counts, and vector
size. Model identity cannot always be proven from storage alone, so the recovery
command uses conservative defaults unless you pass overrides:

```bash
umrs-client experience recover <xp_name> \
  --username <user> \
  --s3cfg ~/.config/umrs/s3cfg_<user> \
  --model-name intfloat/e5-base-v2 \
  --model-type TransformerEncoder \
  --vector-size 768 \
  --output-path recovered-experience.json
```

To merge directly into the active user config:

```bash
umrs-client experience recover <xp_name> \
  --username <user> \
  --s3cfg ~/.config/umrs/s3cfg_<user> \
  --merge --update

umrs-client reload
```

Use `--no-postgres` when the client cannot reach Postgres. The generated JSON is
still useful, but review `model`, `vector_size`, and `sql.table_name` before
reload.

## 7. Endpoint Control For Reusable GPU Services

For demos that benefit from a warm endpoint, use the endpoint lifecycle commands:

```bash
umrs-client endpoint up --model <model-name> --idle-ttl 30m
umrs-client endpoint list
umrs-client endpoint status <job_id>
umrs-client endpoint stop <job_id>
```

The default idle TTL is `30m`. It can be overridden with `--idle-ttl` or by the
selected experience JSON. Active use refreshes the idle expiry; idle endpoints
are cleaned up by UMRS.

## 8. Troubleshooting

| Symptom | Check |
|---------|-------|
| `No bucket` on upload | Run `umrs-client list_buckets`; if missing, add the experience and reload |
| Experience disappeared after reload | It was removed from user JSON; recover it with `experience recover` if data remains |
| Recovery finds buckets but not tables | Check Postgres env vars or rerun without `--no-postgres` from a host that can reach DB |
| Generated recovery JSON has defaults | Storage cannot prove model identity; edit `model_name`, `model_type`, and `vector_size` before reload |
| Demo waits for results | Job may still be running; inspect `umrs-client status` or `umrs-client monitor --tab endpoints` |

## 9. Cleanup

Use JSON removal for reversible cleanup:

```bash
umrs-client experience remove <xp_name>
umrs-client reload
```

Use explicit purge only when the data should be destroyed. Full purge is an
operator/admin action and removes the user's buckets, IAM user, Postgres roles,
databases, and local generated S3 config.
