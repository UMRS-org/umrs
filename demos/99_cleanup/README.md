# Demo 99: Cleanup And Recovery

Use this when you need to either remove an experience from active routing or
tear down all resources created during a demo session.

## Reversible Experience Cleanup

Removing an experience from the user JSON is safe and reversible. It does not
delete MinIO buckets, Postgres databases, tables, or pgvector embeddings.

```bash
umrs experience remove <xp_name>
umrs reload
```

To recover the JSON later from retained storage state:

```bash
umrs experience recover <xp_name> \
   --username <user> \
   --s3cfg ~/.config/umrs/s3cfg_<user> \
   --output-path recovered-experience.json
```

Review recovered model fields before reloading. Storage can prove bucket/table
shape, but may not prove the exact original model name.

## Full Destructive Cleanup

## What it does

1. **Server-side deprovisioning** (requires `umrs-server` CLI):
   - Empties and removes all S3 buckets belonging to `demo_user`
   - Drops all PostgreSQL databases and roles
   - Detaches all MinIO IAM policies and removes the user
   - Removes the server-side user JSON file

2. **Client-side cleanup**:
   - Removes `~/.config/umrs/users/demo_user.json`
   - Removes `~/.s3cfg_demo_user`

## Usage

```bash
# Clean up the default demo_user
# See the numbered steps below in this README

# Clean up a different user
DEMO_USERNAME=other_user
# See the numbered steps below in this README
```

## Prerequisites

- `umrs-server` admin CLI must be accessible (run on the server node)
- For remote cleanup, SSH into the server first

## After cleanup

The system is back to a clean state. Run `00_onboarding` to start fresh.
