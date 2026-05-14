# Demo 99: Full Demo Cleanup

Tears down all resources created during a demo session.

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
bash run_demo.sh

# Clean up a different user
DEMO_USERNAME=other_user bash run_demo.sh
```

## Prerequisites

- `umrs-server` admin CLI must be accessible (run on the server node)
- For remote cleanup, SSH into the server first

## After cleanup

The system is back to a clean state. Run `00_onboarding` to start fresh.
