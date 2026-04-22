# Demo 00: User Onboarding & Initialization

First-run onboarding for a UMRS user. This demo pushes the selected user
config to the server, triggers server-side provisioning, and verifies that
the new buckets are visible afterwards.

See [../README.md](../README.md) for shared setup, connection profiles, and
SSH tunnel guidance.

## Quick Start

```bash
cd "$(git rev-parse --show-toplevel)/umrs/demos/00_onboarding"
bash run_demo.sh
```

## Manual Equivalent To `run_demo.sh`

```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT/umrs/demos/00_onboarding"

CONF="${UMRS_USER_CONF:-$HOME/.config/umrs/users/user1.json}"

# 1. Show the active connection profile.
umrs-client --output text profile show || true

# 2. Verify that the user config exists.
test -f "$CONF"

# 3. Verify client → server connectivity.
umrs-client -c "$CONF" ping

# 4. Trigger onboarding (idempotent).
umrs-client -c "$CONF" reload

# 5. Check the reload result.
umrs-client -c "$CONF" status reload

# 6. Confirm that bucket aliases are visible.
umrs-client -c "$CONF" list_buckets
```

## What `reload` Provisions

- MinIO IAM user and policy for the selected UMRS user
- Postgres role and per-user database/schema assets
- Bucket aliases defined in the user config JSON
- Server-side routing metadata used by later uploads and searches

Re-running the same sequence is safe. If the config has not changed, the
server treats the reload as a no-op.

## Expected Result

After onboarding, `list_buckets` should show the aliases configured in the
user JSON. At that point you can continue with Demo 01 or Demo 02.
