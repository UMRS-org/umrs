# Demo 23: Auth Enforcement

Demonstrates UMRS role-based access control and admin token management.

## What You'll Learn

- How to generate, list, and revoke admin tokens via `umrs-client admin token`
- How the server enforces authorization: user vs admin role
- How maintenance commands (`check`, `recompute`, etc.) require admin privileges
- How the `--admin-token` flag grants admin access

## Quick Start

```bash
# Local-only (token management, no server needed):
DEMO_LOCAL_ONLY=1 bash run_demo.sh

# Full test (requires running UMRS server):
bash run_demo.sh
```

## Prerequisites

1. `umrs-client` installed and in PATH
2. For server tests: UMRS server running and reachable
3. The admin token hash must exist in the server's active UMRS config root.
   On native site deployments started via `deploy/<site>/start.sh`, this is
   often `deploy/<site>/admin_tokens.json` because `UMRS_CONFIG` is exported to
   the site directory. Syncing only `~/.config/umrs/admin_tokens.json` on the
   server machine may be insufficient.

## Steps

| Step | Description | Requires Server |
|------|-------------|-----------------|
| 1 | Generate an admin token | No |
| 2 | List tokens (verify it exists) | No |
| 3 | (local-only: revoke and exit) | No |
| 4 | Ping server | Yes |
| 5 | `check` without token → denied | Yes |
| 6 | `check` with token → allowed | Yes |
| 7 | Pool metrics in status | Yes |
| 8 | Revoke the token | No |
| 9 | Verify token is gone | No |

## Running Manually

```bash
# 1. Generate a local admin token
TOKEN_NAME="demo23-$(date -u +%Y%m%d%H%M%S)"
TOKEN_JSON=$(umrs-client --output json admin token generate "$TOKEN_NAME" --description "Demo 23 test token")
TOKEN=$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["token"])' "$TOKEN_JSON")

# 2. Verify the token exists locally
umrs-client --output json admin token list

# 3. For native site deployments, merge the token hash into the server's
#    active UMRS config root before testing admin access.
#    On rccs-cloud this is deploy/rccs-cloud/admin_tokens.json.
scp ~/.config/umrs/admin_tokens.json rccs-cloud:/tmp/demo23-admin_tokens.json
ssh rccs-cloud "python3 - <<'PY'
import json
from pathlib import Path

name = '${TOKEN_NAME}'
remote = Path('/lvs0/dne1/rccs-dmp/mnagaso/umrs/deploy/rccs-cloud/admin_tokens.json')
uploaded = Path('/tmp/demo23-admin_tokens.json')
remote_tokens = json.loads(remote.read_text()) if remote.exists() else {}
uploaded_tokens = json.loads(uploaded.read_text())
remote_tokens[name] = uploaded_tokens[name]
remote.write_text(json.dumps(remote_tokens, indent=2))
uploaded.unlink(missing_ok=True)
PY"

# 4. Ping the server
umrs-client ping

# 5. Attempt an admin-only maintenance action without a token.
#    On rccs-cloud this is accepted by the client handoff but the server does
#    not materialize a check status.
umrs-client check || true
umrs-client --output json status check

# 6. Retry with the admin token; this should produce a real check summary.
umrs-client --admin-token "$TOKEN" check
umrs-client --admin-token "$TOKEN" --output json status check

# 7. Inspect status all with admin privileges
umrs-client --admin-token "$TOKEN" --output json status all

# 8. Revoke the token locally
umrs-client admin token revoke "$TOKEN_NAME"

# 9. Remove the temporary remote hash entry
ssh rccs-cloud "python3 - <<'PY'
import json
from pathlib import Path

name = '${TOKEN_NAME}'
remote = Path('/lvs0/dne1/rccs-dmp/mnagaso/umrs/deploy/rccs-cloud/admin_tokens.json')
remote_tokens = json.loads(remote.read_text()) if remote.exists() else {}
remote_tokens.pop(name, None)
remote.write_text(json.dumps(remote_tokens, indent=2))
PY"
```

## Key Concepts

### Admin Tokens

Client-side token generation stores hashes locally in
`~/.config/umrs/admin_tokens.json` by default. The server validates against the
token store visible in its own UMRS config root. Only SHA-256 hashes are
stored; the raw token is shown once at generation time.

```bash
# Generate
umrs-client admin token generate my-token --description "CI pipeline"

# List
umrs-client --output text admin token list

# Revoke
umrs-client admin token revoke my-token
```

### Authorization Flow

1. Client connects with `--admin-token <TOKEN>` (or `UMRS_ADMIN_TOKEN` env var)
2. Token is sent during `CREATE_DATA_QUEUE` handshake
3. Server validates the hash against `admin_tokens.json`
4. If valid → role=admin; otherwise → role=user
5. Admin-only commands (maintenance) are rejected for user role

### User-Allowed Commands

Regular users can execute: `STATUS`, `FILE`, `MODEL_LIST`, `MODEL_REGISTER`,
`JOB_SUBMIT`, `JOB_CANCEL`, and others. Maintenance commands (`CHECK`,
`RECOMPUTE`, `UPDATE`, `RELOAD`) require admin role.
