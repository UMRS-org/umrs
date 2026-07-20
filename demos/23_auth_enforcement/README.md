# Demo 23: Auth Enforcement

Demonstrates UMRS role-based access control and admin token management.

## What You'll Learn

- How to generate, list, and revoke admin tokens via `umrs-server admin token`
- How the server enforces authorization: user vs admin role
- How maintenance commands (`check`, `recompute`, etc.) require admin privileges
- How the `--admin-token` flag grants admin access

## Quick Start

```bash
# Local-only (token management only, no server daemon needed):
DEMO_LOCAL_ONLY=1
# See the numbered steps below in this README

# Full test (requires running UMRS server):
# See the numbered steps below in this README

# Remote admin CLI over SSH (optional):
UMRS_ADMIN_SSH_HOST=rccs-cloud \
UMRS_ADMIN_REMOTE_CWD=/lvs0/rccs-dmp/masaru.nagaso/ai4s-platform/umrs \
UMRS_ADMIN_REMOTE_ACTIVATE=/lvs0/rccs-dmp/masaru.nagaso/ai4s-platform/.venv/bin/activate \
# See the numbered steps below in this README
```

## Prerequisites

1. `umrs-client` installed and in PATH
2. `umrs-server` available where the server-side token store lives, or SSH
   access to a host where it is available
3. For server tests: UMRS server running and reachable
4. The admin token must be created in the server's active UMRS config root.
   On native site deployments started via `deploy/<site>/start.sh`, this is
   often the site directory because `UMRS_CONFIG` is exported there.

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

Run the `umrs-server admin ...` commands below on the server host, or in a
shell that shares the UMRS daemon's active config root.

```bash
# 1. Generate an admin token in the server-side token store
TOKEN_NAME="demo23-$(date -u +%Y%m%d%H%M%S)"
TOKEN_JSON=$(umrs-server admin token generate "$TOKEN_NAME" --description "Demo 23 test token" --json)
TOKEN=$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["token"])' "$TOKEN_JSON")

# 2. Verify the token exists in the same server-side token store
umrs-server admin token list --json

# 3. Ping the server
umrs-client ping

# 4. Attempt an admin-only maintenance action without a token.
#    On rccs-cloud this is accepted by the client handoff but the server does
#    not materialize a check status.
umrs-client check || true
umrs-client --output json status check

# 5. Retry with the admin token; this should produce a real check summary.
umrs-client --admin-token "$TOKEN" check
umrs-client --admin-token "$TOKEN" --output json status check

# 6. Inspect status all with admin privileges
umrs-client --admin-token "$TOKEN" --output json status all

# 7. Revoke the token from the same server-side token store
umrs-server admin token revoke "$TOKEN_NAME" -y

# 8. Verify the token is gone
umrs-server admin token list --json
```

## Key Concepts

### Admin Tokens

Admin tokens are managed on the server with `umrs-server admin token ...`.
Hashes are stored in `~/.config/umrs/admin_tokens.json` in the server's UMRS
config root. Only SHA-256 hashes are stored; the raw token is shown once at
generation time.

```bash
# Generate
umrs-server admin token generate my-token --description "CI pipeline"

# List
umrs-server admin token list

# Revoke
umrs-server admin token revoke my-token
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
