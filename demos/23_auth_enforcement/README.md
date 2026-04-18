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
3. For rccs-cloud: the admin token hash must exist on the server machine
   (generate the token on the server, or sync `~/.config/umrs/admin_tokens.json`)

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

## Key Concepts

### Admin Tokens

Admin tokens are managed locally in `~/.config/umrs/admin_tokens.json`.
Only SHA-256 hashes are stored; the raw token is shown once at generation time.

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
