# Demo 17: Multi-User Isolation

Demonstrates that UMRS enforces **per-user data isolation**. Two users
(user1 and user2) each have their own S3 buckets and model registries —
neither can access the other's data.

## Prerequisites

- UMRS server running and reachable (SSH tunnels if remote: ports 5555, 5556-5563, 9000)
- `umrs-client` installed and in PATH (see [demos/README.md](../README.md#prerequisites))
- Two users onboarded on the server (see [Demo 00](../00_onboarding/) for the admin enrollment flow via `umrs-server admin user add`)
- Two user configuration files:
  - `~/.config/umrs/users/user1.json`
  - `~/.config/umrs/users/user2.json`

## Quick Start

```bash
cd demos/17_multi_user
# See the numbered steps below in this README
```

## What This Demo Does

1. **Pre-flight** — checks that both user configs exist and both users can
   reach the server
2. **User1 uploads** — uploads a test file to user1's bucket
3. **User2 uploads** — uploads a test file to user2's bucket
4. **Verify bucket isolation** — user1 tries to list user2's bucket (should
   be denied), and vice versa
5. **Verify model isolation** — checks that user-scoped models are private
   while system-scoped models are shared

## Running Manually

```bash
# 1. Verify both users can reach the server
#    Use `config use` to switch the active user identity.
umrs-client config use --user ~/.config/umrs/users/user1.json
umrs-client ping
umrs-client config use --user ~/.config/umrs/users/user2.json
umrs-client ping

# 2. User1 uploads a test file to their private bucket
#    This file is stored in user1's namespace and should not
#    be visible to user2.
umrs-client config use --user ~/.config/umrs/users/user1.json
echo "This is user1's private data" > /tmp/user1_data.txt
umrs-client data upload /tmp/user1_data.txt user1-textfiles

# 3. User2 uploads a test file to their private bucket
umrs-client config use --user ~/.config/umrs/users/user2.json
echo "This is user2's private data" > /tmp/user2_data.txt
umrs-client data upload /tmp/user2_data.txt user2-textfiles

# 4. User1 lists their own bucket (should succeed)
umrs-client config use --user ~/.config/umrs/users/user1.json
umrs-client bucket list

# 5. User2 lists their own bucket (user1's data is not visible)
umrs-client config use --user ~/.config/umrs/users/user2.json
umrs-client bucket list

# 6. Check user-scoped model isolation
#    User-scoped models are private — each user sees only their own.
umrs-client config use --user ~/.config/umrs/users/user1.json
umrs-client model list --scope user
umrs-client config use --user ~/.config/umrs/users/user2.json
umrs-client model list --scope user

# 7. Check system-scoped models (shared across all users)
#    System models are visible to everyone.
umrs-client model list --scope system
```

## Isolation Model

| Resource | Scope | Visibility |
|----------|-------|------------|
| S3 buckets | Per-user | Owner only |
| Embedding DB tables | Per-user | Owner only |
| User-scoped models | Per-user | Owner only |
| System-scoped models | Global | All users |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `USER1_BUCKET` | `user1-textfiles` | Bucket alias for user1 |
| `USER2_BUCKET` | `user2-textfiles` | Bucket alias for user2 |

## Expected Output

```
[multi_user]   ✓ Both users can reach server
[multi_user]   ✓ User1 uploaded data to user1-textfiles
[multi_user]   ✓ User2 uploaded data to user2-textfiles
[multi_user]   ✓ User1 CANNOT access user2's bucket (expected)
[multi_user]   ✓ User2 CANNOT access user1's bucket (expected)
[multi_user]   ✓ Model isolation verified

Summary:
  ✓ User1 and User2 have separate buckets
  ✓ Cross-user bucket access is denied
  ✓ User-scoped models are isolated
  ✓ System-scoped models are shared
```

## See Also

- [Multi-User Guide](../../doc/multi_users.md) — administration documentation
- [Access Control](../../doc/access_control.md) — RBAC and namespace isolation
- [Demo 01: Health Check](../01_health_check/README.md) — single-user connectivity test
