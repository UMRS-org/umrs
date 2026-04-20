# Demo 17: Multi-User Isolation

Demonstrates that UMRS enforces **per-user data isolation**. Two users
(user1 and user2) each have their own S3 buckets and model registries —
neither can access the other's data.

## Prerequisites

- UMRS server running and reachable (SSH tunnels if remote: ports 5555, 5556-5563, 9000)
- `umrs-client` installed and in PATH
- Two user configuration files:
  - `~/.config/umrs/users/user1.json`
  - `~/.config/umrs/users/user2.json`

## Quick Start

```bash
cd demos/17_multi_user
bash run_demo.sh
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
#    Each user authenticates with their own config file.
#    The -c flag selects which user identity to use.
umrs-client -c ~/.config/umrs/users/user1.json ping
umrs-client -c ~/.config/umrs/users/user2.json ping

# 2. User1 uploads a test file to their private bucket
#    This file is stored in user1's namespace and should not
#    be visible to user2.
echo "This is user1's private data" > /tmp/user1_data.txt
umrs-client -c ~/.config/umrs/users/user1.json upload_file /tmp/user1_data.txt user1-textfiles

# 3. User2 uploads a test file to their private bucket
echo "This is user2's private data" > /tmp/user2_data.txt
umrs-client -c ~/.config/umrs/users/user2.json upload_file /tmp/user2_data.txt user2-textfiles

# 4. User1 lists their own bucket (should succeed)
umrs-client -c ~/.config/umrs/users/user1.json list_buckets

# 5. User1 tries to access user2's bucket (should be denied)
#    Expected: error/denied/forbidden response confirming isolation.
umrs-client -c ~/.config/umrs/users/user1.json list_buckets
# (user2's bucket alias will not appear in user1's bucket list)

# 6. User2 tries to access user1's bucket (should be denied)
umrs-client -c ~/.config/umrs/users/user2.json list_buckets
# (user1's bucket alias will not appear in user2's bucket list)

# 7. Check user-scoped model isolation
#    User-scoped models are private — each user sees only their own.
umrs-client -c ~/.config/umrs/users/user1.json model list --scope user
umrs-client -c ~/.config/umrs/users/user2.json model list --scope user

# 8. Check system-scoped models (shared across all users)
#    System models are visible to everyone.
umrs-client -c ~/.config/umrs/users/user1.json model list --scope system
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
