# UMRS Access Control

This document describes how user access is limited across the three resource
layers: **Server API**, **S3/MinIO storage**, and **Slurm compute**.

---

## 1. Server API — Authentication & Authorization

### Authentication (identity)

| Mechanism | Location | Status |
|-----------|----------|--------|
| Admin token (SHA-256 hashed) | `src/umrs/auth.py` → `AdminTokenManager` | Implemented |
| Per-user config identity | `src/umrs/admin/create_users.py`, JSON config files | Implemented |
| JWT Bearer from Gateway | `src/umrs/api/endpoints.py` → `get_project_id()` | Stub (MVP uses `X-Project-ID` header) |

Admin tokens are generated as `umrs-admin-{random}`, stored as SHA-256 hashes
in `~/.config/umrs/admin_tokens.json` (file mode 0600).

### Authorization (permissions)

**Role-Based Access Control** (`src/umrs/auth.py`):

- `ADMIN` — unrestricted.
- `USER` — limited to own resources.
- `SERVICE` — inter-service calls, unrestricted.

Command-level gates:

| Gate | Commands |
|------|----------|
| `ADMIN_ONLY_COMMANDS` | `RELOAD`, `RECOMPUTE`, `UPDATE`, `CHECK` |
| `USER_ALLOWED_COMMANDS` | `PROCESS_OBJECTS`, `WAIT_OUTPUT`, `FILE`, `STATUS`, … |

Key enforcement functions:

```
AuthContext.can_execute_command(command, target_user)
AuthContext.filter_by_ownership(data)
require_admin(auth, command)           → raises AuthorizationError
require_ownership(auth, target_user)   → raises AuthorizationError
```

### User isolation

`src/umrs/isolation.py` provides:

- `UserContext.from_headers()` — extracts user/project from request headers.
- `AccessControl.validate_path()` — rejects `..`, absolute paths, shell
  metacharacters (directory-traversal prevention).
- `ResourceOwnership` — supports private, shared, and project-level sharing.
- `IsolationValidator` — multi-user invariant checks.

### Security defaults

- **Default-deny**: resources are private unless explicitly shared.
- **Username sanitization**: alphanumeric + underscore/hyphen only.
- **Config validation**: `validate_config_identity()` rejects mismatched
  username/user_id claims.

---

## 2. Job API — Ownership Enforcement

Added in this changeset. Every job-mutating or job-reading endpoint now
requires a `CallerIdentity` (extracted via `get_caller_identity()` dependency).

### CallerIdentity

```python
@dataclass
class CallerIdentity:
    user_id: str
    project_id: str
    is_admin: bool = False
```

Passed as query parameters `caller_user_id`, `caller_project_id`, `is_admin`.
In production these will come from JWT claims decoded by the API Gateway.

### Endpoint enforcement matrix

| Endpoint | Auth check | Non-owner behaviour |
|----------|-----------|---------------------|
| `POST /umrs/jobs` | None (caller self-declares in body) | N/A |
| `GET /umrs/jobs` | `CallerIdentity` | Non-admin: forced to own `project_id`, filtered to own `user_id` |
| `GET /umrs/jobs/{job_id}` | `_enforce_job_access(caller, job, "read")` | HTTP 403 |
| `POST /umrs/jobs/{job_id}/cancel` | `_enforce_job_access(caller, job, "cancel")` | HTTP 403 |
| `POST /umrs/jobs/{job_id}/retry` | `caller.is_admin` required | HTTP 403 |

### Helper: `_enforce_job_access`

```python
def _enforce_job_access(caller, job, action="access"):
    """Raise 403 if caller does not own the job and is not admin."""
```

Checks `job.project_id == caller.project_id AND job.user_id == caller.user_id`,
or `caller.is_admin`. Logs denied attempts at WARNING level.

---

## 3. S3/MinIO Storage — Per-User Bucket Isolation

### User creation flow (server startup)

1. Server reads master config listing all users.
2. For each user, `src/umrs/minio.py` → `minio_s3_create_user()`:
   - Creates a MinIO IAM user with a strong random password (24 chars).
   - Writes the user's `s3cfg` file.
3. For each bucket, `minio_grant_user_access()`:
   - Creates an IAM policy `allow-{bucket}` scoped to `arn:aws:s3:::{bucket}/*`.
   - Attaches the policy to the user via `mc admin policy attach`.

### Bucket naming

```
{username}.{xp_name}.{bucket_alias}[-output]
```

Example: `user1.images-and-texts.bucket-3-output`

Implemented in `src/umrs/s3.py` → `make_path_to_bucket_string()`.

### Access model

| Actor | Can access |
|-------|-----------|
| Admin (`minioadmin`) | All buckets |
| User N | Only buckets matching `userN.*` (enforced by IAM policy) |

### Presigned URLs

Clients never get direct MinIO credentials for other users' buckets. The
server validates bucket ownership, then generates a time-limited presigned URL
via the admin client (`src/umrs/s3_presign.py`).

---

## 4. Slurm Compute — Current State & Known Gaps

### What's implemented

- Every `Job` record carries `project_id` + `user_id`
  (`src/umrs/jobs/models.py`).
- Per-job resource limits (`gpu_count`, `cpu_limit`, `memory_limit`,
  `timeout_seconds`) are translated to `#SBATCH` directives
  (`src/umrs/plugins/slurm_runner.py`).
- Model registry has per-model token quotas.

### Known gaps

| Gap | Impact | Suggested fix |
|-----|--------|---------------|
| All Slurm jobs run under a single service UID | No OS-level isolation between users | Map UMRS users to Slurm accounts; use `--account` |
| No per-user job quotas | A single user can saturate the cluster | Add `max_concurrent_jobs` to user config; enforce in `create_job` |
| SSH tunnels are per-model, not per-user | Users on the same model share a tunnel | Accept for shared-model use-case; document |
| Slurm fairshare/QOS not configured | No priority fairness between users | Enable `PriorityType=priority/multifactor` in `slurm.conf` |

---

## Key source files

| File | Purpose |
|------|---------|
| `src/umrs/auth.py` | Roles, AuthContext, admin tokens, ownership checks |
| `src/umrs/isolation.py` | UserContext, AccessControl, path validation |
| `src/umrs/api/jobs.py` | CallerIdentity, _enforce_job_access, endpoint guards |
| `src/umrs/minio.py` | MinIO user/policy creation |
| `src/umrs/s3.py` | Bucket naming, user+bucket setup |
| `src/umrs/s3_presign.py` | Presigned URL generation |
| `src/umrs/jobs/models.py` | Job model (project_id, user_id) |
| `src/umrs/plugins/slurm_runner.py` | Sbatch script generation with resource limits |
| `tests/test_auth.py` | Auth & role enforcement tests |
| `tests/test_isolation.py` | Isolation invariant tests |
| `tests/test_jobs.py` | Job model + ownership enforcement tests |

---

## Testing

```bash
# Unit tests (no database required)
pytest tests/test_auth.py tests/test_isolation.py tests/test_jobs.py -v

# Integration tests (requires PostgreSQL)
pytest tests/ --db-url postgresql://user:pass@host:5432/umrs_test -v
```
