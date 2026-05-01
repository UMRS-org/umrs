# UMRS Access Control

This document describes the current access-control boundary in UMRS. In the
full R4 platform, the Gateway is the external authentication and policy
boundary. UMRS receives either Gateway-verified headers or MVP/testing
parameters, then enforces ownership and project scoping in the relevant API
handlers.

## Current Identity Sources

| Surface | Current source | Strict mode behavior |
|---|---|---|
| Endpoint and file APIs | `X-Verified-Project-ID` from Gateway, or `X-Project-ID` in MVP mode | `UMRS_AUTH_MODE=strict` requires the Gateway-verified project header |
| Job APIs | `X-User-ID` + `X-Verified-Project-ID`, or query params in MVP mode | `UMRS_AUTH_MODE=strict` requires Gateway-verified caller identity |
| Model registry APIs | `user_id`, `is_admin`, and `project_memberships` query params | Still an MVP/test surface; Gateway should wrap external access |
| ZMQ/control commands | `AuthContext` from UMRS shared auth helpers | Command-level role checks apply |

Authoritative implementation files:

- `src/umrs/shared/auth.py` for roles, admin/service/user checks, admin tokens,
  and XDG config helpers.
- `src/umrs/shared/isolation.py` for path and project isolation helpers.
- `src/umrs/api/endpoints.py` and `src/umrs/api/files.py` for project header
  extraction and strict-mode fallback behavior.
- `src/umrs/api/jobs.py` for `CallerIdentity` and durable-job ownership checks.
- `src/umrs/api/models.py` for model registry access contexts and model policy
  enforcement.

## Gateway Boundary

External clients should call the Platform Gateway, not UMRS directly. Gateway
adds user authentication, project membership validation, rate limiting, audit
logging, and policy checks before forwarding to UMRS.

UMRS still exposes direct HTTP/ZMQ surfaces because they are needed for local
development, harness validation, and trusted operator workflows. In shared
deployments, those surfaces should be bound to trusted networks or reached
through SSH tunnels and Gateway routes.

## Resource Ownership

### Jobs

Every durable job carries `project_id` and `user_id`. Non-admin callers can
read, list, cancel, or retry only jobs they own, unless the endpoint explicitly
allows an operator/admin action.

Current job checks live in `src/umrs/api/jobs.py`:

```python
CallerIdentity.can_access_job(job)
_enforce_job_access(caller, job, action)
```

### Models

The model registry supports system, project, and user scopes. Access checks are
handled by the model registry store and API layer. Model staging and downloads
are submitted as durable `MODEL_DOWNLOAD` jobs; inference containers do not
download models at runtime.

Key files:

- `src/umrs/api/models.py`
- `src/umrs/model_registry/`
- `src/umrs/jobs/model_download.py`
- `src/umrs/jobs/model_download_state.py`

### Files and Presigned URLs

UMRS owns S3/MinIO object access. File APIs validate project-scoped keys before
issuing presigned URLs. External clients should upload through Gateway HTTP;
presigned URLs are for controlled internal/runtime access, especially compute
nodes that need S3 access without holding MinIO credentials.

Key files:

- `src/umrs/api/files.py`
- `src/umrs/shared/storage/`

## Compute Isolation

UMRS orchestrates Mini-RiVault jobs through the selected backend
(`slurm-podman`, Kubernetes, or Compose). The execution authority is selected
once for each job and must not change during the job lifecycle.

Current safeguards:

- Mini-RiVault inference jobs set `TRANSFORMERS_OFFLINE=1` and
  `HF_HUB_OFFLINE=1`.
- Egress proxy environment is injected into Mini-RiVault launches.
- Direct Slurm network blocking is a site infrastructure requirement; see the
  parent repo's `docs/EGRESS_INFRASTRUCTURE.md`.
- RCCS Slurm/Podman launch defaults are captured in `deploy/rccs-cloud/site.yaml`
  and `src/umrs/orchestrator/site_profiles/rccs-cloud.yaml`.

## Tests

Focused coverage lives in:

- `tests/test_auth.py`
- `tests/test_strict_auth_mode.py`
- `tests/test_jobs.py`
- `tests/test_egress_audit_api.py`
- `tests/test_mini_rivault_egress.py`
- `tests/test_slurm_orchestration_authority.py`
