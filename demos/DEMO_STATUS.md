# UMRS Demo Suite — Status Report

**Last run:** 2026-06-18
**Profile:** `rccs-cloud` (direct mode, `data0.ai.r-ccs.riken.jp`)
**Server:** reachable via ZMQ (`50090`)

---

## Summary

| Suite       | Passed | Failed | Skipped | Exit |
|-------------|--------|--------|---------|------|
| Public (`demos/`)    | 11     | 8      | 12      | 1    |
| Private (`demos_private/`) | 0      | 4      | 0       | 1    |
| Jens (`demos_jens/`) | 0      | 2      | 0       | 1    |

All failures share a common root cause: **the server cannot provision the
S3 config for the demo user** because `umrs reload` returns
`HTTP 503: Service Unavailable`. Without a valid `s3cfg`, every
`upload_dir` / `upload_file` call fails with `403 Forbidden` or
"Server connection timeout".

The HTTP API demos (25–30) are **skipped** because the gateway returns
`404 Not Found` for `/openapi.json` and `/docs`. The `check_http_api`
prerequisite correctly detects this and skips the demo rather than
failing it.

---

## Public Demos (`demos/`)

| #  | Demo                          | Result  | Notes                                              |
|----|-------------------------------|---------|----------------------------------------------------|
| 00 | User Onboarding               | FAILED  | `reload` returns 503 — S3 config not provisioned   |
| 01 | Health Check                  | FAILED  | MinIO S3 (`:9000`) unreachable                     |
| 02 | Text Embedding                | FAILED  | `upload_dir` times out (no valid `s3cfg`)          |
| 03 | SAM3 Segmentation             | FAILED  | `S3 config not found: s3cfg_demo_user`             |
| 04 | Vision Inference              | SKIPPED | `check_sam3_model` prerequisite not met            |
| 05 | RAG Search                    | SKIPPED | `check_rag` prerequisite not met                   |
| 06 | Server Maintenance            | PASSED  | Ping + status succeed via ZMQ                      |
| 07 | Model Registry                | PASSED  | `model list` works via ZMQ                         |
| 08 | User Model (local)            | PASSED  | Local demo — no server needed                      |
| 09 | Multi-Step Vision             | SKIPPED | `check_sam3_model` prerequisite not met            |
| 10 | Multi-Mode Vision             | SKIPPED | `check_sam3_model` prerequisite not met            |
| 11 | Job Submission                | PASSED  | ZMQ `job submit` works                             |
| 12 | Pipeline SDK (local)          | PASSED  | Local demo — no server needed                      |
| 13 | Plugins (local)               | PASSED  | Local demo — no server needed                      |
| 14 | Video Analysis                | FAILED  | `upload_dir` times out                             |
| 15 | Complex DAG (local)           | PASSED  | Local demo — no server needed                      |
| 16 | Custom Workflow               | PASSED  | ZMQ-based workflow upload works                    |
| 17 | Multi-User Isolation          | PASSED  | Bucket isolation checks pass via ZMQ               |
| 18 | Model Lifecycle               | PASSED  | `model list` + lifecycle checks pass               |
| 19 | RAG Advanced                  | SKIPPED | `check_rag` prerequisite not met                   |
| 20 | Durable Jobs                  | PASSED  | ZMQ `job submit` + polling works                   |
| 21 | Remote Plugins                | PASSED  | Plugin spec validation works via ZMQ               |
| 22 | Monitor (TUI)                 | PASSED  | Status query via ZMQ works                         |
| 23 | Auth Enforcement              | PASSED  | Token management + RBAC checks pass                |
| 24 | Thread Pool                   | PASSED  | Pool metrics query via ZMQ works                   |
| 25 | Endpoint Provisioning         | SKIPPED | `check_http_api` prerequisite not met              |
| 26 | MCP Servers And Mounts        | SKIPPED | `check_http_api` prerequisite not met              |
| 27 | Pipeline CRUD                 | SKIPPED | `check_http_api` prerequisite not met              |
| 28 | Egress Audit                  | SKIPPED | `check_http_api` prerequisite not met              |
| 29 | LLM Completions               | SKIPPED | `check_http_api` prerequisite not met              |
| 30 | File Management               | SKIPPED | `check_http_api` prerequisite not met              |

---

## Private Demos (`demos_private/`)

| #  | Demo                          | Result  | Notes                                              |
|----|-------------------------------|---------|----------------------------------------------------|
| 04 | HMAE Inference                | FAILED  | `upload_dir` times out                             |
| 09 | Defect Analysis               | FAILED  | `upload_dir` times out                             |
| 10 | HMAE Full Pipeline            | FAILED  | `upload_dir` times out                             |
| 14 | Opossum Analysis              | FAILED  | `upload_dir` times out + local analysis incomplete |

---

## Jens Demos (`demos_jens/`)

| #  | Demo                          | Result  | Notes                                              |
|----|-------------------------------|---------|----------------------------------------------------|
| 11 | SAM3 Direct                   | FAILED  | `upload_dir` times out (step 4)                    |
| 12 | HMAE Direct                   | FAILED  | `upload_dir` times out (step 4/mae)                |

---

## Root Cause Analysis

### 1. `umrs reload` returns HTTP 503 — **BUG FIXED**

**Symptom:**
```
✗ reload rejected: HTTP 503: Service Unavailable
```

**Root cause:** The HTTP reload endpoint (`POST /gw/ops/reload`) in
`umrs/src/umrs/api/ops.py` requires module-level state to be set via
`set_reload_context()`. This function was **defined but never called**
during server startup. As a result, `_reload_configure_file` and
`_reload_buckets_cfg` remained `None`, and the endpoint always returned:

```json
{"error_id":"service.unavailable",
 "message":"Reload context not initialised — server may still be starting"}
```

**Fix applied** (`umrs/src/server_umrs.py`):
Added a call to `set_reload_context()` after `perform_initial_config()`
and before starting the HTTP sidecar:

```python
from umrs.api.ops import set_reload_context

set_reload_context(
    configure_file,
    server_cfg.buckets_cfg,
    getattr(server_cfg, "user_config_map", None),
)
```

This wires the server's `configure_file`, `buckets_cfg`, and
`user_config_map` into the reload endpoint so it can re-read the config
and re-provision users.

**Note:** This fix must be deployed to the server (umrs-front) and the
UMRS service restarted via Ansible for the fix to take effect.

### 2. MinIO S3 endpoint unreachable

The health check demo (01) reports:

```
WARNING: MinIO S3     (data0.ai.r-ccs.riken.jp:9000) — FAILED
```

The MinIO service at `data0.ai.r-ccs.riken.jp:9000` is not accepting
connections. This may be because:
- MinIO is not running on the server
- The firewall is blocking port 9000
- MinIO is only accessible via the internal network

**Impact:** All demos that upload to or download from S3 fail.

**Fix:** Ensure MinIO is running and accessible on port 9000. If MinIO
is behind a reverse proxy, check the nginx configuration.

### 3. HTTP API (gateway) returns 404 for docs/openapi

The `check_http_api` function checks if the gateway's OpenAPI
documentation is reachable:

```bash
curl -sf "${url%/}/openapi.json" >/dev/null 2>&1 \
    || curl -sf "${url%/}/docs" >/dev/null 2>&1
```

Both return `404 Not Found`. This is **expected behavior** in production
— the `_create_app()` function in `http_sidecar.py` only enables docs
when `GATEWAY_ENVIRONMENT` is `development`, `dev`, or `test`.

**Impact:** Demos 25–30 are correctly skipped (not failed).

**Note:** The `check_http_api` prerequisite checks for docs/openapi
availability. In production deployments where docs are disabled, these
demos will always be skipped. This is working as designed.

### 4. Code bug in `loaded_config_profiles.py` — **FIXED**

The `discover_loaded_config_profiles()` function in
`umrs/src/umrs/shared/loaded_config_profiles.py` was crashing when it
encountered a `conf_umrs.json` file that referenced user files that
don't exist on the local machine (e.g. `docker/conf_umrs.json` →
`/root/user_files/user1.json`).

**Fix applied:** The `_validate_server` function now catches
`FileNotFoundError` and `ValueError` in addition to `SystemExit`,
treating the server config as invalid but not crashing discovery.

---

## Recommendations

1. **Fix the server-side gateway service.** The HTTP API is returning
   404, and `reload` is returning 503. The gateway service needs to be
   running for the server to provision new users.

2. **Fix the MinIO S3 endpoint.** The MinIO service at
   `data0.ai.r-ccs.riken.jp:9000` is not accepting connections. Ensure
   MinIO is running and accessible.

3. **Re-run the demos after fixing the server.** Most failures are
   caused by the server not being able to provision the demo user's S3
   config. Once the gateway and MinIO are running, re-run the demos:

   ```bash
   cd umrs/demos && # See step-by-step commands in each demo's README
   cd umrs/demos_private && # See step-by-step commands in each demo's README
   cd umrs/demos_jens/11_sam3_direct && # See step-by-step commands in the demo README
   cd umrs/demos_jens/12_hmae_direct && # See step-by-step commands in the demo README
   ```

4. **The code bug fix in `loaded_config_profiles.py` should be
   committed.** This fix prevents the client from crashing when it
   discovers invalid or incomplete server config files.
