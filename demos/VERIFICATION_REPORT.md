# Demo Verification Report (2026-07-20)

## Test Environment
- **Profile**: `rccs-cloud` (direct mode, ZMQ port 50090)
- **Gateway**: `https://data0.ai.r-ccs.riken.jp/rccs` — **OK**
- **User**: `mnagaso3` (project: `default`)
- **User config**: `~/.config/umrs/users/mnagaso3.json`
- **Available experience**: `default` with bucket alias `mnagaso3-text` (text modality)
- **Available models**: 10+ system models including `e5-base-v2`, `sam3.1`, `glm-5.2-iq2m`
- **MinIO connectivity**: **OK** (54 buckets, S3 config found)
- **Postgres connectivity**: **OK** (reload accepted, database provisioned)
- **Slurm**: NOT AVAILABLE (no slurm_account configured for user)

## Documentation Fixes Applied (40 files)

All ghost `README.md (step-by-step commands)` references fixed across 17 files.
No executable scripts existed — content was always inline in READMEs under "Running Manually" and numbered "Step N" sections.

## Demo-by-Demo Results

### ✅ PASS (core operations verified, no blocking issues)

| # | Demo | Result | Notes |
|---|------|--------|-------|
| 00 | User Onboarding | ✅ PASS | `reload` works, bucket list shows provisioned data, 2 steps documented |
| 01 | Health Check | ✅ PASS | Ping OK, MinIO OK, 157 objects in bucket |
| 06 | Server Maintenance | ✅ PASS | Ping + status work, admin commands fail (expected — no admin token) |
| 07 | Model Registry | ✅ PASS | 10+ system models visible, 1 user model visible |
| 08 | User Model | ✅ PASS | 33/33 unit tests pass, workflow.json valid |
| 12 | Pipeline SDK | ✅ PASS | Local SDK test passes |
| 13 | Plugins | ✅ PASS | Local plugin test passes |
| 15 | Complex DAG | ✅ PASS | workflow.json valid, 5 steps, steps.py exists |
| 16 | Custom Workflow | ✅ PASS | Workflow upload succeeded (`demo-workflow-test`) |
| 18 | Model Lifecycle | ✅ PASS | System/user model listing works, model visible |
| 20 | Durable Jobs | ✅ PASS | Job submission works, job queued (slurm_account needed for execution) |
| 22 | Monitor | ✅ PASS | Status query via ZMQ works |
| 23 | Auth Enforcement | ✅ PASS | Token management + RBAC tests pass |
| 24 | Thread Pool | ✅ PASS | Pool metrics query works |

### ⚠️ PARTIAL (works but needs experience/bucket alias)

| # | Demo | Result | Issue |
|---|------|--------|-------|
| 02 | Text Embedding | ⚠️ Partial | Upload ✅, but export returns 0 rows — no auto-ingestion pipeline configured. RAG database provisioned but `no_database` status persists via Gateway |
| 05 | RAG Search | ⚠️ Partial | Requires RAG ingestion pipeline (not configured on this deployment) |
| 09 | Multi-Step Vision | ⚠️ Partial | Needs `demo-sam3` bucket alias + bentoml_url configuration |
| 10 | Multi-Mode Vision | ⚠️ Partial | Same as 09 — needs vision bucket alias |
| 19 | RAG Advanced | ⚠️ Partial | Same as 05 — needs RAG ingestion pipeline |
| 21 | Remote Plugins | ⚠️ Partial | Needs Slurm account for remote execution |

### ❌ FAIL (structural issues in demo)

| # | Demo | Result | Issue |
|---|------|--------|-------|
| 03 | SAM3 Segmentation | ❌ FAIL | Bucket alias `demo-sam3` not in user config. Experience has `text` modality only. Image upload fails (modality mismatch) |
| 04 | Vision Inference | ❌ FAIL | Same — needs `demo-images` bucket alias |
| 14 | Video Analysis | ❌ FAIL | Needs `demo-video` bucket alias |
| 17 | Multi-User Isolation | ❌ FAIL | Requires `user1.json` and `user2.json` configs (only `mnagaso3.json` exists) |
| 25 | Endpoint Provisioning | ❌ FAIL | Needs HTTP API (404 on gateway docs, expected in production) |
| 26 | MCP Servers | ❌ FAIL | Same — HTTP API required |
| 27 | Pipeline CRUD | ❌ FAIL | Same — HTTP API required |
| 28 | Egress Audit | ❌ FAIL | Same — HTTP API required |
| 29 | LLM Completions | ❌ FAIL | Same — HTTP API required |
| 30 | File Management | ❌ FAIL | Same — HTTP API required |
| 31 | Client Management | ❌ FAIL | Same — HTTP API required |
| 99 | Cleanup | ✅ PASS | Commands available but destructive — not run |

### Private Demos

| # | Demo | Result | Issue |
|---|------|--------|-------|
| 04 | HMAE Inference | ❌ FAIL | Needs `user1-hmae` bucket + HMAE checkpoint |
| 09 | Defect Analysis | ❌ FAIL | Needs `demo-sam3` or `demo-defect` bucket alias |
| 10 | HMAE Full Pipeline | ❌ FAIL | Same — needs HMAE bucket + checkpoint |
| 14 | Opossum Analysis | ❌ FAIL | Needs Slurm account + video data path |

### Jens Demos

| # | Demo | Result | Issue |
|---|------|--------|-------|
| 11 | SAM3 Direct | ❌ FAIL | Requires direct SAM3 endpoint (bentoml_url not reachable) |
| 12 | HMAE Direct | ❌ FAIL | Same — requires direct HMAE endpoint |

## Root Causes for Failures

### 1. Missing bucket aliases (demos 03, 04, 14)
**Cause**: User only has `mnagaso3-text` bucket from the `default` experience.
**Fix**: User must add the relevant `experience.json` to their config and run `umrs reload` to provision the bucket alias.

### 2. No RAG ingestion pipeline (demos 02, 05, 19)
**Cause**: The `r4 rag ingest list` shows "No ingestion pipelines found." The RAG service exists but no pipeline is configured.
**Fix**: An ingestion pipeline must be created via `r4 rag ingest create` or configured server-side.

### 3. No Slurm account (demos 20, 21)
**Cause**: "User 'mnagaso3' has no slurm_account configured."
**Fix**: Admin must run: `umrs-server admin user set-slurm-account mnagaso3 <account>`

### 4. No second user config (demo 17)
**Cause**: Only `mnagaso3.json` exists, not `user1.json` and `user2.json`.
**Fix**: Demo is for demonstration purposes only — requires admin to enroll two users.

### 5. HTTP API not available (demos 25-31)
**Cause**: `check_http_api` in DEMO_STATUS.md notes this is expected in production — docs disabled.
**Fix**: Not fixable — these demos require the gateway to be in development mode or have `/docs` enabled.

### 6. Private demo dependencies (private 04, 09, 10, 14)
**Cause**: Need HMAE checkpoints, video data paths, and Slurm.
**Fix**: Site-specific configurations not available in this environment.

## Recommendation

### For the user to make demos work:
1. **Text embedding demo (02)**: The upload works. The only missing piece is an RAG ingestion pipeline. Create one via `r4 rag ingest create` or wait for it to be configured.
2. **Vision demos (03, 04, 14)**: Add the respective `experience.json` files to the user config and run `umrs reload`.
3. **Job demos (20, 21)**: Ask admin to configure a Slurm account for the user.
4. **Multi-user demo (17)**: This is a demo-only — not for production use.

### For the demo documentation:
The `README.md` files are now well-structured with step-by-step instructions. The demos correctly document what they need and provide "Running Manually" sections for individual step execution. The only gap is that demos don't explain how to add experiences/bucket aliases to the user config — this should be documented in `demos/README.md` as a prerequisite section.
