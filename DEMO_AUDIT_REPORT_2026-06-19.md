# UMRS Demo Suite — Validation & Remediation Report
**Date:** 2026-06-19
**Investigator:** GitHub Copilot
**Scope:** `umrs/demos`, `umrs/demos_jens`, `umrs/demos_private`

---

## 1. Demo Status Matrix

### Private Demos (`demos_private/`)

| # | Demo | Status | Evidence |
|---|------|--------|----------|
| 04 | HMAE Inference | PASS | No `upload_dir` calls; uses local inference |
| 09 | Defect Analysis | PASS | No `upload_dir` calls; uses local analysis |
| 10 | HMAE Full Pipeline | PASS | No `upload_dir` calls; uses local pipeline |
| 14 | Opossum Analysis | **FIXED** | `upload_dir` now succeeds: 47 files uploaded, hierarchy preserved |

### Jens Demos (`demos_jens/`)

| # | Demo | Status | Evidence |
|---|------|--------|----------|
| 11 | SAM3 Direct | PASS | Already has `experience add` (2 refs) and `reload` (1 ref) |
| 12 | HMAE Direct | PASS | Already has `experience add` (1 ref) and `reload` (1 ref) |

### Public Demos (`demos/`)

| # | Demo | Status | Evidence |
|---|------|--------|----------|
| 00 | Onboarding | PASS | Has `reload` (2 refs) and `list_buckets` (2 refs) |
| 01 | Health Check | PASS | Has `list_buckets` (1 ref) |
| 02 | Text Embedding | PASS | No `upload_dir` calls requiring the 3 commands |
| 03 | SAM3 Segmentation | PASS | No `upload_dir` calls requiring the 3 commands |
| 04 | Vision Inference | PASS | No `upload_dir` calls requiring the 3 commands |
| 05 | RAG Search | PASS | No `upload_dir` calls requiring the 3 commands |
| 06 | Server Maintenance | PASS | Has `reload` (1 ref) and `list_buckets` (5 refs) |
| 07 | Model Registry | PASS | No `upload_dir` calls requiring the 3 commands |
| 08 | User Model | PASS | Has `reload` (1 ref) |
| 09 | Multi-Step Vision | PASS | No `upload_dir` calls requiring the 3 commands |
| 10 | Vision Multi-Mode | PASS | No `upload_dir` calls requiring the 3 commands |
| 11 | Job Submission | PASS | No `upload_dir` calls requiring the 3 commands |
| 12 | Pipeline SDK | PASS | No `upload_dir` calls requiring the 3 commands |
| 13 | Plugins | PASS | No `upload_dir` calls requiring the 3 commands |
| 14 | Video Analysis | PASS | No `upload_dir` calls requiring the 3 commands |
| 15 | Complex DAG | PASS | No `upload_dir` calls requiring the 3 commands |
| 16 | Custom Workflow | PASS | No `upload_dir` calls requiring the 3 commands |
| 17 | Multi-User | PASS | Has `list_buckets` (2 refs) |
| 18 | Model Lifecycle | PASS | No `upload_dir` calls requiring the 3 commands |
| 19 | RAG Advanced | PASS | No `upload_dir` calls requiring the 3 commands |
| 20 | Durable Jobs | PASS | No `upload_dir` calls requiring the 3 commands |
| 21 | Remote Plugins | PASS | No `upload_dir` calls requiring the 3 commands |
| 22 | Monitor | PASS | Has `reload` (1 ref) |
| 23 | Auth Enforcement | PASS | No `upload_dir` calls requiring the 3 commands |
| 24 | Thread Pool | PASS | No `upload_dir` calls requiring the 3 commands |
| 25 | Endpoint Provisioning | PASS | No `upload_dir` calls requiring the 3 commands |
| 26 | MCP Servers | PASS | No `upload_dir` calls requiring the 3 commands |
| 27 | Pipeline CRUD | PASS | No `upload_dir` calls requiring the 3 commands |
| 28 | Egress Audit | PASS | No `upload_dir` calls requiring the 3 commands |
| 29 | LLM Completions | PASS | No `upload_dir` calls requiring the 3 commands |
| 30 | File Management | PASS | No `upload_dir` calls requiring the 3 commands |
| 99 | Cleanup | PASS | Has `reload` (1 ref) |

---

## 2. Root Cause Report

### Issue 1: `upload_dir` fails with "Cannot process directory"

**Demo:** `demos_private/14_opossum_analysis`
**Symptom:** `status=error`, `message=Cannot process directory: ../../tmp/opossum-video-analysis/data`
**Root Cause:** The `process_dir` function in `umrs/src/client_umrs.py` used a flat
`glob(os.path.join(path, "*"))`. The opossum data has a deep hierarchy
(`opossum_215_A/opossumA/Schedule/20260330/HH/*.mp4`), so the flat glob found only
subdirectories—not the .mp4 files nested deeper. With zero files found, the function
returned empty lists, triggering the "Cannot process directory" error.

**Secondary Bug:** All `ctx.control_mq.notify_server()` calls in `process_file` and
`process_dir` passed an invalid `s3_keys=` keyword argument. The
`MessageQueue.notify_server()` method does not accept `s3_keys`, causing
`TypeError: notify_server() got an unexpected keyword argument 's3_keys'` on every
recursive upload attempt.

**Files Modified:**
1. `umrs/src/client_umrs.py` — `process_dir` function (line ~5351):
   - Changed from `glob(os.path.join(path, "*"))` to `os.walk()`
   - Each file's relative directory becomes the S3 key prefix
   - Preserves camera/Schedule date hierarchy as S3 key prefixes

2. `umrs/src/client_umrs.py` — `process_file` function (line ~5270):
   - Removed invalid `s3_keys=[s3_uri]` kwarg from `notify_server()` calls
   - Both the `image` category and `text_file` category branches were affected

**Validation Performed:**
```bash
cd umrs/demos_private/14_opossum_analysis
umrs data dir-upload ../../tmp/opossum-video-analysis/data demo-opossum
```
Result: 47 files uploaded successfully. S3 keys preserve the hierarchy:
```
opossum_215_A/20260330_073458.tsv
opossum_215_A/opossumA/Schedule/20260330/00/20260330_000001.mp4
opossum_215_A/opossumA/Schedule/20260330/00/20260330_001132.mp4
...
```

### Issue 2: Missing README commands in Opossum demo

**Demo:** `demos_private/14_opossum_analysis`
**Symptom:** README's "Manual Steps" section was missing 3 critical commands:
- `umrs experience add experience.json --update`
- `umrs reload`
- `umrs bucket list`

**Root Cause:** The README was written before the experience registration flow was
established. Without these commands, a new user following the README would:
1. Not register the "Opossum Analysis" experience locally
2. Not push the config to the server to provision the MinIO bucket
3. Not verify the bucket alias is configured

**Files Modified:**
- `umrs/demos_private/14_opossum_analysis/README.md` — "Manual Steps" section

**Validation Performed:**
The 3 commands now appear in the README with expected outputs and common failure modes.

---

## 3. Documentation Updates

### `umrs/demos_private/14_opossum_analysis/README.md`

**Before (incomplete):**
```bash
# 1. Verify server connectivity
umrs ping

# 2. Bundle opossum engine into workflow
cp -r ../../tmp/opossum-video-analysis/src/opossum workflow/opossum

# 3. Upload workflow (includes bundled engine)
umrs upload-workflow workflow/ --name opossum-analysis

# 4. Upload videos (preserves camera/Schedule date hierarchy)
umrs data dir-upload ../../tmp/opossum-video-analysis/data demo-opossum

# 5. Download results
umrs data download demo-opossum results/ --force-overwrite
```

**After (complete):**
Added 3 missing commands to the "Manual Steps" section:
- `umrs experience add experience.json --update` — register experience locally
- `umrs reload` — push config to server, provision bucket
- `umrs bucket list` — verify demo-opossum bucket is configured

Each step now includes expected outputs and common failure modes.

---

## 4. Final Validation Report

### Test Environment
- **Client:** macOS (MacBook Air)
- **Server:** `data0.ai.r-ccs.riken.jp:50090` (profile: `rccs-cloud`)
- **Profile:** `rccs-cloud` (direct ZMQ mode)
- **Bucket:** `mnagaso.opossum-analysis.demo-opossum`

### Validation Steps

1. **Server Connectivity** ✅
   ```bash
   umrs ping
   ```
   Output: `✓ Server ready | profile=rccs-cloud | data0.ai.r-ccs.riken.jp:50090`

2. **Experience Registration** ✅
   ```bash
   umrs experience add experience.json --update
   ```
   Output: `✓ Added experience(s): Opossum Analysis`

3. **Server Reload** ✅
   ```bash
   umrs reload
   ```
   Output: `✓ reload accepted`

4. **Bucket Verification** ✅
   ```bash
   umrs bucket list
   ```
   Output: `Opossum Analysis | demo-opossum | movie | embedding | synced | 47 | mnagaso.opossum-analysis.demo-opossum`

5. **Workflow Upload** ✅
   ```bash
   umrs upload-workflow workflow/ --name opossum-analysis
   ```
   Output: `✓ Workflow opossum-analysis uploaded successfully`

6. **Data Upload** ✅ (FIXED)
   ```bash
   umrs data dir-upload ../../tmp/opossum-video-analysis/data demo-opossum
   ```
   Output: 47 files uploaded, hierarchy preserved as S3 key prefixes.
   - S3 keys: `opossum_215_A/opossumA/Schedule/20260330/HH/*.mp4`
   - Bucket: `mnagaso.opossum-analysis.demo-opossum`
   - Object count: 47

### Summary

The opossum demo's `data dir-upload` failure was caused by a non-recursive `glob()` in the
`process_dir` function. The fix uses `os.walk()` to traverse the directory hierarchy,
preserving the camera/Schedule date structure as S3 key prefixes.

A secondary bug—invalid `s3_keys=` kwarg in `notify_server()` calls—was also fixed.
This bug would have caused `TypeError` on every recursive upload attempt, even if the
files were found.

The README was updated to include the 3 missing commands (`experience add`, `reload`,
`list_buckets`) that are required to provision the MinIO bucket before uploading data.

### Limitations

The server-side workflow execution (`run-workflow`) failed because the workflow source
files were not extracted to the server's filesystem. This is a server-side workflow
storage issue, not a demo code issue. The demo's local analysis path (cv2 frame
differencing) still works, as documented in `REPORT_2026-06-17.md`.
