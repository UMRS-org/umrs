# Opossum Demo Validation Report
**Date:** 2026-06-19
**Investigator:** GitHub Copilot
**Status:** FIXED

---

## 1. Root Cause Report

### Issue: `upload_dir` fails with "Cannot process directory"

**Symptom:**
```
umrs-client upload_dir ../../tmp/opossum-video-analysis/data demo-opossum
status=error
message=Cannot process directory: ../../tmp/opossum-video-analysis/data
bucket=mnagaso.opossum-analysis.demo-opossum
subdir=None
modality=movie
```

**Root Cause:**
The `process_dir` function in `umrs/src/client_umrs.py` (line ~5351) used a flat
`glob(os.path.join(path, "*"))` to find files. The opossum data has a deep hierarchy
(`opossum_215_A/opossumA/Schedule/20260330/HH/*.mp4`), so the flat glob found only
subdirectories—not the .mp4 files nested deeper. With zero files found, the function
returned empty lists, triggering the "Cannot process directory" error.

**Secondary Bug:**
All `ctx.control_mq.notify_server()` calls in `process_file` and `process_dir` passed
an invalid `s3_keys=` keyword argument. The `MessageQueue.notify_server()` method
signature is `notify_server(self, client_id, notification_type, nb_objects, verbose=False, admin_token=None)`
and does not accept `s3_keys`. This caused `TypeError: notify_server() got an unexpected
keyword argument 's3_keys'` on every recursive upload attempt.

**Files Modified:**
1. `umrs/src/client_umrs.py` — `process_dir` function:
   - Changed from `glob(os.path.join(path, "*"))` to `os.walk()`
   - Each file's relative directory becomes the S3 key prefix
   - Preserves camera/Schedule date hierarchy as S3 key prefixes

2. `umrs/src/client_umrs.py` — `process_file` function:
   - Removed invalid `s3_keys=[s3_uri]` kwarg from `notify_server()` calls
   - Both the `image` category and `text_file` category branches were affected

3. `umrs/demos_private/14_opossum_analysis/README.md` — "Manual Steps" section:
   - Added `umrs-client experience add experience.json --update` (step 2)
   - Added `umrs-client reload` (step 3)
   - Added `umrs-client list_buckets` (step 4)
   - Each step includes expected outputs and common failure modes

**Validation Performed:**
```bash
cd umrs/demos_private/14_opossum_analysis
umrs-client upload_dir ../../tmp/opossum-video-analysis/data demo-opossum
```
Result: 47 files uploaded successfully. S3 keys preserve the hierarchy:
```
opossum_215_A/20260330_073458.tsv
opossum_215_A/opossumA/Schedule/20260330/00/20260330_000001.mp4
opossum_215_A/opossumA/Schedule/20260330/00/20260330_001132.mp4
...
```
Bucket `mnagaso.opossum-analysis.demo-opossum` contains 47 objects.

---

## 2. Documentation Updates

### `umrs/demos_private/14_opossum_analysis/README.md`

**Before (incomplete):**
```bash
# 1. Verify server connectivity
umrs-client ping

# 2. Bundle opossum engine into workflow
cp -r ../../tmp/opossum-video-analysis/src/opossum workflow/opossum

# 3. Upload workflow (includes bundled engine)
umrs-client upload-workflow workflow/ --name opossum-analysis

# 4. Upload videos (preserves camera/Schedule date hierarchy)
umrs-client upload_dir ../../tmp/opossum-video-analysis/data demo-opossum

# 5. Download results
umrs-client download_results demo-opossum results/ --force-overwrite
```

**After (complete):**
Added 3 missing commands to the "Manual Steps" section:
- `umrs-client experience add experience.json --update` — register experience locally
- `umrs-client reload` — push config to server, provision bucket
- `umrs-client list_buckets` — verify demo-opossum bucket is configured

Each step now includes expected outputs and common failure modes.

---

## 3. Demo Status Matrix

| Demo | Status | Evidence |
|------|--------|----------|
| `demos_private/14_opossum_analysis` | **FIXED** | `upload_dir` succeeded: 47 files uploaded, hierarchy preserved |

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
   umrs-client ping
   ```
   Output: `✓ Server ready | profile=rccs-cloud | data0.ai.r-ccs.riken.jp:50090`

2. **Experience Registration** ✅
   ```bash
   umrs-client experience add experience.json --update
   ```
   Output: `✓ Added experience(s): Opossum Analysis`

3. **Server Reload** ✅
   ```bash
   umrs-client reload
   ```
   Output: `✓ reload accepted`

4. **Bucket Verification** ✅
   ```bash
   umrs-client list_buckets
   ```
   Output: `Opossum Analysis | demo-opossum | movie | embedding | synced | 47 | mnagaso.opossum-analysis.demo-opossum`

5. **Workflow Upload** ✅
   ```bash
   umrs-client upload-workflow workflow/ --name opossum-analysis
   ```
   Output: `✓ Workflow opossum-analysis uploaded successfully`

6. **Data Upload** ✅ (FIXED)
   ```bash
   umrs-client upload_dir ../../tmp/opossum-video-analysis/data demo-opossum
   ```
   Output: 47 files uploaded, hierarchy preserved as S3 key prefixes.
   - S3 keys: `opossum_215_A/opossumA/Schedule/20260330/HH/*.mp4`
   - Bucket: `mnagaso.opossum-analysis.demo-opossum`
   - Object count: 47

### Summary

The opossum demo's `upload_dir` failure was caused by a non-recursive `glob()` in the
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
