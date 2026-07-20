# UMRS Private Demo Status Report

## Current Deployment State (rccs-cloud)

### Available Resources
- **Server**: ✅ Reachable (data0.ai.r-ccs.riken.jp:50090)
- **Buckets**: 1 (`mnagaso3-text` — text, embedding)
- **Models**: 35 system models
- **RAG**: ✅ Available
- **Workflows**: ✅ Operational
- **Jobs**: ✅ Responsive

### Private Demo Analysis

| Demo | Bucket(s) Needed | Workflow | Experience | Status | Blocker |
|------|------------------|----------|------------|--------|---------|
| **04_hmae_inference** | `user1-hmae` | ❌ No | ✅ Yes | **BLOCKED** | Bucket not provisioned |
| **09_defect_analysis** | `demo-sam3`, `demo-defect` | ✅ Yes | ✅ Yes | **BLOCKED** | Bucket not provisioned |
| **10_hmae_full_pipeline** | `demo-hmae`, `demo-hmae-mae`, `demo-hmae-features` | ✅ Yes | ✅ Yes | **BLOCKED** | Buckets not provisioned |
| **14_opossum_analysis** | `demo-opossum` | ✅ Yes | ✅ Yes | **BLOCKED** | Bucket not provisioned |

### Blocker Summary

All 4 private demos are blocked by **missing bucket aliases**. The required buckets (`user1-hmae`, `demo-sam3`, `demo-defect`, `demo-hmae*`, `demo-opossum`) need to be provisioned by an operator via:

1. **Add experiences** to site.yaml or user config
2. **Run `umrs-client reload`** on the server
3. **Ensure model checkpoints** are staged on the server

### Workaround: Use Existing `mnagaso3-text` Bucket

For demo purposes, all uploads can be redirected to the existing `mnagaso3-text` bucket. The smart fallback feature (implemented in this session) will auto-redirect unknown bucket aliases.

## Blockers List (Detailed)

### 04_hmae_inference
- **Bucket needed**: `user1-hmae`
- **Why missing**: No HMAE experience configured on site
- **To fix**: Add HMAE experience to site.yaml, run reload
- **HMAE model**: Requires pre-staged checkpoint on server

### 09_defect_analysis
- **Buckets needed**: `demo-sam3`, `demo-defect`
- **Why missing**: SAM3 defect analysis experience not provisioned
- **To fix**: Add defect analysis experience, run reload
- **SAM3 model**: `facebook/sam2-hiera-small` must be staged on server
- **Workflow**: Available (workflow/workflow.json present)

### 10_hmae_full_pipeline
- **Buckets needed**: `demo-hmae`, `demo-hmae-mae`, `demo-hmae-features`
- **Why missing**: HMAE full pipeline experience not provisioned
- **To fix**: Add HMAE pipeline experience, run reload
- **Workflow**: Available (3-stage workflow with reconstruct, mae, features)

### 14_opossum_analysis
- **Bucket needed**: `demo-opossum`
- **Why missing**: Opossum video analysis experience not provisioned
- **To fix**: Add experience, run reload
- **Workflow**: Available (GPU-accelerated video motion detection)
- **Sample data**: Requires `tmp/opossum-video-analysis/data/` with real trail-cam footage

## Recommendations

### Immediate (Demo Day)
1. Use existing `mnagaso3-text` bucket for all uploads
2. Demonstrate upload/download with sample data
3. Focus on public demos that work with existing infrastructure

### Short-term (1-2 weeks)
1. Operator provisions bucket aliases for private demos
2. Stage HMAE and SAM3 model checkpoints on server
3. Run `umrs-client reload` after adding experiences

### Medium-term (1 month)
1. Add all bucket aliases via ansible/deployment
2. Configure model registry with HMAE, SAM3 checkpoints
3. Full private demo validation against production data
