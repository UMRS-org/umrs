# Defect Analysis Meeting Demo

This private shell demo exercises the SAM3 upload, BentoML inference, and S3 result-download path for defect-style image analysis. It does not execute the full bundled multi-step workflow; the workflow bundles remain available under `workflow/` and `workflow_slurm/` for explicit workflow upload/run demos.

## What Runs

- Default normal-user bucket: `demo-sam3`
- Optional private bucket: `demo-defect` after operator provisioning/reload
- Input images: use `umrs/demos/sample_data/images` for a reliable one-object meeting check, or `umrs/tmp/hmae/images` for structural slices
- Runtime service: `bentoml-vision` through the `vision-bentoml` profile
- Adapter path: `umrs/src/umrs/vision_service/default_adapters/sam3_adapter.py`

On the current `umrs-front` deployment, `umrs-client reload` is admin-only. The private `demo-defect` bucket may be present in the local config but not provisioned in MinIO until an operator reloads it. The meeting-safe path is therefore to use `DEFECT_BUCKET=demo-sam3`.

## Meeting Steps

Run from the repository root unless noted.

1. Confirm the direct UMRS profile is reachable:

   ```bash
   PATH="$PWD/.venv/bin:$PATH" umrs-client --profile umrs-front status
   ```

2. Run the strict one-image SAM3-backed defect path:

   ```bash
   cd umrs/demos_private/09_defect_analysis
   PATH="/Users/mnagaso/workspace/RCCS/ai4s-platform/.venv/bin:$PATH" \
     UMRS_PROFILE=umrs-front \
     DEFECT_BUCKET=demo-sam3 \
     IMAGE_DIR="/Users/mnagaso/workspace/RCCS/ai4s-platform/umrs/demos/sample_data/images" \
     MAX_IMAGES=1 \
     RESULT_DOWNLOAD_TIMEOUT=1800 \
     SAM3_REQUIRE_RESULTS=1 \
     bash run_demo.sh
   ```

3. Point out these validated output lines:

   ```text
   Bucket alias : demo-sam3
   bucket=demo-user.sam3-segmentation.demo-sam3
   Defect-analysis results: 2 file(s)
   Defect detections: 1 object(s)
   ```

4. Show downloaded artifacts:

   ```bash
   ls -lh results/
   ```

## Useful Overrides

- Use operator-provisioned private bucket: `DEFECT_BUCKET=demo-defect bash run_demo.sh`
- Use structural slice images: `IMAGE_DIR=../../tmp/hmae/images bash run_demo.sh`
- Require completed outputs: `SAM3_REQUIRE_RESULTS=1 RESULT_DOWNLOAD_TIMEOUT=1800 bash run_demo.sh`