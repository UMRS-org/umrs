# Vision Inference Meeting Demo

This demo uses the same BentoML vision path as the SAM3 segmentation demo, but through the generic `Vision Inference` experience. It uploads general pictures, runs SAM3.1 with the prompt `cat`, and downloads overlay PNG plus compact pickle results.

## What Runs

- BentoML image source: `umrs/src/umrs/vision_service/Dockerfile`
- Runtime service in compose: `umrs/src/umrs/mini-rivault-llm/serve-model.yaml`, service `bentoml-vision`
- BentoML command: `bentoml serve umrs.vision_service.service:VisionModelService --port 3000`
- Default runtime image: `${BENTOML_VISION_IMAGE:-ghcr.io/riken-rccs/umrs-vision:latest}`
- SAM adapter: `umrs/src/umrs/vision_service/default_adapters/sam3_adapter.py`
- BentoML service API: `umrs/src/umrs/vision_service/service.py`

`${SAM3_MODEL_PATH}` is staged on RCCS Cloud and points to the authenticated `facebook/sam3.1` snapshot. The user-facing model name in `experience.json` is `sam3.1`; the actual model files are selected by the server-side environment passed into the Slurm BentoML job.

## How The Image And Prompt Flow

1. `README.md (step-by-step commands)` merges `experience.json` into the active UMRS user config.
2. The demo uses general pictures from `umrs/demos/sample_data/images` unless `IMAGE_DIR` or an argument is provided.
3. `umrs-client upload_dir` is implemented in `umrs/src/client_umrs.py` in `_action_s3_transfer()`. It uploads image bytes to S3, then sends a ZMQ `PROCESS_OBJECTS` notification.
4. UMRS server processing enters `umrs/src/umrs/server/processing.py` through `process_s3_objects()`.
5. For `model_type: rivault://bentoml://...`, processing calls `launch_rivault_model()` in `umrs/src/umrs/server/rivault_launcher.py`.
6. `rivault_launcher.py` selects the `vision-bentoml` profile, submits the Slurm job, opens the SSH tunnel, waits for BentoML `/readyz`, and builds a BentoML-backed compute adapter.
7. The BentoML request is submitted to `/vision/infer/submit`, which maps to `VisionModelService.infer()` in `service.py`.
8. The request fields are:
   - `model_type: "sam3"`
   - `inference_config: {"prompt": "cat"}` from `experience.json`
   - `image_url: "s3://<bucket>/<key>"`
   - `output_bucket: "<output bucket>"`
   - presigned input/output URLs supplied by UMRS for offline container access
9. `SAM3Adapter.infer()` loads SAM3.1 from `/models/sam3`, runs text-prompted segmentation, and uploads result artifacts to the output bucket.

## Meeting Steps

Run from the repository root unless noted.

1. Confirm the direct UMRS profile is reachable:

   ```bash
   PATH="$PWD/.venv/bin:$PATH" umrs-client --profile umrs-front status
   ```

2. Run one-image generic vision inference:

   ```bash
   cd umrs/demos/04_vision_inference
   PATH="/Users/mnagaso/workspace/RCCS/ai4s-platform/.venv/bin:$PATH" \
     UMRS_PROFILE=umrs-front \
     MAX_IMAGES=1 \
   RESULT_DOWNLOAD_TIMEOUT=1500 \
     VISION_REQUIRE_RESULTS=1 \
     REQUIRE_VISIBLE_MODEL=0 \
     # Follow the step-by-step commands in README.md
   ```

3. During the meeting, point out these output lines:

   ```text
   Bucket alias : demo-images
   Image dir    : .../umrs/demos/sample_data/images
   Prompt       : cat
   bucket=demo-user.vision-inference.demo-images
   ```

4. Show downloaded artifacts:

   ```bash
   ls -lh results/
   ```

## Useful Overrides

- Use more images: `MAX_IMAGES=3 # Follow the step-by-step commands in README.md`
- Use a custom input directory: `# Follow the step-by-step commands in README.md /path/to/pictures`
- Require completed outputs: `VISION_REQUIRE_RESULTS=1 RESULT_DOWNLOAD_TIMEOUT=1500 # Follow the step-by-step commands in README.md`