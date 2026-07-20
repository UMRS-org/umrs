# SAM3 Segmentation Meeting Demo

This demo uploads general-purpose pictures, starts the Mini-RiVault BentoML vision stack on Slurm, runs SAM3.1 text-prompted segmentation, and downloads the generated overlay/result files.

## What Runs

- BentoML image source: `umrs/src/umrs/vision_service/Dockerfile`
- Runtime service in compose: `umrs/src/umrs/mini-rivault-llm/serve-model.yaml`, service `bentoml-vision`
- BentoML command: `bentoml serve umrs.vision_service.service:VisionModelService --port 3000`
- Default runtime image: `${BENTOML_VISION_IMAGE:-ghcr.io/riken-rccs/umrs-vision:latest}`
- SAM adapter: `umrs/src/umrs/vision_service/default_adapters/sam3_adapter.py`
- BentoML service API: `umrs/src/umrs/vision_service/service.py`

`serve-model.yaml` enables `bentoml-vision` under the `vision-bentoml` profile and publishes the vision path through the dedicated BentoML vision port. The container binds SAM weights from `${SAM3_MODEL_PATH}` into `/models/sam3`; for this demo `${SAM3_MODEL_PATH}` points at the authenticated `facebook/sam3.1` snapshot staged on RCCS Cloud, and the experience model name is `sam3.1`.

## How The Image And Prompt Flow

1. The demo README merges `experience.json` into the active UMRS user config.
2. The demo uses general pictures from `umrs/demos/sample_data/images` unless `IMAGE_DIR` is explicitly set.
3. `umrs-client data dir-upload` is implemented in `umrs/src/client_umrs.py` in `_action_s3_transfer()`. It uploads the image bytes to the configured S3 bucket, then sends a ZMQ `PROCESS_OBJECTS` notification. The image itself does not travel over ZMQ.
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
9. `SAM3Adapter.infer()` loads the SAM3 model, uses the prompt `cat`, uploads the segmentation overlay PNG and compact pickle results to the output bucket.

## Meeting Steps

Run from the repository root unless noted.

1. Confirm the direct UMRS profile is reachable:

   ```bash
   PATH="$PWD/.venv/bin:$PATH" umrs-client --profile umrs-front status
   ```

2. Run one-image SAM3.1 demo with the general picture set:

   ```bash
   cd umrs/demos/03_sam3_segmentation
   PATH="/Users/mnagaso/workspace/RCCS/ai4s-platform/.venv/bin:$PATH" \
     UMRS_PROFILE=umrs-front \
     MAX_IMAGES=1 \
     RESULT_DOWNLOAD_TIMEOUT=600 \
     SAM3_REQUIRE_RESULTS=0 \
   ```

   See the numbered steps in the main README for the full script.

3. During the meeting, point out these output lines:

   ```text
   Image dir    : .../umrs/demos/sample_data/images
   SAM3 model 'sam3.1'
   Prompt       : cat
   Processing file '.../Cat_1.jpg'
   bucket=demo-user.sam3-segmentation.demo-sam3
   ```

4. Show downloaded artifacts:

   ```bash
   ls -lh results/
   ```

5. If results are still pending because the Slurm job is starting, show server progress:

   ```bash
   ssh umrs-front 'tail -f /home/mnagaso/deploy/ai4s-platform/umrs/deploy/umrs-front/logs/server-rccs.log'
   ```

## Useful Overrides

- Use more general pictures: `MAX_IMAGES=3`
- Use a custom general picture directory: `IMAGE_DIR=/path/to/pictures`
- Require completed outputs for CI-like validation: `SAM3_REQUIRE_RESULTS=1 RESULT_DOWNLOAD_TIMEOUT=900`