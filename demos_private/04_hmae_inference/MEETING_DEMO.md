# HMAE Inference Meeting Demo

This demo uploads Spring-8 concrete PNG volume slices, starts the Mini-RiVault BentoML vision stack on Slurm, runs HMAE inference, and downloads reconstruction or feature outputs.

## What Runs

- BentoML image source: `umrs/src/umrs/vision_service/Dockerfile`
- Runtime service in compose: `umrs/src/umrs/mini-rivault-llm/serve-model.yaml`, service `bentoml-vision`
- BentoML command: `bentoml serve umrs.vision_service.service:VisionModelService --port 3000`
- Default runtime image: `${BENTOML_VISION_IMAGE:-ghcr.io/riken-rccs/umrs-vision:latest}`
- HMAE adapter: `umrs/src/umrs/vision_service/default_adapters/hmae_adapter.py`
- BentoML service API: `umrs/src/umrs/vision_service/service.py`

`serve-model.yaml` binds the HMAE model directory from `${HMAE_MODEL_PATH}` into `/models/hmae`. The container reads `/models/hmae/infer_hmae_vit_b.yaml` and `/models/hmae/latest_checkpoint.pth` through the `HMAE_CONFIG_PATH` and `HMAE_CHECKPOINT` environment variables.

## How The PNG And Task Flow

1. The README merges `experience.json` into the active UMRS user config.
2. The demo now defaults to Spring-8 concrete inputs at `umrs/tmp/hmae/spring8concrete`.
3. `umrs data dir-upload` is implemented in `umrs/src/client_umrs.py` in `_action_s3_transfer()`. It uploads PNG bytes to S3, then sends a ZMQ `PROCESS_OBJECTS` notification. The PNG bytes are not carried over ZMQ.
4. UMRS server processing enters `umrs/src/umrs/server/processing.py` through `process_s3_objects()`.
5. For `model_type: rivault://bentoml://...`, processing calls `launch_rivault_model()` in `umrs/src/umrs/server/rivault_launcher.py`.
6. `rivault_launcher.py` selects the `vision-bentoml` profile, submits the Slurm job, opens the SSH tunnel, waits for BentoML `/readyz`, and builds a BentoML-backed compute adapter.
7. The BentoML request is submitted to `/vision/infer/submit`, which maps to `VisionModelService.infer()` in `service.py`.
8. The request fields are:
   - `model_type: "hmae"`
   - `inference_config: {"task": "reconstruction"}` for the main `demo-hmae` bucket
   - `image_url: "s3://<bucket>/<key>"`
   - `output_bucket: "<output bucket>"`
   - presigned input/output URLs supplied by UMRS for offline container access
9. `HMAEAdapter.infer()` preprocesses the PNG slice, resolves the task, runs reconstruction, MAE reconstruction, or feature extraction, and uploads PNG/pickle result artifacts.

## Meeting Steps

Run from the repository root unless noted.

1. Confirm the direct UMRS profile is reachable:

   ```bash
   PATH="$PWD/.venv/bin:$PATH" umrs --profile umrs-front status
   ```

2. Run one-image HMAE demo with the Spring-8 concrete PNG set:

   ```bash
   cd umrs/demos_private/04_hmae_inference
   PATH="$PWD/.venv/bin:$PATH" \
     UMRS_PROFILE=umrs-front \
     MAX_IMAGES=1 \
   RESULT_DOWNLOAD_TIMEOUT=1800 \
   HMAE_REQUIRE_RESULTS=1 \
     # See the numbered steps in the main README
   ```

3. During the meeting, point out these output lines:

   ```text
   Image dir    : .../umrs/tmp/hmae/spring8concrete
   Processing file '.../volume_170.png'
   bucket=demo-user.hmae-inference.demo-hmae
   ```

4. Show downloaded artifacts:

   ```bash
   ls -lh results/
   ```

5. If results are still pending because the Slurm job is starting, show server progress:

   ```bash
   ssh umrs-front 'tail -f ~/deploy/ai4s-platform/umrs/deploy/umrs-front/logs/server-rccs.log'
   ```

## Useful Overrides

- Use more Spring-8 slices: `MAX_IMAGES=3`
- Run a different HMAE bucket/task: `HMAE_BUCKET=demo-hmae-mae` or `HMAE_BUCKET=demo-hmae-features`
- Require completed outputs for CI-like validation: `HMAE_REQUIRE_RESULTS=1 RESULT_DOWNLOAD_TIMEOUT=1800`