# Jens Demo 11: SAM3 Direct

This isolated demo calls Jens's VisionFrontend BentoML service directly for
SAM3 segmentation.

It is intentionally separate from the active UMRS demo suites because it uses
the Jens-specific direct stack rather than the supported UMRS-mediated flow.

## Quick Start

```bash
cd demos_jens/11_sam3_direct
bash run_demo.sh

# Shared filesystem input
IMAGE_MODE=local bash run_demo.sh /path/to/images

# Custom VisionFrontend endpoint
JENS_SERVER=http://compute01 JENS_PORT=3000 bash run_demo.sh
```

## Defaults

- `IMAGE_MODE=s3`
- sample input: `../../tmp/hmae/images`
- bucket alias: `user1-sam3`

## Notes

- `IMAGE_MODE=presigned` currently falls back to `s3://` references because the
  local `umrs-client` surface does not expose a dedicated presign command here.
- The demo requires VisionFrontend readiness at `/readyz`.