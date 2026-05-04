# Jens Demo 12: HMAE Direct

This isolated demo calls Jens's VisionFrontend BentoML service directly for
HMAE inference.

It is intentionally separate from the active UMRS demo suites because it uses
the Jens-specific direct stack rather than the supported UMRS-mediated flow.

## Quick Start

```bash
cd demos_jens/12_hmae_direct
bash run_demo.sh

# Run all HMAE modes
HMAE_MODE=all bash run_demo.sh

# Shared filesystem input
IMAGE_MODE=local bash run_demo.sh /path/to/volume_slices
```

## Defaults

- `IMAGE_MODE=s3`
- sample input: `../../tmp/hmae/images`
- bucket alias: `user1-hmae`

## Supported HMAE Modes

- `reconstruct`
- `mae`
- `features`
- `all`

## Notes

- `IMAGE_MODE=presigned` currently falls back to `s3://` references because the
  local `umrs-client` surface does not expose a dedicated presign command here.
- The demo requires VisionFrontend readiness at `/readyz`.