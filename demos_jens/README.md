# Isolated Jens Vision Stack Demos

These demos are intentionally isolated from the active `demos/` and
`demos_private/` suites.

They target Jens's standalone VisionFrontend stack directly rather than the
current UMRS-mediated upload -> server processing -> download flow.

## Included Demos

| Demo | Purpose |
|---|---|
| `11_sam3_direct` | Direct SAM3 segmentation via VisionFrontend |
| `12_hmae_direct` | Direct HMAE inference via VisionFrontend |

## Important Behavior

- These demos are **not** run by `demos/run_all.sh` or `demos_private/run_all.sh`.
- They call BentoML VisionFrontend directly for inference.
- When `IMAGE_MODE=s3`, they still use `umrs-client upload_dir` to stage input
  files before calling VisionFrontend.
- The default sample input path is `tmp/hmae/images` so the isolated demos can
  reuse the checked-in local HMAE image assets.

## Prerequisites

- VisionFrontend reachable at `http://localhost:3000/readyz` or equivalent
- Python with `bentoml>=1.4`
- A working UMRS client profile when using `IMAGE_MODE=s3`

## Usage

```bash
cd demos_jens
bash run_all.sh

# Or run one demo directly
bash 11_sam3_direct/run_demo.sh
bash 12_hmae_direct/run_demo.sh
```