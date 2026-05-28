# Demo 12: internal_genAI HMAE Direct

Uploads images through UMRS and runs the standalone `internal_genAI` HMAE
inference path on Slurm/Podman.

The demo supports all three HMAE modes exposed by `internal_genAI`:

- `mae`: masked MAE reconstruction
- `reconstruct`: full reconstruction without masking
- `features`: encoder feature extraction

## Prerequisites

- UMRS server runs with `VISION_STACK=jens`.
- `JENS_RIVAULT_SLURM_SCRIPT` points to `internal_genAI/serve-model.slurm`, or
  that script is staged at the UMRS default Jens wrapper path.
- `internal_genAI/bento/hmaeinfer/checkpoints/latest_checkpoint.pth` is readable
  by rootless Podman.
- For RCCS testing, start the local tunnel first:

```bash
cd deploy/ansible
./bootstrap.sh --profile rccs-cloud --tunnel
```

## Run

```bash
cd umrs/demos_jens/12_hmae_direct
UMRS_PROFILE=rccs-cloud \
  HMAE_MODE=all \
  MAX_IMAGES=1 \
  RESULT_DOWNLOAD_TIMEOUT=1500 \
  bash run_demo.sh
```

Set `HMAE_MODE=mae`, `HMAE_MODE=reconstruct`, or `HMAE_MODE=features` to run
one mode only.
