# Demo 11: internal_genAI SAM3 Direct

Uploads user images through UMRS, lets the UMRS server stage them in S3, then
launches the standalone `internal_genAI` Slurm/Podman stack for SAM3 text-prompt
segmentation.

## Prerequisites

- UMRS server runs with `VISION_STACK=jens`.
- `JENS_RIVAULT_SLURM_SCRIPT` points to `internal_genAI/serve-model.slurm`, or
  that script is staged at the UMRS default Jens wrapper path.
- The `internal_genAI/bento/sam3infer` directory contains readable `sam3.pt`
  and `bpe.gz` files.
- For RCCS testing, start the local tunnel first:

```bash
cd deploy/ansible
./bootstrap.sh --profile rccs-cloud --tunnel
```

## Run

```bash
cd umrs/demos_jens/11_sam3_direct
UMRS_PROFILE=rccs-cloud \
  SAM3_PROMPT=cat \
  MAX_IMAGES=1 \
  RESULT_DOWNLOAD_TIMEOUT=1500 \
  bash run_demo.sh
```

The prompt is text-only for this standalone demo because the attached
`internal_genAI` service exposes SAM3 through `set_text_prompt()`.
