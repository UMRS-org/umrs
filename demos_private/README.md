# Private Demos

This directory contains demos that depend on private / proprietary models
or engines.  **Do not commit this directory to the public repository.**

| # | Demo | Dependency |
|---|------|------------|
| 04 | HMAE Inference | Site-provided HMAE checkpoint and vision runtime |
| 09 | Defect Analysis | HMAE + SAM3 combined pipeline |
| 10 | HMAE Full Pipeline | HMAE reconstruct + MAE + features |
| 14 | Opossum Analysis | External opossum video analysis engine |

## Usage

These demos follow the same conventions as the public demos in `demos/`:

```bash
cd demos_private/
# Follow the step-by-step commands in each demo directory

cd demos_private/04_hmae_inference
# Follow the step-by-step commands in README.md
```

They source `demos/lib/common.sh` so the UMRS server must be reachable.

### RCCS profile (umrs-front shared deployment)

`lib/common.sh` now auto-selects the deployed RCCS connection profile
(`rccs` or `rccs-cloud`) when `UMRS_PROFILE`/`UMRS_ZMQ_HOST` are unset and the
profile exists locally. To target it explicitly:

```bash
export UMRS_PROFILE=rccs-cloud     # or: rccs, depending on your client
umrs-client ping                   # expect: ✓ Server ready | profile=rccs-cloud
```

To run against a different deployment, set `UMRS_PROFILE` (or
`UMRS_ZMQ_HOST`/`UMRS_ZMQ_PORT`) before sourcing the demo commands.

### R4-first wrappers

The `ai4s-platform` superproject ships r4 CLI wrappers for this suite under
`demos_private/` (repo root), including a capstone knowledge/RAG/MCP
enhanced-inference demo. Prefer those when working through the Gateway.


The current private demo suite intentionally omits the Jens direct stack. The
Jens-specific direct demos are isolated under `demos_jens/`, and the supported
private demos here all use the normal UMRS user path instead of calling a
direct VisionFrontend service.

## Public Replacements

Generic, model-agnostic demos have been created in `demos/` to replace
each private demo:

| Private | Public Replacement |
|---------|-------------------|
| `04_hmae_inference` | `demos/04_vision_inference/` |
| `09_defect_analysis` | `demos/09_multistep_vision_pipeline/` |
| `10_hmae_full_pipeline` | `demos/10_vision_multi_mode/` |
| `14_opossum_analysis` | `demos/14_video_analysis/` |
