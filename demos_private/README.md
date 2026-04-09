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
cd demos_private/04_hmae_inference
bash run_demo.sh
```

They source `demos/lib/common.sh` so the UMRS server must be reachable.

## Public Replacements

Generic, model-agnostic demos have been created in `demos/` to replace
each private demo:

| Private | Public Replacement |
|---------|-------------------|
| `04_hmae_inference` | `demos/04_vision_inference/` |
| `09_defect_analysis` | `demos/09_multistep_vision_pipeline/` |
| `10_hmae_full_pipeline` | `demos/10_vision_multi_mode/` |
| `14_opossum_analysis` | `demos/14_video_analysis/` |
