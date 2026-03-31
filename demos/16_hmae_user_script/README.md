# Demo 16: HMAE Inference (user_script)

Equivalent to [Demo 03](../03_hmae_inference/README.md), but uses the
`user_script://` model type with the Pipeline SDK instead of BentoML/Rivault.
The HMAE model is loaded directly by the Step class — no BentoML service or
Rivault sandbox is required.

## Prerequisites

- Services running (see [Demo 00](../00_start_services/README.md))
- HMAE model checkpoint (`.pth`) and config YAML pre-staged on the server node
- Python packages: `torch`, `opencv-python`, `numpy`, `matplotlib`, `PyYAML`
- UMRS client installed (`pip install .` from `src/`)

## Architecture

```
Client → UMRS Server → user_script Pipeline
              │              └── HMAEInferenceStep (GPU)
              │                   ├── Load HMAE from checkpoint (lazy, cached)
              │                   ├── Quadtree decomposition
              │                   ├── Inference (reconstruct / mae / features)
              │                   └── Upload visualization PNG or features JSON
              └── S3 (MinIO)
```

No BentoML, NGINX, or Slurm needed — the model runs directly in the UMRS
server process via the Pipeline SDK.

## Quick Start

```bash
cd demos/16_hmae_user_script
bash run_demo.sh
```

## Configuration

### workflow.json

The `workflow.json` defines the pipeline with a single `HMAEInferenceStep`:

| Field | Description |
|-------|-------------|
| `source.path` | Absolute path to the `scripts/` directory on the server |
| `steps[0].config.task` | UMRS task: `reconstruction`, `mae_reconstruction`, or `feature_extraction` |
| `steps[0].config.cfg_file` | Absolute path to the HMAE YAML config file |
| `steps[0].config.checkpoint_path` | Absolute path to the `.pth` checkpoint |

### vision_models.json

Uses the new modality/task format:

```json
{
  "modalities": [{
    "modality": "image",
    "task": "reconstruction",
    "model": {
      "model_type": "user_script:///path/to/workflow.json",
      "return_type": "file"
    }
  }]
}
```

## HMAE Tasks

| Task | Output | Use Case |
|------|--------|----------|
| `reconstruction` | Side-by-side PNG: original / quadtree / full reconstruction | Visualise model understanding |
| `mae_reconstruction` | Side-by-side PNG + loss JSON: original / masked grid / reconstruction | Quality assessment, anomaly detection |
| `feature_extraction` | JSON with CLS embedding + patch count + feature dim | Downstream classification, similarity search |

To change task, edit `workflow.json`:
```json
"config": {
    "task": "mae_reconstruction"
}
```

## Outputs

For each input image, the step produces (depending on task):
- **reconstruction**: `hmae_out/<filename>_reconstruct.png`
- **mae_reconstruction**: `hmae_out/<filename>_mae.png` + `hmae_out/<filename>_mae_loss.json`
- **feature_extraction**: `hmae_out/<filename>_features.json`

## Differences from Demo 03

| | Demo 03 (BentoML) | Demo 16 (user_script) |
|---|---|---|
| Model type | `rivault://bentoml://...` | `user_script:///path/to/workflow.json` |
| Runtime | BentoML service in Rivault sandbox | Pipeline SDK step in UMRS server |
| Infrastructure | Slurm + BentoML + NGINX | None extra |
| Model loading | Pre-staged by admin, loaded by BentoML | Lazy-loaded from checkpoint in Step |
| Customisation | Change `vision_model_config` | Edit `workflow.json` or `hmae_inference.py` |

## Customising

Edit `scripts/hmae_inference.py` to change inference behaviour, add
post-processing, or support different HMAE model variants. No UMRS code
changes are needed — just update `workflow.json` if the entrypoint changes.
