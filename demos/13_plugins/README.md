# Demo 13 — Plugin Execution System

This demo shows the UMRS plugin system for running user-provided code
in sandboxed environments (Slurm/Singularity on HPC, or Kubernetes Jobs
in cloud).

## What It Does

1. **Define** a `PluginSpec` — source code, entrypoint, resource limits, output destinations
2. **Validate** the spec — checks source URI, entrypoint format, resource limits
3. **Preview** the generated Slurm sbatch script (shows what HPC execution looks like)
4. **Execute** the plugin locally (simulates the sandboxed environment)
5. **Audit** — show the trail of events logged during execution

## Prerequisites

- Python 3.9+
- UMRS installed: `uv pip install -e src/`

## Quick Start

```bash
# Follow the step-by-step commands in README.md
```

Or run directly:

```bash
python3 run_plugin_demo.py --input-dir ../../data/file_of_texts
```

## Running Manually

This demo runs locally and does not require `umrs-client` for the
default local execution backend.

```bash
# 1. Run the plugin demo with sample text data
#    Defines a PluginSpec, validates it, previews the generated sbatch
#    script, executes the plugin locally, and prints the audit trail.
python3 run_plugin_demo.py --input-dir ../../data/file_of_texts

# 2. Run with your own data directory
python3 run_plugin_demo.py --input-dir /path/to/your/data

# 3. For server-side plugin execution (Slurm/K8s backends), you would
#    use umrs-client to upload the plugin spec and trigger execution:
#      umrs-client workflow upload ./my_plugin_dir --name "my-plugin"
#      umrs-client data dir-upload /data/input user1-plugin-bucket
#      umrs-client data download user1-plugin-bucket ./results
```

## Plugin Architecture

```
PluginSpec → PluginRunner → [Sandbox] → PluginResult → ResultWriter → Outputs
                  │                          │
                  │                          └── PluginAuditLog
                  ├── SlurmPluginRunner (HPC)
                  │     └── Singularity container + sbatch
                  │     └── Step executor: python -m umrs.plugins.plugin_runner.entrypoint
                  │     └── Result contract: /workspace/result_manifest.json
                  ├── K8sPluginRunner (Cloud)
                  │     └── Kubernetes Job + security context
                  └── Local (this demo)
                        └── direct execution with workspace dirs
```

## Plugin Spec Fields

| Field | Description | Example |
|-------|-------------|---------|
| `source_type` | Where the plugin code lives | `LOCAL_PATH`, `GIT_REPO`, `S3_WHEEL` |
| `source_uri` | Path/URL to the code | `/path/to/plugin.py` |
| `entrypoint` | Module and function | `word_count_plugin:main` |
| `cpu_limit` | CPU cores | `"1"` |
| `memory_limit` | Memory | `"512Mi"` |
| `gpu_count` | NVIDIA GPUs | `0` |
| `timeout_seconds` | Max execution time | `3600` |
| `output_destinations` | Where to write results | `["s3://bucket/", "umrs://"]` |

## Writing a Plugin

A plugin is a Python script with a `main()` function. The runner sets up:

- `/workspace/` — job directory (host-side bind-mounted)
- `/workspace/result_manifest.json` — step executor writes structured results here
- `/source/` — step source code (bind-mounted read-only from `spec.source_uri`)
- Environment variables:
  - `PLUGIN_ENTRYPOINT`, `PLUGIN_SOURCE_TYPE`, `PLUGIN_SOURCE_URI`, `PLUGIN_TIMEOUT`
  - `UMRS_STEP_NAME`, `UMRS_STEP_ENTRYPOINT`, `UMRS_STEP_CONFIG` (set by the step adapter)
  - `UMRS_INPUT_BUCKET`, `UMRS_INPUT_KEYS`, `UMRS_OUTPUT_BUCKET`, `UMRS_RUN_ID`
  - User-defined vars are prefixed with `USER_` (e.g. `CUDA_VISIBLE_DEVICES` → `USER_CUDA_VISIBLE_DEVICES`)

For remote execution (Slurm/K8s), the in-container step executor
(`python -m umrs.plugins.plugin_runner.entrypoint`) reads these env vars,
imports the step class, runs `setup` → `run` → `teardown`, and writes
`result_manifest.json` with output keys, embeddings, checksums, and metadata.

See [word_count_plugin.py](word_count_plugin.py) for a complete example.

## Execution Backends

### Slurm (HPC)
```python
from umrs.plugins import SlurmPluginRunner, PluginSpec

runner = SlurmPluginRunner(partition="gpu", singularity_image="/path/to/image.sif")
result = await runner.execute(spec)
```
- Generates sbatch script with `#SBATCH` directives
- Runs inside Singularity container with `--containall --cleanenv`
- Env vars delivered via `SINGULARITYENV_*` prefix (survives `--cleanenv`)
- Bind-mounts: workspace dir at `/workspace`, source dir at `/source:ro`
- In-container entrypoint: `python -m umrs.plugins.plugin_runner.entrypoint`
- Metrics written to host-side `{work_dir}/{plugin_id}/metrics.txt`
- Step executor writes `/workspace/result_manifest.json` for result chaining
- Polls `squeue`/`sacct` for completion

### Kubernetes (Cloud)
```python
from umrs.plugins import K8sPluginRunner

runner = K8sPluginRunner(namespace="rivault-plugins")
result = await runner.execute(spec)
```
- Creates K8s Job manifest with security context
- Non-root, read-only filesystem, no privilege escalation
- Resource limits enforced by Kubernetes

## Output Routing

Results can be written to multiple destinations:

| Destination | Writer | Example |
|-------------|--------|---------|
| S3/MinIO | `S3OutputWriter` | `s3://plugin-outputs/` |
| pgvector | `PgVectorOutputWriter` | `pgvector://vectors/` |
| UMRS storage | `UMRSOutputWriter` | `umrs://plugin-outputs/` |

## Related

- Plugin source: `src/umrs/plugins/`
- Pipeline SDK: `demos/12_pipeline_sdk/` (alternative for in-process pipelines)
- Model registry: `demos/07_model_registry/`
