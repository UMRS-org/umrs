# Demo 12 — Plugin Execution System

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
- UMRS installed: `pip install -e src/`

## Quick Start

```bash
bash run_demo.sh
```

Or run directly:

```bash
python3 run_plugin_demo.py --input-dir ../../data/file_of_texts
```

## Plugin Architecture

```
PluginSpec → PluginRunner → [Sandbox] → PluginResult → ResultWriter → Outputs
                  │                          │
                  │                          └── PluginAuditLog
                  ├── SlurmPluginRunner (HPC)
                  │     └── Singularity container + sbatch
                  ├── K8sPluginRunner (Cloud)
                  │     └── Kubernetes Job + security context
                  └── Local (this demo)
                        └── subprocess with workspace dirs
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

- `/workspace/input/` — input data files
- `/workspace/output/` — write results here
- `/workspace/metrics.txt` — write key=value metrics
- Environment variables: `PLUGIN_ENTRYPOINT`, `PLUGIN_SOURCE_*`, `PLUGIN_USER_ID`

See [word_count_plugin.py](word_count_plugin.py) for a complete example.

## Execution Backends

### Slurm (HPC)
```python
from umrs.plugins import SlurmPluginRunner, PluginSpec

runner = SlurmPluginRunner(partition="gpu", singularity_image="/path/to/image.sif")
result = await runner.execute(spec)
```
- Generates sbatch script with `#SBATCH` directives
- Runs inside Singularity container with `--containall`
- Bind-mounts workspace directories
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
- Pipeline SDK: `demos/11_pipeline_sdk/` (alternative for in-process pipelines)
- Model registry: `demos/08_model_registry/`
