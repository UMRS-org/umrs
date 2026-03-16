# Demo 05 — User-Defined Model Workflow

Run custom models / processing pipelines **without modifying UMRS code**.

## Quick start

```bash
bash run_demo.sh
```

This runs 28 unit tests and an integration smoke test using a mock S3.

## How it works

1. **You write a Python script** that implements `UMRSStep`:

```python
from umrs.steps import UMRSStep, StepContext, StepResult

class MyStep(UMRSStep):
    name = "my_step"

    def setup(self, config: dict) -> None:
        self.threshold = config.get("threshold", 0.5)

    def run(self, ctx: StepContext) -> StepResult:
        output_keys = []
        for key in ctx.input_keys:
            data = ctx.download_bytes(key)
            result = self.process(data)
            out_key = f"output/{key}"
            ctx.upload_bytes(out_key, result)
            output_keys.append(out_key)
        return StepResult(output_keys=output_keys)
```

2. **You write a `workflow.json`** that declares steps, requirements, and resources:

```json
{
    "name": "my-pipeline",
    "version": "1.0",
    "source": {"type": "local_path", "path": "/home/user/my_scripts/"},
    "requirements": ["numpy", "Pillow"],
    "resources": {"gpu_count": 1, "memory_limit": "4Gi", "timeout_seconds": 600},
    "steps": [
        {"name": "my_step", "entrypoint": "my_module:MyStep", "config": {"threshold": 0.5}}
    ],
    "output": {"return_type": "file"}
}
```

3. **You reference it in UMRS config** — no code changes needed:

```json
{
    "model_name": "my-pipeline",
    "model_type": "user_script:///home/user/my_scripts/workflow.json",
    "return_type": "file",
    "vector_size": 0
}
```

## Examples in this demo

| File | Description |
|------|-------------|
| `example_scripts/invert_image.py` | Inverts images (negative effect) — demonstrates image I/O via S3 |
| `example_scripts/text_stats.py` | Computes word/line counts — demonstrates text processing |
| `workflow.json` | Image inversion workflow definition |
| `workflow_text.json` | Text statistics workflow definition |

## Multi-step workflows

Chain steps by listing multiple entries in `"steps"`. Each step's `output_keys` become the next step's `input_keys`:

```json
{
    "steps": [
        {"name": "preprocess", "entrypoint": "preproc:ResizeStep", "config": {"size": 256}},
        {"name": "infer",      "entrypoint": "model:InferStep",    "config": {"checkpoint": "/models/net.pt"}},
        {"name": "postprocess","entrypoint": "postproc:FilterStep", "config": {"threshold": 0.5}}
    ]
}
```

## Execution modes

| Mode | When | How |
|------|------|-----|
| **In-process** | `sandbox` not configured | Steps run in the UMRS server process directly |
| **Subprocess** | `sandbox.enabled = true` | Steps run in isolated child processes with resource limits |
| **Slurm** | `rivault://user_script://...` | Steps run on compute nodes via Slurm |

## StepContext helpers

| Method | Description |
|--------|-------------|
| `ctx.download_bytes(key)` | Get raw bytes from S3 |
| `ctx.download_numpy(key)` | Get NumPy array from `.npy` file |
| `ctx.upload_bytes(key, data)` | Write bytes to S3 |
| `ctx.upload_numpy(key, arr)` | Write NumPy array to S3 |
| `ctx.head_object(key)` | Get S3 metadata / ETag |

## References

- Design document: `doc/design_user_defined_models.md`
- Step interface: `src/umrs/steps/interface.py`
- Tests: `tests/test_user_steps.py`
