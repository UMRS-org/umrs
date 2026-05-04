# Demo 08 — User-Defined Model Workflow

Run custom models and processing pipelines **without modifying UMRS code**.
You write Python scripts, describe them in a `workflow.json`, and UMRS loads
and executes them automatically.

## Quick start

```bash
bash run_demo.sh
```

This runs the unit test suite and an integration smoke test using a mock S3.

## Running Manually

```bash
REPO_ROOT="$(git rev-parse --show-toplevel)/umrs"
cd "$REPO_ROOT"

PYTHON="$REPO_ROOT/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="python3"
fi

# 1. Run the step test suite.
PYTHONPATH="$REPO_ROOT/src" "$PYTHON" -m pytest tests/test_user_steps.py -v --tb=short

# 2. Run the same integration smoke test as run_demo.sh.
PYTHONPATH="$REPO_ROOT/src" SCRIPT_DIR="$REPO_ROOT/demos/08_user_model" "$PYTHON" - <<'PY'
import json
import os
from pathlib import Path
from unittest.mock import MagicMock

from umrs.pipeline_sdk.workflow import WorkflowDefinition, WorkflowPipeline, load_steps

repo = Path(os.environ["PYTHONPATH"].split(":")[0]).parent
demo = Path(os.environ["SCRIPT_DIR"])

wf_data = json.loads((demo / "workflow_text.json").read_text())
wf_data["source"]["path"] = str(demo / "example_scripts")
wf = WorkflowDefinition.from_dict(wf_data)
wf.validate()

store = {}

def put_object(Bucket, Key, Body, **kwargs):
    store[f"{Bucket}/{Key}"] = Body if isinstance(Body, bytes) else Body.encode()

def get_object(Bucket, Key):
    data = store[f"{Bucket}/{Key}"]
    body = MagicMock()
    body.read.return_value = data
    return {"Body": body, "ETag": '"aaa"', "ContentLength": len(data), "Metadata": {}}

def head_object(Bucket, Key):
    data = store[f"{Bucket}/{Key}"]
    return {"ETag": '"aaa"', "ContentLength": len(data), "Metadata": {}}

s3 = MagicMock()
s3.put_object.side_effect = put_object
s3.get_object.side_effect = get_object
s3.head_object.side_effect = head_object

store["demo-in/hello.txt"] = b"Hello World!\nThis is a test file.\n"
store["demo-in/readme.txt"] = b"UMRS user-defined workflows rock.\n"

pipeline = WorkflowPipeline.from_workflow(wf, load_steps(wf))
_, checksums, _, saved_keys = pipeline.run_as_embeddings(
    s3, ["hello.txt", "readme.txt"], "demo-in", "demo-out"
)

print(f"Saved keys: {saved_keys}")
print(f"Checksums: {checksums}")
print("Integration smoke test PASSED")
PY
```

---

## Core concepts

### `Step` — the base class for every step

Every processing step inherits from `Step` and implements three methods:

```python
from umrs.pipeline_sdk import Step, StepContext, StepResult

class MyStep(Step):
    # Unique identifier — must match the "name" field in workflow.json
    name = "my_step"

    def setup(self, config: dict) -> None:
        """One-time initialisation. Called once before any run() calls.

        Use this instead of __init__. The loader instantiates the class
        with no arguments, then calls setup(config) with the per-step
        config dict from workflow.json.
        """
        self.threshold = config.get("threshold", 0.5)
        # load model weights, open files, etc.

    def run(self, ctx: StepContext) -> StepResult:
        """Process one batch of S3 objects.

        Called once per workflow execution with all input keys.
        Must return a StepResult describing the outputs.
        """
        output_keys = []
        for key in ctx.input_keys:
            data = ctx.download_bytes(key)
            result = self._process(data)
            out_key = f"output/{key}"
            ctx.upload_bytes(out_key, result)
            output_keys.append(out_key)
        return StepResult(output_keys=output_keys)

    def teardown(self) -> None:
        """Optional cleanup. Called once after all batches finish.

        Release GPU memory, close file handles, etc.
        """
```

---

### `StepContext` — S3 access and shared state

`StepContext` is passed to every `run()` call. It wraps the MinIO S3 client
and exposes helpers so your step never needs to use boto3 directly.

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `ctx.input_bucket` | `str` | Bucket that contains the input objects |
| `ctx.input_keys` | `List[str]` | S3 keys to process in this step |
| `ctx.output_bucket` | `str` | Bucket to write results into |
| `ctx.config` | `dict` | Per-step config from `workflow.json` |
| `ctx.metadata` | `dict` | **Mutable** dict shared across all steps in the same run |
| `ctx.run_id` | `str` | Unique ID for the current workflow execution |

#### Download helpers

```python
# Raw bytes (images, binary files, JSON, …)
data: bytes = ctx.download_bytes("images/sample.png")

# Download from a different bucket
data: bytes = ctx.download_bytes("images/sample.png", bucket="other-bucket")

# NumPy array from a .npy file
arr: np.ndarray = ctx.download_numpy("embeddings/feat.npy")
```

#### Upload helpers

```python
# Raw bytes — returns "s3://<output_bucket>/<key>"
uri = ctx.upload_bytes("results/output.json", json_bytes, content_type="application/json")

# Raw bytes with S3 user metadata
uri = ctx.upload_bytes("results/output.png", png_bytes,
                       content_type="image/png",
                       metadata={"source": "volume_170.png", "step": "1"})

# NumPy array — serialised as .npy, content_type="application/x-npy"
uri = ctx.upload_numpy("embeddings/feat.npy", np.array([...]))

# Write to a different bucket
ctx.upload_bytes("key", data, bucket="archive-bucket")
```

#### Metadata helper

```python
# S3 HEAD — returns ETag, ContentLength, Metadata, etc.
head = ctx.head_object("images/sample.png")
etag = head["ETag"].strip('"')
```

#### Sharing state between steps via `ctx.metadata`

`ctx.metadata` is a plain `dict` that persists for the lifetime of one
workflow run and is passed to every step. Use it to forward information
that doesn't fit naturally into S3 keys:

```python
# Step 1 — store source info for downstream steps
def run(self, ctx: StepContext) -> StepResult:
    ctx.metadata["original_keys"] = list(ctx.input_keys)
    ctx.metadata["original_bucket"] = ctx.input_bucket
    ...

# Step 2 — read what Step 1 stored
def run(self, ctx: StepContext) -> StepResult:
    orig_keys = ctx.metadata.get("original_keys", ctx.input_keys)
    ...
```

---

### `StepResult` — the output of a step

Every `run()` must return a `StepResult`. The executor chains steps by
feeding each step's `output_keys` into the next step's `ctx.input_keys`.

```python
return StepResult(
    # Required: S3 keys produced by this step.
    # These become the next step's ctx.input_keys.
    output_keys=["results/a.json", "results/b.json"],

    # Optional: N×D embedding matrix (numpy.ndarray).
    # The last step that provides embeddings has them stored in pgvector.
    embeddings=np.stack(all_cls_vectors),  # shape (N, 768)

    # Optional: per-key ETag checksums for idempotency.
    checksums=["abc123", "def456"],

    # Optional: per-key metadata dicts (forwarded to the database).
    metadata=[{"source": "a.png"}, {"source": "b.png"}],

    # Optional: arbitrary key-value bag for logging / metrics.
    extra={"total_processed": 2, "errors": 0},
)
```

---

## Writing a custom step — worked examples

### Example 1: Image processing (no ML)

```python
# example_scripts/invert_image.py
from umrs.pipeline_sdk import Step, StepContext, StepResult

class InvertImageStep(Step):
    name = "invert"

    def setup(self, config: dict) -> None:
        self.output_prefix = config.get("output_prefix", "inverted/")

    def run(self, ctx: StepContext) -> StepResult:
        from PIL import Image, ImageOps
        import io

        output_keys = []
        for key in ctx.input_keys:
            raw = ctx.download_bytes(key)
            img = ImageOps.invert(Image.open(io.BytesIO(raw)).convert("RGB"))
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            out_key = f"{self.output_prefix}{key}"
            ctx.upload_bytes(out_key, buf.getvalue(), content_type="image/png")
            output_keys.append(out_key)
        return StepResult(output_keys=output_keys)
```

### Example 2: Call the BentoML vision API

Steps can call any HTTP API during `run()`. This is how the HMAE and SAM3
demos integrate with the BentoML vision service running on the Rivault stack.
All vision models use the unified `/infer` endpoint:

```python
import time, requests
from umrs.pipeline_sdk import Step, StepContext, StepResult

class MyVisionStep(Step):
    name = "my_vision"

    def setup(self, config: dict) -> None:
        # BentoML service URL, e.g. "http://localhost:11434/vision"
        self.bentoml_url = config["bentoml_url"].rstrip("/")
        self.model_type = config.get("model_type", "hmae")
        self.model_config = config.get("model_config", {"task": "feature_extraction"})
        self.timeout = config.get("timeout", 300)

    def run(self, ctx: StepContext) -> StepResult:
        session = requests.Session()
        output_keys = []

        for key in ctx.input_keys:
            image_url = f"s3://{ctx.input_bucket}/{key}"

            # 1. Submit task to the unified /infer endpoint
            resp = session.post(
                f"{self.bentoml_url}/infer/submit",
                json={
                    "model_type": self.model_type,
                    "model_config": self.model_config,
                    "image_url": image_url,
                },
                timeout=30,
            )
            resp.raise_for_status()
            task_id = resp.json()["task_id"]

            # 2. Poll until done
            deadline = time.monotonic() + self.timeout
            while time.monotonic() < deadline:
                status_resp = session.get(
                    f"{self.bentoml_url}/infer/status",
                    params={"task_id": task_id}, timeout=10,
                )
                status = status_resp.json().get("status", "")
                if status in ("success", "completed"):
                    break
                if status in ("failure", "failed"):
                    raise RuntimeError(f"Task {task_id} failed")
                time.sleep(2.0)

            # 3. Retrieve result
            result = session.get(
                f"{self.bentoml_url}/infer/get",
                params={"task_id": task_id}, timeout=30,
            ).json()

            # 4. Store metadata in S3
            import json
            out_key = f"results/{key}.json"
            ctx.upload_bytes(out_key, json.dumps(result).encode(),
                             content_type="application/json")
            output_keys.append(out_key)

        return StepResult(output_keys=output_keys)
```

### Example 3: Return embeddings for pgvector storage

```python
import numpy as np
from umrs.pipeline_sdk import Step, StepContext, StepResult

class EmbedStep(Step):
    name = "embed"

    def run(self, ctx: StepContext) -> StepResult:
        all_vectors = []
        output_keys = []

        for key in ctx.input_keys:
            raw = ctx.download_bytes(key)
            vector = self._encode(raw)           # → np.ndarray shape (768,)
            all_vectors.append(vector)

            emb_key = f"embeddings/{key}.npy"
            ctx.upload_numpy(emb_key, vector)
            output_keys.append(emb_key)

        return StepResult(
            output_keys=output_keys,
            embeddings=np.stack(all_vectors),    # shape (N, 768)
        )
```

---

## `workflow.json` — pipeline definition

```jsonc
{
    "name": "my-pipeline",          // Unique workflow name
    "version": "1.0",
    "description": "...",

    // Where your Python scripts live
    "source": {
        "type": "local_path",       // "local_path" | "s3"
        "path": "/home/user/my_scripts/"
    },

    // pip packages to install before running steps
    "requirements": ["numpy", "Pillow", "requests"],

    // Compute resources
    "resources": {
        "gpu_count": 1,
        "cpu_limit": "4",
        "memory_limit": "16Gi",
        "timeout_seconds": 3600
    },

    // Ordered list of processing steps
    "steps": [
        {
            "name": "preprocess",
            // "module_file:ClassName" — module filename without .py
            "entrypoint": "preproc:ResizeStep",
            // Passed to setup(config) when the step is instantiated
            "config": {"target_size": 1024}
        },
        {
            "name": "infer",
            "entrypoint": "model:InferStep",
            "config": {"bentoml_url": "http://localhost:11434/vision"}
        }
    ],

    // Final output type
    "output": {
        "return_type": "file",      // "file" | "vector"
        "embedding_vector_size": 0  // non-zero when return_type == "vector"
    }
}
```

---

## How UMRS loads and runs your scripts

```
workflow.json ──► ScriptLoader.load_workflow()  reads + validates JSON
                      │
                  ScriptLoader.load_steps()
                      │  for each step:
                      │    • sys.path.insert(0, source.path)
                      │    • importlib.import_module("module_file")
                      │    • cls = getattr(module, "ClassName")
                      │    • assert issubclass(cls, Step)
                      │    • instance = cls(); instance.setup(config)
                      ▼
              WorkflowExecutor.run(s3, input_keys, input_bucket, output_bucket)
                      │
                      │  step 0:  ctx.input_keys  = original input_keys
                      │            ctx.input_bucket = input_bucket
                      │            result = step0.run(ctx)
                      │
                      │  step 1:  ctx.input_keys  = result.output_keys   ◄── chaining
                      │            ctx.input_bucket = output_bucket
                      │            result = step1.run(ctx)
                      │
                      │  step N:  final result.output_keys, embeddings,
                      │            checksums, metadata returned to UMRS
                      ▼
              stored in pgvector / returned to client
```

Intermediate step outputs are written under `_staging/<run_id>/step_<i>_<name>/`
inside the output bucket, then the final step writes to the bucket root.

---

## Registering your workflow in UMRS

Add it as a modality in the user config JSON:

```json
{
    "modality_name": "image",
    "s3": {
        "bucket_alias": "my-results",
        "data": { "type": "file", "mime_type": "image/png" },
        "need_output_bucket": true
    },
    "model": {
        "model_name": "My Pipeline",
        "model_type": "user_script:///home/user/my_scripts/workflow.json",
        "vector_size": "0",
        "return_type": "file"
    },
    "sql": {
        "store_embeddings": "no",
        "table_name": "my_pipeline_results",
        "meta_data": []
    }
}
```

After adding the entry, activate it using the mechanism supported by your site.
On local/dev deployments that may be `umrs-client reload` or a server restart.
On operator-managed sites, ask the site/operator to activate the updated config.
The server will call your workflow whenever new objects arrive in the
`my-results` input bucket.

---

## BentoML vision API endpoints

All vision models are served through a single unified `/infer` endpoint. When your
step calls the Rivault BentoML vision service, use the `model_type` and
`model_config` fields to select the model and mode:

### Submit a task

```
POST /vision/infer/submit
     Body: {"model_type": "sam3" | "hmae",
            "model_config": { ... adapter-specific params ... },
            "image_url": "s3://bucket/key",
            "output_bucket": "results"    (optional)}
     → {"task_id": "..."}
```

### Poll status / retrieve result

```
GET  /vision/infer/status?task_id=...
     → {"status": "in_progress|completed|failed"}

GET  /vision/infer/get?task_id=...
     → { ... adapter-specific result ... }
```

### SAM3 segmentation example

```json
{"model_type": "sam3",
 "model_config": {"prompt": "defect", "threshold": 0.5, "mask_threshold": 0.5},
 "image_url": "s3://bucket/key.png"}
```

**Result**: `{"num_objects": N, "boxes": [...], "scores": [...], "overlay_s3_key": "s3://..."}`

### HMAE inference example (3 tasks)

```json
{"model_type": "hmae",
 "model_config": {"task": "reconstruction"},
 "image_url": "s3://bucket/key.png"}
```

`task` can be `"reconstruction"`, `"mae_reconstruction"`, or `"feature_extraction"`.

**Results by task**:
- `reconstruction` → `{"output_s3_key": "s3://..."}`
- `mae_reconstruction` → `{"output_s3_key": "s3://...", "mae_loss": 0.18}`
- `feature_extraction` → `{"output_s3_key": "s3://...features.pt", "cls_shape": [1, 768], "patch_shape": [1024, 768]}`

### Health check

```
GET /vision/healthz   → HTTP 200 when the service is ready
```

**Base URL**: `http://localhost:11434/vision` (via NGINX on the Rivault stack).

Task status values: `in_progress`, `completed`, `failed`, `cancelled`.
Always accept both `"success"` and `"completed"` for backward compatibility.

---

## Execution modes

| Mode | `workflow.json` key | Description | Status |
|------|---------------------|-------------|--------|
| **In-process** | *(default)* | Steps run directly in the UMRS server process | Production-ready |
| **Slurm** | `"execution_mode": "slurm"` | Steps run as Slurm batch jobs in Singularity containers on HPC. Each step is wrapped into a `PluginSpec`, submitted via `SlurmPluginRunner`, and executed by `python -m umrs.plugins.plugin_runner.entrypoint` inside the container. Results are chained between steps via `result_manifest.json`. | Functional (Phase 1 complete) |
| **Kubernetes** | `"execution_mode": "k8s"` | Kubernetes Job with security context. Same step executor and result manifest contract as Slurm. | Structural (same contract, needs cluster validation) |
| **Subprocess** | `"execution_mode": "subprocess"` | Same-host isolated sandbox. Not yet implemented — raises `NotImplementedError`. | Planned |

---

## Files in this demo

| File | Description |
|------|-------------|
| `example_scripts/invert_image.py` | Image inversion step — minimal image I/O example |
| `example_scripts/text_stats.py` | Text statistics step — word/line/char counts |
| `example_adapters/grayscale_adapter.py` | Custom VisionModelAdapter — grayscale conversion |
| `workflow.json` | Image inversion workflow |
| `workflow_text.json` | Text statistics workflow |

---

## Custom Vision Adapters

In addition to embedding pipelines (`user_script://`), you can add **custom
vision adapters** to the BentoML vision service without modifying UMRS code.

### How it works

The BentoML vision service scans directories listed in `VISION_ADAPTERS_DIR`
(colon-separated) for Python files defining `VisionModelAdapter` subclasses.
Each adapter is registered by filename stem (e.g., `grayscale_adapter.py` →
`model_type="grayscale_adapter"`).

Built-in adapters (SAM3, HMAE) are loaded from `rivault/default_adapters/`
using the same mechanism — no special treatment.

### Writing a custom adapter

```python
# my_adapter.py
from umrs.vision_service.adapters.base import VisionModelAdapter

class MyAdapter(VisionModelAdapter):
    def cache_key(self, config):
        return f"my_model:{config.get('checkpoint', 'default')}"

    def load(self, config):
        # Load model weights, return opaque model object
        ...

    def infer(self, model, image, bucket, key, config, s3, output_bucket):
        # Run inference, upload results to S3, return result dict
        ...
```

### Deploying your adapter

1. Place your `.py` file(s) in a directory on the compute node
2. Optionally add a `requirements.txt` for extra dependencies
3. Set `USER_ADAPTERS_DIR` before launching the Slurm job:

```bash
export USER_ADAPTERS_DIR=/home/user/my_adapters
sbatch serve-model.slurm
```

4. Use your adapter's `model_type` in requests:

```json
{
    "model_type": "my_adapter",
    "inference_config": {"checkpoint": "/models/user/weights.pth"},
    "image_url": "s3://bucket/image.png"
}
```

### Example: Grayscale adapter

See [`example_adapters/grayscale_adapter.py`](example_adapters/grayscale_adapter.py)
for a complete working example.

### Two extension models compared

| | Embedding Pipeline (`user_script://`) | Vision Adapter (`VisionModelAdapter`) |
|--|------|------|
| **Runs inside** | UMRS server process | BentoML container (GPU) |
| **Best for** | Multi-step processing pipelines | Single-model GPU inference |
| **Config** | `workflow.json` + Step classes | Single `.py` file |
| **Dependencies** | Listed in `workflow.json` | `requirements.txt` in adapter dir |
| **Model type** | `user_script:///path/to/workflow.json` | Adapter filename stem |

## See also

- [Demo 09: Multi-Step Vision Pipeline](../09_multistep_vision_pipeline/README.md) — public multi-step image workflow
- [Demo 10: Vision Multi-Mode](../10_vision_multi_mode/README.md) — branched workflow with multiple output modes
- [Step ABC source](../../src/umrs/pipeline_sdk/step.py)
- [WorkflowPipeline source](../../src/umrs/pipeline_sdk/workflow.py)
- [VisionModelAdapter base](../../src/umrs/vision_service/adapters/base.py)
- [Design document](../../doc/design_user_defined_models.md)
- [Unit tests](../../tests/test_user_steps.py)
