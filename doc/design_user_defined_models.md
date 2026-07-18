# Design: User-Defined Model Workflows (UDM)

> **Status**: Implemented; current paths verified May 2026

---

## 1. Problem Statement

Before the user-defined workflow path existed, adding a new model to UMRS
required modifying internal code in multiple files:

1. **`src/umrs/inference/loading.py`** — add an `elif` branch in `load_model()`
2. **`bentoml_adapter.py`** — write a `make_bentoml_<name>_compute()` wrapper
3. **`rivault_launcher.py`** — add endpoint detection logic
4. **`vision_service/service.py`** — add a `@bentoml.task` method
5. Rebuild the Docker image and redeploy

This means **every new model = a code change + image rebuild + redeployment**,
even when the model is a straightforward Python script that reads an image,
runs inference, and writes output.

### What users want

- Define a processing job as **a Python script** in their own workspace
- Optionally chain multiple scripts into a **multi-step workflow** (e.g., preprocess → infer → postprocess)
- Have UMRS **schedule and monitor** the job using existing infrastructure (Slurm / K8s / Compose)
- Read inputs from S3, write outputs to S3 — using the same data plane
- **Never touch UMRS source code**

---

## 2. Design Principles

| # | Principle | Rationale |
|---|-----------|-----------|
| 1 | **Script-as-config** | A model is just a Python entry point + a JSON workflow definition — no UMRS code changes |
| 2 | **Same data plane** | I/O goes through S3 (MinIO); results land in the same buckets/DB that UMRS already manages |
| 3 | **Same orchestration** | Jobs run through the existing Orchestrator backends (Slurm, K8s, Compose) |
| 4 | **Sandboxed execution** | User code runs in isolated environments with resource limits and audit trails |
| 5 | **Leverage existing infra** | Reuse `PluginSpec`, `PluginRunner`, `PipelineDefinition`, and `OrchestrationService` — don't reinvent |
| 6 | **Incremental adoption** | Existing hardcoded models keep working; UDM is an opt-in addition |

---

## 3. Architecture Overview

```
                          ┌──────────────────────────────────────────┐
                          │           USER SPACE                     │
                          │                                          │
                          │  workflow.json          my_model.py      │
                          │  ┌──────────────┐      ┌──────────────┐  │
                          │  │ steps:       │      │ class MyStep │  │
                          │  │  - preprocess│      │  (Step)  │  │
                          │  │  - infer     │      │  def run():  │  │
                          │  │  - postproc  │      │    ...       │  │
                          │  └──────┬───────┘      └──────────────┘  │
                          └─────────┼────────────────────────────────┘
                                    │ upload / mount
                                    ▼
┌───────────────────────────────────────────────────────────────────────┐
│                          UMRS SERVER                                  │
│                                                                       │
│  src/umrs/inference/loading.py                                        │
│  ┌─────────────────────────────────────────────────────────┐          │
│  │ model_type == "user_script://"                          │          │
│  │   → ScriptLoader.load(source)                           │          │
│  │   → WorkflowExecutor.create(workflow_def, loaded_steps) │          │
│  │   → Return EmbeddingModel(compute=executor.run)         │          │
│  └─────────────────────────────────────────────────────────┘          │
│                              │                                        │
│                              ▼                                        │
│  ┌─────────────────────────────────────────────┐                      │
│  │ WorkflowExecutor                            │                      │
│  │                                             │                      │
│  │  for step in workflow.steps:                │                      │
│  │    ctx.input_keys = previous.output_keys    │                      │
│  │    step_instance = loaded_steps[step.name]  │                      │
│  │    step_instance.run(ctx)                   │                      │
│  │    ctx.output_keys = step_instance.outputs  │                      │
│  └──────────────┬──────────────────────────────┘                      │
│                 │                                                      │
│     ┌───────────┴───────────────┐                                     │
│     │ S3 (MinIO)                │  ← same data plane                  │
│     │ input:  user-bucket/raw/  │                                     │
│     │ output: user-bucket/out/  │                                     │
│     └───────────────────────────┘                                     │
│                                                                       │
│  Orchestrator (Slurm / K8s / Compose)                                 │
│     → PluginRunner executes each step in sandboxed env                │
└───────────────────────────────────────────────────────────────────────┘
```

---

## 4. User-Facing Interface

### 4.1 Step Interface (`Step` ABC)

Users implement this Python interface for each processing step:

```python
# --- file: my_model.py (user-provided) ---
from umrs.pipeline_sdk import Step, StepContext, StepResult

class PreprocessStep(Step):
    """Resize and normalize images before inference."""

    name = "preprocess"

    def setup(self, config: dict) -> None:
        """Called once at initialization. Load models, configs, etc."""
        self.target_size = config.get("target_size", 1024)

    def run(self, ctx: StepContext) -> StepResult:
        """Process one batch of S3 objects.

        Args:
            ctx.s3: boto3 S3 client (MinIO)
            ctx.input_bucket: str — source bucket
            ctx.input_keys: List[str] — S3 keys to process
            ctx.output_bucket: str — destination bucket
            ctx.config: dict — step-specific config from workflow.json
            ctx.metadata: dict — carries metadata across steps

        Returns:
            StepResult with output_keys and optional embeddings/metadata
        """
        results = []
        for key in ctx.input_keys:
            img = ctx.download(key)                    # helper → S3 get
            processed = self._resize(img)
            out_key = f"preprocessed/{key}"
            ctx.upload(out_key, processed)             # helper → S3 put
            results.append(out_key)

        return StepResult(
            output_keys=results,
            metadata={"step": "preprocess", "count": len(results)},
        )

class InferenceStep(Step):
    name = "infer"

    def setup(self, config: dict) -> None:
        import torch
        self.model = torch.load(config["checkpoint_path"])
        self.device = torch.device(config.get("device", "cuda"))
        self.model.to(self.device).eval()

    def run(self, ctx: StepContext) -> StepResult:
        embeddings = []
        for key in ctx.input_keys:
            img = ctx.download(key)
            tensor = self._to_tensor(img).to(self.device)
            with torch.no_grad():
                emb = self.model(tensor)
            embeddings.append(emb.cpu().numpy())

        return StepResult(
            embeddings=np.stack(embeddings),
            output_keys=ctx.input_keys,  # pass-through
        )
```

### 4.2 Workflow Definition (`workflow.json`)

```json
{
    "name": "my-ct-pipeline",
    "version": "1.0",
    "description": "CT volume preprocessing + custom HMAE inference + feature extraction",

    "source": {
        "type": "local_path",
        "path": "/home/user/my_models/"
    },

    "requirements": [
        "torch>=2.0",
        "numpy",
        "Pillow"
    ],

    "env": {
        "CUDA_VISIBLE_DEVICES": "0"
    },

    "resources": {
        "gpu_count": 1,
        "cpu_limit": "4",
        "memory_limit": "8Gi",
        "timeout_seconds": 3600
    },

    "steps": [
        {
            "name": "preprocess",
            "entrypoint": "my_model:PreprocessStep",
            "config": {
                "target_size": 1024,
                "normalize": true
            }
        },
        {
            "name": "infer",
            "entrypoint": "my_model:InferenceStep",
            "config": {
                "checkpoint_path": "/models/my_checkpoint.pth",
                "device": "cuda"
            }
        },
        {
            "name": "postprocess",
            "entrypoint": "my_model:PostprocessStep",
            "config": {
                "threshold": 0.5
            }
        }
    ],

    "output": {
        "return_type": "file",
        "embedding_vector_size": 768
    }
}
```

### 4.3 Model Config (user's `vision_models.json` or bucket config)

```json
{
    "model_name": "my-ct-pipeline",
    "model_type": "user_script:///home/user/my_models/workflow.json",
    "return_type": "file",
    "embedding_vector_size": 768
}
```

**Alternative source schemes:**

| Scheme | Example | How it works |
|--------|---------|--------------|
| `user_script:///path/to/workflow.json` | Local mount | Direct path on host (dev, Slurm bind-mount) |
| `user_script://s3://bucket/workflow.json` | S3 | Downloaded from MinIO user space at load time |
| `user_script://git://github.com/user/repo@commit` | Git | Cloned at specific commit (via existing `PluginSpec.GIT_REPO`) |

---

## 5. Internal Architecture

### 5.1 New Components

| Component | Location | Purpose |
|-----------|----------|---------|
| `Step` ABC | `src/umrs/pipeline_sdk/step.py` | Contract for user scripts |
| `StepContext` | `src/umrs/pipeline_sdk/step.py` | S3 helpers + metadata carrier |
| `StepResult` | `src/umrs/pipeline_sdk/step.py` | Output keys + embeddings + metadata |
| `WorkflowPipeline` | `src/umrs/pipeline_sdk/workflow.py` | Loads workflow.json, imports steps, chains execution |
| `EmbeddingAdapter` | `src/umrs/pipeline_sdk/adapter.py` | Bridges WorkflowPipeline → compute_embeddings contract |
| `WorkflowDefinition` | `src/umrs/pipeline_sdk/workflow.py` | Data model for `workflow.json` |
| `VisionModelAdapter` | `src/umrs/vision_service/adapters/base.py` | ABC for pluggable vision models (BentoML) |

### 5.2 Integration Points (minimal changes to existing code)

#### A. `src/umrs/inference/loading.py` — implemented loader branches

```python
elif model_type.startswith("user_script://"):
    from umrs.pipeline_sdk.adapter import EmbeddingAdapter
    from umrs.pipeline_sdk.workflow import WorkflowPipeline

    source_uri = model_type[len("user_script://"):]
    pipeline = WorkflowPipeline(source_uri)
    adapter = EmbeddingAdapter(pipeline)
    compute_embeddings_func = adapter.compute_embeddings

elif model_type.startswith("user_workflow://"):
    from umrs.inference.plugins.user_workflow import UserWorkflow, UserWorkflowRunner
    from umrs.pipeline_sdk.workflow import WorkflowPipeline

    source_uri = model_type[len("user_workflow://"):]
    pipeline = WorkflowPipeline(source_uri)
    workflow = UserWorkflow.from_workflow_definition(pipeline.workflow)
    compute_embeddings_func = UserWorkflowRunner(workflow).run
```

#### B. `src/umrs/inference/embeddings.py` — no changes needed

`compute_and_store()` already calls `model.compute_embeddings(s3, keys, bucket, out_bucket)` generically.
The `WorkflowExecutor.run()` method implements that exact signature.

#### C. `rivault_launcher.py` — optional integration for remote execution

For GPU-backed user workflows running on Slurm, the `user_script://` type can **optionally** be wrapped with `rivault://`:

```
rivault://user_script:///home/user/my_models/workflow.json
```

This triggers the standard Slurm job + SSH tunnel flow, but instead of starting BentoML, the job would:
1. Set up an isolated Python virtualenv with the user's `requirements`
2. Mount the user's script directory
3. Execute the `WorkflowExecutor` as a subprocess

This reuses the existing `PluginRunner` (Slurm or K8s backend).

### 5.3 Execution Modes

| Mode | Trigger | Where code runs | GPU access |
|------|---------|-----------------|------------|
| **In-process** | `user_script:///path/...` | UMRS server process (dev only) | Server GPU |
| **Subprocess** | `user_script:///path/...` + `sandbox: true` | Isolated subprocess with venv | Server GPU |
| **Slurm job** | `rivault://user_script:///path/...` | Compute node via sbatch | Dedicated GPU |
| **K8s job** | `k8s://user_script://s3://...` | K8s pod | Scheduled GPU |

### 5.4 Step Sequencing

```
WorkflowExecutor.run(s3, s3_keys, input_bucket, output_bucket)
│
├─ Step 1: preprocess
│   ctx.input_keys = s3_keys (original inputs)
│   ctx.output_bucket = <staging>/step_0_preprocess/
│   step.run(ctx) → StepResult(output_keys=[...])
│
├─ Step 2: infer
│   ctx.input_keys = step_1.output_keys (chained!)
│   ctx.input_bucket = <staging>/step_0_preprocess/
│   ctx.output_bucket = <staging>/step_1_infer/
│   step.run(ctx) → StepResult(embeddings=np.array, output_keys=[...])
│
├─ Step 3: postprocess
│   ctx.input_keys = step_2.output_keys
│   ctx.output_bucket = output_bucket (final destination)
│   step.run(ctx) → StepResult(output_keys=[...])
│
└─ Return:
    embeddings = last step with embeddings (or None)
    saved_keys = final step.output_keys
    checksums, metadata = aggregated from all steps
```

**Staging bucket**: Each step's intermediate outputs go to a namespaced prefix within the user's output bucket:
```
s3://output-bucket/_staging/<workflow_run_id>/step_0_preprocess/...
s3://output-bucket/_staging/<workflow_run_id>/step_1_infer/...
s3://output-bucket/final/...   ← last step writes here
```

Optional: staging artifacts are cleaned up after successful completion (configurable).

---

## 6. ScriptLoader — Source Resolution & Validation

### 6.1 Load Flow

```
ScriptLoader.load_workflow(source_uri)
  │
  ├─ Local path: read workflow.json directly
  ├─ S3: download workflow.json + script directory to temp dir
  └─ Git: clone at commit, read workflow.json
  │
  ├─ Parse JSON → WorkflowDefinition
  ├─ Validate schema (required fields, allowed values)
  ├─ Validate entrypoints (module:class format)
  └─ Return WorkflowDefinition

ScriptLoader.load_steps(workflow_def)
  │
  ├─ For each step in workflow_def.steps:
  │   ├─ Resolve entrypoint "my_model:InferenceStep"
  │   │   → importlib.import_module("my_model")
  │   │   → getattr(module, "InferenceStep")
  │   ├─ Verify class inherits Step
  │   ├─ Instantiate, call setup(step.config)
  │   └─ Collect into ordered list
  └─ Return List[Step]
```

### 6.2 Security Constraints

| Check | What | When |
|-------|------|------|
| **Schema validation** | All required fields present, types correct | Load time |
| **Source pinning** | Git sources MUST specify commit hash (not branch) | Load time |
| **Entrypoint format** | Must be `module:ClassName`, no shell injection | Load time |
| **ABC compliance** | Class must inherit `Step` | Load time |
| **Resource limits** | CPU/memory/timeout enforced by Slurm/K8s/cgroups | Runtime |
| **No network** (optional) | Sandboxed mode blocks outbound (`--network=none`) | Runtime |
| **Filesystem isolation** | Read-only mount for scripts, write only to S3 staging | Runtime |
| **Audit trail** | All executions logged via existing `PluginAuditEntry` | Runtime |

---

## 7. Relationship to Existing Subsystems

### 7.1 Reuse from Plugins (`src/umrs/plugins/`)

| Existing | Reused for |
|----------|------------|
| `PluginSpec` | `WorkflowDefinition` wraps multiple steps, each is essentially a `PluginSpec` |
| `PluginRunner` (Slurm) | Execute user steps as Slurm jobs with Singularity |
| `PluginRunner` (K8s) | Execute user steps as K8s Jobs |
| `PluginResult` | Each `StepResult` maps to a `PluginResult` |
| `PluginAuditEntry` | Every step execution is audited |
| `S3OutputWriter` | Step outputs written to S3 |
| `PgVectorOutputWriter` | Embeddings stored in pgvector |

### 7.2 Reuse from Pipelines (`src/umrs/pipelines/`)

| Existing | Reused for |
|----------|------------|
| `PipelineDefinition.pre_processors` | Mapped to workflow steps |
| `PipelineDefinition.post_processors` | Mapped to workflow steps |
| `EmbedderSpec.CUSTOM` type | User script is the "custom" embedder |
| `ChunkerSpec` / `RerankerSpec` | Can be referenced from workflow steps (optional) |

### 7.3 Reuse from Orchestrator (`src/umrs/orchestrator/`)

| Existing | Reused for |
|----------|------------|
| `OrchestrationService.get_or_provision()` | Provision compute for user workflows |
| `DurableOrchestrationService` | Crash-safe execution of multi-step workflows |
| Backend auto-detection | Same detection for where to run user scripts |
| `Janitor` | Clean up expired workflow staging artifacts |

---

## 8. Example End-to-End Flow

### User's perspective:

```bash
# 1. User writes their model script
cat > ~/my_models/denoise.py << 'EOF'
from umrs.pipeline_sdk import Step, StepContext, StepResult
import numpy as np

class DenoiseStep(Step):
    name = "denoise"

    def setup(self, config):
        from skimage.restoration import denoise_nl_means
        self.h = config.get("h", 0.1)

    def run(self, ctx):
        output_keys = []
        for key in ctx.input_keys:
            img = ctx.download_numpy(key)
            denoised = denoise_nl_means(img, h=self.h)
            out_key = f"denoised/{key}"
            ctx.upload_numpy(out_key, denoised)
            output_keys.append(out_key)
        return StepResult(output_keys=output_keys)
EOF

# 2. User writes workflow definition
cat > ~/my_models/workflow.json << 'EOF'
{
    "name": "denoise-pipeline",
    "version": "1.0",
    "source": {"type": "local_path", "path": "/home/user/my_models/"},
    "requirements": ["scikit-image>=0.20"],
    "resources": {"gpu_count": 0, "memory_limit": "4Gi", "timeout_seconds": 600},
    "steps": [
        {
            "name": "denoise",
            "entrypoint": "denoise:DenoiseStep",
            "config": {"h": 0.1}
        }
    ],
    "output": {"return_type": "file"}
}
EOF

# 3. User adds to their UMRS model config
# In their user config JSON:
#   "model_type": "user_script:///home/user/my_models/workflow.json"

# 4. Upload data, UMRS processes using their script
umrs-client data dir-upload --bucket my-images ~/data/scans/
umrs-client data update  # triggers processing via user's denoise pipeline
```

### What happens internally:

1. `src/umrs/inference/loading.py` sees `user_script://` or `user_workflow://`
    and creates a `WorkflowPipeline`.
2. `WorkflowPipeline` reads `/home/user/my_models/workflow.json`, validates the
    schema, imports `denoise.py`, verifies `DenoiseStep(Step)`, and calls `setup()`.
3. `EmbeddingAdapter` or `UserWorkflowRunner` wraps the step chain into a
    `compute_embeddings()`-compatible function.
4. `EmbeddingModel` is returned with `type="user_script"` or `type="user_workflow"`.
6. When S3 objects arrive, `compute_and_store()` calls `executor.run(s3, keys, bucket, out_bucket)`
7. `DenoiseStep.run()` downloads each image, denoises, uploads to output bucket
8. Results stored in DB + S3 as usual

---

## 9. Multi-Step Workflow Example (Vision + Embedding)

```json
{
    "name": "ct-analysis-pipeline",
    "version": "2.0",
    "source": {"type": "local_path", "path": "/home/user/ct_pipeline/"},
    "requirements": ["torch>=2.0", "numpy", "Pillow", "scikit-image"],
    "resources": {"gpu_count": 1, "memory_limit": "16Gi", "timeout_seconds": 7200},
    "steps": [
        {
            "name": "normalize",
            "entrypoint": "preprocessing:NormalizeCTStep",
            "config": {"window_center": 40, "window_width": 400}
        },
        {
            "name": "segment",
            "entrypoint": "segmentation:SegmentStep",
            "config": {"checkpoint": "/models/unet_ct.pth", "threshold": 0.5}
        },
        {
            "name": "extract_features",
            "entrypoint": "features:FeatureExtractStep",
            "config": {"checkpoint": "/models/resnet_ct.pth", "pool": "avg"}
        }
    ],
    "output": {
        "return_type": "file",
        "embedding_vector_size": 2048
    }
}
```

Step chaining:
```
S3 keys (raw CT scans)
  → [normalize] → s3://.../staging/<run>/step_0/normalized_*.npy
  → [segment]   → s3://.../staging/<run>/step_1/mask_*.png
  → [features]  → s3://.../staging/<run>/step_2/features_*.pt
  → Final: embeddings stored in pgvector, files in output bucket
```

---

## 10. Implementation Plan

### Phase 1: Core Interface & In-Process Execution (U-201 through U-205) ✅

**Goal**: Users can define single-step and multi-step workflows that run in the UMRS server process.

| Task | What | Files | Status |
|------|------|-------|--------|
| U-201 | `Step` ABC, `StepContext`, `StepResult` | `src/umrs/pipeline_sdk/step.py`, `context.py` | ✅ |
| U-202 | `WorkflowDefinition` model + JSON schema validation | `src/umrs/pipeline_sdk/models.py` | ✅ |
| U-203 | `ScriptLoader` (local path first, then S3/git) | `src/umrs/pipeline_sdk/loader.py` | ✅ |
| U-204 | `WorkflowExecutor` with step chaining + S3 staging | `src/umrs/pipeline_sdk/executor.py` | ✅ |
| U-205 | `user_script://` and `user_workflow://` model loading | `src/umrs/inference/loading.py`, `src/umrs/pipeline_sdk/workflow.py` | ✅ |

### Phase 2: Sandboxed & Remote Execution (U-206) ✅

**Goal**: User scripts run in isolated environments with resource limits.

| Sub-task | What | Status |
|----------|------|--------|
| U-206a | Subprocess sandbox (`SubprocessSandbox` in `sandbox.py`) | ✅ |
| U-206b | Slurm execution (`SlurmStepRunner` in `slurm_runner.py`) | ✅ |
| U-206c | K8s execution (`K8sStepRunner` in `k8s_runner.py`) | ✅ |
| U-206d | Resource limit enforcement + timeout | ✅ |
| U-206e | Audit trail integration (`StepAuditLog`) | ✅ |

Execution mode is selected via the `ExecutionMode` enum and configured in
`workflow.json` → `resources.extra.execution_mode` or the model config's
`execution_mode` field.  Four modes: `in_process`, `subprocess`, `slurm`, `k8s`.

### Phase 3: Demo & Docs (U-207, U-208) ✅

| Task | What | Status |
|------|------|--------|
| U-207 | Demo: `demos/07_user_model/` with example scripts | ✅ |
| U-208 | User guide section + example script templates | ✅ |

### Phase 4: Staging Cleanup ✅

Intermediate artifacts under `_staging/<run_id>/` are automatically deleted
after a successful workflow run.  Cleanup is best-effort and non-fatal.

---

## 11. Open Questions

| # | Question | Options | Recommendation |
|---|----------|---------|----------------|
| 1 | Should user scripts run **in-process** or always **subprocess**? | In-process (faster, simpler for dev), subprocess (safer), configurable | Default subprocess in production, in-process for `sandbox: false` in dev |
| 2 | How to handle **GPU sharing**? One step at a time or parallel? | Sequential (safe), parallel (fast but complex) | Sequential per workflow; parallelism across workflows is handled by orchestrator |
| 3 | Should intermediate staging artifacts be **auto-cleaned**? | Always clean, configurable retention, never clean | Configurable: `staging_retention: "on_success"` / `"always"` / `"never"` (default: `"on_success"`) |
| 4 | Should `Step` support **streaming** (one object at a time) or **batch**? | Batch (current), streaming, both | Batch first (matches existing compute_embeddings contract); add streaming later |
| 5 | How to handle **step failures** mid-workflow? | Fail entire workflow, retry step, skip step | Fail workflow by default; add `on_failure: "skip"` per step later |
| 6 | Should we support **DAG workflows** (parallel branches) or only linear chains? | Linear (simpler), DAG (flexible) | Linear chains first; DAG is a future enhancement |
| 7 | `umrs.steps` package — pip-installable separately for user dev? | Yes (light SDK), No (import from full package) | Yes — publish `umrs-steps` as a lightweight SDK with just the ABCs + StepContext |

---

## 12. Custom Vision Adapters (BentoML)

In addition to the Pipeline SDK (`user_script://`) for embedding pipelines,
users can extend the BentoML vision service with **custom vision adapters**.

### 12.1 Architecture

```
/adapters/system/           ← Built-in adapters (shipped with source tree)
  sam3_adapter.py            → ADAPTER_REGISTRY["sam3_adapter"]
  hmae_adapter.py            → ADAPTER_REGISTRY["hmae_adapter"]
/adapters/user/             ← User-provided adapters (bind-mounted from host)
  my_model.py                → ADAPTER_REGISTRY["my_model"]
  requirements.txt           ← auto-installed at startup
```

At BentoML container startup:
1. `VISION_ADAPTERS_DIR` (colon-separated) is scanned for `*.py` files
2. If a `requirements.txt` exists in a directory, its packages are auto-installed
   via `uv pip install` (fallback to `pip`)
3. Each file's `VisionModelAdapter` subclass is registered by filename stem
4. The `/infer` endpoint dispatches by `model_type` → `ADAPTER_REGISTRY[model_type]`

### 12.2 VisionModelAdapter Interface

```python
from umrs.vision_service.adapters.base import VisionModelAdapter

class MyAdapter(VisionModelAdapter):
    def cache_key(self, config):
        # Unique string for model cache (same config → same cache entry)
        return f"my_model:{config.get('checkpoint', 'default')}"

    def load(self, config):
        # Load model weights → return opaque model object
        ...

    def infer(self, model, image, bucket, key, config, s3, output_bucket):
        # Run inference on PIL image, upload results to S3
        # Return JSON-serializable result dict
        ...
```

### 12.3 Deployment

```bash
# On the compute node, before starting the Slurm job:
export USER_ADAPTERS_DIR=/home/user/my_adapters    # adapter .py files
export USER_MODEL_PATH=/home/user/model_weights    # model weights dir
sbatch serve-model.slurm
```

Inside the container:
- Adapters are at `/adapters/user/`
- Model weights are at `/models/user/`

### 12.4 User Config

```json
{
    "model_type": "rivault://bentoml://http://localhost:{LOCAL_PORT}/vision/infer/submit",
    "vision_model_type": "my_model",
    "vision_model_config": { "checkpoint": "/models/user/weights.pth" }
}
```

### 12.5 Two Extension Models Compared

| | Embedding Pipeline (`user_script://`) | Vision Adapter (`VisionModelAdapter`) |
|--|------|------|
| **Runs inside** | UMRS server process | BentoML container (GPU) |
| **Best for** | Multi-step processing, custom embeddings | Single-model GPU inference |
| **Config** | `workflow.json` + Step classes | Single `.py` file |
| **Dependencies** | `requirements` in workflow.json | `requirements.txt` in adapter dir |
| **Model type** | `user_script:///path/to/workflow.json` | Adapter filename stem |
| **GPU access** | Via BentoML HTTP API (from steps) | Direct (adapter runs on GPU node) |

---

## 13. Compatibility Notes

- **Backward compatible**: All existing `model_type` values (`SentenceTransformer`, `openai://`, `rivault://bentoml://`, etc.) continue to work unchanged
- **Opt-in**: Users who don't need custom models never encounter UDM code
- **Composable with rivault://**: `rivault://user_script://...` chains infrastructure (Slurm launch) with user code execution
- **Extends the current Pipeline SDK**: Workflow steps map to the implemented `src/umrs/pipeline_sdk/` execution model and can be exposed through `user_script://` or `user_workflow://` model types.
- **Extends PipelineDefinition**: `pre_processors` / `post_processors` fields can now reference user script entrypoints instead of only plugin IDs
