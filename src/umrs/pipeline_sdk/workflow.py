"""
WorkflowPipeline — reads a ``workflow.json`` file and executes it as a Pipeline.

This is the bridge between the declarative workflow.json format (originally
driven by ``UMRSStep`` / ``WorkflowExecutor``) and the unified Pipeline SDK.
Users who prefer JSON configuration over writing a full Pipeline subclass
can use ``WorkflowPipeline`` directly.

Data-model classes (``WorkflowDefinition``, ``StepSpec``, etc.) are defined
here as well — they were previously in ``umrs.steps.models``.

Usage::

    from umrs.pipeline_sdk.workflow import WorkflowPipeline

    # Load and run
    pipeline = WorkflowPipeline("/path/to/workflow.json")
    results = pipeline.run()

Server integration (via EmbeddingAdapter)::

    from umrs.pipeline_sdk.adapter import EmbeddingAdapter

    pipeline = WorkflowPipeline(source_uri)
    compute_fn = EmbeddingAdapter(pipeline).compute_embeddings
"""

from __future__ import annotations

import importlib
import importlib.util
import io
import json
import logging
import re
import sys
import tempfile
import threading
import uuid
import zipfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from umrs.pipeline_sdk.base import (
    ComputeBackend,
    DataItem,
    Pipeline,
    PipelineConfig,
    Result,
    TaskHandle,
)
from umrs.pipeline_sdk.step import Step

logger = logging.getLogger(__name__)

# Regex: "module_name:ClassName" — no whitespace, no shell chars.
_ENTRYPOINT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*:[A-Za-z_][A-Za-z0-9_]*$")

# Thread-safe cache for imported step classes.
# Include the resolved module path and mtime so a workflow re-upload to the
# same directory invalidates the stale class definition automatically.
_step_class_cache: dict[tuple[str, str, str, int | None], type[Step]] = {}
_step_cache_lock = threading.Lock()


# =========================================================================
# Data-model classes (formerly umrs.steps.models)
# =========================================================================


class ExecutionMode(Enum):
    """How workflow steps are executed."""

    IN_PROCESS = "in_process"  # Direct call in server process (dev/default)
    SUBPROCESS = "subprocess"  # Isolated subprocess per step
    SLURM = "slurm"  # One Slurm job per step (HPC GPU)
    K8S = "k8s"  # One Kubernetes Job per step (cloud)


class SourceType(Enum):
    """Where the user's Python scripts live."""

    LOCAL_PATH = "local_path"
    S3 = "s3"
    GIT = "git"
    UPLOADED = "uploaded"  # server-side directory from client upload


@dataclass
class SourceSpec:
    """Location of the user's Python scripts."""

    type: SourceType
    path: str
    ref: str | None = None  # git commit hash

    def validate(self) -> None:
        if self.type == SourceType.GIT and not self.ref:
            raise ValueError("Git source requires 'ref' (commit hash)")
        if self.type == SourceType.LOCAL_PATH and not self.path.startswith("/"):
            raise ValueError("Local source path must be absolute")
        if self.type == SourceType.S3 and not self.path.startswith("s3://"):
            raise ValueError("S3 source must start with s3://")
        if self.type == SourceType.UPLOADED and not self.path:
            raise ValueError("Uploaded source requires a non-empty path")

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"type": self.type.value, "path": self.path}
        if self.ref:
            d["ref"] = self.ref
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SourceSpec:
        return cls(
            type=SourceType(data["type"]),
            path=data["path"],
            ref=data.get("ref"),
        )


@dataclass
class ResourceSpec:
    """Compute resource limits for the whole workflow."""

    gpu_count: int = 0
    cpu_limit: str = "1"
    memory_limit: str = "1Gi"
    timeout_seconds: int = 3600
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = {
            "gpu_count": self.gpu_count,
            "cpu_limit": self.cpu_limit,
            "memory_limit": self.memory_limit,
            "timeout_seconds": self.timeout_seconds,
        }
        if self.extra:
            d["extra"] = self.extra
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ResourceSpec:
        return cls(
            gpu_count=data.get("gpu_count", 0),
            cpu_limit=data.get("cpu_limit", "1"),
            memory_limit=data.get("memory_limit", "1Gi"),
            timeout_seconds=data.get("timeout_seconds", 3600),
            extra=data.get("extra", {}),
        )


# Capability tags the StepRouter understands today. Unknown values are
# accepted (with a warning) so user workflows aren't brittle when new
# capabilities ship.
KNOWN_CAPABILITY_TAGS: frozenset = frozenset(
    {
        "vision",
        "tool_calling",
        "structured_output",
        "embedding",
        "gpu",
    }
)

# Allowed values for ``StepSpec.model_type``. Mirrors
# ``umrs.model_registry.models.ModelType`` but redeclared as plain strings to
# avoid pulling registry deps into pipeline_sdk.
KNOWN_MODEL_TYPES: frozenset = frozenset({"llm", "vision", "embedding"})


@dataclass
class StepSpec:
    """One processing step inside a workflow."""

    name: str
    entrypoint: str  # "module:ClassName"
    config: dict[str, Any] = field(default_factory=dict)
    extends: str | None = None  # "name@version" reference to a registered step
    depends_on: list[str] | None = None  # step names this step depends on
    max_retries: int = 0  # per-step retry count (0 = no retries)
    retry_delay: float = 1.0  # seconds between retries
    # ---- routing / resource hints (all optional, default-omitted in to_dict) ----
    requires: list[str] = field(default_factory=list)  # capability tags
    model: str | None = None  # symbolic model alias (e.g. "vlm-large")
    lora: str | None = None  # adapter name (passthrough)
    model_type: str | None = None  # "llm" | "vision" | "embedding"
    routing_hints: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.name:
            raise ValueError("Step name cannot be empty")
        if ":" not in self.entrypoint:
            raise ValueError(
                f"Step '{self.name}': entrypoint must be 'module:ClassName', "
                f"got '{self.entrypoint}'"
            )
        if self.requires:
            if not isinstance(self.requires, list) or any(
                not isinstance(tag, str) or not tag for tag in self.requires
            ):
                raise ValueError(
                    f"Step '{self.name}': requires must be a list of non-empty strings"
                )
            unknown = [tag for tag in self.requires if tag not in KNOWN_CAPABILITY_TAGS]
            if unknown:
                logger.warning(
                    "Step '%s' declares unknown capability tag(s): %s; "
                    "router will treat them as opaque hints.",
                    self.name,
                    ", ".join(sorted(unknown)),
                )
        if self.model_type is not None and self.model_type not in KNOWN_MODEL_TYPES:
            raise ValueError(
                f"Step '{self.name}': model_type must be one of "
                f"{sorted(KNOWN_MODEL_TYPES)}, got '{self.model_type}'"
            )

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "name": self.name,
            "entrypoint": self.entrypoint,
            "config": self.config,
        }
        if self.extends:
            d["extends"] = self.extends
        if self.depends_on is not None:
            d["depends_on"] = self.depends_on
        if self.max_retries > 0:
            d["max_retries"] = self.max_retries
            d["retry_delay"] = self.retry_delay
        # New routing fields are omitted when at default to keep existing
        # workflow.json fixtures byte-stable on round-trip.
        if self.requires:
            d["requires"] = list(self.requires)
        if self.model is not None:
            d["model"] = self.model
        if self.lora is not None:
            d["lora"] = self.lora
        if self.model_type is not None:
            d["model_type"] = self.model_type
        if self.routing_hints:
            d["routing_hints"] = dict(self.routing_hints)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StepSpec:
        return cls(
            name=data["name"],
            entrypoint=data["entrypoint"],
            config=data.get("config", {}),
            extends=data.get("extends"),
            depends_on=data.get("depends_on"),
            max_retries=data.get("max_retries", 0),
            retry_delay=data.get("retry_delay", 1.0),
            requires=list(data.get("requires") or []),
            model=data.get("model"),
            lora=data.get("lora"),
            model_type=data.get("model_type"),
            routing_hints=dict(data.get("routing_hints") or {}),
        )


def _normalize_execution_parallel_steps(
    steps_data: list[dict[str, Any]],
    execution_data: dict[str, Any],
) -> list[dict[str, Any]]:
    """Translate legacy ``execution.parallel`` shorthand into ``depends_on``.

    The workflow engine only executes DAGs expressed via ``depends_on``.
    Demo workflows and docs may still use ``execution.parallel`` to mean that
    a contiguous group of steps should receive the same upstream input keys and
    fan back into the next serial step.
    """

    normalized_steps = [dict(step) for step in steps_data]
    parallel_names = execution_data.get("parallel")
    if not parallel_names:
        return normalized_steps

    if not isinstance(parallel_names, list) or any(
        not isinstance(name, str) for name in parallel_names
    ):
        raise ValueError("execution.parallel must be a list of step names")

    if len(set(parallel_names)) != len(parallel_names):
        raise ValueError("execution.parallel contains duplicate step names")

    step_names = [step.get("name") for step in normalized_steps if "name" in step]
    missing = [name for name in parallel_names if name not in step_names]
    if missing:
        missing_csv = ", ".join(missing)
        raise ValueError(
            f"execution.parallel references unknown step(s): {missing_csv}"
        )

    if any("depends_on" in step for step in normalized_steps):
        return normalized_steps

    parallel_set = set(parallel_names)
    previous_serial_name: str | None = None
    current_parallel_group: list[str] = []

    for step in normalized_steps:
        step_name = step.get("name")

        if step_name in parallel_set:
            step["depends_on"] = [previous_serial_name] if previous_serial_name else []
            current_parallel_group.append(step_name)
            continue

        if current_parallel_group:
            step["depends_on"] = list(current_parallel_group)
            current_parallel_group = []
        else:
            step["depends_on"] = [previous_serial_name] if previous_serial_name else []

        previous_serial_name = step_name

    return normalized_steps


@dataclass
class OutputSpec:
    """Describes the final output format of the workflow."""

    return_type: str = "file"  # "file" | "vector"
    embedding_vector_size: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "return_type": self.return_type,
            "embedding_vector_size": self.embedding_vector_size,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OutputSpec:
        return cls(
            return_type=data.get("return_type", "file"),
            embedding_vector_size=data.get("embedding_vector_size", 0),
        )


@dataclass
class WorkflowDefinition:
    """Complete definition of a user-provided multi-step workflow.

    Typically loaded from a ``workflow.json`` file via :meth:`from_file`.
    """

    name: str
    source: SourceSpec
    steps: list[StepSpec]

    version: str = "1.0"
    description: str = ""
    requirements: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    resources: ResourceSpec = field(default_factory=ResourceSpec)
    output: OutputSpec = field(default_factory=OutputSpec)
    execution_mode: ExecutionMode = ExecutionMode.IN_PROCESS

    def validate(self) -> None:
        """Run all validation checks; raise ``ValueError`` on failure."""
        if not self.name:
            raise ValueError("Workflow name is required")
        if not self.steps:
            raise ValueError("Workflow must contain at least one step")

        self.source.validate()

        seen_names: set[str] = set()
        for step in self.steps:
            step.validate()
            if step.name in seen_names:
                raise ValueError(f"Duplicate step name: '{step.name}'")
            seen_names.add(step.name)

        # Validate depends_on references and detect cycles
        has_deps = any(s.depends_on is not None for s in self.steps)
        if has_deps:
            for step in self.steps:
                if step.depends_on is not None:
                    for dep in step.depends_on:
                        if dep not in seen_names:
                            raise ValueError(
                                f"Step '{step.name}' depends on unknown step '{dep}'"
                            )
                        if dep == step.name:
                            raise ValueError(f"Step '{step.name}' depends on itself")
            from umrs.pipeline_sdk.dag import DAGScheduler, DAGValidationError

            try:
                DAGScheduler(self.steps).validate()
            except DAGValidationError as exc:
                raise ValueError(str(exc)) from exc

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "source": self.source.to_dict(),
            "requirements": self.requirements,
            "env": self.env,
            "resources": self.resources.to_dict(),
            "steps": [s.to_dict() for s in self.steps],
            "output": self.output.to_dict(),
        }
        if self.execution_mode != ExecutionMode.IN_PROCESS:
            d["execution_mode"] = self.execution_mode.value
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkflowDefinition:
        mode_str = data.get("execution_mode", "in_process")
        steps_data = _normalize_execution_parallel_steps(
            data["steps"],
            data.get("execution", {}),
        )
        return cls(
            name=data["name"],
            version=data.get("version", "1.0"),
            description=data.get("description", ""),
            source=SourceSpec.from_dict(data["source"]),
            requirements=data.get("requirements", []),
            env=data.get("env", {}),
            resources=ResourceSpec.from_dict(data.get("resources", {})),
            steps=[StepSpec.from_dict(s) for s in steps_data],
            output=OutputSpec.from_dict(data.get("output", {})),
            execution_mode=ExecutionMode(mode_str),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> WorkflowDefinition:
        """Load and validate a workflow from a JSON file."""
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Workflow file not found: {path}")
        with open(path) as fh:
            data = json.load(fh)
        wf = cls.from_dict(data)
        wf.validate()
        return wf


# =========================================================================
# Script loader (formerly umrs.steps.loader)
# =========================================================================


class ScriptLoadError(Exception):
    """Raised when a user script cannot be loaded."""


def _resolve_source(
    workflow: WorkflowDefinition,
    s3: Any | None = None,
) -> Path:
    """Return a local directory containing the user's Python modules."""
    src = workflow.source

    if src.type == SourceType.LOCAL_PATH:
        p = Path(src.path)
        if not p.is_dir():
            raise ScriptLoadError(f"Source directory does not exist: {p}")
        return p

    if src.type == SourceType.S3:
        if s3 is None:
            raise ScriptLoadError("S3 client is required to download script source")
        return _download_s3_source(s3, src.path)

    if src.type == SourceType.GIT:
        raise NotImplementedError("Git source type is not yet implemented")

    if src.type == SourceType.UPLOADED:
        p = Path(src.path)
        if not p.is_dir():
            raise ScriptLoadError(f"Uploaded source directory does not exist: {p}")
        return p

    raise ScriptLoadError(f"Unknown source type: {src.type}")


def _download_s3_source(s3: Any, s3_uri: str) -> Path:
    """Download a zip archive from S3 and extract to a temp dir."""
    if not s3_uri.startswith("s3://"):
        raise ScriptLoadError(f"Invalid S3 URI: {s3_uri}")

    without_scheme = s3_uri[len("s3://") :]
    bucket, _, key = without_scheme.partition("/")
    if not key:
        raise ScriptLoadError(f"S3 URI missing key: {s3_uri}")

    tmp_dir = Path(tempfile.mkdtemp(prefix="umrs_steps_"))
    logger.info("Downloading user scripts from %s to %s", s3_uri, tmp_dir)

    obj = s3.get_object(Bucket=bucket, Key=key)
    data = obj["Body"].read()

    if key.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            zf.extractall(tmp_dir)
    else:
        dest = tmp_dir / Path(key).name
        dest.write_bytes(data)

    return tmp_dir


def _import_step_class(spec: StepSpec, script_dir: Path) -> type[Step]:
    """Import and validate a single step class.

    Uses ``importlib.util.spec_from_file_location`` for thread-safe
    loading without mutating ``sys.path``.  Results are cached by
    ``(entrypoint, script_dir)`` to avoid repeated filesystem access.
    """
    ep = spec.entrypoint
    if not _ENTRYPOINT_RE.match(ep):
        raise ScriptLoadError(
            f"Step '{spec.name}': invalid entrypoint format '{ep}'. "
            "Expected 'module_name:ClassName'."
        )

    module_name, class_name = ep.split(":", 1)
    str_dir = str(script_dir)

    # Resolve the module file on disk
    # Support dotted modules: "pkg.sub.mod" -> pkg/sub/mod.py
    rel_parts = module_name.replace(".", "/")
    module_file = script_dir / f"{rel_parts}.py"
    if not module_file.is_file():
        # Try as a package: pkg/sub/mod/__init__.py
        module_file = script_dir / rel_parts / "__init__.py"

    if module_file.is_file():
        resolved_module = str(module_file.resolve())
        module_mtime = module_file.stat().st_mtime_ns
    else:
        resolved_module = module_name
        module_mtime = None

    cache_key = (ep, str_dir, resolved_module, module_mtime)

    with _step_cache_lock:
        cached = _step_class_cache.get(cache_key)
        if cached is not None:
            return cached

    if module_file.is_file():
        # Use importlib.util for thread-safe import without sys.path mutation
        loader_spec = importlib.util.spec_from_file_location(
            module_name,
            str(module_file),
            submodule_search_locations=[str(script_dir)],
        )
        if loader_spec is None or loader_spec.loader is None:
            raise ScriptLoadError(
                f"Step '{spec.name}': cannot create module spec for "
                f"'{module_name}' at {module_file}"
            )
        mod = importlib.util.module_from_spec(loader_spec)
        sys.modules[module_name] = mod
        try:
            loader_spec.loader.exec_module(mod)
        except Exception as exc:
            sys.modules.pop(module_name, None)
            raise ScriptLoadError(
                f"Step '{spec.name}': cannot load module '{module_name}' "
                f"from {module_file}: {exc}"
            ) from exc
    else:
        # Fallback: module may already be importable (e.g. installed package)
        try:
            mod = importlib.import_module(module_name)
        except Exception as exc:
            raise ScriptLoadError(
                f"Step '{spec.name}': cannot import module '{module_name}' "
                f"from {script_dir}: {exc}"
            ) from exc

    cls = getattr(mod, class_name, None)
    if cls is None:
        raise ScriptLoadError(
            f"Step '{spec.name}': class '{class_name}' not found in "
            f"module '{module_name}'"
        )

    if not (isinstance(cls, type) and issubclass(cls, Step)):
        raise ScriptLoadError(f"Step '{spec.name}': {class_name} does not inherit Step")

    with _step_cache_lock:
        _step_class_cache[cache_key] = cls

    return cls


def load_workflow(source_uri: str) -> WorkflowDefinition:
    """Load a :class:`WorkflowDefinition` from *source_uri*.

    *source_uri* is the part after the ``user_script://`` scheme.
    Searches first relative to CWD, then in the uploaded-workflows
    directory (``tmp/user_workflows/<user>/<source_uri>``).
    """
    if source_uri.startswith("s3://"):
        raise NotImplementedError(
            "S3-hosted workflow files are not yet supported. "
            "Place workflow.json on a mounted local path for now."
        )
    path = Path(source_uri)
    if path.is_file():
        return WorkflowDefinition.from_file(path)

    # Fallback: search in uploaded-workflows directory
    from umrs.shared.config import get_default_tmp_subdir

    uploaded_root = Path(get_default_tmp_subdir()) / "user_workflows"
    if uploaded_root.is_dir():
        for user_dir in uploaded_root.iterdir():
            if not user_dir.is_dir():
                continue
            candidate = user_dir / source_uri
            if candidate.is_file():
                return WorkflowDefinition.from_file(candidate)

    raise FileNotFoundError(
        f"Workflow not found: {source_uri} (checked CWD and {uploaded_root})"
    )


def load_steps(
    workflow: WorkflowDefinition,
    s3: Any | None = None,
) -> list[Step]:
    """Import and instantiate every step declared in *workflow*."""
    script_dir = _resolve_source(workflow, s3)
    instances: list[Step] = []
    for spec in workflow.steps:
        cls = _import_step_class(spec, script_dir)
        instance = cls()
        instance.setup(spec.config)
        instances.append(instance)
    return instances


# =========================================================================
# WorkflowPipeline — the unified Pipeline subclass
# =========================================================================


class WorkflowPipeline(Pipeline):
    """A Pipeline backed by a ``workflow.json`` file.

    Reads step definitions from JSON, imports user Step classes, and
    chains them inside the ``compute()`` phase.  For standalone use,
    call ``.run()``.  For server integration, wrap with
    :class:`~umrs.pipeline_sdk.adapter.EmbeddingAdapter`.

    Args:
        source_uri: Path to workflow.json (after ``user_script://``).
        s3: Optional boto3-compatible S3 client for S3-hosted scripts.
        s3_keys: Input S3 keys (set when used server-side via adapter).
        input_bucket: Source bucket (set when used server-side).
        output_bucket: Destination bucket (set when used server-side).
    """

    def __init__(
        self,
        source_uri: str,
        s3: Any | None = None,
        s3_keys: list[str] | None = None,
        input_bucket: str = "",
        output_bucket: str = "",
    ) -> None:
        super().__init__()
        self.workflow = load_workflow(source_uri)
        self._step_instances = load_steps(self.workflow, s3)
        self._s3 = s3
        self._s3_keys = s3_keys or []
        self._input_bucket = input_bucket
        self._output_bucket = output_bucket

    @classmethod
    def from_workflow(
        cls,
        workflow: WorkflowDefinition,
        steps: list[Step],
        s3: Any | None = None,
        s3_keys: list[str] | None = None,
        input_bucket: str = "",
        output_bucket: str = "",
    ) -> WorkflowPipeline:
        """Construct from a pre-built WorkflowDefinition + step instances.

        Useful when the workflow definition was patched (e.g. for testing)
        or when steps have been loaded separately.
        """
        obj = cls.__new__(cls)
        Pipeline.__init__(obj)
        obj.workflow = workflow
        obj._step_instances = steps
        obj._s3 = s3
        obj._s3_keys = s3_keys or []
        obj._input_bucket = input_bucket
        obj._output_bucket = output_bucket
        return obj

    # ── Pipeline lifecycle hooks ─────────────────────────────────────────

    def configure(self) -> PipelineConfig:
        return PipelineConfig(
            name=self.workflow.name,
            compute_backend=ComputeBackend.LOCAL,
            timeout_seconds=float(self.workflow.resources.timeout_seconds),
            extra={
                "workflow": self.workflow,
                "return_type": self.workflow.output.return_type,
                "embedding_vector_size": self.workflow.output.embedding_vector_size,
            },
        )

    def ingest(self, config: PipelineConfig) -> list[DataItem]:
        return [
            DataItem(key=k, source=f"s3://{self._input_bucket}/{k}")
            for k in self._s3_keys
        ]

    def compute(
        self,
        config: PipelineConfig,
        items: list[DataItem],
    ) -> list[TaskHandle]:
        """Chain steps: step N+1's input = step N's output.

        When steps declare ``depends_on``, uses DAG scheduling for
        parallel execution of independent steps.
        """
        from umrs.pipeline_sdk.dag import DAGScheduler, execute_dag

        run_id = uuid.uuid4().hex[:12]
        input_keys = [item.key for item in items]
        shared_metadata: dict[str, Any] = {}

        scheduler = DAGScheduler(self.workflow.steps)
        step_results = execute_dag(
            scheduler,
            self._step_instances,
            self.workflow.steps,
            s3=self._s3,
            input_bucket=self._input_bucket,
            input_keys=input_keys,
            output_bucket=self._output_bucket,
            run_id=run_id,
            shared_metadata=shared_metadata,
        )

        # Extract final results from the last level's steps
        last_embeddings, last_checksums, last_metadata, current_keys = (
            self._aggregate_dag_results(step_results, scheduler, input_keys)
        )

        # Teardown all steps
        for step in self._step_instances:
            try:
                step.teardown()
            except Exception:
                logger.warning(
                    "teardown failed for step '%s'", step.name, exc_info=True
                )

        # Clean up staging artifacts
        self._cleanup_staging(self._s3, self._input_bucket, run_id)

        # Store results for collect() phase
        self._run_result = (
            last_embeddings,
            last_checksums,
            last_metadata,
            current_keys,
        )

        return [
            TaskHandle(
                task_id=run_id,
                endpoint="workflow",
                metadata={
                    "embeddings": last_embeddings,
                    "checksums": last_checksums,
                    "step_metadata": last_metadata,
                    "output_keys": current_keys,
                    "input_keys": input_keys,
                },
            )
        ]

    def collect(
        self,
        config: PipelineConfig,
        handles: list[TaskHandle],
    ) -> list[Result]:
        handle = handles[0]
        meta = handle.metadata
        return [
            Result(
                task_handle=handle,
                status="success",
                data={
                    "embeddings": meta.get("embeddings"),
                    "checksums": meta.get("checksums", []),
                    "metadata": meta.get("step_metadata", []),
                    "output_keys": meta.get("output_keys", []),
                },
            )
        ]

    # ── Embedding-contract bridge ────────────────────────────────────────

    def run_as_embeddings(
        self,
        s3: Any,
        s3_keys: list[str],
        path_to_bucket: str,
        output_bucket: str,
    ) -> tuple[list[Any], list[str], list[dict[str, Any]], list[str | None]]:
        """Execute like ``WorkflowExecutor.run()`` — drop-in for
        ``EmbeddingModel.compute_embeddings``.

        Returns ``(embeddings, checksums, metadata_list, saved_keys)``.

        For non-``IN_PROCESS`` execution modes, delegates to
        :class:`~umrs.plugins.user_workflow.UserWorkflowRunner`.
        """
        if self.workflow.execution_mode != ExecutionMode.IN_PROCESS:
            from umrs.plugins.user_workflow import (
                UserWorkflow,
                UserWorkflowRunner,
            )

            uw = UserWorkflow.from_workflow_definition(self.workflow)
            runner = UserWorkflowRunner(uw, s3=s3)
            return runner.run(s3, s3_keys, path_to_bucket, output_bucket)

        from umrs.pipeline_sdk.dag import DAGScheduler, execute_dag

        run_id = uuid.uuid4().hex[:12]
        logger.info(
            "Workflow '%s' run=%s: %d steps, %d input keys",
            self.workflow.name,
            run_id,
            len(self._step_instances),
            len(s3_keys),
        )

        shared_metadata: dict[str, Any] = {}

        scheduler = DAGScheduler(self.workflow.steps)
        step_results = execute_dag(
            scheduler,
            self._step_instances,
            self.workflow.steps,
            s3=s3,
            input_bucket=path_to_bucket,
            input_keys=list(s3_keys),
            output_bucket=output_bucket,
            run_id=run_id,
            shared_metadata=shared_metadata,
        )

        last_embeddings, last_checksums, last_metadata, current_keys = (
            self._aggregate_dag_results(step_results, scheduler, list(s3_keys))
        )

        # Build return tuple matching the compute_embeddings contract
        if not last_checksums:
            last_checksums = self._collect_checksums(s3, path_to_bucket, s3_keys)
        if not last_metadata:
            last_metadata = self._collect_metadata(s3, path_to_bucket, s3_keys)

        if last_embeddings is not None:
            embeddings_list: list[Any] = list(last_embeddings)
        else:
            embeddings_list = [None] * len(s3_keys)

        saved_keys: list[str | None] = (
            list(current_keys)
            if current_keys != list(s3_keys)
            else [None] * len(s3_keys)
        )

        # Teardown
        for step in self._step_instances:
            try:
                step.teardown()
            except Exception:
                logger.warning(
                    "teardown failed for step '%s'", step.name, exc_info=True
                )

        self._cleanup_staging(s3, path_to_bucket, run_id)

        logger.info("Workflow '%s' run=%s complete.", self.workflow.name, run_id)
        return embeddings_list, last_checksums, last_metadata, saved_keys

    # ── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _aggregate_dag_results(
        step_results: dict[str, Any],
        scheduler: Any,
        input_keys: list[str],
    ) -> tuple[Any, list[str], list[dict[str, Any]], list[str]]:
        """Extract final outputs from the DAG execution results.

        Walks execution levels in reverse to find the last step(s) that
        produced embeddings, checksums, metadata, or output_keys.

        Returns ``(embeddings, checksums, metadata_list, output_keys)``.
        """
        last_embeddings: Any = None
        last_checksums: list[str] = []
        last_metadata: list[dict[str, Any]] = []
        current_keys: list[str] = list(input_keys)

        # Iterate levels forward so later steps override earlier ones
        for level in scheduler.execution_levels():
            for spec_idx in level:
                spec = scheduler._specs[spec_idx]
                result = step_results.get(spec.name)
                if result is None:
                    continue
                if result.embeddings is not None:
                    last_embeddings = result.embeddings
                if result.checksums:
                    last_checksums = result.checksums
                if result.metadata:
                    last_metadata = result.metadata
                if result.output_keys:
                    current_keys = result.output_keys

        return last_embeddings, last_checksums, last_metadata, current_keys

    @staticmethod
    def _collect_checksums(
        s3: Any,
        bucket: str,
        keys: list[str],
    ) -> list[str]:
        checksums = []
        for key in keys:
            try:
                obj = s3.head_object(Bucket=bucket, Key=key)
                checksums.append(obj["ETag"].strip('"'))
            except Exception:
                checksums.append("")
        return checksums

    @staticmethod
    def _collect_metadata(
        s3: Any,
        bucket: str,
        keys: list[str],
    ) -> list[dict[str, Any]]:
        meta = []
        for key in keys:
            try:
                obj = s3.head_object(Bucket=bucket, Key=key)
                meta.append(obj.get("Metadata", {}))
            except Exception:
                meta.append({})
        return meta

    @staticmethod
    def _cleanup_staging(
        s3: Any | None,
        bucket: str,
        run_id: str,
    ) -> None:
        """Delete intermediate staging objects for this run."""
        if s3 is None or not bucket:
            return
        prefix = f"_staging/{run_id}/"
        try:
            paginator = s3.get_paginator("list_objects_v2")
            to_delete: list[dict[str, str]] = []
            for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                for obj in page.get("Contents", []):
                    to_delete.append({"Key": obj["Key"]})

            while to_delete:
                batch, to_delete = to_delete[:1000], to_delete[1000:]
                s3.delete_objects(
                    Bucket=bucket,
                    Delete={"Objects": batch, "Quiet": True},
                )
        except Exception:
            logger.debug("Staging cleanup failed for run=%s", run_id, exc_info=True)
