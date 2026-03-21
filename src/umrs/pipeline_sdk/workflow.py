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
import io
import json
import logging
import re
import sys
import tempfile
import uuid
import zipfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

from umrs.pipeline_sdk.base import (
    ComputeBackend,
    DataItem,
    Phase,
    Pipeline,
    PipelineConfig,
    Result,
    TaskHandle,
)
from umrs.pipeline_sdk.step import Step, StepContext, StepResult

logger = logging.getLogger(__name__)

# Regex: "module_name:ClassName" — no whitespace, no shell chars.
_ENTRYPOINT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*:[A-Za-z_][A-Za-z0-9_]*$")


# =========================================================================
# Data-model classes (formerly umrs.steps.models)
# =========================================================================

class SourceType(Enum):
    """Where the user's Python scripts live."""
    LOCAL_PATH = "local_path"
    S3 = "s3"
    GIT = "git"


@dataclass
class SourceSpec:
    """Location of the user's Python scripts."""
    type: SourceType
    path: str
    ref: Optional[str] = None  # git commit hash

    def validate(self) -> None:
        if self.type == SourceType.GIT and not self.ref:
            raise ValueError("Git source requires 'ref' (commit hash)")
        if self.type == SourceType.LOCAL_PATH and not self.path.startswith("/"):
            raise ValueError("Local source path must be absolute")
        if self.type == SourceType.S3 and not self.path.startswith("s3://"):
            raise ValueError("S3 source must start with s3://")

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"type": self.type.value, "path": self.path}
        if self.ref:
            d["ref"] = self.ref
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SourceSpec":
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
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
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
    def from_dict(cls, data: Dict[str, Any]) -> "ResourceSpec":
        return cls(
            gpu_count=data.get("gpu_count", 0),
            cpu_limit=data.get("cpu_limit", "1"),
            memory_limit=data.get("memory_limit", "1Gi"),
            timeout_seconds=data.get("timeout_seconds", 3600),
            extra=data.get("extra", {}),
        )


@dataclass
class StepSpec:
    """One processing step inside a workflow."""
    name: str
    entrypoint: str  # "module:ClassName"
    config: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.name:
            raise ValueError("Step name cannot be empty")
        if ":" not in self.entrypoint:
            raise ValueError(
                f"Step '{self.name}': entrypoint must be 'module:ClassName', "
                f"got '{self.entrypoint}'"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "entrypoint": self.entrypoint,
            "config": self.config,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StepSpec":
        return cls(
            name=data["name"],
            entrypoint=data["entrypoint"],
            config=data.get("config", {}),
        )


@dataclass
class OutputSpec:
    """Describes the final output format of the workflow."""
    return_type: str = "file"  # "file" | "vector"
    embedding_vector_size: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "return_type": self.return_type,
            "embedding_vector_size": self.embedding_vector_size,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OutputSpec":
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
    steps: List[StepSpec]

    version: str = "1.0"
    description: str = ""
    requirements: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    resources: ResourceSpec = field(default_factory=ResourceSpec)
    output: OutputSpec = field(default_factory=OutputSpec)

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

    def to_dict(self) -> Dict[str, Any]:
        return {
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

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkflowDefinition":
        return cls(
            name=data["name"],
            version=data.get("version", "1.0"),
            description=data.get("description", ""),
            source=SourceSpec.from_dict(data["source"]),
            requirements=data.get("requirements", []),
            env=data.get("env", {}),
            resources=ResourceSpec.from_dict(data.get("resources", {})),
            steps=[StepSpec.from_dict(s) for s in data["steps"]],
            output=OutputSpec.from_dict(data.get("output", {})),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "WorkflowDefinition":
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
    s3: Optional[Any] = None,
) -> Path:
    """Return a local directory containing the user's Python modules."""
    src = workflow.source

    if src.type == SourceType.LOCAL_PATH:
        p = Path(src.path)
        if not p.is_dir():
            raise ScriptLoadError(
                f"Source directory does not exist: {p}"
            )
        return p

    if src.type == SourceType.S3:
        if s3 is None:
            raise ScriptLoadError(
                "S3 client is required to download script source"
            )
        return _download_s3_source(s3, src.path)

    if src.type == SourceType.GIT:
        raise NotImplementedError("Git source type is not yet implemented")

    raise ScriptLoadError(f"Unknown source type: {src.type}")


def _download_s3_source(s3: Any, s3_uri: str) -> Path:
    """Download a zip archive from S3 and extract to a temp dir."""
    if not s3_uri.startswith("s3://"):
        raise ScriptLoadError(f"Invalid S3 URI: {s3_uri}")

    without_scheme = s3_uri[len("s3://"):]
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


def _import_step_class(spec: StepSpec, script_dir: Path) -> Type[Step]:
    """Import and validate a single step class."""
    ep = spec.entrypoint
    if not _ENTRYPOINT_RE.match(ep):
        raise ScriptLoadError(
            f"Step '{spec.name}': invalid entrypoint format '{ep}'. "
            "Expected 'module_name:ClassName'."
        )

    module_name, class_name = ep.split(":", 1)

    str_dir = str(script_dir)
    added = False
    if str_dir not in sys.path:
        sys.path.insert(0, str_dir)
        added = True

    try:
        mod = importlib.import_module(module_name)
    except Exception as exc:
        raise ScriptLoadError(
            f"Step '{spec.name}': cannot import module '{module_name}' "
            f"from {script_dir}: {exc}"
        ) from exc
    finally:
        if added:
            sys.path.remove(str_dir)

    cls = getattr(mod, class_name, None)
    if cls is None:
        raise ScriptLoadError(
            f"Step '{spec.name}': class '{class_name}' not found in "
            f"module '{module_name}'"
        )

    if not (isinstance(cls, type) and issubclass(cls, Step)):
        raise ScriptLoadError(
            f"Step '{spec.name}': {class_name} does not inherit Step"
        )

    return cls


def load_workflow(source_uri: str) -> WorkflowDefinition:
    """Load a :class:`WorkflowDefinition` from *source_uri*.

    *source_uri* is the part after the ``user_script://`` scheme.
    """
    if source_uri.startswith("s3://"):
        raise NotImplementedError(
            "S3-hosted workflow files are not yet supported. "
            "Place workflow.json on a mounted local path for now."
        )
    path = Path(source_uri)
    return WorkflowDefinition.from_file(path)


def load_steps(
    workflow: WorkflowDefinition,
    s3: Optional[Any] = None,
) -> List[Step]:
    """Import and instantiate every step declared in *workflow*."""
    script_dir = _resolve_source(workflow, s3)
    instances: List[Step] = []
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
        s3: Optional[Any] = None,
        s3_keys: Optional[List[str]] = None,
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
        steps: List[Step],
        s3: Optional[Any] = None,
        s3_keys: Optional[List[str]] = None,
        input_bucket: str = "",
        output_bucket: str = "",
    ) -> "WorkflowPipeline":
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

    def ingest(self, config: PipelineConfig) -> List[DataItem]:
        return [
            DataItem(key=k, source=f"s3://{self._input_bucket}/{k}")
            for k in self._s3_keys
        ]

    def compute(
        self, config: PipelineConfig, items: List[DataItem],
    ) -> List[TaskHandle]:
        """Chain steps: step N+1's input = step N's output."""
        run_id = uuid.uuid4().hex[:12]
        current_keys = [item.key for item in items]
        current_bucket = self._input_bucket
        shared_metadata: Dict[str, Any] = {}

        last_embeddings: Optional[Any] = None
        last_checksums: List[str] = []
        last_metadata: List[Dict[str, Any]] = []

        for idx, (step, spec) in enumerate(
            zip(self._step_instances, self.workflow.steps)
        ):
            is_last = idx == len(self._step_instances) - 1
            step_output_bucket = (
                self._output_bucket if is_last else self._input_bucket
            )
            step_prefix = (
                "" if is_last
                else f"_staging/{run_id}/step_{idx}_{spec.name}/"
            )

            self._log.info(
                "  step %d/%d [%s]: %d keys → bucket=%s",
                idx + 1, len(self._step_instances), spec.name,
                len(current_keys), step_output_bucket,
            )

            ctx = StepContext(
                s3=self._s3,
                input_bucket=current_bucket,
                input_keys=current_keys,
                output_bucket=step_output_bucket,
                config=spec.config,
                metadata=shared_metadata,
                run_id=run_id,
            )
            result = step.run(ctx)

            if result.output_keys:
                current_keys = result.output_keys
                current_bucket = step_output_bucket

            if result.embeddings is not None:
                last_embeddings = result.embeddings
            if result.checksums:
                last_checksums = result.checksums
            if result.metadata:
                last_metadata = result.metadata

        # Teardown all steps
        for step in self._step_instances:
            try:
                step.teardown()
            except Exception:
                logger.warning("teardown failed for step '%s'", step.name,
                               exc_info=True)

        # Clean up staging artifacts
        self._cleanup_staging(self._s3, self._input_bucket, run_id)

        # Store results for collect() phase
        self._run_result = (
            last_embeddings, last_checksums, last_metadata, current_keys
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
                    "input_keys": [item.key for item in items],
                },
            )
        ]

    def collect(
        self, config: PipelineConfig, handles: List[TaskHandle],
    ) -> List[Result]:
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
        s3_keys: List[str],
        path_to_bucket: str,
        output_bucket: str,
    ) -> Tuple[List[Any], List[str], List[Dict[str, Any]], List[Optional[str]]]:
        """Execute like ``WorkflowExecutor.run()`` — drop-in for
        ``EmbeddingModel.compute_embeddings``.

        Returns ``(embeddings, checksums, metadata_list, saved_keys)``.
        """
        run_id = uuid.uuid4().hex[:12]
        logger.info(
            "Workflow '%s' run=%s: %d steps, %d input keys",
            self.workflow.name, run_id, len(self._step_instances),
            len(s3_keys),
        )

        current_keys = list(s3_keys)
        current_bucket = path_to_bucket
        shared_metadata: Dict[str, Any] = {}

        last_embeddings: Optional[Any] = None
        last_checksums: List[str] = []
        last_metadata: List[Dict[str, Any]] = []

        for idx, (step, spec) in enumerate(
            zip(self._step_instances, self.workflow.steps)
        ):
            is_last = idx == len(self._step_instances) - 1
            step_output_bucket = (
                output_bucket if is_last else path_to_bucket
            )

            logger.info(
                "  step %d/%d [%s]: %d keys → bucket=%s",
                idx + 1, len(self._step_instances), spec.name,
                len(current_keys), step_output_bucket,
            )

            ctx = StepContext(
                s3=s3,
                input_bucket=current_bucket,
                input_keys=current_keys,
                output_bucket=step_output_bucket,
                config=spec.config,
                metadata=shared_metadata,
                run_id=run_id,
            )
            result = step.run(ctx)

            if result.output_keys:
                current_keys = result.output_keys
                current_bucket = step_output_bucket

            if result.embeddings is not None:
                last_embeddings = result.embeddings
            if result.checksums:
                last_checksums = result.checksums
            if result.metadata:
                last_metadata = result.metadata

        # Build return tuple matching the compute_embeddings contract
        if not last_checksums:
            last_checksums = self._collect_checksums(s3, path_to_bucket, s3_keys)
        if not last_metadata:
            last_metadata = self._collect_metadata(s3, path_to_bucket, s3_keys)

        if last_embeddings is not None:
            embeddings_list: List[Any] = list(last_embeddings)
        else:
            embeddings_list = [None] * len(s3_keys)

        saved_keys: List[Optional[str]] = (
            list(current_keys)
            if current_keys != list(s3_keys)
            else [None] * len(s3_keys)
        )

        # Teardown
        for step in self._step_instances:
            try:
                step.teardown()
            except Exception:
                logger.warning("teardown failed for step '%s'", step.name,
                               exc_info=True)

        self._cleanup_staging(s3, path_to_bucket, run_id)

        logger.info("Workflow '%s' run=%s complete.", self.workflow.name, run_id)
        return embeddings_list, last_checksums, last_metadata, saved_keys

    # ── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _collect_checksums(
        s3: Any, bucket: str, keys: List[str],
    ) -> List[str]:
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
        s3: Any, bucket: str, keys: List[str],
    ) -> List[Dict[str, Any]]:
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
        s3: Optional[Any], bucket: str, run_id: str,
    ) -> None:
        """Delete intermediate staging objects for this run."""
        if s3 is None or not bucket:
            return
        prefix = f"_staging/{run_id}/"
        try:
            paginator = s3.get_paginator("list_objects_v2")
            to_delete: List[Dict[str, str]] = []
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
            logger.debug("Staging cleanup failed for run=%s", run_id,
                         exc_info=True)
