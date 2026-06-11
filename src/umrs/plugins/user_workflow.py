"""
Unified UserWorkflow model and runner (Phase 3).

``UserWorkflow`` merges the best of two worlds:

- **WorkflowDefinition** (Pipeline SDK) — multi-step JSON workflows
- **PluginSpec** (Plugin system)          — Slurm/K8s sandboxed execution

``UserWorkflowRunner`` orchestrates step execution across backends:

- ``in_process`` — run steps directly in the server process (current default)
- ``subprocess`` — run each step in an isolated subprocess
- ``slurm``      — one Slurm job per step (HPC GPU via Singularity)
- ``k8s``        — one Kubernetes Job per step

Usage::

    from umrs.plugins.user_workflow import UserWorkflow, UserWorkflowRunner

    uw = UserWorkflow.from_workflow_definition(wf_def)
    runner = UserWorkflowRunner(uw)
    embeddings, checksums, metadata, keys = runner.run(
        s3, s3_keys, input_bucket, output_bucket
    )
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from umrs.pipeline_sdk.step import Step, StepContext
from umrs.pipeline_sdk.workflow import (
    ExecutionMode,
    OutputSpec,
    ResourceSpec,
    SourceSpec,
    StepSpec,
    WorkflowDefinition,
    _resolve_source,
    load_steps,
)

logger = logging.getLogger(__name__)


# =========================================================================
# UserWorkflow — superset dataclass
# =========================================================================


@dataclass
class UserWorkflow:
    """Unified workflow specification.

    Extends :class:`WorkflowDefinition` fields with execution-context
    metadata borrowed from :class:`PluginSpec`:

    - ``execution_mode``      — how to run (in_process | subprocess | slurm | k8s)
    - ``env_vars``            — extra environment variables for remote execution
    - ``output_destinations`` — approved output URIs (s3://, pgvector://, umrs://)
    - ``user_id``             — owning user (for isolation/audit)
    - ``project_id``          — project context
    """

    # ── WorkflowDefinition fields ────────────────────────────────────────
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

    # ── PluginSpec-like additions ────────────────────────────────────────
    env_vars: dict[str, str] = field(default_factory=dict)
    output_destinations: list[str] = field(default_factory=list)
    user_id: str | None = None
    project_id: str | None = None

    # ── Convenience constructors ─────────────────────────────────────────

    @classmethod
    def from_workflow_definition(
        cls,
        wf: WorkflowDefinition,
        *,
        env_vars: dict[str, str] | None = None,
        output_destinations: list[str] | None = None,
        user_id: str | None = None,
        project_id: str | None = None,
    ) -> UserWorkflow:
        """Wrap an existing :class:`WorkflowDefinition`."""
        return cls(
            name=wf.name,
            source=wf.source,
            steps=wf.steps,
            version=wf.version,
            description=wf.description,
            requirements=wf.requirements,
            env=wf.env,
            resources=wf.resources,
            output=wf.output,
            execution_mode=wf.execution_mode,
            env_vars=env_vars or {},
            output_destinations=output_destinations or [],
            user_id=user_id,
            project_id=project_id,
        )

    def to_workflow_definition(self) -> WorkflowDefinition:
        """Down-cast to a plain :class:`WorkflowDefinition`."""
        return WorkflowDefinition(
            name=self.name,
            source=self.source,
            steps=self.steps,
            version=self.version,
            description=self.description,
            requirements=self.requirements,
            env=self.env,
            resources=self.resources,
            output=self.output,
            execution_mode=self.execution_mode,
        )

    def to_dict(self) -> dict[str, Any]:
        d = self.to_workflow_definition().to_dict()
        if self.env_vars:
            d["env_vars"] = self.env_vars
        if self.output_destinations:
            d["output_destinations"] = self.output_destinations
        if self.user_id:
            d["user_id"] = self.user_id
        if self.project_id:
            d["project_id"] = self.project_id
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UserWorkflow:
        wf = WorkflowDefinition.from_dict(data)
        return cls.from_workflow_definition(
            wf,
            env_vars=data.get("env_vars", {}),
            output_destinations=data.get("output_destinations", []),
            user_id=data.get("user_id"),
            project_id=data.get("project_id"),
        )


# =========================================================================
# UserWorkflowRunner — orchestrator
# =========================================================================


class UserWorkflowRunner:
    """Execute a :class:`UserWorkflow` using the configured backend.

    This is the single entry-point for all execution modes.  Callers
    interact with the same API regardless of whether steps run in-process,
    via Slurm, or on Kubernetes.
    """

    def __init__(self, workflow: UserWorkflow, s3: Any | None = None) -> None:
        self.workflow = workflow
        self._s3 = s3
        self._step_instances: list[Step] | None = None

    # ── Public API ───────────────────────────────────────────────────────

    def run(
        self,
        s3: Any,
        s3_keys: list[str],
        input_bucket: str,
        output_bucket: str,
    ) -> tuple[list[Any], list[str], list[dict[str, Any]], list[str | None]]:
        """Execute workflow and return ``(embeddings, checksums, metadata, saved_keys)``.

        This matches the ``EmbeddingModel.compute_embeddings`` contract.
        """
        mode = self.workflow.execution_mode

        if mode == ExecutionMode.IN_PROCESS:
            return self._run_in_process(s3, s3_keys, input_bucket, output_bucket)
        elif mode == ExecutionMode.SUBPROCESS:
            return self._run_subprocess(s3, s3_keys, input_bucket, output_bucket)
        elif mode == ExecutionMode.SLURM:
            return self._run_slurm(s3, s3_keys, input_bucket, output_bucket)
        elif mode == ExecutionMode.K8S:
            return self._run_k8s(s3, s3_keys, input_bucket, output_bucket)
        else:
            raise ValueError(f"Unknown execution mode: {mode}")

    # ── In-process execution (current behaviour) ─────────────────────────

    def _run_in_process(
        self,
        s3: Any,
        s3_keys: list[str],
        input_bucket: str,
        output_bucket: str,
    ) -> tuple[list[Any], list[str], list[dict[str, Any]], list[str | None]]:
        """Run steps directly — identical to the original WorkflowPipeline."""
        steps = self._ensure_steps(s3)

        run_id = uuid.uuid4().hex[:12]
        logger.info(
            "UserWorkflowRunner[in_process] '%s' run=%s: %d steps, %d keys",
            self.workflow.name,
            run_id,
            len(steps),
            len(s3_keys),
        )

        current_keys = list(s3_keys)
        current_bucket = input_bucket
        shared_metadata: dict[str, Any] = {}

        last_embeddings: Any | None = None
        last_checksums: list[str] = []
        last_metadata: list[dict[str, Any]] = []

        for idx, (step, spec) in enumerate(
            zip(steps, self.workflow.steps, strict=False)
        ):
            # Keep all step outputs in the workflow output bucket so
            # intermediate artifacts do not retrigger the watched input bucket.
            step_output_bucket = output_bucket

            logger.info(
                "  step %d/%d [%s]: %d keys → bucket=%s",
                idx + 1,
                len(steps),
                spec.name,
                len(current_keys),
                step_output_bucket,
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

        # Teardown
        for step in steps:
            try:
                step.teardown()
            except Exception:
                logger.warning(
                    "teardown failed for step '%s'", step.name, exc_info=True
                )

        # Cleanup staging
        self._cleanup_staging(s3, input_bucket, run_id)

        # Build return tuple
        if not last_checksums:
            last_checksums = self._collect_checksums(s3, input_bucket, s3_keys)
        if not last_metadata:
            last_metadata = self._collect_metadata(s3, input_bucket, s3_keys)

        embeddings_list: list[Any] = (
            list(last_embeddings)
            if last_embeddings is not None
            else [None] * len(s3_keys)
        )
        saved_keys: list[str | None] = (
            list(current_keys)
            if current_keys != list(s3_keys)
            else [None] * len(s3_keys)
        )

        logger.info(
            "UserWorkflowRunner[in_process] '%s' run=%s complete.",
            self.workflow.name,
            run_id,
        )
        return embeddings_list, last_checksums, last_metadata, saved_keys

    # ── Subprocess execution ─────────────────────────────────────────────

    def _run_subprocess(
        self,
        s3: Any,
        s3_keys: list[str],
        input_bucket: str,
        output_bucket: str,
    ) -> tuple[list[Any], list[str], list[dict[str, Any]], list[str | None]]:
        """Run each step in an isolated subprocess."""
        raise NotImplementedError(
            "Subprocess execution mode is planned but not yet implemented. "
            "Use 'in_process' or 'slurm' instead."
        )

    # ── Slurm execution ──────────────────────────────────────────────────

    def _run_slurm(
        self,
        s3: Any,
        s3_keys: list[str],
        input_bucket: str,
        output_bucket: str,
    ) -> tuple[list[Any], list[str], list[dict[str, Any]], list[str | None]]:
        """Submit each step as a Slurm job via SlurmPluginRunner.

        Each step is wrapped into a :class:`PluginSpec` using
        :func:`step_to_plugin_spec` and submitted as an independent batch
        job.  Steps are chained sequentially: output keys from step N
        become input keys for step N+1.

        After each job completes the runner reads
        ``{work_dir}/{plugin_id}/result_manifest.json`` written by the
        in-container step executor to chain output_keys, embeddings,
        checksums, and metadata to the next step.
        """
        from .slurm_runner import SlurmPluginRunner
        from .step_adapter import step_to_plugin_spec

        run_id = uuid.uuid4().hex[:12]
        logger.info(
            "UserWorkflowRunner[slurm] '%s' run=%s: %d steps, %d keys",
            self.workflow.name,
            run_id,
            len(self.workflow.steps),
            len(s3_keys),
        )

        wf_def = self.workflow.to_workflow_definition()
        script_dir = str(_resolve_source(wf_def, s3))

        runner = SlurmPluginRunner(
            partition=self.workflow.resources.extra.get("slurm_partition", "a100"),
        )

        current_keys = list(s3_keys)
        current_bucket = input_bucket

        last_embeddings: Any | None = None
        last_checksums: list[str] = []
        last_metadata: list[dict[str, Any]] = []

        for idx, spec in enumerate(self.workflow.steps):
            step_output_bucket = output_bucket

            logger.info(
                "  [slurm] step %d/%d [%s]: submitting job",
                idx + 1,
                len(self.workflow.steps),
                spec.name,
            )

            plugin_spec = step_to_plugin_spec(
                step_name=spec.name,
                entrypoint=spec.entrypoint,
                script_dir=script_dir,
                step_config=spec.config,
                input_bucket=current_bucket,
                input_keys=current_keys,
                output_bucket=step_output_bucket,
                run_id=run_id,
                cpu_limit=self.workflow.resources.cpu_limit,
                memory_limit=self.workflow.resources.memory_limit,
                gpu_count=self.workflow.resources.gpu_count,
                timeout_seconds=self.workflow.resources.timeout_seconds,
                env_vars={**self.workflow.env, **self.workflow.env_vars},
                user_id=self.workflow.user_id,
                output_destinations=self.workflow.output_destinations,
            )

            # runner.execute() is async — run it in an event loop.
            result = asyncio.run(runner.execute(plugin_spec))

            if result.status.value != "succeeded":
                raise RuntimeError(
                    f"Step '{spec.name}' failed on Slurm: "
                    f"{result.error_message or result.stderr}"
                )

            manifest = self._resolve_step_manifest(runner, result)
            if manifest:
                if manifest.get("output_keys"):
                    current_keys = manifest["output_keys"]
                    current_bucket = step_output_bucket
                if manifest.get("embeddings") is not None:
                    last_embeddings = manifest["embeddings"]
                if manifest.get("checksums"):
                    last_checksums = manifest["checksums"]
                if manifest.get("metadata"):
                    last_metadata = manifest["metadata"]

            logger.info(
                "  [slurm] step %d/%d [%s]: completed",
                idx + 1,
                len(self.workflow.steps),
                spec.name,
            )

        # Build return tuple
        embeddings_list: list[Any] = (
            list(last_embeddings)
            if last_embeddings is not None
            else [None] * len(s3_keys)
        )
        saved_keys: list[str | None] = (
            list(current_keys)
            if current_keys != list(s3_keys)
            else [None] * len(s3_keys)
        )

        return embeddings_list, last_checksums, last_metadata, saved_keys

    # ── Kubernetes execution ─────────────────────────────────────────────

    def _run_k8s(
        self,
        s3: Any,
        s3_keys: list[str],
        input_bucket: str,
        output_bucket: str,
    ) -> tuple[list[Any], list[str], list[dict[str, Any]], list[str | None]]:
        """Submit each step as a Kubernetes Job via K8sPluginRunner.

        Follows the same pattern as ``_run_slurm`` but uses the K8s
        backend. Output chaining only happens when the backend exposes
        a retrievable step manifest.
        """
        from .k8s_runner import K8sPluginRunner
        from .step_adapter import step_to_plugin_spec

        run_id = uuid.uuid4().hex[:12]
        logger.info(
            "UserWorkflowRunner[k8s] '%s' run=%s: %d steps, %d keys",
            self.workflow.name,
            run_id,
            len(self.workflow.steps),
            len(s3_keys),
        )

        wf_def = self.workflow.to_workflow_definition()
        script_dir = str(_resolve_source(wf_def, s3))

        runner = K8sPluginRunner(
            namespace=self.workflow.resources.extra.get("k8s_namespace", "umrs"),
        )

        current_keys = list(s3_keys)
        current_bucket = input_bucket

        last_embeddings: Any | None = None
        last_checksums: list[str] = []
        last_metadata: list[dict[str, Any]] = []

        for idx, spec in enumerate(self.workflow.steps):
            step_output_bucket = output_bucket

            logger.info(
                "  [k8s] step %d/%d [%s]: submitting job",
                idx + 1,
                len(self.workflow.steps),
                spec.name,
            )

            plugin_spec = step_to_plugin_spec(
                step_name=spec.name,
                entrypoint=spec.entrypoint,
                script_dir=script_dir,
                step_config=spec.config,
                input_bucket=current_bucket,
                input_keys=current_keys,
                output_bucket=step_output_bucket,
                run_id=run_id,
                cpu_limit=self.workflow.resources.cpu_limit,
                memory_limit=self.workflow.resources.memory_limit,
                gpu_count=self.workflow.resources.gpu_count,
                timeout_seconds=self.workflow.resources.timeout_seconds,
                env_vars={**self.workflow.env, **self.workflow.env_vars},
                user_id=self.workflow.user_id,
                output_destinations=self.workflow.output_destinations,
            )

            # runner.execute() is async — run it in an event loop.
            result = asyncio.run(runner.execute(plugin_spec))

            if result.status.value != "succeeded":
                raise RuntimeError(
                    f"Step '{spec.name}' failed on K8s: "
                    f"{result.error_message or result.stderr}"
                )

            manifest = self._resolve_step_manifest(runner, result)
            if manifest:
                if manifest.get("output_keys"):
                    current_keys = manifest["output_keys"]
                    current_bucket = step_output_bucket
                if manifest.get("embeddings") is not None:
                    last_embeddings = manifest["embeddings"]
                if manifest.get("checksums"):
                    last_checksums = manifest["checksums"]
                if manifest.get("metadata"):
                    last_metadata = manifest["metadata"]

            logger.info(
                "  [k8s] step %d/%d [%s]: completed",
                idx + 1,
                len(self.workflow.steps),
                spec.name,
            )

        # Build return tuple
        embeddings_list: list[Any] = (
            list(last_embeddings)
            if last_embeddings is not None
            else [None] * len(s3_keys)
        )
        saved_keys: list[str | None] = (
            list(current_keys)
            if current_keys != list(s3_keys)
            else [None] * len(s3_keys)
        )

        return embeddings_list, last_checksums, last_metadata, saved_keys

    # ── Internal helpers ─────────────────────────────────────────────────

    @staticmethod
    def _resolve_step_manifest(
        runner: Any,
        result: Any,
    ) -> dict[str, Any] | None:
        """Resolve a step manifest from backend-specific result surfaces."""
        metrics = getattr(result, "metrics", None)
        if isinstance(metrics, dict):
            manifest = metrics.get("result_manifest")
            if isinstance(manifest, dict):
                return manifest

        work_dir = getattr(runner, "work_dir", None)
        if work_dir is not None:
            return UserWorkflowRunner._read_step_manifest(
                Path(work_dir),
                result.plugin_id,
            )

        logger.info(
            "Runner %s does not expose a retrievable step manifest; "
            "skipping output chaining",
            type(runner).__name__,
        )
        return None

    @staticmethod
    def _read_step_manifest(
        work_dir: Path,
        plugin_id: str,
    ) -> dict[str, Any] | None:
        """Read result_manifest.json written by the in-container step executor.

        Returns the parsed manifest dict, or ``None`` if not found.
        """
        manifest_path = work_dir / plugin_id / "result_manifest.json"
        if not manifest_path.is_file():
            logger.warning(
                "No result manifest at %s — step chaining unavailable",
                manifest_path,
            )
            return None
        try:
            return json.loads(manifest_path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to read manifest %s: %s", manifest_path, exc)
            return None

    def _ensure_steps(self, s3: Any | None = None) -> list[Step]:
        """Load step instances lazily (in_process mode only)."""
        if self._step_instances is None:
            wf_def = self.workflow.to_workflow_definition()
            self._step_instances = load_steps(wf_def, s3)
        return self._step_instances

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
