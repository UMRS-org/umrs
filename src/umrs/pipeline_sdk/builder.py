"""
B8 — Pipeline Builder (Programmatic API)

Fluent API for constructing pipeline workflows programmatically, as an
alternative to writing workflow.json files manually.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


class PipelineBuilderError(Exception):
    """Raised when pipeline construction fails."""


@dataclass
class StepBuilder:
    """Per-step configuration builder."""
    name: str
    entrypoint: str
    depends_on: List[str] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)
    max_retries: int = 0
    retry_delay: float = 1.0
    extends: Optional[str] = None

    def with_config(self, key: str, value: Any) -> "StepBuilder":
        self.config[key] = value
        return self

    def with_retry(self, max_retries: int, delay: float = 1.0) -> "StepBuilder":
        self.max_retries = max_retries
        self.retry_delay = delay
        return self

    def extending(self, base_step: str) -> "StepBuilder":
        self.extends = base_step
        return self

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "name": self.name,
            "entrypoint": self.entrypoint,
        }
        if self.depends_on:
            d["depends_on"] = self.depends_on
        if self.config:
            d["config"] = self.config
        if self.max_retries > 0:
            d["max_retries"] = self.max_retries
            d["retry_delay"] = self.retry_delay
        if self.extends:
            d["extends"] = self.extends
        return d


class PipelineBuilder:
    """Fluent builder for WorkflowDefinition-compatible dicts.

    Usage::

        workflow = (
            PipelineBuilder("my-analysis")
            .add_step("preprocess", "steps/preprocess.py:Preprocess")
            .add_step("embed", "steps/embed.py:Embed", depends_on=["preprocess"])
            .add_step("search", "steps/search.py:Search", depends_on=["embed"])
            .set_execution_mode("IN_PROCESS")
            .set_source("/path/to/steps/")
            .build()
        )
    """

    VALID_EXECUTION_MODES = {"IN_PROCESS", "SUBPROCESS", "SLURM", "K8S"}
    VALID_SOURCE_TYPES = {"LOCAL_PATH", "S3", "GIT", "UPLOADED"}

    def __init__(self, name: str) -> None:
        self._name = name
        self._steps: List[StepBuilder] = []
        self._source_type: str = "LOCAL_PATH"
        self._source_path: str = "."
        self._source_ref: Optional[str] = None
        self._execution_mode: str = "IN_PROCESS"
        self._version: str = "1.0"
        self._description: str = ""
        self._requirements: List[str] = []
        self._env: Dict[str, str] = {}
        self._resources: Dict[str, Any] = {}
        self._output: Dict[str, Any] = {}

    def add_step(
        self,
        name: str,
        entrypoint: str,
        depends_on: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None,
        max_retries: int = 0,
    ) -> "PipelineBuilder":
        step = StepBuilder(
            name=name,
            entrypoint=entrypoint,
            depends_on=depends_on or [],
            config=config or {},
            max_retries=max_retries,
        )
        self._steps.append(step)
        return self

    def add_step_builder(self, step: StepBuilder) -> "PipelineBuilder":
        self._steps.append(step)
        return self

    def set_execution_mode(self, mode: str) -> "PipelineBuilder":
        if mode not in self.VALID_EXECUTION_MODES:
            raise PipelineBuilderError(
                f"Invalid execution mode: {mode}. Must be one of {self.VALID_EXECUTION_MODES}"
            )
        self._execution_mode = mode
        return self

    def set_source(
        self,
        path: str,
        source_type: str = "LOCAL_PATH",
        ref: Optional[str] = None,
    ) -> "PipelineBuilder":
        if source_type not in self.VALID_SOURCE_TYPES:
            raise PipelineBuilderError(
                f"Invalid source type: {source_type}. Must be one of {self.VALID_SOURCE_TYPES}"
            )
        self._source_type = source_type
        self._source_path = path
        self._source_ref = ref
        return self

    def set_version(self, version: str) -> "PipelineBuilder":
        self._version = version
        return self

    def set_description(self, description: str) -> "PipelineBuilder":
        self._description = description
        return self

    def add_requirement(self, package: str) -> "PipelineBuilder":
        self._requirements.append(package)
        return self

    def set_env(self, key: str, value: str) -> "PipelineBuilder":
        self._env[key] = value
        return self

    def set_resources(
        self,
        gpu_count: int = 0,
        cpu_limit: str = "1",
        memory_limit: str = "1Gi",
        timeout_seconds: int = 3600,
    ) -> "PipelineBuilder":
        self._resources = {
            "gpu_count": gpu_count,
            "cpu_limit": cpu_limit,
            "memory_limit": memory_limit,
            "timeout_seconds": timeout_seconds,
        }
        return self

    def set_output(
        self,
        return_type: str = "file",
        embedding_vector_size: int = 0,
    ) -> "PipelineBuilder":
        self._output = {
            "return_type": return_type,
            "embedding_vector_size": embedding_vector_size,
        }
        return self

    def validate(self) -> List[str]:
        """Validate the pipeline configuration. Returns list of errors."""
        errors: List[str] = []

        if not self._name:
            errors.append("Pipeline name is required")

        if not self._steps:
            errors.append("Pipeline must have at least one step")

        # Check for duplicate step names
        names = [s.name for s in self._steps]
        if len(names) != len(set(names)):
            dupes = [n for n in names if names.count(n) > 1]
            errors.append(f"Duplicate step names: {set(dupes)}")

        # Validate DAG: all depends_on references exist
        name_set = set(names)
        for step in self._steps:
            for dep in step.depends_on:
                if dep not in name_set:
                    errors.append(f"Step '{step.name}' depends on unknown step '{dep}'")

        # Check for cycles (simple DFS)
        cycle_err = self._check_cycles()
        if cycle_err:
            errors.append(cycle_err)

        return errors

    def _check_cycles(self) -> Optional[str]:
        adj: Dict[str, List[str]] = {s.name: s.depends_on for s in self._steps}
        visited: Set[str] = set()
        in_stack: Set[str] = set()

        def dfs(node: str) -> bool:
            visited.add(node)
            in_stack.add(node)
            for dep in adj.get(node, []):
                if dep in in_stack:
                    return True
                if dep not in visited:
                    if dfs(dep):
                        return True
            in_stack.discard(node)
            return False

        for name in adj:
            if name not in visited:
                if dfs(name):
                    return f"Cycle detected in DAG involving step '{name}'"
        return None

    def build(self) -> Dict[str, Any]:
        """Build the workflow definition dict. Validates DAG integrity."""
        errors = self.validate()
        if errors:
            raise PipelineBuilderError(f"Pipeline validation failed: {'; '.join(errors)}")

        workflow: Dict[str, Any] = {
            "name": self._name,
            "source": {
                "type": self._source_type,
                "path": self._source_path,
            },
            "execution_mode": self._execution_mode,
            "steps": [s.to_dict() for s in self._steps],
        }

        if self._source_ref:
            workflow["source"]["ref"] = self._source_ref
        if self._version:
            workflow["version"] = self._version
        if self._description:
            workflow["description"] = self._description
        if self._requirements:
            workflow["requirements"] = self._requirements
        if self._env:
            workflow["env"] = self._env
        if self._resources:
            workflow["resources"] = self._resources
        if self._output:
            workflow["output"] = self._output

        return workflow

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.build(), indent=indent)
