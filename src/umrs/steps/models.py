"""
Workflow definition data model and JSON schema validation (U-202).

A *workflow* is defined by a JSON file (``workflow.json``) that
describes the source location, Python requirements, resource limits,
and an ordered list of processing steps.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class SourceType(Enum):
    """Where the user's Python scripts live."""

    LOCAL_PATH = "local_path"
    S3 = "s3"
    GIT = "git"


# ---------------------------------------------------------------------------
# Component dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SourceSpec:
    """Location of the user's Python scripts."""

    type: SourceType
    path: str                          # filesystem path / s3 URI / git URL
    ref: Optional[str] = None          # git commit hash (required for git)

    def validate(self) -> None:
        if self.type == SourceType.GIT and not self.ref:
            raise ValueError("Git source requires 'ref' (commit hash)")
        if self.type == SourceType.LOCAL_PATH and not self.path.startswith("/"):
            raise ValueError("Local source path must be absolute")
        if self.type == SourceType.S3 and not self.path.startswith("s3://"):
            raise ValueError("S3 source must start with s3://")

    # -- serialisation -------------------------------------------------------

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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gpu_count": self.gpu_count,
            "cpu_limit": self.cpu_limit,
            "memory_limit": self.memory_limit,
            "timeout_seconds": self.timeout_seconds,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResourceSpec":
        return cls(
            gpu_count=data.get("gpu_count", 0),
            cpu_limit=data.get("cpu_limit", "1"),
            memory_limit=data.get("memory_limit", "1Gi"),
            timeout_seconds=data.get("timeout_seconds", 3600),
        )


@dataclass
class StepSpec:
    """One processing step inside a workflow."""

    name: str
    entrypoint: str                    # "module:ClassName"
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

    return_type: str = "file"                # "file" | "vector"
    embedding_vector_size: int = 0           # non-zero when return_type == "vector"

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


# ---------------------------------------------------------------------------
# Top-level workflow definition
# ---------------------------------------------------------------------------

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

    # -- validation ----------------------------------------------------------

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

    # -- serialisation -------------------------------------------------------

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
