"""
UMRS Pipeline SDK
=================

Provides the abstract ``Pipeline`` base class, ``Step`` base class for
workflow steps, ``WorkflowPipeline`` for JSON-driven workflows, and
connectors for building user-defined AI pipelines on UMRS + Rivault.

Quick start (full Pipeline)::

    from umrs.pipeline_sdk import Pipeline, PipelineConfig, DataItem, TaskHandle, Result

Quick start (workflow.json steps)::

    from umrs.pipeline_sdk import Step, StepContext, StepResult

See ``demos/12_pipeline_sdk/README.md`` for the full guide on creating a new pipeline,
and ``demos/08_user_model/`` for a workflow.json example.
"""

from .adapter import EmbeddingAdapter
from .base import (
    ComputeBackend,
    DataItem,
    Phase,
    Pipeline,
    PipelineConfig,
    Result,
    TaskHandle,
)
from .step import Step, StepContext, StepResult
from .workflow import (
    OutputSpec,
    ResourceSpec,
    ScriptLoadError,
    SourceSpec,
    SourceType,
    StepSpec,
    WorkflowDefinition,
    WorkflowPipeline,
)

__all__ = [
    # Pipeline SDK core
    "ComputeBackend",
    "DataItem",
    "EmbeddingAdapter",
    "Phase",
    "Pipeline",
    "PipelineConfig",
    "Result",
    "TaskHandle",
    # Step interface
    "Step",
    "StepContext",
    "StepResult",
    # Workflow (JSON-driven)
    "OutputSpec",
    "ResourceSpec",
    "ScriptLoadError",
    "SourceSpec",
    "SourceType",
    "StepSpec",
    "WorkflowDefinition",
    "WorkflowPipeline",
]
