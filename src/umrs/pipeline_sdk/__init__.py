"""
UMRS Pipeline SDK
=================

Provides the abstract ``Pipeline`` base class and connectors for building
user-defined AI pipelines on UMRS + Rivault.

Quick start::

    from umrs.pipeline_sdk import Pipeline, PipelineConfig, DataItem, TaskHandle, Result

See ``templates/README.md`` for the full guide on creating a new pipeline,
and ``templates/examples/`` for working examples.
"""

from .base import (
    ComputeBackend,
    DataItem,
    Phase,
    Pipeline,
    PipelineConfig,
    Result,
    TaskHandle,
)

__all__ = [
    "ComputeBackend",
    "DataItem",
    "Phase",
    "Pipeline",
    "PipelineConfig",
    "Result",
    "TaskHandle",
]
