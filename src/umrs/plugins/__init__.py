"""
Plugin system for UMRS (X-101 through X-106).

Allows users to provide custom code executed in sandboxed environments.
Supports both Slurm and Kubernetes as execution backends.
"""

from .models import (
    PluginSpec,
    PluginSourceType,
    PluginResult,
    PluginAuditEntry,
    PluginStatus,
)
from .runner import PluginRunner
from .slurm_runner import SlurmPluginRunner
from .k8s_runner import K8sPluginRunner
from .result_writer import PluginResultWriter
from .audit import PluginAuditLog
from .step_adapter import step_to_plugin_spec
from .user_workflow import UserWorkflow, UserWorkflowRunner

__all__ = [
    "PluginSpec",
    "PluginSourceType",
    "PluginResult",
    "PluginAuditEntry",
    "PluginStatus",
    "PluginRunner",
    "SlurmPluginRunner",
    "K8sPluginRunner",
    "PluginResultWriter",
    "PluginAuditLog",
    "step_to_plugin_spec",
    "UserWorkflow",
    "UserWorkflowRunner",
]
