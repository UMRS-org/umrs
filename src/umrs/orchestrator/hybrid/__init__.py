"""
Hybrid Orchestration Module (U-1101 to U-1124).

Provides composable orchestration through:
- ExecutionTarget: Resource allocation (Slurm, K8s, Compose)
- RuntimeWrapper: Container launch (Singularity, Docker, K8s)
- ModelRefResolver: Model path resolution per authority

Design Principle: Separate resource allocation from container launch to enable
clean composition across deployment scenarios without orchestrator recursion.
"""

from .targets import (
    ExecutionTarget,
    SlurmTarget,
    K8sTarget,
    ComposeTarget,
    AllocationInfo,
    AllocationStatus,
)
from .wrappers import (
    RuntimeWrapper,
    SingularityWrapper,
    DockerWrapper,
    K8sJobWrapper,
    K8sBootstrapWrapper,
    K8sOnSlurmNotEnabledError,
    NetworkUnreachableError,
    LaunchSpec,
    MountSpec,
)
from .resolvers import (
    ModelRefResolver,
    FilesystemResolver,
    PVCResolver,
    HuggingFaceResolver,
    ResolvedModel,
)
from .composer import HybridOrchestrator
from .k8s_slurm_scheduler import (
    K8sToSlurmTranslator,
    K8sOnSlurmScheduler,
)

__all__ = [
    # Targets
    "ExecutionTarget",
    "SlurmTarget",
    "K8sTarget",
    "ComposeTarget",
    "AllocationInfo",
    "AllocationStatus",
    # Wrappers
    "RuntimeWrapper",
    "SingularityWrapper",
    "DockerWrapper",
    "K8sJobWrapper",
    "K8sBootstrapWrapper",
    "K8sOnSlurmNotEnabledError",
    "NetworkUnreachableError",
    "LaunchSpec",
    "MountSpec",
    # Resolvers
    "ModelRefResolver",
    "FilesystemResolver",
    "PVCResolver",
    "HuggingFaceResolver",
    "ResolvedModel",
    # Composer
    "HybridOrchestrator",
    # K8s-on-Slurm (U-1131-1134)
    "K8sToSlurmTranslator",
    "K8sOnSlurmScheduler",
]
