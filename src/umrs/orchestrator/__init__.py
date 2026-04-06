"""
UMRS Orchestrator Module

Provides a unified interface for managing Mini-RiVault inference endpoints
across different orchestration backends (Slurm, Kubernetes, Docker Compose).

Usage:
    from umrs.orchestrator import get_orchestrator, get_registry

    orchestrator = get_orchestrator()  # Auto-detect from RIVAULT_ORCHESTRATOR env
    job = await orchestrator.submit_job(model_path="/models/rivault-gpt-oss")
    endpoint = await orchestrator.wait_for_ready(job.job_id)
    # ... use endpoint ...
    await orchestrator.cleanup(job.job_id)

Endpoint Registry:
    from umrs.orchestrator import get_registry

    registry = get_registry()
    registry.register(project_id, model, job_id, host, port, backend)
    endpoint = registry.get(project_id, model)

Readiness Polling (U-107 - Authoritative):
    from umrs.orchestrator import get_readiness_poller

    poller = get_readiness_poller()
    result = await poller.wait_for_ready(job_id, host, port)

Backend Detection (U-105):
    from umrs.orchestrator import detect_backend, BackendType

    result = detect_backend()
    print(f"Detected: {result.backend}")

Port Allocation (U-106):
    from umrs.orchestrator import get_port_manager

    port = get_port_manager().allocate(job_id, host)
"""

from .base import (
    OrchestratorBackend,
    JobInfo,
    JobStatus,
    EndpointInfo,
    OrchestratorConfig,
)
from .factory import get_orchestrator, register_backend
from .endpoint_registry import (
    EndpointRegistry,
    EndpointStatus,
    RegisteredEndpoint,
    get_registry,
)
from .readiness import (
    ReadinessPoller,
    ReadinessResult,
    ReadinessState,
    ReadinessConfig,
    get_readiness_poller,
)
from .detection import (
    BackendType,
    BackendDetector,
    DetectionResult,
    detect_backend,
    get_backend_type,
)
from .ports import (
    PortManager,
    PortAllocationConfig,
    PortAllocation,
    get_port_manager,
)
from .service import (
    OrchestrationService,
    ProvisionRequest,
    ProvisionResult,
    get_orchestration_service,
)
from .janitor import (
    Janitor,
    JanitorConfig,
    CleanupResult,
    get_janitor,
    start_janitor,
    stop_janitor,
)
from .durable import (
    DurableOrchestrationService,
    DurableProvisionRequest,
    get_durable_orchestration_service,
    init_durable_orchestration_service,
)
from .site_config import (
    SiteConfig,
    get_site_config,
    load_site_config,
    reset_site_config,
)

__all__ = [
    # Backend abstractions
    "OrchestratorBackend",
    "JobInfo",
    "JobStatus",
    "EndpointInfo",
    "OrchestratorConfig",
    "get_orchestrator",
    "register_backend",
    # Endpoint registry
    "EndpointRegistry",
    "EndpointStatus",
    "RegisteredEndpoint",
    "get_registry",
    # Readiness polling (U-107)
    "ReadinessPoller",
    "ReadinessResult",
    "ReadinessState",
    "ReadinessConfig",
    "get_readiness_poller",
    # Backend detection (U-105)
    "BackendType",
    "BackendDetector",
    "DetectionResult",
    "detect_backend",
    "get_backend_type",
    # Port allocation (U-106)
    "PortManager",
    "PortAllocationConfig",
    "PortAllocation",
    "get_port_manager",
    # Orchestration service (integration)
    "OrchestrationService",
    "ProvisionRequest",
    "ProvisionResult",
    "get_orchestration_service",
    # Janitor (U-111, U-112)
    "Janitor",
    "JanitorConfig",
    "CleanupResult",
    "get_janitor",
    "start_janitor",
    "stop_janitor",
    # Durable Orchestration (U-701+)
    "DurableOrchestrationService",
    "DurableProvisionRequest",
    "get_durable_orchestration_service",
    "init_durable_orchestration_service",
    # Site configuration
    "SiteConfig",
    "get_site_config",
    "load_site_config",
    "reset_site_config",
]
