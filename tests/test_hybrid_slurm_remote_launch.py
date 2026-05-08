"""Tests for hybrid Slurm runtime-wrapper launch boundaries."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from umrs.model_registry.models import ExecutionAuthority
from umrs.orchestrator.hybrid.composer import HybridOrchestrator
from umrs.orchestrator.hybrid.targets import AllocationInfo, AllocationStatus
from umrs.orchestrator.hybrid.wrappers.base import ContainerEndpoint, LaunchSpec
from umrs.orchestrator.hybrid.wrappers.singularity import SingularityWrapper


def _run(coro):
    """Run an async helper in a fresh event loop."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _slurm_allocation(node: str = "gpu001") -> AllocationInfo:
    return AllocationInfo(
        job_id="12345",
        node=node,
        status=AllocationStatus.ALLOCATED,
        metadata={"execution_authority": "slurm"},
    )


def test_singularity_wrapper_launch_uses_srun_for_slurm_allocation():
    """Slurm allocations launch Singularity through the allocated job/node."""
    wrapper = SingularityWrapper(nvidia_support=False)
    spec = LaunchSpec(
        image="/images/mini-rivault.sif",
        command=["litellm"],
        args=["--port", "11434"],
    )

    cmd = wrapper._build_command(
        _slurm_allocation(),
        spec,
        "/images/mini-rivault.sif",
    )

    assert cmd[:8] == [
        "srun",
        "--jobid=12345",
        "--nodes=1",
        "--ntasks=1",
        "--overlap",
        "--nodelist",
        "gpu001",
        "singularity",
    ]
    assert "exec" in cmd
    assert "/images/mini-rivault.sif" in cmd


def test_hybrid_orchestrator_tags_allocation_authority():
    """Composer tags allocations before handing them to wrappers."""
    orchestrator = HybridOrchestrator.__new__(HybridOrchestrator)
    orchestrator._authority = ExecutionAuthority.SLURM
    allocation = AllocationInfo(
        job_id="12345",
        node="gpu001",
        status=AllocationStatus.ALLOCATED,
    )

    orchestrator._mark_allocation_authority(allocation)

    assert allocation.metadata["execution_authority"] == "slurm"


def test_singularity_wrapper_keeps_non_slurm_launch_local():
    """Non-Slurm allocations keep the existing local Singularity command."""
    wrapper = SingularityWrapper(nvidia_support=False)
    allocation = AllocationInfo(
        job_id="compose-1",
        node="localhost",
        status=AllocationStatus.ALLOCATED,
    )

    cmd = wrapper._build_command(
        allocation,
        LaunchSpec(image="/images/dev.sif"),
        "/images/dev.sif",
    )

    assert cmd[0] == "singularity"
    assert "srun" not in cmd


def test_singularity_remote_liveness_uses_srun_on_allocated_node():
    """Remote launch liveness probes run through the same Slurm job/node."""
    wrapper = SingularityWrapper()
    endpoint = ContainerEndpoint(
        node="gpu001",
        port=11434,
        container_id="singularity-12345",
        metadata={
            "remote_launch": True,
            "job_id": "12345",
            "image": "/images/mini-rivault.sif",
        },
    )
    proc = AsyncMock()
    proc.communicate.return_value = (b"", b"")
    proc.returncode = 0

    with patch("asyncio.create_subprocess_exec", return_value=proc) as mock_exec:
        assert _run(wrapper.is_running(endpoint)) is True

    args = mock_exec.call_args.args
    assert args[:8] == (
        "srun",
        "--jobid=12345",
        "--nodes=1",
        "--ntasks=1",
        "--overlap",
        "--nodelist",
        "gpu001",
        "pgrep",
    )


def test_singularity_launch_rejects_slurm_allocation_without_node():
    """Hybrid Slurm cannot launch without an allocated node coordinate."""
    wrapper = SingularityWrapper()
    allocation = _slurm_allocation(node="")

    with pytest.raises(Exception, match="no node"):
        _run(wrapper.launch(allocation, LaunchSpec(image="/images/mini.sif")))
