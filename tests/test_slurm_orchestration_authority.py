"""Guards for the selected Slurm orchestration authority."""

from __future__ import annotations

from pathlib import Path

import pytest


def test_factory_does_not_register_hybrid_slurm_by_default() -> None:
    from umrs.orchestrator.factory import _BACKEND_REGISTRY, _lazy_register_builtins

    _lazy_register_builtins()

    assert "slurm" in _BACKEND_REGISTRY
    assert "slurm-podman" in _BACKEND_REGISTRY
    assert not {"hybrid", "hybrid-slurm", "slurm-hybrid"} & set(_BACKEND_REGISTRY)


def test_factory_rejects_hybrid_slurm_backend_name() -> None:
    from umrs.orchestrator.factory import get_orchestrator

    with pytest.raises(ValueError, match="experimental, not for HPC production"):
        get_orchestrator("hybrid-slurm")


def test_hybrid_slurm_requires_explicit_experimental_opt_in(monkeypatch) -> None:
    from umrs.orchestrator.hybrid.composer import HybridOrchestrator

    monkeypatch.delenv("RIVAULT_ENABLE_EXPERIMENTAL_HYBRID_SLURM", raising=False)

    with pytest.raises(RuntimeError, match="experimental, not for HPC production"):
        HybridOrchestrator(profile="slurm-singularity")


def test_hybrid_slurm_opt_in_is_explicit(monkeypatch) -> None:
    from umrs.orchestrator.hybrid.composer import HybridOrchestrator
    from umrs.model_registry.models import ExecutionAuthority

    monkeypatch.setenv("RIVAULT_ENABLE_EXPERIMENTAL_HYBRID_SLURM", "true")

    orchestrator = HybridOrchestrator(profile="slurm-singularity")

    assert orchestrator.execution_authority == ExecutionAuthority.SLURM
    assert orchestrator.backend_name == "hybrid-slurm"


def test_hpc_runner_defaults_to_slurm_podman() -> None:
    script = (Path(__file__).resolve().parents[2] / "tests/run_hpc_tests.sh").read_text()

    assert 'RIVAULT_ORCHESTRATOR="${RIVAULT_ORCHESTRATOR:-slurm-podman}"' in script
    assert 'check_prerequisite "Podman"' in script
    assert "RIVAULT_ENABLE_EXPERIMENTAL_HYBRID_SLURM" not in script
