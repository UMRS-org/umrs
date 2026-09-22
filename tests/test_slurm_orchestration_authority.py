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


def test_hybrid_slurm_singularity_profile_retired_adr061(monkeypatch) -> None:
    """ADR-061: the slurm-singularity profile refuses (wrapper retired),
    opt-in env or not."""
    from umrs.orchestrator.hybrid.composer import HybridOrchestrator

    monkeypatch.delenv("RIVAULT_ENABLE_EXPERIMENTAL_HYBRID_SLURM", raising=False)
    with pytest.raises(RuntimeError):
        HybridOrchestrator(profile="slurm-singularity")

    monkeypatch.setenv("RIVAULT_ENABLE_EXPERIMENTAL_HYBRID_SLURM", "true")
    with pytest.raises(RuntimeError, match=r"retired \(ADR-061\)"):
        HybridOrchestrator(profile="slurm-singularity")


def test_factory_slurm_backend_name_retired_adr061() -> None:
    """ADR-061: the bare 'slurm' (Singularity) backend selection refuses by name."""
    from umrs.orchestrator.factory import get_orchestrator

    with pytest.raises(ValueError, match=r"retired \(ADR-061\)"):
        get_orchestrator("slurm")


def test_hpc_runner_defaults_to_slurm_podman() -> None:
    script = (Path(__file__).resolve().parents[2] / "tests/run_hpc_tests.sh").read_text()

    assert 'RIVAULT_ORCHESTRATOR="${RIVAULT_ORCHESTRATOR:-slurm-podman}"' in script
    assert 'check_prerequisite "Podman"' in script
    assert "RIVAULT_ENABLE_EXPERIMENTAL_HYBRID_SLURM" not in script
