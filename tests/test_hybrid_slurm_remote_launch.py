"""ADR-061 retirement pins: the Singularity sandbox lane refuses by name.

The wrapper class stays importable so references fail at CONSTRUCTION with
a named error (not an ImportError smudge); the hybrid composer's
slurm-singularity profile dies the same way, opt-in env or not.
"""

from __future__ import annotations

import pytest


def test_singularity_wrapper_refuses_at_construction():
    from umrs.orchestrator.hybrid.wrappers.singularity import SingularityWrapper

    with pytest.raises(RuntimeError, match=r"retired \(ADR-061\)"):
        SingularityWrapper()


def test_singularity_wrapper_refuses_with_arguments():
    from umrs.orchestrator.hybrid.wrappers.singularity import SingularityWrapper

    with pytest.raises(RuntimeError, match=r"retired \(ADR-061\)"):
        SingularityWrapper(default_image="/x.sif", nvidia_support=False)


def test_hybrid_composer_slurm_singularity_profile_refuses(monkeypatch):
    from umrs.orchestrator.hybrid.composer import HybridOrchestrator

    monkeypatch.setenv("RIVAULT_ENABLE_EXPERIMENTAL_HYBRID_SLURM", "true")
    with pytest.raises(RuntimeError, match=r"retired \(ADR-061\)"):
        HybridOrchestrator(profile="slurm-singularity")


def test_retirement_message_names_the_podman_lane():
    from umrs.orchestrator.hybrid.wrappers.singularity import SingularityWrapper

    with pytest.raises(RuntimeError) as exc:
        SingularityWrapper()
    assert "Podman" in str(exc.value)
    assert "slurm-remote" in str(exc.value)
