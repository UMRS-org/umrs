"""Tests for shipped UMRS site profiles."""

from __future__ import annotations

from unittest.mock import patch

from umrs.orchestrator.base import OrchestratorConfig
from umrs.orchestrator.site_config import reset_site_config
from umrs.orchestrator.site_profiles import (
    available_site_profiles,
    inject_site_profile_env,
    load_site_profile,
)


EXPECTED_PROFILES = {"rccs-cloud", "lab-podman", "compose-dev", "k8s-cluster"}


def test_shipped_site_profiles_are_discoverable():
    assert EXPECTED_PROFILES <= set(available_site_profiles())


def test_rccs_profile_expands_user_paths(monkeypatch):
    monkeypatch.setenv("USER", "test-operator")

    profile = load_site_profile("rccs-cloud")

    assert profile.scheduler_kind == "slurm-podman"
    assert profile.runtime_kind == "podman-compose"
    assert profile.slurm_modules == []
    assert "/test-operator/" in profile.slurm_log_dir
    assert "/test-operator/" in profile.slurm_job_info_dir


def test_selected_site_profile_injects_env(monkeypatch):
    monkeypatch.setenv("UMRS_SITE_PROFILE", "lab-podman")
    monkeypatch.setenv("USER", "alice")

    profile = inject_site_profile_env(overwrite=True)

    assert profile is not None
    assert profile.name == "lab-podman"
    assert "alice" in profile.slurm_log_dir
    assert profile.to_env()["SLURM_LOG_DIR"] == "/var/tmp/rivault-alice/logs"


def test_orchestrator_config_uses_selected_site_profile(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    env = {
        "UMRS_SITE_PROFILE": "compose-dev",
        "USER": "devuser",
    }
    reset_site_config()

    with patch.dict("os.environ", env, clear=True):
        cfg = OrchestratorConfig.from_env()

    assert cfg.slurm_log_dir == "./tmp/rivault/logs"
    assert cfg.slurm_job_info_dir == "./tmp/rivault/jobs"
    assert cfg.slurm_modules == ["cuda/12.1", "singularity"]
