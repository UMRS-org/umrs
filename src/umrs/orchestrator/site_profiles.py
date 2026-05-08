"""Declarative deployment site profiles for orchestrator defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROFILE_ENV_VARS = ("UMRS_SITE_PROFILE", "RIVAULT_SITE_PROFILE")
PROFILE_DATA_DIR = Path(__file__).with_name("site_profiles")


@dataclass(frozen=True)
class SiteProfile:
    """Small profile describing site-level runtime assumptions."""

    name: str
    description: str = ""
    scheduler_kind: str = ""
    runtime_kind: str = ""
    slurm_modules: list[str] = field(default_factory=list)
    slurm_log_dir: str = ""
    slurm_job_info_dir: str = ""

    def to_env(self) -> dict[str, str]:
        """Export profile defaults as environment variables."""
        env: dict[str, str] = {}
        if self.scheduler_kind:
            env["RIVAULT_ORCHESTRATOR"] = self.scheduler_kind
        if self.runtime_kind:
            env["RIVAULT_RUNTIME"] = self.runtime_kind
        if self.slurm_modules:
            env["SLURM_MODULES"] = " ".join(self.slurm_modules)
        if self.slurm_log_dir:
            env["SLURM_LOG_DIR"] = self.slurm_log_dir
        if self.slurm_job_info_dir:
            env["SLURM_JOB_INFO_DIR"] = self.slurm_job_info_dir
        return env


def available_site_profiles() -> list[str]:
    """Return shipped site profile names."""
    return sorted(path.stem for path in PROFILE_DATA_DIR.glob("*.yaml"))


def selected_site_profile_name() -> str | None:
    """Return the selected profile name from environment, if any."""
    for env_name in PROFILE_ENV_VARS:
        value = os.environ.get(env_name, "").strip()
        if value:
            return value
    return None


def load_site_profile(name: str) -> SiteProfile:
    """Load a shipped site profile by name."""
    import yaml

    path = PROFILE_DATA_DIR / f"{name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Unknown UMRS site profile: {name}")

    with path.open("r") as fh:
        raw = _expand_env_refs(yaml.safe_load(fh) or {})
    return _profile_from_raw(raw)


def load_selected_site_profile() -> SiteProfile | None:
    """Load the profile named by UMRS_SITE_PROFILE/RIVAULT_SITE_PROFILE."""
    name = selected_site_profile_name()
    if not name:
        return None
    return load_site_profile(name)


def inject_site_profile_env(overwrite: bool = False) -> SiteProfile | None:
    """Inject selected site-profile defaults into os.environ."""
    profile = load_selected_site_profile()
    if profile is None:
        return None
    for key, value in profile.to_env().items():
        if overwrite or key not in os.environ:
            os.environ[key] = value
    return profile


def _expand_env_refs(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    if isinstance(value, list):
        return [_expand_env_refs(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_env_refs(item) for key, item in value.items()}
    return value


def _profile_from_raw(raw: dict[str, Any]) -> SiteProfile:
    modules = raw.get("slurm_modules", [])
    if isinstance(modules, str):
        modules = modules.replace(",", " ").split()
    return SiteProfile(
        name=str(raw.get("name", "")),
        description=str(raw.get("description", "")),
        scheduler_kind=str(raw.get("scheduler_kind", "")),
        runtime_kind=str(raw.get("runtime_kind", "")),
        slurm_modules=list(modules or []),
        slurm_log_dir=str(raw.get("slurm_log_dir", "")),
        slurm_job_info_dir=str(raw.get("slurm_job_info_dir", "")),
    )
