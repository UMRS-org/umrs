import os

from umrs.shared.config import get_resolved_umrs_subdir

ENABLE_PRETTY_PRINT = False  # disabled by default; enable with CLI flag
VISION_STACK = os.environ.get("VISION_STACK", "umrs")
RIVAULT_REMOTE_PORT = int(os.environ.get("RIVAULT_REMOTE_PORT", -1))


def get_default_db_subdir():
    return get_resolved_umrs_subdir("tasks_database")


def _default_rivault_script_path(filename):
    return get_resolved_umrs_subdir("rivault_data", filename)


def get_rivault_slurm_script(fully_qualified_domain_name):
    env_script = os.environ.get("RIVAULT_SLURM_SCRIPT")
    if env_script:
        return env_script
    if "plafrim" in fully_qualified_domain_name:
        return _default_rivault_script_path("serve-model-plafrim.slurm")
    return _default_rivault_script_path("serve-model.slurm")


def get_jens_rivault_slurm_script():
    return os.environ.get(
        "JENS_RIVAULT_SLURM_SCRIPT",
        _default_rivault_script_path("serve-model-jens.slurm"),
    )


def get_rivault_remote_port():
    port = RIVAULT_REMOTE_PORT
    if port > 0:
        return port
    # Fall back to SiteConfig (loaded from site.yaml or env-only defaults)
    try:
        from umrs.orchestrator.site_config import get_site_config

        return get_site_config().network.litellm_port
    except Exception:
        return port


__all__ = [
    "ENABLE_PRETTY_PRINT",
    "VISION_STACK",
    "RIVAULT_REMOTE_PORT",
    "get_default_db_subdir",
    "get_rivault_slurm_script",
    "get_jens_rivault_slurm_script",
    "get_rivault_remote_port",
]
