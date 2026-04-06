import os

from umrs.shared.config import DEFAULT_UMRS_CFG_DIR

DEFAULT_DB_SUBDIR = f"{DEFAULT_UMRS_CFG_DIR}/tasks_database"
DEFAULT_RIVAULT_SLURM_SCRIPT = os.environ.get(
    "RIVAULT_SLURM_SCRIPT",
    f"{DEFAULT_UMRS_CFG_DIR}/rivault_data/serve-model.slurm",
)
PLAFRIM_RIVAULT_SLURM_SCRIPT = f"{DEFAULT_UMRS_CFG_DIR}/rivault_data/serve-model-plafrim.slurm"
JENS_RIVAULT_SLURM_SCRIPT = os.environ.get(
    "JENS_RIVAULT_SLURM_SCRIPT",
    f"{DEFAULT_UMRS_CFG_DIR}/rivault_data/serve-model-jens.slurm",
)
VISION_STACK = os.environ.get("VISION_STACK", "umrs")
RIVAULT_REMOTE_PORT = int(os.environ.get("RIVAULT_REMOTE_PORT", -1))


def get_default_db_subdir():
    return DEFAULT_DB_SUBDIR


def get_rivault_slurm_script(fully_qualified_domain_name):
    env_script = os.environ.get("RIVAULT_SLURM_SCRIPT")
    if env_script:
        return env_script
    if "plafrim" in fully_qualified_domain_name:
        return PLAFRIM_RIVAULT_SLURM_SCRIPT
    return DEFAULT_RIVAULT_SLURM_SCRIPT


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
    "DEFAULT_DB_SUBDIR",
    "DEFAULT_RIVAULT_SLURM_SCRIPT",
    "PLAFRIM_RIVAULT_SLURM_SCRIPT",
    "JENS_RIVAULT_SLURM_SCRIPT",
    "VISION_STACK",
    "RIVAULT_REMOTE_PORT",
    "get_default_db_subdir",
    "get_rivault_slurm_script",
    "get_rivault_remote_port",
]