import os

from umrs.shared.config import get_resolved_umrs_subdir

ENABLE_PRETTY_PRINT = False  # disabled by default; enable with CLI flag
VISION_STACK = os.environ.get("VISION_STACK", "umrs")
RIVAULT_REMOTE_PORT = int(os.environ.get("RIVAULT_REMOTE_PORT", "-1"))
_UMRS_SERVER_DB_USER = "umrs_server_db_user"
_UMRS_SERVER_DB_NAME = "umrs"
_DB_HOST = "localhost"
_DB_PORT = "5432"
_DB_SUPERUSER = "postgres"
_DB_CREDENTIALS_FILENAME = os.path.join("credentials", "db_credentials.ini")


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
        # Silent -1 → downstream tunnel arms fall back to port 4000 with no
        # evidence of why the configured port was lost. Keep the fallback but
        # make the degradation observable (audit: observability gap).
        import logging

        logging.getLogger(__name__).warning(
            "Site config unreadable while resolving rivault remote port; "
            "returning RIVAULT_REMOTE_PORT=%d (downstream fallback applies)",
            port,
            exc_info=True,
        )
        return port


def get_umrs_server_db_user() -> str:
    return _UMRS_SERVER_DB_USER


def get_umrs_server_db_name() -> str:
    return os.environ.get("DB_UMRS_NAME", _UMRS_SERVER_DB_NAME)


from umrs.shared.pg_config import (
    get_db_config,
)

__all__ = [
    "ENABLE_PRETTY_PRINT",
    "RIVAULT_REMOTE_PORT",
    "VISION_STACK",
    "get_db_config",
    "get_default_db_subdir",
    "get_jens_rivault_slurm_script",
    "get_rivault_remote_port",
    "get_rivault_slurm_script",
    "get_umrs_server_db_name",
    "get_umrs_server_db_user",
]
