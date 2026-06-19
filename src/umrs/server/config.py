import configparser
import os

from umrs.shared.config import get_resolved_umrs_subdir

ENABLE_PRETTY_PRINT = False  # disabled by default; enable with CLI flag
VISION_STACK = os.environ.get("VISION_STACK", "umrs")
RIVAULT_REMOTE_PORT = int(os.environ.get("RIVAULT_REMOTE_PORT", -1))
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
        return port




def get_umrs_server_db_user() -> str:
    return _UMRS_SERVER_DB_USER


def get_umrs_server_db_name() -> str:
    return os.environ.get("DB_UMRS_NAME", _UMRS_SERVER_DB_NAME)


def get_db_config() -> dict:
    """Read DB connection config from credentials file, with env var overrides.

    Credentials file: ~/.config/umrs/credentials/db_credentials.ini
    Format:
        [postgres]
        superuser_password = <postgres superuser password>
        umrs_server_password = <umrs_server_db_user password>
        user_password = <per-experiment user password>
        # host = localhost
        # port = 5432
        # superuser = postgres

    Env vars override file values:
        DB_SUPERUSER_PASSWORD, DB_UMRS_SERVER_PASSWORD, DB_USER_PASSWORD,
        DB_HOST, DB_PORT, DB_SUPERUSER / POSTGRES_USER, DB_UMRS_NAME.

    Raises FileNotFoundError if the credentials file does not exist.
    Raises ValueError if a required password is missing.
    """
    creds_file = get_resolved_umrs_subdir(_DB_CREDENTIALS_FILENAME)

    if not os.path.isfile(creds_file):
        raise FileNotFoundError(
            f"DB credentials file not found: {creds_file}\n"
            f"Create it with the following content:\n"
            f"  [postgres]\n"
            f"  superuser_password = <postgres superuser password>\n"
            f"  umrs_server_password = <umrs_server_db_user password>\n"
            f"  user_password = <per-experiment user password>\n"
            f"  # host = localhost\n"
            f"  # port = 5432\n"
            f"  # superuser = postgres\n"
        )

    config = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
    config.read(creds_file)

    def _get(env_key: str, ini_key: str, default: str | None = None) -> str | None:
        # Env var takes priority over credentials file
        env_val = os.environ.get(env_key)
        if env_val:
            return env_val
        return config.get("postgres", ini_key, fallback=default)

    superuser_password = _get("DB_SUPERUSER_PASSWORD", "superuser_password",
                               os.environ.get("POSTGRES_PASSWORD"))
    if not superuser_password:
        raise ValueError(
            "superuser_password is required in [postgres] section of the credentials file"
        )

    umrs_server_password = _get("DB_UMRS_SERVER_PASSWORD", "umrs_server_password")
    if not umrs_server_password:
        raise ValueError(
            "umrs_server_password is required in [postgres] section of the credentials file"
        )

    return {
        "host":                 _get("DB_HOST", "host", _DB_HOST),
        "port":                 _get("DB_PORT", "port", _DB_PORT),
        "superuser":            _get("DB_SUPERUSER", "superuser",
                                     os.environ.get("POSTGRES_USER", _DB_SUPERUSER)),
        "superuser_password":   superuser_password,
        "umrs_server_password": umrs_server_password,
        "user_password":        _get("DB_USER_PASSWORD", "user_password"),
        "db_name":              _get("DB_UMRS_NAME", "db_name", _UMRS_SERVER_DB_NAME),
    }


__all__ = [
    "ENABLE_PRETTY_PRINT",
    "VISION_STACK",
    "RIVAULT_REMOTE_PORT",
    "get_default_db_subdir",
    "get_rivault_slurm_script",
    "get_jens_rivault_slurm_script",
    "get_rivault_remote_port",
    "get_umrs_server_db_user",
    "get_umrs_server_db_name",
    "get_db_config",
]
