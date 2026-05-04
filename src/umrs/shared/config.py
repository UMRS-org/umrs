import os
import stat
from collections import Counter

DEFAULT_UMRS_CFG_DIR = os.path.expanduser("~/.config/umrs")
DEFAULT_USERS_CFG_SUBDIR = "users"
DEFAULT_CONNECTIONS_CFG_SUBDIR = "connections"
DEFAULT_TMP_SUBDIR = "tmp"
DEFAULT_SERVER_USER_FILES_SUBDIR = "user_files"
OWNER_ONLY_DIRECTORY_MODE = stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR
UMRS_CFG_ENV_VAR_NAME = "UMRS_CONFIG"
UMRS_USER_FILES_ENV_VAR_NAME = "UMRS_USER_FILES_DIR"
DEFAULT_JSON_SERVER_FILE = "conf_umrs.json"
DEFAULT_DOCKER_S3CFG = "/app/docker/s3cfg"
DEFAULT_CURRENT_CONNECTION_PROFILE_FILE = "current_connection_profile"
DEFAULT_CURRENT_SERVER_CONFIG_FILE = "current_server_config"
DEFAULT_CURRENT_USER_CONFIG_FILE = "current_user_config"
DEFAULT_CURRENT_S3_CONFIG_FILE = "current_s3_config"


def get_default_umrs_cfg_dir():
    return DEFAULT_UMRS_CFG_DIR


def get_resolved_umrs_cfg_dir():
    override = os.environ.get(UMRS_CFG_ENV_VAR_NAME)
    if override:
        return os.path.realpath(os.path.expanduser(override))
    xdg_config_home = os.environ.get("XDG_CONFIG_HOME")
    if xdg_config_home:
        return os.path.realpath(os.path.join(os.path.expanduser(xdg_config_home), "umrs"))
    return DEFAULT_UMRS_CFG_DIR


def resolve_config_path(path, config_file=None):
    expanded = os.path.expanduser(path)
    if os.path.isabs(expanded):
        return os.path.realpath(expanded)
    if config_file:
        config_root = os.path.dirname(os.path.abspath(os.path.expanduser(config_file)))
        return os.path.realpath(os.path.join(config_root, expanded))
    return os.path.realpath(expanded)


def ensure_private_directory(path):
    resolved = os.path.realpath(os.path.expanduser(path))
    os.makedirs(resolved, exist_ok=True)
    try:
        os.chmod(resolved, OWNER_ONLY_DIRECTORY_MODE)
    except OSError:
        # Best effort only; some environments do not support chmod semantics.
        pass
    if not os.path.isdir(resolved):
        raise OSError(f"{resolved} is not a directory")
    if not os.access(resolved, os.W_OK):
        raise OSError(f"{resolved} is not writable")
    return resolved


def get_umrs_user_files_env_var_name():
    return UMRS_USER_FILES_ENV_VAR_NAME


def get_server_user_files_default_subdir():
    return DEFAULT_SERVER_USER_FILES_SUBDIR


def _resolve_explicit_user_files_dir(config_file=None, server_cfg=None):
    override = os.environ.get(UMRS_USER_FILES_ENV_VAR_NAME)
    if override:
        return resolve_config_path(override, config_file=config_file)

    if server_cfg:
        configured_dir = server_cfg.get("user_files_dir")
        if configured_dir:
            return resolve_config_path(configured_dir, config_file=config_file)

    return None


def _infer_user_files_dir_from_map(config_file=None, user_config_map=None):
    if not user_config_map:
        return None

    parent_dirs = [
        os.path.dirname(resolve_config_path(path, config_file=config_file))
        for path in user_config_map.values()
        if path
    ]
    if not parent_dirs:
        return None
    return Counter(parent_dirs).most_common(1)[0][0]


def _resolve_user_files_fallback_root(config_file=None):
    if config_file:
        return os.path.dirname(os.path.abspath(os.path.expanduser(config_file)))
    return get_resolved_umrs_cfg_dir()


def resolve_server_user_files_dir(config_file=None, server_cfg=None, user_config_map=None):
    explicit_dir = _resolve_explicit_user_files_dir(config_file=config_file, server_cfg=server_cfg)
    if explicit_dir:
        return explicit_dir

    inferred_dir = _infer_user_files_dir_from_map(config_file=config_file, user_config_map=user_config_map)
    if inferred_dir:
        return inferred_dir

    config_root = _resolve_user_files_fallback_root(config_file=config_file)
    return os.path.realpath(os.path.join(config_root, DEFAULT_SERVER_USER_FILES_SUBDIR))


def get_umrs_cfg_env_var_name():
    return UMRS_CFG_ENV_VAR_NAME


def get_default_json_server_file():
    return DEFAULT_JSON_SERVER_FILE


def get_resolved_umrs_subdir(*parts):
    return os.path.realpath(os.path.join(get_resolved_umrs_cfg_dir(), *parts))


def get_resolved_json_server_path():
    return get_resolved_umrs_subdir(get_default_json_server_file())


def get_resolved_model_staging_root():
    override = os.environ.get("UMRS_MODEL_STAGING_ROOT")
    if override:
        return os.path.realpath(os.path.expanduser(override))
    return get_resolved_umrs_subdir("models")


def get_users_default_cfg_subdir():
    return DEFAULT_USERS_CFG_SUBDIR


def get_connections_default_cfg_subdir():
    return DEFAULT_CONNECTIONS_CFG_SUBDIR


def get_current_connection_profile_file():
    return DEFAULT_CURRENT_CONNECTION_PROFILE_FILE


def get_current_server_config_file():
    return DEFAULT_CURRENT_SERVER_CONFIG_FILE


def get_current_user_config_file():
    return DEFAULT_CURRENT_USER_CONFIG_FILE


def get_current_s3_config_file():
    return DEFAULT_CURRENT_S3_CONFIG_FILE


def get_default_tmp_subdir():
    return DEFAULT_TMP_SUBDIR


def get_default_docker_s3cfg():
    return DEFAULT_DOCKER_S3CFG


__all__ = [
    "DEFAULT_UMRS_CFG_DIR",
    "DEFAULT_USERS_CFG_SUBDIR",
    "DEFAULT_CONNECTIONS_CFG_SUBDIR",
    "DEFAULT_TMP_SUBDIR",
    "DEFAULT_SERVER_USER_FILES_SUBDIR",
    "OWNER_ONLY_DIRECTORY_MODE",
    "UMRS_CFG_ENV_VAR_NAME",
    "UMRS_USER_FILES_ENV_VAR_NAME",
    "DEFAULT_JSON_SERVER_FILE",
    "DEFAULT_DOCKER_S3CFG",
    "DEFAULT_CURRENT_CONNECTION_PROFILE_FILE",
    "DEFAULT_CURRENT_SERVER_CONFIG_FILE",
    "DEFAULT_CURRENT_USER_CONFIG_FILE",
    "DEFAULT_CURRENT_S3_CONFIG_FILE",
    "get_default_umrs_cfg_dir",
    "get_resolved_umrs_cfg_dir",
    "resolve_config_path",
    "ensure_private_directory",
    "get_umrs_cfg_env_var_name",
    "get_umrs_user_files_env_var_name",
    "get_default_json_server_file",
    "get_resolved_umrs_subdir",
    "get_resolved_json_server_path",
    "get_resolved_model_staging_root",
    "get_users_default_cfg_subdir",
    "get_server_user_files_default_subdir",
    "resolve_server_user_files_dir",
    "get_connections_default_cfg_subdir",
    "get_current_connection_profile_file",
    "get_current_server_config_file",
    "get_current_user_config_file",
    "get_current_s3_config_file",
    "get_default_tmp_subdir",
    "get_default_docker_s3cfg",
]
