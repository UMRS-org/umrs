import os

# Ultimately this should be the only hard-coded variables in the project
DEFAULT_UMRS_CFG_DIR = os.path.expanduser("~/.config/umrs") 
DEFAULT_USERS_CFG_SUBDIR = "users" 
DEFAULT_TMP_SUBDIR = "tmp" 
UMRS_CFG_ENV_VAR_NAME = "UMRS_CONFIG"
DEFAULT_JSON_SERVER_FILE ="conf_umrs.json"


def get_default_umrs_cfg_dir():
    return DEFAULT_UMRS_CFG_DIR

def get_umrs_cfg_env_var_name():
    return UMRS_CFG_ENV_VAR_NAME

def get_default_json_server_file():
    return DEFAULT_JSON_SERVER_FILE

def get_users_default_cfg_subdir():
    return DEFAULT_USERS_CFG_SUBDIR

def get_default_tmp_subdir():
    return DEFAULT_TMP_SUBDIR
