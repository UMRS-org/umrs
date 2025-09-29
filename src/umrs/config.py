import os

# Ultimately this should be the only hard-coded variables in the project
DEFAULT_UMRS_CFG_DIR = os.path.expanduser("~/.config/umrs") 
UMRS_CFG_ENV_VAR_NAME = "UMRS_CONFIG"
DEFAULT_JSON_SERVER_FILE ="conf_umrs.json"

def get_default_umrs_cfg_dir():
    return DEFAULT_UMRS_CFG_DIR

def get_umrs_cfg_env_var_name():
    return UMRS_CFG_ENV_VAR_NAME

def get_default_json_server_file():
    return DEFAULT_JSON_SERVER_FILE