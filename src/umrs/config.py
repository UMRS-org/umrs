import os

# Ultimately this should be the only hard-coded variables in the project
DEFAULT_UMRS_CFG_DIR = os.path.expanduser("~/.config/umrs")
DEFAULT_USERS_CFG_SUBDIR = "users"
DEFAULT_TMP_SUBDIR = "tmp"
UMRS_CFG_ENV_VAR_NAME = "UMRS_CONFIG"
DEFAULT_JSON_SERVER_FILE ="conf_umrs.json"
DEFAULT_DOCKER_S3CFG = "/app/docker/s3cfg"
DEFAULT_DB_SUBDIR = f"{DEFAULT_UMRS_CFG_DIR}/tasks_database"
# The rivault runtime scripts now live in src/umrs/mini-rivault-llm/ (LLM only).
# The sbatch wrapper at ~/.config/umrs/rivault_data/serve-model.slurm sets host
# env overrides and then delegates to the actual serve-model.slurm inside
# RIVAULT_DIR (resolved from the Python package by rivault_launcher.py).
# Override with $RIVAULT_SLURM_SCRIPT if your host uses a different path.
DEFAULT_RIVAULT_SLURM_SCRIPT = os.environ.get(
    "RIVAULT_SLURM_SCRIPT",
    f"{DEFAULT_UMRS_CFG_DIR}/rivault_data/serve-model.slurm",
)
PLAFRIM_RIVAULT_SLURM_SCRIPT = f"{DEFAULT_UMRS_CFG_DIR}/rivault_data/serve-model-plafrim.slurm"
# Jens's standalone vision stack (mini-rivault-vision submodule).
# Override with $JENS_RIVAULT_SLURM_SCRIPT if your host uses a different path.
JENS_RIVAULT_SLURM_SCRIPT = os.environ.get(
    "JENS_RIVAULT_SLURM_SCRIPT",
    f"{DEFAULT_UMRS_CFG_DIR}/rivault_data/serve-model-jens.slurm",
)
# VISION_STACK selects which vision backend to launch:
#   "umrs"  (default) — vision_service/ via vision-bentoml profile in mini-rivault-llm
#   "jens"  — mini-rivault-vision submodule (Jens's standalone compose)
VISION_STACK = os.environ.get("VISION_STACK", "umrs")
RIVAULT_REMOTE_PORT = int(os.environ.get("RIVAULT_REMOTE_PORT", -1))

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

def get_default_db_subdir():
    return DEFAULT_DB_SUBDIR

def get_default_docker_s3cfg():
    return DEFAULT_DOCKER_S3CFG

def get_rivault_slurm_script(fully_qualified_domain_name):
    env_script = os.environ.get("RIVAULT_SLURM_SCRIPT")
    if env_script:
        return env_script
    if "plafrim" in fully_qualified_domain_name:
        return PLAFRIM_RIVAULT_SLURM_SCRIPT
    return DEFAULT_RIVAULT_SLURM_SCRIPT

def get_rivault_remote_port():
    return RIVAULT_REMOTE_PORT
