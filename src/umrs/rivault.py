from .models_loading import load_model
from umrs.config import *


import subprocess
import time

def get_slurm_job_state(job_id):
    result = subprocess.run(
        ["squeue", "-h", "-j", str(job_id), "-o", "%T"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(f"squeue failed: {result.stderr.strip()}")

    state = result.stdout.strip()

    if state == "":
        # Job not in queue anymore (COMPLETED, FAILED, CANCELLED, etc.)
        return None

    return state

def is_slurm_job_running(job_id):
    state = get_slurm_job_state(job_id)
    return state == "RUNNING"

def get_slurm_node(slurm_script):
    # Launch script
    proc = subprocess.Popen(["sbatch", slurm_script], stdout=subprocess.PIPE)

    # Read job SLURM ID
    output, _ = proc.communicate()
    job_id = None
    if output:
        text = output.decode()
        # sbatch renvoie typiquement "Submitted batch job 12345"
        if "Submitted batch job" in text:
            job_id = int(text.strip().split()[-1])
            print("Job submitted with ID:", job_id)
            while not is_slurm_job_running(job_id):
                time.sleep(0.5)

def launch_rivault_model(model_cfg):
    # --- 1. get a node form SLURM ---    
    rivault_slurm_script = get_rivault_slurm_script()
    get_slurm_node(rivault_slurm_script)
   
    # --- 2. Load environment variables (from the SLURM env file) ---
    NGINX_PORT = os.getenv("NGINX_LITELLM_DOCKER_PORT")
    MASTER_KEY = os.getenv("LITELLM_MASTER_KEY")
    
    # model_type = "https:.....:{NGINX_PORT}"

     # --- 3. Build the OpenAI-compatible model ---
    if model_cfg["model_type"].startswith("openai://"):
        model_cfg["model_type"] = model_cfg["model_type"].format(NGINX_PORT=NGINX_PORT)
        return load_model(model_cfg, "")
    else:
        raise RuntimeError(f"Do not know what to do with model type {model_cfg['model_type']}")