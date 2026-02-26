from .models_loading import load_model
from umrs.config import *


import subprocess
import time
import requests
import socket
import os
import diskcache
from contextlib import redirect_stderr, redirect_stdout
import io
import threading

def get_slurm_job_state(job_id):
    result = subprocess.run(
        ["squeue", "-h", "-j", str(job_id), "-o", "%T"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        # No such job or any problem. 
        return None

    state = result.stdout.strip()

    if state == "":
        # Job not in queue anymore (COMPLETED, FAILED, CANCELLED, etc.)
        return None

    return state

def is_slurm_job_running(job_id):
    state = get_slurm_job_state(job_id)
    return state == "RUNNING"

def create_plafrim_slurm_script(slurm_script):
    tmp_dir = get_default_tmp_subdir()
    full_tmp_path = os.path.abspath(os.path.expanduser(tmp_dir))

    # Create directory if it does not exist
    if not os.path.exists(full_tmp_path):
        os.makedirs(full_tmp_path, exist_ok=True)

    output_file = os.path.join(full_tmp_path, "slurm-%j.out")
    error_file  = os.path.join(full_tmp_path, "slurm-%j.err")
    
    script_content=f'''#!/bin/bash
#SBATCH -N 1
#SBATCH -n 36
#SBATCH -C bora
#SBATCH --exclusive
#SBATCH --time=0:15:00
#SBATCH --job-name=vllm_fastapi
#SBATCH --output={output_file}
#SBATCH --error={error_file}

echo "=============================="
echo "SLURM JOB ID: $SLURM_JOB_ID"
echo "Allocated nodes: $SLURM_NODELIST"
echo "Running on node: $(hostname)"
echo "Fully qualified name: $(hostname -f)"
echo "=============================="

# Activate virtual environment
source ~/venvs_vllm/bin/activate

echo "Python used: $(which python)"
echo "Starting FastAPI model..."

# Launch your model
python /home/jeannot/umrs/tests/OpenAI-API/launch_fastapi.py
'''
    with open(slurm_script, "w") as f:
        f.write(script_content)

    return output_file, error_file

def _ensure_output_dir():
    """Create and return the tmp directory for Slurm output files."""
    cfg_dir = get_default_umrs_cfg_dir()
    tmp_dir = get_default_tmp_subdir()
    full_tmp_path = os.path.join(os.path.expanduser(cfg_dir), tmp_dir)
    os.makedirs(full_tmp_path, exist_ok=True)
    return full_tmp_path

def get_slurm_node(slurm_script, job_id_to_use=None):
    output_file = error_file = None
    if "plafrim" in slurm_script:
        output_file, error_file = create_plafrim_slurm_script(slurm_script)

    # For non-plafrim scripts, set up output/error files so we can
    # extract the compute node address from the Slurm output.
    sbatch_extra_args = []
    if output_file is None:
        out_dir = _ensure_output_dir()
        output_file = os.path.join(out_dir, "slurm-%j.out")
        error_file = os.path.join(out_dir, "slurm-%j.err")
        sbatch_extra_args = [f"--output={output_file}", f"--error={error_file}"]
    
    # If we already have a job_id, we only need the output and error file paths. No need to call SLURM.
    if job_id_to_use is not None:
        job_id = job_id_to_use
    else :
        # Launch script
        sbatch_cmd = ["sbatch"] + sbatch_extra_args + [slurm_script]
        print(f"{sbatch_cmd}")

        proc = subprocess.Popen(sbatch_cmd, stdout=subprocess.PIPE)

        print("Waiting for job to start...")
        # Read job SLURM ID
        output, _ = proc.communicate()
        job_id = None
        if output:
            text = output.decode()
            # sbatch returns "Submitted batch job 12345"
            if "Submitted batch job" in text:
                job_id = int(text.strip().split()[-1])
                print("Job submitted with ID:", job_id)
                while not is_slurm_job_running(job_id):
                    time.sleep(0.5)

    # Replace %j placeholder with actual JOB_ID
    if output_file:
        output_file = output_file.replace("%j", str(job_id))
    if error_file:
        error_file = error_file.replace("%j", str(job_id))

    print(f"Job {job_id} is running. Output will be in {output_file} and {error_file}")

    return output_file, error_file, job_id

def get_slurm_script_from_host():
    # Get the hostname
    hostname = socket.gethostname()

    # Get the full domain name
    fqdn = socket.getfqdn()
    
    return get_rivault_slurm_script(fqdn)

def extract_node_address(slurm_output_file):
    """
    Extract the fully qualified node name from a SLURM output file.
    
    Args:
        slurm_output_file (str): Path to the SLURM output file.
    
    Returns:
        str or None: Fully qualified node name, e.g. 'bora007.plafrim.cluster'.
    """
    print(f"Checking {slurm_output_file}...")
    # Wait until the file exists
    i = 0
    while not os.path.exists(slurm_output_file):
        if i % 100 == 0:
            print(f"{slurm_output_file=} not found, waiting...")
        i += 1
        time.sleep(0.5)


    print(f"{slurm_output_file} ready")
    while True: 
        with open(slurm_output_file, "r") as f:
            for line in f:
                line = line.strip()
                # print(f"{line=}")
                if line.startswith("Fully qualified name:"):
                    # Split by ":" and remove spaces
                    return line.split(":", 1)[1].strip()
        time.sleep(0.5)


def wait_for_local_port(port, host="127.0.0.1", timeout=5.0):
    """Wait until a local TCP port is open."""
    f = io.StringIO()
    with redirect_stdout(f), redirect_stderr(f):
        start = time.time()
        while True:
            try:
                with socket.create_connection((host, port), timeout=0.5):
                    return True
            except OSError:
                time.sleep(0.1)
                if time.time() - start > timeout:
                    raise TimeoutError(f"Port {port} on {host} did not open in {timeout}s")

def forward_to_remote(remote_host: str, remote_port: int = 8001) -> tuple[int, threading.Thread, subprocess.Popen]:
    """
    Find a free local port and forward it via SSH to remote_host:remote_port.

    Returns:
        tuple containing:
        - local port number (int) you can use to access remote_host:remote_port
        - the Thread managing the SSH tunnel
        - the subprocess.Popen object for the SSH process
    """
    # 1. Find a free local port
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('', 0))  # 0 means choose any free port
    local_port = s.getsockname()[1]
    s.close()

    # 2. Build the SSH command (blocking, attached to its thread)
    # -f: go to background
    # -N: no command
    # -L: local forward
    ssh_cmd = [
        "ssh",
        "-N",              # do not execute remote command
        "-L", f"{local_port}:{remote_host}:{remote_port}",
        "localhost"
    ]

     # 3. Function to run SSH as subprocess
    def run_ssh():
        nonlocal ssh_proc
        ssh_proc = subprocess.Popen(
            ssh_cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        ssh_proc.wait()

    ssh_proc = None
    thread = threading.Thread(target=run_ssh, daemon=True)
    thread.start()

    wait_for_local_port(local_port)
    print(f"port {local_port} ready...")
    return local_port, thread, ssh_proc


def silent_requests_get(url, timeout=1.0):
    f = io.StringIO()
    try:
        with redirect_stdout(f), redirect_stderr(f):
            resp = requests.get(url, timeout=timeout)
            return resp
    except requests.RequestException:
        return None

def wait_for_model(openai_url, model_name="intfloat/e5-base-v2", timeout=60.0, interval=0.2):
    """
    Wait until the OpenAI-compatible server is up and the specified model is available.
    
    openai_url: e.g., "openai://http://localhost:56126/v1"
    model_name: name of the model to check
    timeout: maximum time to wait in seconds
    interval: delay between checks
    """
    start = time.time()
    model_url = openai_url[len("openai://"):] +  "/models"
    
    print(f"Checking model on {model_url=}...")
    while True:
        try:
            resp = silent_requests_get(model_url, timeout=1.0)
            # print(f"{resp=}")
            if resp is not None and resp.status_code == 200:
                data = resp.json().get("data", [])
                if any(m.get("id") == model_name for m in data):
                    return True
        except requests.RequestException as e:
            # print(f"{type(e).__name__}: {e}")
            pass 

        if time.time() - start > timeout:
            raise TimeoutError(f"Model {model_name} not available at {model_url} after {timeout}s")
        
        time.sleep(interval)



# persistent in-memory cache (can persist to disk if needed later, e.g. after crash)
jobid_cache = diskcache.Cache(f"{get_default_tmp_subdir()}/job_id_store", size_limit = 10*1024*1024)

# Ephemeral in-memory cache for models
model_cache = {}

def launch_rivault_model(model_cfg):
    model_type_key = model_cfg["model_type"]
    # --- 0. Check cache ---
    job_id = jobid_cache.get(model_type_key, None)
    if job_id is not None:
        if is_slurm_job_running(job_id):
            # job is running. Is the model in memory? 
            if model_type_key in model_cache:
                model, _, _ = model_cache[model_type_key]
                print("===> Found a running job and a loaded model in cache. Reusing it.")
                return model
            else:
                print("===> Found a running job in cache but no in memory model. This can happen if the server has restarted. Will try to load it again.")
        else:
            # Job not runnning
            print("===> Found a job_id in cache but the job is not running. Cleaning up cache and launching a new job.")
            jobid_cache.pop(model_type_key, None)
            if model_type_key in model_cache:
                _, thread_tunnel, proc_tunnel = model_cache.pop(model_type_key)
                # Terminate the SSH subprocess
                proc_tunnel.terminate()   # sends SIGTERM
                proc_tunnel.wait()        # wait for it to exit

                # Thread will exit automatically once the subprocess ends
                thread_tunnel.join()      # wait for thread to finish cleanly

            job_id = None
                
    # --- 1. get a node form SLURM ---    
    rivault_slurm_script = get_slurm_script_from_host()
    output_file, error_file, job_id = get_slurm_node(rivault_slurm_script, job_id_to_use=job_id)

    print(f"{output_file=} {job_id=}")

    # --- 2. Extract compute node address and create SSH tunnel ---
    server_address = extract_node_address(output_file) if output_file else "localhost"
    remote_port = get_rivault_remote_port()
    local_port, thread_tunnel, proc_tunnel = forward_to_remote(server_address, remote_port)
    NGINX_PORT = os.getenv("NGINX_LITELLM_DOCKER_PORT")
    MASTER_KEY = os.getenv("LITELLM_MASTER_KEY")
    
    # model_type = "https:.....:{NGINX_PORT}"

     # --- 3. Build the OpenAI-compatible model ---
    if model_cfg["model_type"].startswith("openai://"):
        # replace a placeholder {NGINX_PORT} with the actual value
        model_cfg["model_type"] = model_cfg["model_type"].replace("{NGINX_PORT}", str(NGINX_PORT))
        # if you also want to replace a placeholder {local_port} with a variable 'local_port'
        model_cfg["model_type"] = model_cfg["model_type"].replace("{LOCAL_PORT}", str(local_port))
        wait_for_model(model_cfg["model_type"], model_cfg["model_name"])
        print(f"Configuring: {model_cfg['model_type']}...")
        # Load the model
        model = load_model(model_cfg, "")
        
         # --- 4. Cache the loaded model and job_id ---
        model_cache[model_type_key] = (model,  thread_tunnel, proc_tunnel)
        jobid_cache[model_type_key] = job_id  # persistant
        return model
    else:
        raise RuntimeError(f"Do not know what to do with model type {model_cfg['model_type']}")


if __name__ == "__main__":
    # --- 1. get a node from SLURM ---
    rivault_slurm_script = get_slurm_script_from_host()
    output_file, error_file, job_id = get_slurm_node(rivault_slurm_script)
    print(f"{output_file=} {job_id=}")
    server_adress = extract_node_address(output_file) if output_file else "localhost"
    print(f"{server_adress=}")
    remote_port = get_rivault_remote_port()
    local_port = forward_to_remote(server_adress, remote_port)
    print(f"{local_port=}")
    
