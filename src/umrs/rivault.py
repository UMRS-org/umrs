from umrs.models_loading import load_model
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

def extract_files_from_slurm_script(slurm_script, default_output="slurm-%j.out", default_error="slurm-%j.err"):
 
    with open(slurm_script, "r") as f:
        lines = f.readlines()

    # here we want to check if there is a shebang (#!) at the beginning of the script, 
    # and if so we want to keep it at the top of the script when we modify it to add the output and error file if they are not present. 
    start_idx = 0

    # Skip leading empty lines and detect shebang
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "":
            continue
        if stripped.startswith("#!"):
            start_idx = i + 1
        else:
            # first non-empty line that's not a shebang
            start_idx = i
        break

    # check existing SBATCH lines
    # in new_lines we will keep all the lines except the ones with --output and --error that do not contain %j, 
    # we will replace them with the correct ones with %j, and if there is no line with --output or --error 
    # we will add them at start_idx (after the shebang and any initial comments)
    new_lines = []  # keep shebang if present
    output_file = error_file = None

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#SBATCH") and "--output=" in stripped:
            path = stripped.split("--output=", 1)[1].strip()
            if "%j" in path:
                output_file = path
                new_lines.append(line)  # keep correct line
            # else: skip line (do not append). We need %j to replace with actual job id later, otherwise we won't be able to monitor the file.
        elif stripped.startswith("#SBATCH") and "--error=" in stripped:
            path = stripped.split("--error=", 1)[1].strip()
            if "%j" in path:
                error_file = path
                new_lines.append(line)
            # else: skip line. We need %j to replace with actual job id later, otherwise we won't be able to monitor the file.
        else:
            new_lines.append(line)
    
    print(f"{new_lines}")

    # Now create lines to insert if needed with %j placeholder, and insert them after the shebang and any initial comments (after start_idx)
    insert_lines = []
    if output_file is None:
        output_file = default_output
        insert_lines.append(f"#SBATCH --output={output_file}\n")
    if error_file is None:
        error_file = default_error
        insert_lines.append(f"#SBATCH --error={error_file}\n")
    
    print("insert_lines=", insert_lines)
    # Insert these lines at start_idx (after shebang and any initial comments)
    final_lines = new_lines[:start_idx] + insert_lines + new_lines[start_idx:]
    with open(slurm_script, "w") as f:
        f.writelines(final_lines)
    
    return output_file, error_file

def get_slurm_node(slurm_script, job_id_to_use=None):
    output_file = error_file = None
    if "plafrim" in slurm_script:
        create_plafrim_slurm_script(slurm_script)

    # OK now we need to check and extact the output_file and the error_file from the slurm script, so we can monitor them to extract the node address.
    # we need to ensure they exist in the slurm script, otherwise we won't be able to monitor them. and that they contain %j so we can replace it with the actual job id.
    # this is exactly what does the following function, it returns the output_file and error_file paths with the %j placeholder, so we can replace it later with the actual job id once we have it.
    out_dir = _ensure_output_dir()
    default_output_file = os.path.join(out_dir, "slurm-%j.out") # Will be insterted in the script if there is no --output line with %j placeholder
    default_error_file = os.path.join(out_dir, "slurm-%j.err") # Will be insterted in the script if there is no --error line with %j placeholder
    output_file, error_file = extract_files_from_slurm_script(slurm_script, default_output=default_output_file, default_error=default_error_file)

    # If we already have a job_id, we only need the output and error file paths. No need to call SLURM.
    if job_id_to_use is not None:
        job_id = job_id_to_use
    else :
        # Launch script
        sbatch_cmd = ["sbatch"] + [slurm_script]
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

def ssh_tunnel_forward_to_remote(remote_host: str, remote_port: int = 8001) -> tuple[int, threading.Thread, subprocess.Popen]:
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
    if remote_port < 0:
        remote_port = model_cfg.get("server_port", None)
        if remote_port is None:
            raise ValueError("Remote port for RivaULT not specified. Please set the RIVAULT_REMOTE_PORT environment variable or specify 'server_port' in the model configuration.")
    local_port, thread_tunnel, proc_tunnel = ssh_tunnel_forward_to_remote(server_address, remote_port)
    NGINX_PORT = os.getenv("NGINX_LITELLM_DOCKER_PORT")
    MASTER_KEY = os.getenv("LITELLM_MASTER_KEY")
    
    # model_type = "https:.....:{NGINX_PORT}"

     # --- 3. Build the OpenAI-compatible model ---
    if model_cfg["model_type"].startswith(("openai://","bentoml://")):
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
    output_file, error_file = extract_files_from_slurm_script("~/tmp/myslurm.slurm")
    print(f"{output_file=} {error_file=}")
    exit(0)

    # --- 1. get a node from SLURM ---
    rivault_slurm_script = get_slurm_script_from_host()
    output_file, error_file, job_id = get_slurm_node(rivault_slurm_script)
    print(f"{output_file=} {job_id=}")
    server_adress = extract_node_address(output_file) if output_file else "localhost"
    print(f"{server_adress=}")
    remote_port = get_rivault_remote_port()
    local_port = forward_to_remote(server_adress, remote_port)
    print(f"{local_port=}")
    
