from umrs.models_loading import load_model, EmbeddingModel
from umrs.config import *


import subprocess
import time
import re
import requests
import socket
import os
import diskcache
from contextlib import redirect_stderr, redirect_stdout
import io
import threading

# persistent in-memory cache (can persist to disk if needed later, e.g. after crash)
jobid_cache = diskcache.Cache(f"{get_default_tmp_subdir()}/job_id_store", size_limit = 10*1024*1024)

# Ephemeral in-memory cache for models
model_cache = {}


def load_model_as_bentoml(model_name, return_type, vector_size, input_type, compute_func, modality):
    """Create an EmbeddingModel backed by a BentoML compute adapter."""
    em = EmbeddingModel(
        model_name=model_name,
        type="bentoml",
        return_type=return_type,
        embedding_vector_size=vector_size,
        max_tokens=None,
        model=None,
        input_type=input_type,
        compute_embeddings=compute_func,
        modality=modality,
        spacy_model=None,
        tokenizer=None,
    )
    return em

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
    """Check if a Slurm job is currently in RUNNING state."""
    state = get_slurm_job_state(job_id)
    return state == "RUNNING"

def wait_for_running(job_id, timeout=300):
    """Wait until a Slurm job reaches RUNNING state, with timeout."""
    start = time.time()
    while True:
        state = get_slurm_job_state(job_id)
        if state == "RUNNING":
            return True
        if state is None or state not in ("PENDING", "CONFIGURING"):
            return False
        if time.time() - start > timeout:
            return False
        time.sleep(0.5)


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

def get_slurm_node(slurm_script, model_type_key, job_id_to_use=None, sbatch_env=None):
    output_file = error_file = None
    if "plafrim" in slurm_script:
        create_plafrim_slurm_script(slurm_script)

    out_dir = _ensure_output_dir()
    output_file = os.path.join(out_dir, "slurm-%j.out") # Will be insterted in the script if there is no --output line with %j placeholder
    error_file = os.path.join(out_dir, "slurm-%j.err") # Will be insterted in the script if there is no --error line with %j placeholder
    # output_file, error_file = extract_files_from_slurm_script(slurm_script, default_output=default_output_file, default_error=default_error_file)

    sbatch_extra_args = [f"--output={output_file}", f"--error={error_file}"]
    if sbatch_env:
        env_pairs = ",".join(f"{k}={v}" for k, v in sbatch_env.items())
        sbatch_extra_args.append(f"--export=ALL,{env_pairs}")

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
        if proc.returncode != 0:
            raise RuntimeError(
                f"sbatch failed with return code {proc.returncode}: "
                f"{output.decode() if output else '(no output)'}")
        if output:
            text = output.decode()
            # sbatch returns "Submitted batch job 12345"
            match = re.search(r'Submitted batch job (\d+)', text)
            if match:
                job_id = int(match.group(1))
                print("Job submitted with ID:", job_id)
                jobid_cache[model_type_key] = job_id  # persist immediately
                if not wait_for_running(job_id):
                    raise ValueError(f"Job {job_id} did not start running. Current state: {get_slurm_job_state(job_id)}")
        if job_id is None:
            raise ValueError(
                f"Could not parse job ID from sbatch output: "
                f"{output.decode() if output else '(no output)'}")

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

def extract_node_address(slurm_output_file, timeout=300):
    """
    Extract the fully qualified node name from a SLURM output file.

    Args:
        slurm_output_file (str): Path to the SLURM output file.
        timeout (float): Maximum time to wait in seconds.

    Returns:
        str or None: Fully qualified node name, e.g. 'bora007.plafrim.cluster'.
    """
    print(f"Checking {slurm_output_file}...")
    start = time.time()
    # Wait until the file exists
    i = 0
    while not os.path.exists(slurm_output_file):
        if i % 100 == 0:
            print(f"{slurm_output_file=} not found, waiting...")
        i += 1
        if time.time() - start > timeout:
            raise TimeoutError(f"Slurm output file {slurm_output_file} was not created within {timeout}s")
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
        if time.time() - start > timeout:
            raise TimeoutError(f"Could not find 'Fully qualified name:' in {slurm_output_file} within {timeout}s")
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


def silent_requests_get(url, timeout=1.0, headers=None):
    f = io.StringIO()
    try:
        with redirect_stdout(f), redirect_stderr(f):
            resp = requests.get(url, timeout=timeout, headers=headers)
            return resp
    except requests.RequestException:
        return None

def wait_for_model(openai_url, model_name="intfloat/e5-base-v2", timeout=300.0, interval=0.2, api_key=None):
    """
    Wait until the OpenAI-compatible server is up and the specified model is available.

    openai_url: e.g., "openai://http://localhost:56126/v1"
    model_name: name of the model to check
    timeout: maximum time to wait in seconds
    interval: delay between checks
    api_key: optional API key for authenticated endpoints (e.g. LiteLLM master key)
    """
    start = time.time()
    model_url = openai_url[len("openai://"):] +  "/models"

    print(f"Checking model on {model_url=}...")
    while True:
        try:
            headers = {}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            resp = silent_requests_get(model_url, timeout=1.0, headers=headers)
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


def warmup_embedding_model(openai_url, model_name, api_key=None, timeout=120.0, interval=2.0):
    """
    Send a test embedding request to ensure the model is truly ready for inference.

    wait_for_model only checks /v1/models metadata; this verifies that the
    model can actually serve embedding requests (weights loaded, kernels compiled).
    """
    from openai import OpenAI, OpenAIError
    base_url = openai_url[len("openai://"):]
    client = OpenAI(base_url=base_url, api_key=api_key or "EMPTY")
    start = time.time()
    last_err = None
    while True:
        try:
            client.embeddings.create(model=model_name, input=["warmup"])
            print(f"Warmup embedding succeeded after {time.time() - start:.1f}s")
            return
        except (OpenAIError, Exception) as e:
            last_err = e
        if time.time() - start > timeout:
            raise RuntimeError(
                f"Model {model_name} at {base_url} did not become ready for "
                f"inference after {timeout}s (last error: {last_err})"
            )
        time.sleep(interval)


def launch_rivault_model(model_cfg):
    model_type_key = model_cfg["model_name"] + ":" + model_cfg["model_type"]
    # --- 0. Check cache ---
    job_id = jobid_cache.get(model_type_key, None)
    if job_id is not None:
        print(f"Found job_id {job_id} in cache for model type {model_type_key}. Checking if it's running...")
        if wait_for_running(job_id):
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

    # --- 0b. BentoML models share a single Rivault stack. If another BentoML
    #          model already has a running job, reuse it instead of submitting
    #          a second Slurm job that would just PEND forever. ---
    if job_id is None and model_cfg["model_type"].startswith("bentoml://"):
        for cached_key in list(jobid_cache):
            if cached_key == model_type_key:
                continue
            # Only reuse jobs from other BentoML models (same Rivault stack)
            if ":bentoml://" not in str(cached_key):
                continue
            cached_job_id = jobid_cache.get(cached_key)
            if cached_job_id is not None and is_slurm_job_running(cached_job_id):
                print(f"===> Reusing running BentoML Rivault job {cached_job_id} from '{cached_key}' for '{model_type_key}'.")
                job_id = cached_job_id
                break

    # --- 1. get a node form SLURM ---
    rivault_slurm_script = get_slurm_script_from_host()

    # BentoML vision models need the vision-bentoml compose profile.
    sbatch_env = None
    if model_cfg["model_type"].startswith("bentoml://"):
        sbatch_env = {"RIVAULT_PROFILES": "harness vllm vision-bentoml"}

    output_file, error_file, job_id = get_slurm_node(rivault_slurm_script, model_type_key, job_id_to_use=job_id, sbatch_env=sbatch_env)

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

     # --- 3. Build the OpenAI-compatible model or BentoML vision model ---
    if model_cfg["model_type"].startswith("bentoml://"):
        # BentoML vision service: resolve placeholders and verify health.
        # Traffic is routed through NGINX /vision/ prefix so all services
        # are reachable via the single SSH-tunneled port.
        model_cfg["model_type"] = model_cfg["model_type"].replace("{NGINX_PORT}", str(NGINX_PORT))
        model_cfg["model_type"] = model_cfg["model_type"].replace("{LOCAL_PORT}", str(local_port))
        bentoml_url = model_cfg["model_type"][len("bentoml://"):]
        print(f"Configuring BentoML vision service at: {bentoml_url}...")

        # Wait for BentoML service health
        from umrs.bentoml_client import BentoMLTaskClient

        # Detect endpoint name from the URL path (e.g. ".../hmae_reconstruct/submit" → "hmae_reconstruct")
        url_path = bentoml_url.rstrip("/")
        # Strip trailing /submit if present
        if url_path.endswith("/submit"):
            url_path = url_path[:-len("/submit")]
        endpoint_name = url_path.rsplit("/", 1)[-1]

        # Derive the service base URL by stripping the endpoint path
        # e.g. "http://localhost:33241/vision/sam3_segment/submit" → "http://localhost:33241/vision"
        service_base_url = url_path[:url_path.rfind("/" + endpoint_name)]

        client = BentoMLTaskClient(service_base_url)
        _wait_start = time.time()
        while not client.is_healthy():
            if time.time() - _wait_start > 300:
                raise RuntimeError(f"BentoML service at {bentoml_url} not healthy after 300s")
            time.sleep(2)
        print(f"BentoML vision service is healthy at {bentoml_url}")

        from umrs.models.bentoml_adapter import make_bentoml_hmae_compute, make_bentoml_sam3_compute

        mode = model_cfg.get("mode", "reconstruct")
        model_name = model_cfg["model_name"]
        vector_size = model_cfg.get("vector_size", "768")
        return_type = model_cfg.get("return_type", "file")
        modality = model_cfg.get("_modality", "")

        if "hmae" in endpoint_name:
            compute_func = make_bentoml_hmae_compute(client, endpoint_name, mode)
            input_type = "raw_image"
        elif "sam3" in endpoint_name or "segment" in endpoint_name:
            compute_func = make_bentoml_sam3_compute(client, endpoint_name)
            input_type = "image"
        else:
            raise RuntimeError(f"Unknown BentoML endpoint: {endpoint_name}")

        model = load_model_as_bentoml(model_name, return_type, vector_size, input_type, compute_func, modality)

        model_cache[model_type_key] = (model, thread_tunnel, proc_tunnel)
        jobid_cache[model_type_key] = job_id
        return model

    elif model_cfg["model_type"].startswith("openai://"):
        # replace a placeholder {NGINX_PORT} with the actual value
        model_cfg["model_type"] = model_cfg["model_type"].replace("{NGINX_PORT}", str(NGINX_PORT))
        # if you also want to replace a placeholder {local_port} with a variable 'local_port'
        model_cfg["model_type"] = model_cfg["model_type"].replace("{LOCAL_PORT}", str(local_port))
        wait_for_model(model_cfg["model_type"], model_cfg["model_name"], api_key=MASTER_KEY)
        warmup_embedding_model(model_cfg["model_type"], model_cfg["model_name"], api_key=MASTER_KEY)
        print(f"Configuring: {model_cfg['model_type']}...")
        # Store the API key in model_cfg so load_model can forward it
        model_cfg["api_key"] = MASTER_KEY
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

