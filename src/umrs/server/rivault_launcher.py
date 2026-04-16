from umrs.inference.loading import load_model, EmbeddingModel
from umrs.shared.config import *
from umrs.server.config import *


import subprocess
import time
import re
import requests
import socket
import os
from contextlib import redirect_stderr, redirect_stdout
import io
import threading

from umrs.orchestrator.endpoint_registry import (
    EndpointStatus,
    get_registry,
)

# Ephemeral in-memory cache for models
model_cache = {}


# ---------------------------------------------------------------------------
# Job registry helpers — thin wrapper on EndpointRegistry so the rest of
# rivault_launcher can keep its simple  key → job_id  semantics while gaining
# structured state and optional PostgresBackend persistence.
# ---------------------------------------------------------------------------

_RIVAULT_PROJECT = "rivault"


def _get_cached_job(model_type_key):
    """Return the Slurm job ID (int) cached for *model_type_key*, or None."""
    ep = get_registry().get(
        _RIVAULT_PROJECT, model_type_key,
        prefer_warm=False, min_ttl_sec=0,
    )
    if ep and ep.status not in (EndpointStatus.TERMINATED, EndpointStatus.FAILED):
        return int(ep.job_id)
    return None


def _cache_job(model_type_key, job_id, host="", port=0):
    """Register / update a job in the endpoint registry."""
    get_registry().register(
        _RIVAULT_PROJECT, model_type_key, str(job_id),
        host=host, port=port, backend="slurm-podman",
        ttl_sec=14400,  # 4 hours (matches Slurm --time)
    )


def _remove_cached_job(model_type_key):
    """Remove and return the job_id (int) for *model_type_key*, or None."""
    ep = get_registry().get(
        _RIVAULT_PROJECT, model_type_key,
        prefer_warm=False, min_ttl_sec=0,
    )
    if ep:
        get_registry().remove(ep.job_id)
        return int(ep.job_id)
    return None


def _list_cached_keys():
    """Return all model_type_keys currently tracked."""
    return [ep.model for ep in get_registry().list_all(project_id=_RIVAULT_PROJECT)]


def release_backend_job(model_type_key):
    """Cancel the Slurm job and clean up SSH tunnel for a specific model.

    Removes the job from the endpoint registry / model_cache, runs scancel,
    and tears down the SSH tunnel subprocess + thread.
    """
    job_id = _remove_cached_job(model_type_key)
    if model_type_key in model_cache:
        _, thread_tunnel, proc_tunnel = model_cache.pop(model_type_key)
        try:
            proc_tunnel.terminate()
            proc_tunnel.wait(timeout=10)
        except Exception as e:
            print(f"[lifecycle] Warning: failed to terminate SSH tunnel process for '{model_type_key}': {e}")
        try:
            thread_tunnel.join(timeout=5)
        except Exception as e:
            print(f"[lifecycle] Warning: failed to join SSH tunnel thread for '{model_type_key}': {e}")
    if job_id is not None:
        try:
            subprocess.run(
                ["scancel", str(job_id)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
            )
            print(f"[lifecycle] Cancelled Slurm job {job_id} for '{model_type_key}'")
        except FileNotFoundError:
            print(f"[lifecycle] scancel not found — Slurm job {job_id} not cancelled (non-Slurm env?)")
        except Exception as e:
            print(f"[lifecycle] Error cancelling Slurm job {job_id}: {e}")


def release_all_backend_jobs():
    """Cancel ALL cached Slurm jobs and clean up SSH tunnels.

    Called when service_lifecycle is 'auto_release' and no more
    processing threads are active.
    """
    keys = _list_cached_keys()
    if not keys:
        return
    print(f"[lifecycle] Releasing {len(keys)} backend job(s)...")
    for key in keys:
        release_backend_job(key)


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
        timeout=15,
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


def get_rivault_startup_timeout_sec(default=600):
    """Return the configured RiVault startup timeout, falling back safely."""
    raw_timeout = os.environ.get("RIVAULT_STARTUP_TIMEOUT_SEC")
    if not raw_timeout:
        return default

    try:
        return max(int(raw_timeout), 1)
    except ValueError:
        print(
            "Invalid RIVAULT_STARTUP_TIMEOUT_SEC="
            f"{raw_timeout!r}; falling back to {default}s"
        )
        return default


def wait_for_running(job_id, timeout=None):
    """Wait until a Slurm job reaches RUNNING state, with timeout."""
    if timeout is None:
        timeout = get_rivault_startup_timeout_sec()

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

    # Inject partition from site config when the Slurm script lacks its own
    # #SBATCH --partition directive (avoids the "No partition specified" error).
    try:
        from umrs.orchestrator.site_config import get_site_config
        _site = get_site_config()
        _partition = getattr(_site.scheduler, "partition", "")
        if _partition:
            # Pick the first candidate; full selection logic lives in the
            # async orchestrator backend — here we just need *any* valid value.
            _first = _partition.split(",")[0].strip()
            if _first:
                sbatch_extra_args.append(f"--partition={_first}")
        _gpus = getattr(_site.scheduler, "gpus", 0)
        if _gpus:
            sbatch_extra_args.append(f"--gres=gpu:{_gpus}")
    except Exception:
        pass  # site config unavailable — let #SBATCH directives or Slurm default decide

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
                _cache_job(model_type_key, job_id)  # persist immediately
                startup_timeout = get_rivault_startup_timeout_sec()
                if not wait_for_running(job_id, timeout=startup_timeout):
                    raise ValueError(
                        f"Job {job_id} did not start running within "
                        f"{startup_timeout}s. Current state: {get_slurm_job_state(job_id)}"
                    )
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
    # Inject site.yaml values into os.environ so they propagate via sbatch
    # to serve-model.slurm (which uses ${VAR:-default} patterns).
    try:
        from umrs.orchestrator.site_config import get_site_config
        get_site_config().inject_env(overwrite=False)
    except Exception as e:
        print(f"Warning: failed to inject site config: {e}")  # No site.yaml → existing env vars / script defaults used

    model_type_key = model_cfg["model_name"] + ":" + model_cfg["model_type"]
    # --- 0. Check cache ---
    job_id = _get_cached_job(model_type_key)
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
            _remove_cached_job(model_type_key)
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
        for cached_key in _list_cached_keys():
            if cached_key == model_type_key:
                continue
            # Only reuse jobs from other BentoML models (same Rivault stack)
            if ":bentoml://" not in str(cached_key):
                continue
            cached_job_id = _get_cached_job(cached_key)
            if cached_job_id is not None and is_slurm_job_running(cached_job_id):
                print(f"===> Reusing running BentoML Rivault job {cached_job_id} from '{cached_key}' for '{model_type_key}'.")
                job_id = cached_job_id
                break

    # --- 1. get a node form SLURM ---
    # Select slurm script based on VISION_STACK setting.
    vision_stack = os.environ.get("VISION_STACK", VISION_STACK)

    if model_cfg["model_type"].startswith("bentoml://") and vision_stack == "jens":
        rivault_slurm_script = JENS_RIVAULT_SLURM_SCRIPT
        print(f"[VISION_STACK=jens] Using Jens's standalone stack: {rivault_slurm_script}")
    else:
        rivault_slurm_script = get_slurm_script_from_host()

    # Compute RIVAULT_DIR is handled by the wrapper scripts themselves:
    # they auto-discover from the Python package or use the slurmd-exported value.
    # We only pass extra sbatch env if explicitly needed.
    sbatch_env = None

    output_file, error_file, job_id = get_slurm_node(rivault_slurm_script, model_type_key, job_id_to_use=job_id, sbatch_env=sbatch_env)

    print(f"{output_file=} {job_id=}")

    # --- 2. Extract compute node address and create SSH tunnel ---
    server_address = extract_node_address(output_file) if output_file else "localhost"

    # BentoML vision models use a dedicated NGINX port (no path prefix)
    # to avoid httpx absolute-path resolution issues with /readyz etc.
    if model_cfg["model_type"].startswith("bentoml://"):
        if vision_stack == "jens":
            # Jens's stack: NGINX publishes on NGINX_LITELLM_DOCKER_PORT → BentoML:4000.
            # The port is written into env.<uuid> by serve-model.slurm; for now
            # we fall back to reading it from the UMRS env or default to 3001.
            remote_port = int(os.environ.get("BENTOML_VISION_DOCKER_PORT",
                                             os.environ.get("NGINX_LITELLM_DOCKER_PORT", 3001)))
        else:
            remote_port = int(os.environ.get("BENTOML_VISION_DOCKER_PORT", 3001))
    else:
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
        # BentoML vision service: connect via the dedicated NGINX port
        # (BENTOML_VISION_DOCKER_PORT, default 3001) which proxies to
        # bentoml-vision:3000 at root path.  This avoids the path-prefix
        # issues that arise with /vision/ location blocks — BentoML's
        # SyncHTTPClient uses absolute paths (/readyz, /schema.json,
        # /infer/submit) which would bypass the /vision/ location.
        service_base_url = f"http://localhost:{local_port}"
        print(f"Configuring BentoML vision service at: {service_base_url}...")

        import bentoml
        import httpx

        # BentoML's wait_until_server_ready only catches TimeoutException and
        # ConnectError.  During cold start the proxy may accept connections
        # then reset them (ReadError), which crashes the health-check loop.
        # Retry the entire client construction on transient read errors.
        ready_timeout = 300
        start_time = __import__("time").monotonic()
        while True:
            remaining = ready_timeout - (__import__("time").monotonic() - start_time)
            if remaining <= 0:
                raise TimeoutError(
                    f"BentoML service at {service_base_url} not ready "
                    f"after {ready_timeout}s"
                )
            try:
                # Jens's VisionFrontend checks AUTH_TOKEN via Bearer header;
                # pass the token so the BentoML client sends it on every request.
                token = MASTER_KEY if (vision_stack == "jens" and MASTER_KEY) else None
                client = bentoml.SyncHTTPClient(
                    service_base_url, server_ready_timeout=remaining,
                    token=token,
                )
                break
            except (httpx.ReadError, httpx.RemoteProtocolError):
                __import__("time").sleep(5)
                continue

        print(f"BentoML vision service is healthy at {service_base_url}")

        # BentoML 1.4.x bug: _submit() references self._opened_files but it
        # lives on self._file_manager._opened_files.  Alias it so submit works.
        if not hasattr(client, "_opened_files") and hasattr(client, "_file_manager"):
            client._opened_files = client._file_manager._opened_files

        from umrs.vision_service.bentoml_adapter import make_bentoml_vision_compute

        # Build model_config for the generic /infer endpoint.
        # vision_model_type must be set explicitly in the model config
        # (e.g. "sam3", "hmae", or any custom adapter name).
        vision_model_type = model_cfg.get("vision_model_type", "")
        if not vision_model_type:
            raise ValueError(
                "Model config requires 'vision_model_type' field "
                "(e.g. 'sam3', 'hmae', or a custom adapter name). "
                f"Config: {model_cfg}"
            )

        vision_model_config = dict(model_cfg.get("vision_model_config", {}))
        task = model_cfg.get("_task")

        if vision_stack == "jens":
            # Jens's standalone stack: /vision_task endpoint with pickle protocol.
            from umrs.vision_service.bentoml_adapter import make_jens_vision_compute
            compute_func = make_jens_vision_compute(
                client, vision_model_type, vision_model_config,
                task=task, auth_token=MASTER_KEY,
            )
        else:
            # UMRS generic stack: /infer endpoint with JSON protocol.
            compute_func = make_bentoml_vision_compute(
                client, vision_model_type, vision_model_config,
                task=task,
            )

        model_name = model_cfg["model_name"]
        vector_size = model_cfg.get("vector_size", "768")
        return_type = model_cfg.get("return_type", "file")
        modality = model_cfg.get("_modality", "")
        input_type = model_cfg.get("input_type", "image")

        model = load_model_as_bentoml(model_name, return_type, vector_size, input_type, compute_func, modality)

        model_cache[model_type_key] = (model, thread_tunnel, proc_tunnel)
        _cache_job(model_type_key, job_id, host=server_address, port=local_port)
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
        _cache_job(model_type_key, job_id, host=server_address, port=local_port)
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

