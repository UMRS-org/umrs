import io
import logging
import os
import re
import shlex
import socket
import subprocess
import threading
import time
from contextlib import redirect_stderr, redirect_stdout

import requests

logger = logging.getLogger(__name__)

from umrs.inference.loading import EmbeddingModel, load_model
from umrs.orchestrator.endpoint_registry import (
    EndpointStatus,
    get_registry,
)
from umrs.server.config import (
    VISION_STACK,
    get_jens_rivault_slurm_script,
    get_rivault_remote_port,
    get_rivault_slurm_script,
)
from umrs.shared.config import get_default_tmp_subdir, get_default_umrs_cfg_dir

# Ephemeral in-memory cache for models
model_cache = {}
_REMOTE_OUTPUT_PREFIX = "remote:"
_RIVAULT_PROJECT = "rivault"
_FRONTEND_ONLY_PROFILES = {"services"}


# ---------------------------------------------------------------------------
# Job registry helpers — thin wrapper on EndpointRegistry so the rest of
# rivault_launcher can keep its simple  key → job_id  semantics while gaining
# structured state and optional PostgresBackend persistence.
# ---------------------------------------------------------------------------


def _get_cached_job(model_type_key):
    """Return the Slurm job ID (int) cached for *model_type_key*, or None."""
    ep = get_registry().get(
        _RIVAULT_PROJECT,
        model_type_key,
        prefer_warm=False,
        min_ttl_sec=0,
    )
    if ep and ep.status not in (EndpointStatus.TERMINATED, EndpointStatus.FAILED):
        return int(ep.job_id)
    return None


def _cache_job(model_type_key, job_id, host="", port=0, metadata=None):
    """Register / update a job in the endpoint registry."""
    merged_metadata = {}
    existing = get_registry().get(
        _RIVAULT_PROJECT,
        model_type_key,
        prefer_warm=False,
        min_ttl_sec=0,
    )
    if existing and existing.metadata:
        merged_metadata.update(existing.metadata)
    if metadata:
        merged_metadata.update(metadata)

    get_registry().register(
        _RIVAULT_PROJECT,
        model_type_key,
        str(job_id),
        host=host,
        port=port,
        backend="slurm-podman",
        ttl_sec=14400,  # 4 hours (matches Slurm --time)
        metadata=merged_metadata,
    )


def _remove_cached_job(model_type_key):
    """Remove and return the job_id (int) for *model_type_key*, or None."""
    ep = get_registry().get(
        _RIVAULT_PROJECT,
        model_type_key,
        prefer_warm=False,
        min_ttl_sec=0,
    )
    if ep:
        get_registry().remove(ep.job_id)
        return int(ep.job_id)
    return None


def _list_cached_keys():
    """Return all model_type_keys currently tracked."""
    return [ep.model for ep in get_registry().list_all(project_id=_RIVAULT_PROJECT)]


def _parse_profile_env():
    raw = os.environ.get("RIVAULT_PROFILES", "").strip()
    if raw:
        return tuple(p for p in raw.split() if p)
    return None


def _is_bentoml_model_type(model_type):
    return "bentoml://" in str(model_type)


def _active_vision_stack():
    return os.environ.get("VISION_STACK", VISION_STACK)


def _uses_jens_vision_stack():
    return _active_vision_stack() == "jens"


def _is_openai_model_type(model_type):
    return "openai://" in str(model_type)


def _bentoml_profiles_from(profiles):
    profile_list = [p for p in profiles if p not in _FRONTEND_ONLY_PROFILES]
    if "harness" not in profile_list:
        profile_list.insert(0, "harness")
    if "vision-bentoml" not in profile_list:
        profile_list.append("vision-bentoml")
    return tuple(profile_list)


def _read_configured_profiles():
    """Read profiles from env var or site config. Returns None if unset."""
    env_profiles = _parse_profile_env()
    if env_profiles:
        return env_profiles
    try:
        from umrs.orchestrator.site_config import get_site_config

        runtime_profiles = getattr(get_site_config().runtime, "profiles", None) or []
        profiles = tuple(str(p).strip() for p in runtime_profiles if str(p).strip())
        if profiles:
            return profiles
    except Exception:
        pass
    return None


def _get_active_rivault_profiles():
    """Return the effective RiVault compose profiles for the current site.

    This returns the global default profiles from env or site config. For
    model-aware profile selection, use ``_resolve_profiles_for_model()``.
    """
    return _read_configured_profiles() or ("harness", "vllm")


def _resolve_profiles_for_model(model_cfg):
    """Select the minimal compose profiles needed for *model_cfg*."""
    model_type = model_cfg.get("model_type", "")
    if _is_bentoml_model_type(model_type):
        env_profiles = _parse_profile_env()
        if env_profiles:
            return _bentoml_profiles_from(env_profiles)

        configured = _read_configured_profiles()
        if configured:
            return _bentoml_profiles_from(configured)

        return ("harness", "vision-bentoml")

    configured = _read_configured_profiles()
    if configured:
        # LLM / embedding jobs are served by vLLM only and must never start the
        # BentoML vision sidecar, even when the site-wide default profile list
        # (used for co-located deployments) includes ``vision-bentoml``.
        llm_profiles = tuple(p for p in configured if p != "vision-bentoml")
        return llm_profiles or ("harness", "vllm")
    return ("harness", "vllm")


def _job_capabilities_for_profiles(profiles):
    """Encode reusable RiVault capabilities derived from compose profiles."""
    profile_set = set(profiles)
    return {
        "rivault_profiles": list(profiles),
        "supports_openai": "vllm" in profile_set,
        "supports_bentoml": "vision-bentoml" in profile_set,
    }


def _endpoint_supports_bentoml(endpoint, default_profiles=()):
    """Return whether a cached RiVault endpoint exposes the BentoML vision path."""
    metadata = endpoint.metadata or {}
    if "supports_bentoml" in metadata:
        return bool(metadata["supports_bentoml"])

    cached_profiles = metadata.get("rivault_profiles")
    if isinstance(cached_profiles, str):
        profiles = tuple(profile for profile in cached_profiles.split() if profile)
    elif isinstance(cached_profiles, (list, tuple)):
        profiles = tuple(
            str(profile).strip() for profile in cached_profiles if str(profile).strip()
        )
    else:
        profiles = tuple(default_profiles)

    if profiles:
        return "vision-bentoml" in set(profiles)

    return _is_bentoml_model_type(endpoint.model)


def release_backend_job(model_type_key):
    """Cancel the Slurm job and clean up SSH tunnel for a specific model.

    Removes the job from the endpoint registry / model_cache, runs scancel,
    and tears down the SSH tunnel subprocess + thread.
    """
    job_id = _remove_cached_job(model_type_key)
    if model_type_key in model_cache:
        _, thread_tunnel, proc_tunnel = model_cache.pop(model_type_key)
        if proc_tunnel is not None:
            try:
                proc_tunnel.terminate()
                proc_tunnel.wait(timeout=10)
            except Exception as e:
                logger.info(
                    f"[lifecycle] Warning: failed to terminate SSH tunnel process for '{model_type_key}': {e}"
                )
        if thread_tunnel is not None:
            try:
                thread_tunnel.join(timeout=5)
            except Exception as e:
                logger.info(
                    f"[lifecycle] Warning: failed to join SSH tunnel thread for '{model_type_key}': {e}"
                )

    if job_id is not None:
        try:
            site = _get_site_config_if_available()
            if _is_remote_slurm_site(site):
                _run_remote_slurm_command(
                    site, f"scancel {shlex.quote(str(job_id))}", timeout=10
                )
            else:
                subprocess.run(
                    ["scancel", str(job_id)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=10,
                )
            logger.info(f"[lifecycle] Cancelled Slurm job {job_id} for '{model_type_key}'")
        except FileNotFoundError:
            logger.info(
                f"[lifecycle] scancel not found — Slurm job {job_id} not cancelled (non-Slurm env?)"
            )
        except Exception as e:
            logger.info(
                f"[lifecycle] Warning: failed to cancel Slurm job {job_id} for '{model_type_key}': {e}"
            )


def release_all_backend_jobs():
    """Cancel ALL cached Slurm jobs and clean up SSH tunnels.

    Called when service_lifecycle is 'auto_release' and no more
    processing threads are active.
    """
    keys = _list_cached_keys()
    if not keys:
        return
    logger.info(f"[lifecycle] Releasing {len(keys)} backend job(s)...")
    for key in keys:
        release_backend_job(key)


def load_model_as_bentoml(
    model_name, return_type, vector_size, input_type, compute_func, modality
):
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
    site = _get_site_config_if_available()
    if _is_remote_slurm_site(site):
        return _get_remote_slurm_job_state(site, job_id)

    result = subprocess.run(
        ["squeue", "-h", "-j", str(job_id), "-o", "%T"],
        capture_output=True,
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


def _get_site_config_if_available():
    try:
        from umrs.orchestrator.site_config import get_site_config

        return get_site_config()
    except Exception:
        return None


def _is_remote_slurm_site(site):
    return bool(
        site and getattr(getattr(site, "scheduler", None), "kind", "") == "slurm-remote"
    )


def _remote_ssh_args(site):
    extra = getattr(site, "extra", {}) or {}
    ssh_target = str(extra.get("ssh_target") or "").strip()
    if not ssh_target:
        raise RuntimeError("scheduler.kind=slurm-remote requires extra.ssh_target")

    args = ["ssh"]
    ssh_user = str(extra.get("ssh_user") or "").strip()
    if ssh_user:
        args.extend(["-l", ssh_user])
    ssh_port = int(extra.get("ssh_port") or 22)
    if ssh_port != 22:
        args.extend(["-p", str(ssh_port)])
    connect_timeout = int(extra.get("ssh_connect_timeout") or 15)
    args.extend(["-o", f"ConnectTimeout={connect_timeout}"])
    args.append(ssh_target)
    return args


def _run_remote_slurm_command(site, command, timeout=60):
    return subprocess.run(
        _remote_ssh_args(site) + [command],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _get_remote_slurm_job_state(site, job_id):
    result = _run_remote_slurm_command(
        site,
        f"squeue -h -j {shlex.quote(str(job_id))} -o %T",
        timeout=15,
    )
    if result.returncode != 0:
        return None

    state = result.stdout.strip()
    return state or None


def _remote_slurm_output_ref(path):
    return f"{_REMOTE_OUTPUT_PREFIX}{path}"


def _remote_path_from_ref(path):
    if path.startswith(_REMOTE_OUTPUT_PREFIX):
        return path[len(_REMOTE_OUTPUT_PREFIX) :]
    return ""


def _read_remote_file(site, path):
    result = _run_remote_slurm_command(site, f"cat {shlex.quote(path)}", timeout=30)
    if result.returncode != 0:
        raise FileNotFoundError(path)
    return result.stdout


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
        logger.info(
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
    error_file = os.path.join(full_tmp_path, "slurm-%j.err")

    script_content = f"""#!/bin/bash
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
"""
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


def get_slurm_node(
    slurm_script,
    model_type_key,
    job_id_to_use=None,
    sbatch_env=None,
    job_metadata=None,
    slurm_overrides=None,
):
    site = _get_site_config_if_available()
    if _is_remote_slurm_site(site):
        return get_remote_slurm_node(
            site,
            model_type_key,
            job_id_to_use,
            sbatch_env,
            job_metadata,
            slurm_overrides=slurm_overrides,
        )

    output_file = error_file = None
    if "plafrim" in slurm_script:
        create_plafrim_slurm_script(slurm_script)

    out_dir = _ensure_output_dir()
    output_file = os.path.join(
        out_dir, "slurm-%j.out"
    )  # Will be insterted in the script if there is no --output line with %j placeholder
    error_file = os.path.join(
        out_dir, "slurm-%j.err"
    )  # Will be insterted in the script if there is no --error line with %j placeholder
    # output_file, error_file = extract_files_from_slurm_script(slurm_script, default_output=default_output_file, default_error=default_error_file)

    sbatch_extra_args = [f"--output={output_file}", f"--error={error_file}"]

    # Per-request overrides (experience config "slurm" block) take precedence
    # over site.yaml defaults. Site defaults remain untouched.
    overrides = dict(slurm_overrides or {})
    partition = str(overrides.get("partition") or "").strip()

    # Inject partition from site config when the Slurm script lacks its own
    # #SBATCH --partition directive (avoids the "No partition specified" error).
    try:
        from umrs.orchestrator.site_config import get_site_config

        _site = get_site_config()
        if not partition:
            _partition = getattr(_site.scheduler, "partition", "")
            if _partition:
                # Pick the first candidate; full selection logic lives in the
                # async orchestrator backend — here we just need *any* valid value.
                partition = _partition.split(",")[0].strip()
        _gpus = getattr(_site.scheduler, "gpus", 0)
        if _gpus:
            sbatch_extra_args.append(f"--gres=gpu:{_gpus}")
        # Site-level extra sbatch flags (e.g. --reservation=..., --nodelist=...),
        # matching the async orchestrator backends.
        sbatch_extra_args.extend(getattr(_site.scheduler, "extra_sbatch", []) or [])
    except Exception:
        pass  # site config unavailable — let #SBATCH directives or Slurm default decide

    if partition:
        sbatch_extra_args.append(f"--partition={partition}")
    if overrides.get("reservation"):
        sbatch_extra_args.append(f"--reservation={overrides['reservation']}")
    if overrides.get("nodelist"):
        sbatch_extra_args.append(f"--nodelist={overrides['nodelist']}")
    # Per-request extra flags come last so they win over site-level duplicates
    # (sbatch uses the last occurrence of a repeated option).
    sbatch_extra_args.extend(overrides.get("extra_sbatch") or [])

    if sbatch_env:
        env_pairs = ",".join(f"{k}={v}" for k, v in sbatch_env.items())
        sbatch_extra_args.append(f"--export=ALL,{env_pairs}")

    # If we already have a job_id, we only need the output and error file paths. No need to call SLURM.
    if job_id_to_use is not None:
        job_id = job_id_to_use
    else:
        # Launch script
        sbatch_cmd = ["sbatch"] + sbatch_extra_args + [slurm_script]
        logger.info(f"{sbatch_cmd}")

        proc = subprocess.Popen(sbatch_cmd, stdout=subprocess.PIPE)

        logger.info("Waiting for job to start...")
        # Read job SLURM ID
        output, _ = proc.communicate()
        job_id = None
        if proc.returncode != 0:
            raise RuntimeError(
                f"sbatch failed with return code {proc.returncode}: "
                f"{output.decode() if output else '(no output)'}"
            )
        if output:
            text = output.decode()
            # sbatch returns "Submitted batch job 12345"
            match = re.search(r"Submitted batch job (\d+)", text)
            if match:
                job_id = int(match.group(1))
                logger.info("Job submitted with ID:", job_id)
                _cache_job(
                    model_type_key, job_id, metadata=job_metadata
                )  # persist immediately
                startup_timeout = get_rivault_startup_timeout_sec()
                if not wait_for_running(job_id, timeout=startup_timeout):
                    raise ValueError(
                        f"Job {job_id} did not start running within "
                        f"{startup_timeout}s. Current state: {get_slurm_job_state(job_id)}"
                    )
        if job_id is None:
            raise ValueError(
                f"Could not parse job ID from sbatch output: "
                f"{output.decode() if output else '(no output)'}"
            )

    # Replace %j placeholder with actual JOB_ID
    if output_file:
        output_file = output_file.replace("%j", str(job_id))
    if error_file:
        error_file = error_file.replace("%j", str(job_id))

    logger.info(f"Job {job_id} is running. Output will be in {output_file} and {error_file}")

    return output_file, error_file, job_id


def _first_site_partition(site):
    partition = getattr(getattr(site, "scheduler", None), "partition", "")
    return partition.split(",")[0].strip() if partition else ""


def _remote_wrapper_script(site):
    extra = getattr(site, "extra", {}) or {}
    wrapper_script = str(extra.get("slurm_wrapper_script") or "").strip()
    if not wrapper_script:
        raise RuntimeError(
            "scheduler.kind=slurm-remote requires extra.slurm_wrapper_script"
        )
    return wrapper_script


def _remote_job_info_dir(site):
    extra = getattr(site, "extra", {}) or {}
    job_info_dir = str(extra.get("slurm_job_info_dir") or "").strip()
    if job_info_dir:
        return job_info_dir.rstrip("/")
    return "/tmp"


def _remote_sbatch_args(site, sbatch_env, slurm_overrides=None):
    job_info_dir = _remote_job_info_dir(site)
    scheduler = site.scheduler
    overrides = dict(slurm_overrides or {})
    args = [
        "sbatch",
        f"--output={job_info_dir}/slurm-%j.out",
        f"--error={job_info_dir}/slurm-%j.err",
    ]

    partition = (
        str(overrides.get("partition") or "").strip() or _first_site_partition(site)
    )
    if partition:
        args.append(f"--partition={partition}")
    if getattr(scheduler, "time_limit", ""):
        args.append(f"--time={scheduler.time_limit}")
    if getattr(scheduler, "gpus", 0):
        args.append(f"--gres=gpu:{scheduler.gpus}")
    if getattr(scheduler, "cpus", 0):
        args.append(f"--cpus-per-task={scheduler.cpus}")
    if getattr(scheduler, "memory", ""):
        args.append(f"--mem={scheduler.memory}")
    else:
        args.append("--mem=0")
    if getattr(scheduler, "account", ""):
        args.append(f"--account={scheduler.account}")
    if getattr(scheduler, "qos", ""):
        args.append(f"--qos={scheduler.qos}")
    # Site-level extra sbatch flags (e.g. --reservation=..., --nodelist=...),
    # matching the async orchestrator backends.
    args.extend(getattr(scheduler, "extra_sbatch", []) or [])

    # Per-request overrides come last so they win over site-level duplicates.
    if overrides.get("reservation"):
        args.append(f"--reservation={overrides['reservation']}")
    if overrides.get("nodelist"):
        args.append(f"--nodelist={overrides['nodelist']}")
    args.extend(overrides.get("extra_sbatch") or [])

    env_overrides = dict(sbatch_env or {})
    wrapper_script = _remote_wrapper_script(site)
    env_overrides.setdefault("RIVAULT_WRAPPER_DIR", os.path.dirname(wrapper_script))
    env_overrides.setdefault("JOB_INFO_DIR", job_info_dir)
    env_overrides.setdefault("RIVAULT_CONTROL_DIR", f"{job_info_dir}/control")

    if env_overrides:
        export_parts = ["ALL"] + [
            f"{key}={value}" for key, value in env_overrides.items()
        ]
        args.append(f"--export={','.join(export_parts)}")

    args.append(wrapper_script)
    return args


def _rivault_model_path_env():
    keys = (
        "SAM3_MODEL_PATH",
        "HMAE_MODEL_PATH",
        "HMAE_CONFIG_PATH",
        "HMAE_CHECKPOINT",
    )
    return {key: os.environ[key] for key in keys if os.environ.get(key)}


def get_remote_slurm_node(
    site,
    model_type_key,
    job_id_to_use=None,
    sbatch_env=None,
    job_metadata=None,
    slurm_overrides=None,
):
    job_info_dir = _remote_job_info_dir(site)
    if job_id_to_use is not None:
        job_id = int(job_id_to_use)
    else:
        sbatch_args = _remote_sbatch_args(site, sbatch_env, slurm_overrides)
        command = shlex.join(sbatch_args)
        logger.info(f"[remote-slurm] {command}")
        result = _run_remote_slurm_command(site, command, timeout=60)
        if result.returncode != 0:
            raise RuntimeError(
                f"remote sbatch failed: {result.stderr.strip() or result.stdout.strip()}"
            )

        match = re.search(r"(?:Submitted batch job\s+)?(\d+)", result.stdout.strip())
        if not match:
            raise ValueError(
                f"Could not parse remote job ID from sbatch output: {result.stdout!r}"
            )
        job_id = int(match.group(1))
        logger.info("Remote job submitted with ID:", job_id)
        _cache_job(model_type_key, job_id, metadata=job_metadata)
        startup_timeout = get_rivault_startup_timeout_sec()
        if not wait_for_running(job_id, timeout=startup_timeout):
            raise ValueError(
                f"Remote job {job_id} did not start running within "
                f"{startup_timeout}s. Current state: {get_slurm_job_state(job_id)}"
            )

    output_file = _remote_slurm_output_ref(f"{job_info_dir}/slurm-{job_id}.out")
    error_file = _remote_slurm_output_ref(f"{job_info_dir}/slurm-{job_id}.err")
    logger.info(
        f"Remote job {job_id} is running. Output will be in {output_file} and {error_file}"
    )
    return output_file, error_file, job_id


def get_slurm_script_from_host():
    # Get the hostname
    socket.gethostname()

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
    logger.info(f"Checking {slurm_output_file}...")
    site = _get_site_config_if_available()
    remote_path = _remote_path_from_ref(slurm_output_file)
    start = time.time()
    # Wait until the file exists
    i = 0
    while True:
        file_is_ready = False
        if remote_path:
            try:
                _read_remote_file(site, remote_path)
                file_is_ready = True
            except FileNotFoundError:
                file_is_ready = False
        else:
            file_is_ready = os.path.exists(slurm_output_file)
        if file_is_ready:
            break
        if i % 100 == 0:
            logger.info(f"{slurm_output_file=} not found, waiting...")
        i += 1
        if time.time() - start > timeout:
            raise TimeoutError(
                f"Slurm output file {slurm_output_file} was not created within {timeout}s"
            )
        time.sleep(0.5)

    logger.info(f"{slurm_output_file} ready")
    while True:
        if remote_path:
            content = _read_remote_file(site, remote_path)
            lines = content.splitlines()
        else:
            with open(slurm_output_file) as slurm_file:
                lines = slurm_file.readlines()
        for line in lines:
            line = line.strip()
            # print(f"{line=}")
            if line.startswith("Fully qualified name:"):
                # Split by ":" and remove spaces
                return line.split(":", 1)[1].strip()
        if time.time() - start > timeout:
            raise TimeoutError(
                f"Could not find 'Fully qualified name:' in {slurm_output_file} within {timeout}s"
            )
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
                    raise TimeoutError(
                        f"Port {port} on {host} did not open in {timeout}s"
                    ) from None


def ssh_tunnel_forward_to_remote(
    remote_host: str,
    remote_port: int = 8001,
    ssh_target: str = "localhost",
    ssh_user: str = "",
    ssh_port: int = 22,
) -> tuple[int, threading.Thread, subprocess.Popen]:
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
    s.bind(("", 0))  # 0 means choose any free port
    local_port = s.getsockname()[1]
    s.close()

    # 2. Build the SSH command (blocking, attached to its thread)
    # -f: go to background
    # -N: no command
    # -L: local forward
    ssh_cmd = ["ssh", "-N"]  # do not execute remote command
    if ssh_port != 22:
        ssh_cmd.extend(["-p", str(ssh_port)])
    if ssh_user:
        ssh_cmd.extend(["-l", ssh_user])
    ssh_cmd.extend(
        [
            "-L",
            f"{local_port}:{remote_host}:{remote_port}",
            ssh_target,
        ]
    )

    # 3. Function to run SSH as subprocess
    def run_ssh():
        nonlocal ssh_proc
        ssh_proc = subprocess.Popen(
            ssh_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        ssh_proc.wait()

    ssh_proc = None
    thread = threading.Thread(target=run_ssh, daemon=True)
    thread.start()

    wait_for_local_port(local_port)
    logger.info(f"port {local_port} ready...")
    return local_port, thread, ssh_proc


def silent_requests_get(url, timeout=1.0, headers=None):
    f = io.StringIO()
    try:
        with redirect_stdout(f), redirect_stderr(f):
            resp = requests.get(url, timeout=timeout, headers=headers)
            return resp
    except requests.RequestException:
        return None


def wait_for_model(
    openai_url,
    model_name="intfloat/e5-base-v2",
    timeout=300.0,
    interval=0.2,
    api_key=None,
):
    """
    Wait until the OpenAI-compatible server is up and the specified model is available.

    openai_url: e.g., "openai://http://localhost:56126/v1"
    model_name: name of the model to check
    timeout: maximum time to wait in seconds
    interval: delay between checks
    api_key: optional API key for authenticated endpoints (e.g. LiteLLM master key)
    """
    start = time.time()
    model_url = openai_url[len("openai://") :] + "/models"

    logger.info(f"Checking model on {model_url=}...")
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
        except requests.RequestException:
            # print(f"{type(e).__name__}: {e}")
            pass

        if time.time() - start > timeout:
            raise TimeoutError(
                f"Model {model_name} not available at {model_url} after {timeout}s"
            )

        time.sleep(interval)


def warmup_embedding_model(
    openai_url, model_name, api_key=None, timeout=120.0, interval=2.0
):
    """
    Send a test embedding request to ensure the model is truly ready for inference.

    wait_for_model only checks /v1/models metadata; this verifies that the
    model can actually serve embedding requests (weights loaded, kernels compiled).
    """
    from openai import OpenAI, OpenAIError

    base_url = openai_url[len("openai://") :]
    client = OpenAI(base_url=base_url, api_key=api_key or "EMPTY")
    start = time.time()
    last_err = None
    while True:
        try:
            client.embeddings.create(model=model_name, input=["warmup"])
            logger.info(f"Warmup embedding succeeded after {time.time() - start:.1f}s")
            return
        except (OpenAIError, Exception) as e:
            last_err = e
        if time.time() - start > timeout:
            raise RuntimeError(
                f"Model {model_name} at {base_url} did not become ready for "
                f"inference after {timeout}s (last error: {last_err})"
            )
        time.sleep(interval)


def _resolve_remote_port(model_cfg):
    """Determine the remote port for SSH tunnel based on model type."""
    if _is_bentoml_model_type(model_cfg["model_type"]):
        if _uses_jens_vision_stack():
            remote_port = get_rivault_remote_port()
            if remote_port > 0:
                return remote_port
            return int(os.environ.get("NGINX_LITELLM_DOCKER_PORT", 4000))
        # BentoML vision models use a dedicated NGINX port (no path prefix)
        # to avoid httpx absolute-path resolution issues with /readyz etc.
        port = os.environ.get("BENTOML_VISION_DOCKER_PORT")
        return int(port or 3001)

    remote_port = get_rivault_remote_port()
    if remote_port < 0:
        remote_port = model_cfg.get("server_port")
        if remote_port is None:
            raise ValueError(
                "Remote port for RivaULT not specified. "
                "Please set the RIVAULT_REMOTE_PORT environment variable "
                "or specify 'server_port' in the model configuration."
            )
    return remote_port


def _check_bentoml_health(service_url: str, ready_timeout: int = 300):
    """Block until the BentoML service is ready (``/readyz`` returns 200)."""
    import httpx

    start_time = time.monotonic()
    while True:
        remaining = ready_timeout - (time.monotonic() - start_time)
        if remaining <= 0:
            raise TimeoutError(
                f"BentoML service at {service_url} not ready after {ready_timeout}s"
            )
        try:
            resp = httpx.get(f"{service_url}/readyz", timeout=10)
            if resp.status_code == 200:
                return
        except (httpx.ReadError, httpx.RemoteProtocolError, httpx.ConnectError):
            pass
        time.sleep(5)


def _build_bentoml_model(model_cfg, local_port):
    """Build an EmbeddingModel backed by a BentoML vision service."""
    service_url = f"http://localhost:{local_port}"
    logger.info(f"Configuring BentoML vision service at: {service_url}...")

    _check_bentoml_health(service_url)
    logger.info(f"BentoML vision service is healthy at {service_url}")

    vision_model_type = model_cfg.get("vision_model_type", "")
    if not vision_model_type:
        raise ValueError(
            "Model config requires 'vision_model_type' field "
            "(e.g. 'sam3', 'hmae', or a custom adapter name). "
            f"Config: {model_cfg}"
        )

    vision_model_config = dict(model_cfg.get("vision_model_config", {}))
    task = model_cfg.get("_task")

    if _uses_jens_vision_stack():
        from umrs.vision_service.bentoml_adapter import make_jens_vision_compute

        compute_func = make_jens_vision_compute(
            service_url,
            vision_model_type,
            vision_model_config,
            task=task,
            token=os.environ.get("JENS_VISION_TOKEN")
            or os.environ.get("LITELLM_MASTER_KEY"),
        )
    else:
        from umrs.vision_service.bentoml_adapter import make_bentoml_vision_compute

        compute_func = make_bentoml_vision_compute(
            service_url,
            vision_model_type,
            vision_model_config,
            task=task,
        )

    return load_model_as_bentoml(
        model_cfg["model_name"],
        model_cfg.get("return_type", "file"),
        model_cfg.get("vector_size", "768"),
        model_cfg.get("input_type", "image"),
        compute_func,
        model_cfg.get("_modality", ""),
    )


def _build_openai_model(model_cfg, local_port, nginx_port, master_key):
    """Build an EmbeddingModel backed by an OpenAI-compatible endpoint."""
    model_cfg["model_type"] = model_cfg["model_type"].replace(
        "{NGINX_PORT}", str(nginx_port)
    )
    model_cfg["model_type"] = model_cfg["model_type"].replace(
        "{LOCAL_PORT}", str(local_port)
    )
    wait_for_model(model_cfg["model_type"], model_cfg["model_name"], api_key=master_key)
    warmup_embedding_model(
        model_cfg["model_type"], model_cfg["model_name"], api_key=master_key
    )
    logger.info(f"Configuring: {model_cfg['model_type']}...")
    model_cfg["api_key"] = master_key
    return load_model(model_cfg, "")


def _check_cache_or_reuse_job(model_type_key, model_cfg):
    """Check cache for existing job/model, clean up stale entries.

    Returns (model, job_id):
    - model is set if a loaded model can be reused (caller should return it).
    - job_id is the running Slurm job to reuse, or None if a new job is needed.
    """
    job_id = _get_cached_job(model_type_key)
    default_profiles = _get_active_rivault_profiles()

    if job_id is not None:
        logger.info(
            f"Found job_id {job_id} in cache for model type {model_type_key}. Checking if it's running..."
        )
        if wait_for_running(job_id):
            if model_type_key in model_cache:
                model, _, _ = model_cache[model_type_key]
                logger.info(
                    "===> Found a running job and a loaded model in cache. Reusing it."
                )
                return model, job_id
            logger.info(
                "===> Found a running job in cache but no in memory model. "
                "This can happen if the server has restarted. Will try to load it again."
            )
        else:
            logger.info(
                "===> Found a job_id in cache but the job is not running. "
                "Cleaning up cache and launching a new job."
            )
            _remove_cached_job(model_type_key)
            if model_type_key in model_cache:
                _, thread_tunnel, proc_tunnel = model_cache.pop(model_type_key)
                proc_tunnel.terminate()
                proc_tunnel.wait()
                thread_tunnel.join()
            job_id = None

    # BentoML models share a single Rivault stack. If another BentoML
    # model already has a running job, reuse it instead of submitting
    # a second Slurm job that would just PEND forever.
    if job_id is None and _is_bentoml_model_type(model_cfg["model_type"]):
        for endpoint in get_registry().list_all(project_id=_RIVAULT_PROJECT):
            if endpoint.model == model_type_key:
                continue
            try:
                cached_job_id = int(endpoint.job_id)
            except (TypeError, ValueError):
                continue
            if not is_slurm_job_running(cached_job_id):
                continue
            if not _endpoint_supports_bentoml(
                endpoint, default_profiles=default_profiles
            ):
                continue

            logger.info(
                f"===> Reusing running RiVault job {cached_job_id} from "
                f"'{endpoint.model}' for '{model_type_key}' "
                "(vision-capable stack)."
            )
            job_id = cached_job_id
            break

    return None, job_id


def launch_rivault_model(model_cfg):
    # Inject site.yaml values into os.environ so they propagate via sbatch
    # to serve-model.slurm (which uses ${VAR:-default} patterns).
    try:
        from umrs.orchestrator.site_config import get_site_config

        get_site_config().inject_env(overwrite=False)
    except Exception as e:
        logger.info(f"Warning: failed to inject site config: {e}")

    # Include vision_model_config task/prompt in cache key so model closures
    # keep the correct effective_config for HMAE modes and SAM3 prompts.
    vmc = model_cfg.get("vision_model_config", {})
    config_discriminator = vmc.get("task") or vmc.get("prompt")
    task_suffix = f":{config_discriminator}" if config_discriminator else ""
    model_type_key = (
        model_cfg["model_name"] + ":" + model_cfg["model_type"] + task_suffix
    )
    job_profiles = _resolve_profiles_for_model(model_cfg)
    job_metadata = _job_capabilities_for_profiles(job_profiles)
    logger.info(f"[rivault] resolved profiles for {model_cfg['model_type']}: {job_profiles}")

    # --- 0. Check cache / reuse running job ---
    cached_model, job_id = _check_cache_or_reuse_job(model_type_key, model_cfg)
    if cached_model is not None:
        return cached_model

    # --- 1. get a node from SLURM ---
    # Select slurm script based on VISION_STACK setting.
    vision_stack = _active_vision_stack()

    if _is_bentoml_model_type(model_cfg["model_type"]) and vision_stack == "jens":
        rivault_slurm_script = get_jens_rivault_slurm_script()
        logger.info(
            f"[VISION_STACK=jens] Using Jens's standalone stack: {rivault_slurm_script}"
        )
    else:
        rivault_slurm_script = get_slurm_script_from_host()

    sbatch_env = {"RIVAULT_PROFILES": " ".join(job_profiles)}
    sbatch_env.update(_rivault_model_path_env())
    # Per-modality Slurm placement overrides injected by provisioning from
    # the experience config's "slurm" block (partition / reservation / ...).
    slurm_overrides = model_cfg.get("slurm") or {}
    if slurm_overrides:
        logger.info(f"[rivault] per-request slurm overrides: {slurm_overrides}")

    output_file, error_file, job_id = get_slurm_node(
        rivault_slurm_script,
        model_type_key,
        job_id_to_use=job_id,
        sbatch_env=sbatch_env,
        job_metadata=job_metadata,
        slurm_overrides=slurm_overrides,
    )
    logger.info(f"{output_file=} {job_id=}")

    # --- 2. Extract compute node address and create SSH tunnel ---
    server_address = extract_node_address(output_file) if output_file else "localhost"
    remote_port = _resolve_remote_port(model_cfg)
    site = _get_site_config_if_available()
    if _is_remote_slurm_site(site):
        extra = getattr(site, "extra", {}) or {}
        local_port, thread_tunnel, proc_tunnel = ssh_tunnel_forward_to_remote(
            server_address,
            remote_port,
            ssh_target=str(extra.get("ssh_target") or "localhost"),
            ssh_user=str(extra.get("ssh_user") or ""),
            ssh_port=int(extra.get("ssh_port") or 22),
        )
    else:
        local_port, thread_tunnel, proc_tunnel = ssh_tunnel_forward_to_remote(
            server_address, remote_port
        )

    NGINX_PORT = os.getenv("NGINX_LITELLM_DOCKER_PORT")
    MASTER_KEY = os.getenv("LITELLM_MASTER_KEY")

    # --- 3. Build model (BentoML vision or OpenAI embedding) ---
    if _is_bentoml_model_type(model_cfg["model_type"]):
        model = _build_bentoml_model(model_cfg, local_port)
    elif _is_openai_model_type(model_cfg["model_type"]):
        model = _build_openai_model(model_cfg, local_port, NGINX_PORT, MASTER_KEY)
    else:
        raise RuntimeError(f"Unknown model type: {model_cfg['model_type']}")

    # --- 4. Cache the loaded model and job_id ---
    model_cache[model_type_key] = (model, thread_tunnel, proc_tunnel)
    _cache_job(
        model_type_key,
        job_id,
        host=server_address,
        port=local_port,
        metadata=job_metadata,
    )
    return model
