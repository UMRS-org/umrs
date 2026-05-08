"""
Step ↔ Plugin adapter (Phase 3).

Wraps a pipeline-SDK ``Step`` class as a ``PluginSpec`` so it can be
dispatched to Slurm or Kubernetes via the existing plugin runners.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from .models import PluginSourceType, PluginSpec

logger = logging.getLogger(__name__)

_RESERVED_STEP_ENV_PREFIXES = ("UMRS_", "PLUGIN_")


def step_to_plugin_spec(
    step_name: str,
    entrypoint: str,
    script_dir: str,
    step_config: dict[str, Any],
    input_bucket: str,
    input_keys: list,
    output_bucket: str,
    run_id: str,
    *,
    cpu_limit: str = "1",
    memory_limit: str = "1Gi",
    gpu_count: int = 0,
    timeout_seconds: int = 3600,
    env_vars: dict[str, str] | None = None,
    user_id: str | None = None,
    output_destinations: list | None = None,
) -> PluginSpec:
    """Create a ``PluginSpec`` that runs a single pipeline Step remotely.

    The resulting spec uses ``LOCAL_PATH`` as source and sets environment
    variables that a *step executor* entrypoint reads at runtime:

    - ``UMRS_STEP_NAME`` — step name to load from ``entrypoint``
    - ``UMRS_STEP_ENTRYPOINT`` — ``module:ClassName``
    - ``UMRS_STEP_CONFIG`` — JSON-serialised step config dict
    - ``UMRS_INPUT_BUCKET`` — S3 bucket for input data
    - ``UMRS_INPUT_KEYS`` — JSON list of S3 keys
    - ``UMRS_OUTPUT_BUCKET`` — S3 bucket for output data
    - ``UMRS_RUN_ID`` — unique run identifier
    """
    user_env: dict[str, str] = {}
    if env_vars:
        for key, value in env_vars.items():
            if key.startswith(_RESERVED_STEP_ENV_PREFIXES):
                logger.warning("Ignoring reserved step env override: %s", key)
                continue
            user_env[key] = value

    step_env: dict[str, str] = {
        "UMRS_STEP_NAME": step_name,
        "UMRS_STEP_ENTRYPOINT": entrypoint,
        "UMRS_STEP_CONFIG": json.dumps(step_config),
        "UMRS_INPUT_BUCKET": input_bucket,
        "UMRS_INPUT_KEYS": json.dumps(input_keys),
        "UMRS_OUTPUT_BUCKET": output_bucket,
        "UMRS_RUN_ID": run_id,
    }
    step_env = {**user_env, **step_env}

    return PluginSpec(
        source_type=PluginSourceType.LOCAL_PATH,
        source_uri=script_dir,
        entrypoint=entrypoint,
        env_vars=step_env,
        cpu_limit=cpu_limit,
        memory_limit=memory_limit,
        gpu_count=gpu_count,
        timeout_seconds=timeout_seconds,
        output_destinations=output_destinations or [],
        user_id=user_id,
        metadata={
            "step_name": step_name,
            "run_id": run_id,
            "adapter": "StepPluginAdapter",
        },
    )
