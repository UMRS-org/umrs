"""
Step ↔ Plugin adapter (Phase 3).

Wraps a pipeline-SDK ``Step`` class as a ``PluginSpec`` so it can be
dispatched to Slurm or Kubernetes via the existing plugin runners.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from .models import PluginSpec, PluginSourceType

logger = logging.getLogger(__name__)


def step_to_plugin_spec(
    step_name: str,
    entrypoint: str,
    script_dir: str,
    step_config: Dict[str, Any],
    input_bucket: str,
    input_keys: list,
    output_bucket: str,
    run_id: str,
    *,
    cpu_limit: str = "1",
    memory_limit: str = "1Gi",
    gpu_count: int = 0,
    timeout_seconds: int = 3600,
    env_vars: Optional[Dict[str, str]] = None,
    user_id: Optional[str] = None,
    output_destinations: Optional[list] = None,
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
    step_env: Dict[str, str] = {
        "UMRS_STEP_NAME": step_name,
        "UMRS_STEP_ENTRYPOINT": entrypoint,
        "UMRS_STEP_CONFIG": json.dumps(step_config),
        "UMRS_INPUT_BUCKET": input_bucket,
        "UMRS_INPUT_KEYS": json.dumps(input_keys),
        "UMRS_OUTPUT_BUCKET": output_bucket,
        "UMRS_RUN_ID": run_id,
    }
    if env_vars:
        step_env.update(env_vars)

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
