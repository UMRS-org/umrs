"""Rendering helpers for Slurm/Podman Vision Model serving jobs.

Generates sbatch scripts that launch the generic vision model service
inside a Podman container on Slurm compute nodes with GPU access via CDI.

This is the Podman-first equivalent of ``slurm_script.py`` (which uses
Singularity for LLM mini-RiVault jobs).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class VisionJobConfig:
    """Configuration for a Slurm vision model serving job."""

    # Slurm resource allocation
    partition: str = "gpu"
    time_limit: str = "04:00:00"
    gpus: int = 1
    cpus: int = 8
    mem: str = "32G"
    nodelist_hint: Optional[str] = None
    slurm_log_dir: str = "/tmp/umrs-vision-logs"
    slurm_modules: List[str] = field(default_factory=list)

    # Container configuration
    container_image: str = "umrs-vision-service:latest"
    container_runtime: str = "podman"  # podman | docker
    bentoml_port: int = 3000

    # Model and adapter paths
    model_path: str = "/models"
    adapter_path: str = ""  # Custom adapter directory (optional)
    model_type: str = ""  # Primary model type to load (e.g., "sam3", "hmae", "huggingface")

    # Environment overrides
    max_vision_models: int = 3
    vision_require_presigned_urls: bool = True
    hf_offline: bool = True
    extra_env: dict = field(default_factory=dict)

    # Orchestrator integration
    job_info_dir: str = "/tmp/umrs-vision-jobs"
    startup_timeout_sec: int = 120

    # S3 configuration for presigned URL generation
    s3_endpoint_url: str = ""
    s3_presign_endpoint: str = ""


def render_vision_sbatch_script(cfg: VisionJobConfig, **kwargs: Any) -> str:
    """Generate sbatch script for a Podman-based vision model serving job.

    The generated script:
    1. Allocates GPU node via Slurm
    2. Launches Podman container with GPU access (CDI)
    3. Mounts models and adapters as read-only volumes
    4. Waits for BentoML health endpoint
    5. Writes job info file for UMRS orchestrator discovery
    """
    return "\n".join(
        part.rstrip()
        for part in (
            _vision_header(cfg),
            _vision_config(cfg),
            _vision_job_info(cfg),
            _vision_cleanup(cfg),
            _vision_validation(cfg),
            _vision_launch(cfg),
            _vision_readiness(cfg),
        )
    )


def _vision_header(cfg: VisionJobConfig) -> str:
    nodelist = f"\n#SBATCH --nodelist={cfg.nodelist_hint}" if cfg.nodelist_hint else ""
    return f"""#!/bin/bash
#SBATCH --job-name=vision-service
#SBATCH --partition={cfg.partition}{nodelist}
#SBATCH --time={cfg.time_limit}
#SBATCH --gres=gpu:{cfg.gpus}
#SBATCH --cpus-per-task={cfg.cpus}
#SBATCH --mem={cfg.mem}
#SBATCH --output={cfg.slurm_log_dir}/vision-job-%j.out
#SBATCH --error={cfg.slurm_log_dir}/vision-job-%j.err

# =============================================================================
# UMRS Generic Vision Model Service — Slurm Job (Podman)
# =============================================================================

set -e"""


def _vision_config(cfg: VisionJobConfig) -> str:
    env_lines = []
    env_lines.append(f'CONTAINER_IMAGE="{cfg.container_image}"')
    env_lines.append(f"BENTOML_PORT={cfg.bentoml_port}")
    env_lines.append(f'MODEL_PATH="{cfg.model_path}"')
    env_lines.append(f'ADAPTER_PATH="{cfg.adapter_path}"')
    env_lines.append(f'MODEL_TYPE="{cfg.model_type}"')
    env_lines.append(f"MAX_VISION_MODELS={cfg.max_vision_models}")
    env_lines.append(f'JOB_INFO_DIR="{cfg.job_info_dir}"')
    env_lines.append(f'CONTAINER_RUNTIME="{cfg.container_runtime}"')

    return f"""
# Configuration
{chr(10).join(env_lines)}

echo "============================================="
echo "UMRS Vision Model Service"
echo "============================================="
echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $(hostname)"
echo "Model Path:    $MODEL_PATH"
echo "Model Type:    $MODEL_TYPE"
echo "Container:     $CONTAINER_IMAGE"
echo "Runtime:       $CONTAINER_RUNTIME"
echo "Port:          $BENTOML_PORT"
echo "=============================================" """


def _vision_job_info(cfg: VisionJobConfig) -> str:
    return """
# Create job info directory
mkdir -p "$JOB_INFO_DIR"

# Write initial job info for UMRS orchestrator discovery
JOB_INFO_FILE="$JOB_INFO_DIR/vision-job-${SLURM_JOB_ID}.info"
cat > "$JOB_INFO_FILE" << EOF
VISION_JOB_ID=${SLURM_JOB_ID}
VISION_NODE=$(hostname)
VISION_PORT=${BENTOML_PORT}
VISION_MODEL_PATH=${MODEL_PATH}
VISION_MODEL_TYPE=${MODEL_TYPE}
VISION_START_TIME=$(date -Iseconds)
VISION_STATUS=starting
VISION_CONTAINER_IMAGE=${CONTAINER_IMAGE}
EOF"""


def _vision_cleanup(cfg: VisionJobConfig) -> str:
    return """
# Cleanup function
cleanup() {
    echo "Cleaning up vision service..."
    if [[ -f "$JOB_INFO_FILE" ]]; then
        sed -i 's/VISION_STATUS=.*/VISION_STATUS=stopped/' "$JOB_INFO_FILE"
        echo "VISION_STOP_TIME=$(date -Iseconds)" >> "$JOB_INFO_FILE"
    fi
    # Stop the container gracefully
    $CONTAINER_RUNTIME stop vision-${SLURM_JOB_ID} 2>/dev/null || true
    $CONTAINER_RUNTIME rm -f vision-${SLURM_JOB_ID} 2>/dev/null || true
    sleep 2
    rm -f "$JOB_INFO_FILE"
    echo "Cleanup complete."
}
trap cleanup EXIT INT TERM"""


def _vision_validation(cfg: VisionJobConfig) -> str:
    return """
# Validate prerequisites
if ! command -v "$CONTAINER_RUNTIME" &> /dev/null; then
    echo "Error: $CONTAINER_RUNTIME not found in PATH"
    sed -i 's/VISION_STATUS=.*/VISION_STATUS=failed/' "$JOB_INFO_FILE"
    exit 1
fi

# Check that container image is available
if ! $CONTAINER_RUNTIME image exists "$CONTAINER_IMAGE" 2>/dev/null; then
    echo "Warning: Image $CONTAINER_IMAGE not cached locally, will attempt pull..."
fi

if [[ -n "$MODEL_PATH" && ! -d "$MODEL_PATH" ]]; then
    echo "Error: Model path not found: $MODEL_PATH"
    sed -i 's/VISION_STATUS=.*/VISION_STATUS=failed/' "$JOB_INFO_FILE"
    exit 1
fi"""


def _vision_launch(cfg: VisionJobConfig) -> str:
    # Build environment variable flags
    env_flags = []
    env_flags.append('-e "VISION_ADAPTERS_DIR=/adapters/system:/adapters/user"')
    env_flags.append(f'-e "MAX_VISION_MODELS={cfg.max_vision_models}"')
    env_flags.append(f'-e "TRANSFORMERS_OFFLINE={1 if cfg.hf_offline else 0}"')
    env_flags.append(f'-e "HF_HOME=/models/.hf_cache"')

    if cfg.vision_require_presigned_urls:
        env_flags.append('-e "VISION_REQUIRE_PRESIGNED_URLS=true"')
    if cfg.s3_endpoint_url:
        env_flags.append(f'-e "S3_ENDPOINT_URL={cfg.s3_endpoint_url}"')
    if cfg.s3_presign_endpoint:
        env_flags.append(f'-e "VISION_S3_PRESIGN_ENDPOINT={cfg.s3_presign_endpoint}"')

    for k, v in cfg.extra_env.items():
        env_flags.append(f'-e "{k}={v}"')

    env_str = " \\\n    ".join(env_flags)

    # Build volume mount flags
    volume_flags = []
    volume_flags.append('-v "${MODEL_PATH}:/models:ro"')
    if cfg.adapter_path:
        volume_flags.append(f'-v "{cfg.adapter_path}:/adapters/user:ro"')

    volume_str = " \\\n    ".join(volume_flags)

    modules_str = ""
    if cfg.slurm_modules:
        modules_str = "\n".join(
            f"module load {m} 2>/dev/null || true" for m in cfg.slurm_modules
        )
        modules_str = f"\n# Load site modules\n{modules_str}\n"

    return f"""{modules_str}
# Launch vision model service container
echo "Starting vision service container..."
$CONTAINER_RUNTIME run \\
    --name vision-${{SLURM_JOB_ID}} \\
    --rm \\
    --device nvidia.com/gpu=all \\
    {volume_str} \\
    {env_str} \\
    -p ${{BENTOML_PORT}}:3000 \\
    "$CONTAINER_IMAGE" &

CONTAINER_PID=$!
echo "Container PID: $CONTAINER_PID" """


def _vision_readiness(cfg: VisionJobConfig) -> str:
    max_checks = cfg.startup_timeout_sec // 5
    return f"""
# Wait for readiness (check /health endpoint)
echo "Waiting for vision service to be ready..."
READY=false
for i in $(seq 1 {max_checks}); do
    if curl -sf "http://localhost:${{BENTOML_PORT}}/health" > /dev/null 2>&1; then
        READY=true
        break
    fi
    echo "  Attempt $i/{max_checks} — waiting..."
    sleep 5
done

if $READY; then
    echo ""
    echo "============================================="
    echo "Vision Service Ready!"
    echo "============================================="
    echo "Node:     $(hostname)"
    echo "API:      http://$(hostname):${{BENTOML_PORT}}/"
    echo "Health:   http://$(hostname):${{BENTOML_PORT}}/health"
    echo "Models:   http://$(hostname):${{BENTOML_PORT}}/models"
    echo "Infer:    http://$(hostname):${{BENTOML_PORT}}/infer/submit"
    echo "Job ID:   $SLURM_JOB_ID"
    echo "============================================="

    sed -i 's/VISION_STATUS=.*/VISION_STATUS=ready/' "$JOB_INFO_FILE"
    echo "VISION_READY_TIME=$(date -Iseconds)" >> "$JOB_INFO_FILE"
    echo "VISION_URL=http://$(hostname):${{BENTOML_PORT}}" >> "$JOB_INFO_FILE"

    # Keep running until Slurm kills us or container exits
    wait $CONTAINER_PID
else
    echo "Error: Vision service failed to start within {cfg.startup_timeout_sec}s"
    sed -i 's/VISION_STATUS=.*/VISION_STATUS=failed/' "$JOB_INFO_FILE"
    $CONTAINER_RUNTIME logs vision-${{SLURM_JOB_ID}} 2>/dev/null || true
    exit 1
fi
"""
