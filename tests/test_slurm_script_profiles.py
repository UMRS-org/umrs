"""Tests for profile-driven Slurm script rendering."""

from __future__ import annotations

from umrs.orchestrator.base import OrchestratorConfig
from umrs.orchestrator.slurm_script import module_load_lines, render_sbatch_script


def test_slurm_script_uses_configured_log_dir_and_modules():
    cfg = OrchestratorConfig(
        model_path="/models/test",
        slurm_partition="gpu",
        slurm_log_dir="/shared/rivault/logs",
        slurm_job_info_dir="/shared/rivault/jobs",
        slurm_modules=["cuda/12.4", "apptainer"],
        singularity_image="/images/mini-rivault.sif",
        vllm_enable_prefix_caching=True,
    )

    script = render_sbatch_script(cfg, "/models/test")

    assert "#SBATCH --output=/shared/rivault/logs/job-%j.out" in script
    assert "#SBATCH --error=/shared/rivault/logs/job-%j.err" in script
    assert 'JOB_INFO_DIR="/shared/rivault/jobs"' in script
    assert "module load cuda/12.4 2>/dev/null || true" in script
    assert "module load apptainer 2>/dev/null || true" in script
    assert "module load cuda/12.1" not in script
    assert "VLLM_ENABLE_PREFIX_CACHING=true" in script
    assert "--env VLLM_ENABLE_PREFIX_CACHING=${VLLM_ENABLE_PREFIX_CACHING}" in script


def test_empty_slurm_modules_render_no_site_modules_comment():
    assert module_load_lines([]) == "# No site modules configured"
