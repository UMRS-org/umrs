"""
Slurm plugin runner (X-103).

Executes plugins as Slurm jobs with Singularity/Apptainer containers.
Follows HPC-first design principles.
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from .models import PluginResult, PluginSpec, PluginStatus
from .runner import PluginRunner

logger = logging.getLogger(__name__)

# Slurm configuration
SLURM_PARTITION = os.environ.get("SLURM_PARTITION", "gpu")
SLURM_DEFAULT_TIME = os.environ.get("SLURM_TIME", "02:00:00")
SINGULARITY_IMAGE_PATH = os.environ.get(
    "PLUGIN_SINGULARITY_IMAGE", "/opt/rivault/images/plugin-runner.sif"
)


class SlurmPluginRunner(PluginRunner):
    """
    Execute plugins as Slurm jobs (X-103).

    Uses Singularity/Apptainer containers for isolation.
    Follows HPC-first design:
    - No runtime downloads
    - Pre-staged images
    - Static networking
    """

    def __init__(
        self,
        partition: str = SLURM_PARTITION,
        reservation: str = "",
        singularity_image: str = SINGULARITY_IMAGE_PATH,
        work_dir: str | None = None,
        **kwargs,
    ):
        """
        Initialize Slurm plugin runner.

        Args:
            partition: Slurm partition for jobs.
            reservation: Slurm reservation ID (e.g. "demo_day_2026").
            singularity_image: Path to plugin runner Singularity image.
            work_dir: Working directory for job files.
        """
        super().__init__(**kwargs)
        self.partition = partition
        self.reservation = reservation
        self.singularity_image = singularity_image
        self.work_dir = Path(work_dir or tempfile.gettempdir()) / "plugin_jobs"
        self.work_dir.mkdir(parents=True, exist_ok=True)

        # Track submitted jobs: plugin_id -> slurm_job_id
        self._slurm_jobs: dict[str, str] = {}

    async def _execute_backend(
        self,
        plugin_id: str,
        spec: PluginSpec,
    ) -> PluginResult:
        """Execute plugin as Slurm job."""
        # Create job directory
        job_dir = self.work_dir / plugin_id
        job_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Generate sbatch script
            script_path = job_dir / "job.sh"
            self._generate_sbatch_script(script_path, plugin_id, spec)

            # Submit job
            slurm_job_id = await self._submit_job(script_path)
            self._slurm_jobs[plugin_id] = slurm_job_id

            logger.info(f"Submitted Slurm job {slurm_job_id} for plugin {plugin_id}")

            # Wait for completion
            result = await self._wait_for_completion(
                plugin_id, slurm_job_id, job_dir, spec
            )

            return result

        except Exception as e:
            logger.exception(f"Slurm execution failed for {plugin_id}")
            return PluginResult(
                plugin_id=plugin_id,
                status=PluginStatus.FAILED,
                error_message=str(e),
            )

    def _generate_sbatch_script(
        self,
        script_path: Path,
        plugin_id: str,
        spec: PluginSpec,
    ) -> None:
        """Generate sbatch script for plugin execution."""
        # Calculate time limit
        hours = spec.timeout_seconds // 3600
        minutes = (spec.timeout_seconds % 3600) // 60
        time_limit = f"{hours:02d}:{minutes:02d}:00"

        # GPU configuration
        gpu_line = ""
        if spec.gpu_count > 0:
            gpu_line = f"#SBATCH --gpus={spec.gpu_count}"

        # Environment variables — use SINGULARITYENV_ prefix so they
        # survive --cleanenv (Singularity propagates SINGULARITYENV_X as X
        # inside the container).
        env_vars = self._generate_container_env(spec)
        env_exports = "\n".join(
            f"export SINGULARITYENV_{k}='{v}'" for k, v in env_vars.items()
        )

        # Host-side job directory (metrics are written here, outside the
        # container, so _collect_results() can find them).
        host_job_dir = self.work_dir / plugin_id

        # Singularity bind mounts
        binds = [
            f"{host_job_dir}:/workspace",
        ]

        # Bind-mount the step source directory so user scripts are
        # accessible inside the container at /source.
        if spec.source_uri and spec.source_type.value == "local_path":
            binds.append(f"{spec.source_uri}:/source:ro")

        # Add S3 output paths if configured
        for dest in spec.output_destinations:
            if dest.startswith("s3://"):
                # Mount MinIO/S3 credentials
                binds.append("/etc/minio:/etc/minio:ro")
                break

        bind_args = ",".join(binds)

        reservation_line = (
            f"#SBATCH --reservation={self.reservation}" if self.reservation else ""
        )

        script_content = f"""#!/bin/bash
#SBATCH --job-name=plugin-{plugin_id[:8]}
{reservation_line}
#SBATCH --partition={self.partition}
#SBATCH --time={time_limit}
#SBATCH --cpus-per-task={spec.cpu_limit}
#SBATCH --mem={spec.memory_limit}
{gpu_line}
#SBATCH --output={host_job_dir}/slurm-%j.out
#SBATCH --error={host_job_dir}/slurm-%j.err

# Environment setup (SINGULARITYENV_ prefix → available inside container)
{env_exports}

# Record start time on host (where _collect_results reads)
echo "PLUGIN_START=$(date -Iseconds)" >> {host_job_dir}/metrics.txt

# Execute plugin in Singularity container
singularity exec \\
    --containall \\
    --cleanenv \\
    --bind {bind_args} \\
    {self.singularity_image} \\
    python -m umrs.plugins.plugin_runner.entrypoint

EXIT_CODE=$?

# Record end time and exit code on host
echo "PLUGIN_END=$(date -Iseconds)" >> {host_job_dir}/metrics.txt
echo "PLUGIN_EXIT_CODE=$EXIT_CODE" >> {host_job_dir}/metrics.txt

exit $EXIT_CODE
"""
        script_path.write_text(script_content)
        os.chmod(script_path, 0o755)

    async def _submit_job(self, script_path: Path) -> str:
        """Submit sbatch script and return job ID."""
        proc = await asyncio.create_subprocess_exec(
            "sbatch",
            "--parsable",
            str(script_path),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            raise RuntimeError(f"sbatch failed: {stderr.decode()}")

        # Parse job ID from output (format: "12345" or "12345;cluster")
        job_id = stdout.decode().strip().split(";")[0]
        return job_id

    async def _wait_for_completion(
        self,
        plugin_id: str,
        slurm_job_id: str,
        job_dir: Path,
        spec: PluginSpec,
    ) -> PluginResult:
        """Wait for Slurm job completion and collect results."""
        poll_interval = 5  # seconds
        max_wait = spec.timeout_seconds + 300  # Extra buffer
        elapsed = 0

        while elapsed < max_wait:
            # Check job status
            status = await self._get_slurm_status(slurm_job_id)

            if status in ("COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL"):
                break

            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

        # Collect results
        return await self._collect_results(plugin_id, slurm_job_id, job_dir, status)

    async def _get_slurm_status(self, job_id: str) -> str:
        """Get Slurm job status."""
        proc = await asyncio.create_subprocess_exec(
            "squeue",
            "--job",
            job_id,
            "--noheader",
            "--format",
            "%T",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()

        status = stdout.decode().strip()
        if not status:
            # Job no longer in queue, check sacct
            return await self._get_sacct_status(job_id)

        return status

    async def _get_sacct_status(self, job_id: str) -> str:
        """Get completed job status from sacct."""
        proc = await asyncio.create_subprocess_exec(
            "sacct",
            "--job",
            job_id,
            "--noheader",
            "--format",
            "State",
            "--parsable2",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()

        # Parse first line (main job, not steps)
        lines = stdout.decode().strip().split("\n")
        if lines:
            return lines[0].strip()
        return "UNKNOWN"

    async def _collect_results(
        self,
        plugin_id: str,
        slurm_job_id: str,
        job_dir: Path,
        slurm_status: str,
    ) -> PluginResult:
        """Collect results from completed job."""
        # Read output files
        stdout_file = list(job_dir.glob("slurm-*.out"))
        stderr_file = list(job_dir.glob("slurm-*.err"))

        stdout = ""
        stderr = ""
        if stdout_file:
            stdout = stdout_file[0].read_text()[-50000:]  # Last 50KB
        if stderr_file:
            stderr = stderr_file[0].read_text()[-50000:]

        # Read metrics
        metrics: dict[str, Any] = {"slurm_job_id": slurm_job_id}
        metrics_file = job_dir / "metrics.txt"
        if metrics_file.exists():
            for line in metrics_file.read_text().splitlines():
                if "=" in line:
                    key, value = line.split("=", 1)
                    metrics[key] = value

        # Determine status
        exit_code = int(metrics.get("PLUGIN_EXIT_CODE", -1))
        if slurm_status == "COMPLETED" and exit_code == 0:
            status = PluginStatus.SUCCEEDED
        elif slurm_status == "CANCELLED":
            status = PluginStatus.CANCELLED
        else:
            status = PluginStatus.FAILED

        # Find output files
        output_files = [
            str(f.relative_to(job_dir)) for f in job_dir.glob("output/*") if f.is_file()
        ]

        return PluginResult(
            plugin_id=plugin_id,
            status=status,
            exit_code=exit_code if exit_code >= 0 else None,
            stdout=stdout,
            stderr=stderr,
            output_files=output_files,
            metrics=metrics,
        )

    async def cancel(self, plugin_id: str) -> bool:
        """Cancel a running plugin job."""
        slurm_job_id = self._slurm_jobs.get(plugin_id)
        if not slurm_job_id:
            return False

        try:
            proc = await asyncio.create_subprocess_exec(
                "scancel",
                slurm_job_id,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            await proc.communicate()
            return proc.returncode == 0
        except Exception:
            logger.exception(f"Failed to cancel Slurm job {slurm_job_id}")
            return False

    async def get_status(self, plugin_id: str) -> PluginStatus | None:
        """Get current status of a plugin execution."""
        slurm_job_id = self._slurm_jobs.get(plugin_id)
        if not slurm_job_id:
            return None

        slurm_status = await self._get_slurm_status(slurm_job_id)

        status_map = {
            "PENDING": PluginStatus.PENDING,
            "RUNNING": PluginStatus.RUNNING,
            "COMPLETED": PluginStatus.SUCCEEDED,
            "FAILED": PluginStatus.FAILED,
            "CANCELLED": PluginStatus.CANCELLED,
            "TIMEOUT": PluginStatus.FAILED,
        }

        return status_map.get(slurm_status, PluginStatus.RUNNING)
