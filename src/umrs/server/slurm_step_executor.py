"""Slurm step executor — runs user workflow steps as Slurm jobs.

This executor follows the same pattern as the existing
:class:`SlurmPodmanBackend` for vLLM/BentoML: it submits a Slurm job
to a GPU compute node, which runs the step inside a podman container.

Flow:
    1. Serialize step entrypoint, config, and StepContext to ``input.pkl``
    2. Generate an sbatch script that:
       - Allocates GPU resources on a compute node
       - Runs ``podman run --rm --gpus all -v <workdir>:/work <image>``
         ``python -m umrs.server.step_worker /work/input.pkl /work/output.pkl``
    3. Submit via ``sbatch``
    4. Poll ``squeue`` for completion
    5. Read ``output.pkl`` and return the :class:`StepResult`

The executor is selected by setting ``UMRS_STEP_EXECUTOR=slurm``.
"""

from __future__ import annotations

import logging
import os
import pickle
import shlex
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any

from umrs.pipeline_sdk.step import Step, StepContext, StepResult

from .step_executor import (
    EXECUTOR_GPU_METADATA_KEY,
    EXECUTOR_IMAGE_METADATA_KEY,
    StepExecutionError,
    StepExecutor,
    _drive_lifecycle,
    _load_step_class,
)

logger = logging.getLogger(__name__)


@dataclass
class SlurmStepExecutor(StepExecutor):
    """Run workflow steps as Slurm jobs on GPU compute nodes.

    This executor is designed for the RCCS deployment pattern where
    UMRS runs on a login node and submits Slurm jobs to GPU compute
    nodes. Each step runs inside a podman container with GPU access.

    Configuration via environment variables:

    * ``UMRS_SLURM_PARTITION`` — Slurm partition (default: ``gpu``)
    * ``UMRS_SLURM_GPUS`` — Number of GPUs to request (default: ``1``)
    * ``UMRS_SLURM_CPUS`` — CPUs per task (default: ``8``)
    * ``UMRS_SLURM_MEM`` — Memory per node (default: ``32G``)
    * ``UMRS_SLURM_TIME`` — Time limit (default: ``01:00:00``)
    * ``UMRS_SLURM_ACCOUNT`` — Slurm account (optional)
    * ``UMRS_STEP_IMAGE`` — Container image to run the step in
    * ``UMRS_STEP_GPU`` — Whether to attach GPU (default: ``0``)
    """

    partition: str = field(
        default_factory=lambda: os.environ.get("UMRS_SLURM_PARTITION", "gpu")
    )
    gpus: int = field(
        default_factory=lambda: int(os.environ.get("UMRS_SLURM_GPUS", "1"))
    )
    cpus: int = field(
        default_factory=lambda: int(os.environ.get("UMRS_SLURM_CPUS", "8"))
    )
    memory: str = field(default_factory=lambda: os.environ.get("UMRS_SLURM_MEM", "32G"))
    time_limit: str = field(
        default_factory=lambda: os.environ.get("UMRS_SLURM_TIME", "01:00:00")
    )
    account: str = field(
        default_factory=lambda: os.environ.get("UMRS_SLURM_ACCOUNT", "")
    )
    image: str = field(default_factory=lambda: os.environ.get("UMRS_STEP_IMAGE", ""))
    timeout_seconds: int = field(
        default_factory=lambda: int(
            os.environ.get("UMRS_STEP_EXECUTOR_TIMEOUT", "3600")
        )
    )
    poll_interval: int = field(default=5)
    keep_workdir: bool = field(default=False)

    def execute(
        self,
        *,
        entrypoint: str,
        config: dict[str, Any],
        ctx: StepContext,
    ) -> StepResult:
        """Execute a step as a Slurm job on a GPU compute node."""
        # Determine the container image
        image = self._resolve_image(ctx)

        # Create a temporary working directory
        workdir = tempfile.mkdtemp(prefix="slurm-step-")
        logger.info(
            "SlurmStepExecutor: workdir=%s, image=%s, entrypoint=%s",
            workdir,
            image,
            entrypoint,
        )

        try:
            # Serialize the step payload
            input_path = os.path.join(workdir, "input.pkl")
            output_path = os.path.join(workdir, "output.pkl")

            payload = {
                "entrypoint": entrypoint,
                "config": config,
                "ctx_input_bucket": ctx.input_bucket,
                "ctx_input_keys": ctx.input_keys,
                "ctx_output_bucket": ctx.output_bucket,
                "ctx_config": ctx.config,
                "ctx_metadata": ctx.metadata,
                "ctx_run_id": ctx.run_id,
            }

            with open(input_path, "wb") as f:
                pickle.dump(payload, f)

            # Generate the sbatch script
            sbatch_script = self._generate_sbatch_script(
                workdir=workdir,
                image=image,
                input_path=input_path,
                output_path=output_path,
                ctx=ctx,
            )

            # Write the sbatch script
            sbatch_path = os.path.join(workdir, "run_step.sbatch")
            with open(sbatch_path, "w") as f:
                f.write(sbatch_script)

            # Submit the Slurm job
            job_id = self._submit_sbatch(sbatch_path, workdir)
            logger.info(
                "SlurmStepExecutor: submitted job %s for step %s",
                job_id,
                entrypoint,
            )

            # Poll for completion
            result = self._poll_completion(
                job_id=job_id,
                output_path=output_path,
                workdir=workdir,
            )
            return result

        finally:
            if not self.keep_workdir:
                import shutil

                shutil.rmtree(workdir, ignore_errors=True)

    def _resolve_image(self, ctx: StepContext) -> str:
        """Resolve the container image to use for the step."""
        # Check per-step image override
        if EXECUTOR_IMAGE_METADATA_KEY in ctx.metadata:
            override = ctx.metadata[EXECUTOR_IMAGE_METADATA_KEY]
            if not isinstance(override, str):
                raise StepExecutionError(
                    "setup",
                    f"per-step image override must be a string, got "
                    f"{type(override).__name__}",
                )
            if not override.strip():
                raise StepExecutionError(
                    "setup",
                    "per-step image override is empty; remove the hint "
                    "or supply a valid image reference",
                )
            return override

        # Fall back to deployment-wide default
        if not self.image:
            raise StepExecutionError(
                "setup",
                "SlurmStepExecutor requires a container image. "
                "Set UMRS_STEP_IMAGE or routing_hints.image.",
            )
        return self.image

    def _generate_sbatch_script(
        self,
        *,
        workdir: str,
        image: str,
        input_path: str,
        output_path: str,
        ctx: StepContext,
    ) -> str:
        """Generate the sbatch script for the Slurm job."""
        # Determine if GPU is needed
        gpu_hint = ctx.metadata.get(EXECUTOR_GPU_METADATA_KEY)
        needs_gpu = gpu_hint if gpu_hint is not None else False

        # Build sbatch directives
        lines = ["#!/bin/bash"]
        lines.append(f"#SBATCH --job-name=umrs-step")
        lines.append(f"#SBATCH --partition={self.partition}")
        lines.append(f"#SBATCH --time={self.time_limit}")
        lines.append(f"#SBATCH --cpus-per-task={self.cpus}")
        lines.append(f"#SBATCH --mem={self.memory}")
        lines.append(f"#SBATCH --output={workdir}/slurm-%j.out")
        lines.append(f"#SBATCH --error={workdir}/slurm-%j.err")

        if self.account:
            lines.append(f"#SBATCH --account={self.account}")

        if needs_gpu:
            lines.append(f"#SBATCH --gres=gpu:{self.gpus}")

        lines.append("")
        lines.append("set -e")
        lines.append(f'echo "Starting step at $(date)"')
        lines.append(f'echo "Workdir: {workdir}"')
        lines.append(f'echo "Image: {image}"')
        lines.append("")

        # Build the podman run command
        gpu_flag = "--gpus all" if needs_gpu else ""
        input_rel = os.path.relpath(input_path, workdir)
        output_rel = os.path.relpath(output_path, workdir)

        podman_cmd = (
            f"podman run --rm "
            f"-v {workdir}:/work "
            f"{gpu_flag} "
            f"{image} "
            f"python -m umrs.server.step_worker "
            f"/work/{input_rel} /work/{output_rel}"
        )

        lines.append(f'echo "Running: {podman_cmd}"')
        lines.append(podman_cmd)
        lines.append(f'echo "Step completed at $(date)"')
        lines.append("")

        return "\n".join(lines)

    def _submit_sbatch(self, sbatch_path: str, workdir: str) -> str:
        """Submit the sbatch script and return the job ID."""
        cmd = ["sbatch", sbatch_path]
        logger.info("Submitting: %s", " ".join(cmd))

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode != 0:
            raise StepExecutionError(
                "worker",
                f"sbatch submission failed (rc={result.returncode}): "
                f"{result.stderr.strip()}",
            )

        # Parse job ID from "Submitted batch job 12345"
        output = result.stdout.strip()
        if "Submitted batch job" not in output:
            raise StepExecutionError(
                "worker",
                f"Unexpected sbatch output: {output}",
            )

        job_id = output.split()[-1]
        return job_id

    def _poll_completion(
        self,
        *,
        job_id: str,
        output_path: str,
        workdir: str,
    ) -> StepResult:
        """Poll squeue until the job completes, then read the output."""
        start_time = time.monotonic()
        deadline = start_time + self.timeout_seconds

        while time.monotonic() < deadline:
            # Check job status
            status = self._check_job_status(job_id)

            if status == "COMPLETED":
                # Read the output pickle
                if not os.path.isfile(output_path):
                    raise StepExecutionError(
                        "worker",
                        f"Job {job_id} completed but output file "
                        f"not found: {output_path}",
                    )

                with open(output_path, "rb") as f:
                    result = pickle.load(f)

                if not isinstance(result, StepResult):
                    raise StepExecutionError(
                        "worker",
                        f"Expected StepResult, got {type(result).__name__}",
                    )

                logger.info(
                    "SlurmStepExecutor: job %s completed successfully",
                    job_id,
                )
                return result

            elif status in ("FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY"):
                # Read the Slurm error log for diagnostics
                error_log = os.path.join(workdir, f"slurm-{job_id}.err")
                error_msg = ""
                if os.path.isfile(error_log):
                    with open(error_log, "r") as f:
                        error_msg = f.read()[-1000:]

                raise StepExecutionError(
                    "worker",
                    f"Slurm job {job_id} failed with status {status}. "
                    f"Error log: {error_msg}",
                )

            # Still running, wait and poll again
            time.sleep(self.poll_interval)

        # Timeout
        raise StepExecutionError(
            "worker",
            f"Slurm job {job_id} timed out after {self.timeout_seconds}s",
        )

    def _check_job_status(self, job_id: str) -> str:
        """Check the status of a Slurm job via squeue/sacct."""
        # Try squeue first (for running/pending jobs)
        cmd = [
            "squeue",
            "--job",
            job_id,
            "--format=%T",
            "--noheader",
        ]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=15,
            )
            if result.returncode == 0 and result.stdout.strip():
                # Job is still in queue. Normalize identically to the sacct
                # branch below: array/het jobs emit one line per element —
                # take the first line, then strip trailing info tokens.
                state = result.stdout.strip().split("\n")[0].split()[0].upper()
                # Map Slurm states to our simplified status
                if state in ("PENDING", "CONFIGURING"):
                    return "PENDING"
                elif state in ("RUNNING", "COMPLETING"):
                    return "RUNNING"
                else:
                    return state
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        # Job not in squeue, check sacct for final state
        cmd = [
            "sacct",
            "--job",
            job_id,
            "--format=State",
            "--noheader",
            "--parsable2",
        ]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=15,
            )
            if result.returncode == 0 and result.stdout.strip():
                state = result.stdout.strip().split("\n")[0].upper()
                # Remove any extra info like "COMPLETED by 12345"
                state = state.split()[0]
                return state
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        # Can't determine status, assume running
        return "RUNNING"
