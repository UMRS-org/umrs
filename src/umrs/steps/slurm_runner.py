"""
Slurm step runner for user-defined workflow steps (U-206b).

Executes UMRSStep instances as Slurm batch jobs with Singularity containers.
Follows the same synchronous interface as ``SubprocessSandbox`` so it can be
used interchangeably in ``WorkflowExecutor``.

Design:
  * Generates an sbatch script with bind-mounts for user code + workspace
  * Reuses the same worker script pattern as ``sandbox.py``
  * Polls ``squeue`` / ``sacct`` for job completion
  * Reads back the JSON result from the job workspace
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from umrs.steps.sandbox import SandboxConfig, StepAuditLog, _WORKER_SCRIPT

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SlurmStepConfig:
    """Configuration for the Slurm step runner."""

    partition: str = os.environ.get("SLURM_PARTITION", "gpu")
    singularity_image: str = os.environ.get(
        "UDM_SINGULARITY_IMAGE",
        "/opt/rivault/images/umrs-step-runner.sif",
    )
    timeout_seconds: int = 3600
    memory_limit: str = "8G"
    cpu_count: int = 4
    gpu_count: int = 0
    work_dir: str = ""  # empty → uses tempdir
    audit_dir: Optional[str] = None
    poll_interval: int = 5  # seconds between squeue checks
    extra_sbatch_flags: List[str] = field(default_factory=list)
    extra_bind_mounts: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class SlurmStepRunner:
    """Execute a single UMRSStep as a Slurm batch job."""

    def __init__(self, config: Optional[SlurmStepConfig] = None) -> None:
        self.cfg = config or SlurmStepConfig()
        self.audit = StepAuditLog(self.cfg.audit_dir)
        self._work_root = Path(
            self.cfg.work_dir or tempfile.gettempdir()
        ) / "umrs_slurm_steps"
        self._work_root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------

    def run_step(
        self,
        *,
        entrypoint: str,
        step_config: Dict[str, Any],
        script_dir: str,
        input_bucket: str,
        input_keys: List[str],
        output_bucket: str,
        s3_params: Dict[str, str],
        run_id: str = "",
        shared_metadata: Optional[Dict[str, Any]] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> "StepResult":
        """Run one step as a Slurm batch job.

        Same signature as :meth:`SubprocessSandbox.run_step` for drop-in use.
        """
        from umrs.steps.context import StepResult

        run_id = run_id or os.urandom(6).hex()
        job_dir = self._work_root / f"step_{run_id}"
        job_dir.mkdir(parents=True, exist_ok=True)

        # -- Write params JSON for the worker script -----------------------
        params = {
            "entrypoint": entrypoint,
            "config": step_config,
            "script_dir": "/user_code",          # container path
            "umrs_src_dir": "/opt/umrs_src",     # container path
            "input_bucket": input_bucket,
            "input_keys": input_keys,
            "output_bucket": output_bucket,
            "s3": s3_params,
            "run_id": run_id,
            "metadata": shared_metadata or {},
            "memory_limit_bytes": 0,  # Slurm manages memory
            "env": env or {},
            "result_path": "/workspace/result.json",
        }
        (job_dir / "params.json").write_text(json.dumps(params))
        (job_dir / "_worker.py").write_text(_WORKER_SCRIPT)

        # -- Generate sbatch script ----------------------------------------
        sbatch_path = job_dir / "job.sh"
        self._generate_sbatch_script(sbatch_path, job_dir, script_dir)

        self.audit.log({
            "event": "slurm_step_submit",
            "run_id": run_id,
            "entrypoint": entrypoint,
            "input_keys_count": len(input_keys),
            "partition": self.cfg.partition,
        })

        # -- Submit and wait -----------------------------------------------
        t0 = time.monotonic()
        slurm_job_id = self._submit(sbatch_path)
        logger.info(
            "Slurm job %s submitted for step '%s' (run=%s)",
            slurm_job_id, entrypoint, run_id,
        )

        try:
            final_status = self._poll(slurm_job_id)
        except TimeoutError:
            self._cancel(slurm_job_id)
            elapsed = time.monotonic() - t0
            self.audit.log({
                "event": "slurm_step_timeout",
                "run_id": run_id,
                "slurm_job_id": slurm_job_id,
                "elapsed_seconds": elapsed,
            })
            raise RuntimeError(
                f"Slurm step '{entrypoint}' timed out after "
                f"{self.cfg.timeout_seconds}s (job {slurm_job_id})"
            )

        elapsed = time.monotonic() - t0

        if final_status != "COMPLETED":
            stderr = self._read_stderr(job_dir)
            self.audit.log({
                "event": "slurm_step_failed",
                "run_id": run_id,
                "slurm_job_id": slurm_job_id,
                "slurm_status": final_status,
                "elapsed_seconds": elapsed,
            })
            raise RuntimeError(
                f"Slurm step '{entrypoint}' failed (status={final_status}, "
                f"job={slurm_job_id}):\n{stderr[-4000:]}"
            )

        # -- Read result ---------------------------------------------------
        result_path = job_dir / "result.json"
        if not result_path.is_file():
            raise RuntimeError(
                f"Slurm step '{entrypoint}' did not produce a result file "
                f"(job {slurm_job_id})"
            )

        with open(result_path) as fh:
            raw = json.load(fh)

        embeddings = None
        if "embeddings_b64" in raw:
            import base64
            import io
            import numpy as np
            buf = io.BytesIO(base64.b64decode(raw["embeddings_b64"]))
            embeddings = np.load(buf, allow_pickle=False)

        self.audit.log({
            "event": "slurm_step_succeeded",
            "run_id": run_id,
            "slurm_job_id": slurm_job_id,
            "output_keys_count": len(raw.get("output_keys", [])),
            "elapsed_seconds": elapsed,
        })

        return StepResult(
            output_keys=raw.get("output_keys", []),
            embeddings=embeddings,
            checksums=raw.get("checksums", []),
            metadata=raw.get("metadata", []),
            extra=raw.get("extra", {}),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_sbatch_script(
        self,
        dest: Path,
        job_dir: Path,
        script_dir: str,
    ) -> None:
        """Write the sbatch script that runs the worker inside Singularity."""
        hours = self.cfg.timeout_seconds // 3600
        minutes = (self.cfg.timeout_seconds % 3600) // 60
        time_limit = f"{hours:02d}:{minutes:02d}:00"

        gpu_line = ""
        if self.cfg.gpu_count > 0:
            gpu_line = f"#SBATCH --gpus={self.cfg.gpu_count}"

        extra_flags = "\n".join(
            f"#SBATCH {f}" for f in self.cfg.extra_sbatch_flags
        )

        # Bind mounts: user code (ro), workspace (rw), UMRS src (ro)
        umrs_src = str(Path(__file__).resolve().parent.parent.parent)
        binds = [
            f"{script_dir}:/user_code:ro",
            f"{job_dir}:/workspace",
            f"{umrs_src}:/opt/umrs_src:ro",
        ] + self.cfg.extra_bind_mounts
        bind_args = ",".join(binds)

        content = f"""#!/bin/bash
#SBATCH --job-name=udm-step-{job_dir.name[:12]}
#SBATCH --partition={self.cfg.partition}
#SBATCH --time={time_limit}
#SBATCH --cpus-per-task={self.cfg.cpu_count}
#SBATCH --mem={self.cfg.memory_limit}
{gpu_line}
{extra_flags}
#SBATCH --output={job_dir}/slurm-%j.out
#SBATCH --error={job_dir}/slurm-%j.err

singularity exec \\
    --containall \\
    --cleanenv \\
    --bind {bind_args} \\
    {self.cfg.singularity_image} \\
    python /workspace/_worker.py /workspace/params.json

EXIT_CODE=$?
echo "EXIT_CODE=$EXIT_CODE" >> /workspace/metrics.txt
exit $EXIT_CODE
"""
        dest.write_text(content)
        os.chmod(dest, 0o755)

    def _submit(self, script_path: Path) -> str:
        """Submit sbatch and return the Slurm job ID."""
        proc = subprocess.run(
            ["sbatch", "--parsable", str(script_path)],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"sbatch failed: {proc.stderr}")
        return proc.stdout.strip().split(";")[0]

    def _poll(self, job_id: str) -> str:
        """Block until the Slurm job finishes. Returns final status string."""
        deadline = time.monotonic() + self.cfg.timeout_seconds + 300
        while time.monotonic() < deadline:
            status = self._squeue_status(job_id)
            if status in (
                "COMPLETED", "FAILED", "CANCELLED",
                "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY",
            ):
                return status
            time.sleep(self.cfg.poll_interval)
        raise TimeoutError(f"Slurm job {job_id} did not finish in time")

    def _squeue_status(self, job_id: str) -> str:
        """Check squeue; fall back to sacct if the job already left the queue."""
        proc = subprocess.run(
            ["squeue", "--job", job_id, "--noheader", "--format", "%T"],
            capture_output=True, text=True,
        )
        status = proc.stdout.strip()
        if status:
            return status
        # Job has left the queue — check sacct
        proc2 = subprocess.run(
            ["sacct", "--job", job_id, "--noheader",
             "--format", "State", "--parsable2"],
            capture_output=True, text=True,
        )
        lines = proc2.stdout.strip().split("\n")
        return lines[0].strip() if lines else "UNKNOWN"

    def _cancel(self, job_id: str) -> None:
        """Best-effort cancel."""
        subprocess.run(["scancel", job_id], capture_output=True)

    @staticmethod
    def _read_stderr(job_dir: Path) -> str:
        """Read the latest Slurm stderr file from the job dir."""
        err_files = sorted(job_dir.glob("slurm-*.err"))
        if err_files:
            return err_files[-1].read_text()[-4000:]
        return "(no stderr file found)"
