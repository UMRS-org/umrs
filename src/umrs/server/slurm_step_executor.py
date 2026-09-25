"""Slurm step executor — runs user workflow steps as Slurm jobs.

This executor follows the same pattern as the existing
:class:`SlurmPodmanBackend` for vLLM/BentoML: it submits a Slurm job
to a GPU compute node, which runs the step inside a podman container.

Flow:
    1. Serialize step entrypoint, config, and StepContext to ``input.pkl``
       using the same nested ``ctx`` payload as PodmanExecutor (includes
       ``parent_results`` / ``input_keys`` for multi-step DAGs)
    2. Generate an sbatch script that:
       - Allocates GPU resources on a compute node
       - Runs ``podman run`` with ``UMRS_STEP_INPUT`` / ``UMRS_STEP_OUTPUT``
         pointing at the bind-mounted pickle files
       - Invokes ``python -m umrs.server.step_worker``
    3. Submit via ``sbatch``
    4. Poll ``squeue`` for completion
    5. Read ``output.pkl`` reply dict and return the :class:`StepResult`

The executor is selected by setting ``UMRS_STEP_EXECUTOR=slurm``.
"""

from __future__ import annotations

import logging
import os
import pickle
import re
import shlex
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any

from umrs.pipeline_sdk.step import StepContext, StepResult

from .step_executor import (
    _PAYLOAD_MAX_BYTES,
    EXECUTOR_GPU_METADATA_KEY,
    EXECUTOR_IMAGE_METADATA_KEY,
    EXECUTOR_PAYLOAD_S3_METADATA_KEY,
    EXECUTOR_SOURCE_DIR_ENV_VAR,
    EXECUTOR_SOURCE_DIR_METADATA_KEY,
    StepExecutionError,
    StepExecutor,
    _serialise_ctx_payload,
    _serialise_s3_client_config,
)
from .step_reply import (
    check_reply_size,
    json_reply_path,
    reply_max_bytes,
    result_from_json,
    result_from_legacy_pickle,
)

logger = logging.getLogger(__name__)


def measure_slurm_accounting(
    job_id: str,
    run: Any = None,
) -> dict[str, Any]:
    """D-36 leg B: sacct-measured cost for a FINISHED slurm job.

    One ``sacct -o Elapsed,AllocTRES%gres`` read → seconds × GPUs.
    Empty dict when sacct is unavailable or answers garbage — unmeasured,
    never approximated (ADR-067's never-approximate rule).
    """
    from umrs.shared.slurm_accounting import (
        measured_gpu_seconds,
        parse_alloc_tres_gpus,
        parse_slurm_elapsed,
    )

    cmd = [
        "sacct",
        "--job",
        str(job_id),
        "--format=Elapsed,AllocTRES%gres",
        "--noheader",
        "--parsable2",
    ]
    runner = run or _run_local
    try:
        result = runner(cmd, timeout=15)
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return {}
    if result.returncode != 0 or not result.stdout.strip():
        return {}
    parts = result.stdout.strip().splitlines()[0].split("|")
    elapsed = parse_slurm_elapsed(parts[0]) if parts else None
    gpus = parse_alloc_tres_gpus(parts[1]) if len(parts) > 1 else 0
    if elapsed is None:
        return {}
    return {
        "job_id": str(job_id),
        "elapsed_seconds": elapsed,
        "gpus": gpus,
        "gpu_seconds": measured_gpu_seconds(elapsed, gpus),
    }


def _run_local(cmd: list[str], *, timeout: int) -> subprocess.CompletedProcess:
    """Local Slurm CLI call (the submit host has sbatch/squeue/sacct)."""
    return subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout, check=False
    )


def stamp_slurm_accounting(result: Any, accounting: dict[str, Any]) -> Any:
    """Stamp measured accounting onto a StepResult's ``extra`` bag.

    Only stamps when the measurement is real (non-empty) — unmeasured
    steps carry nothing, never an invented zero."""
    if not accounting or not isinstance(result.extra, dict):
        return result
    result.extra["slurm"] = dict(accounting)
    return result


#: An OCI image reference: registry/name[:tag][@digest] characters only. The
#: image lands in a generated shell script, so anything else is refused.
_IMAGE_REF_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/:@+-]{0,511}")


def _checked_image(image: str) -> str:
    """*image* when it is a plain image reference; StepExecutionError otherwise."""
    if not _IMAGE_REF_RE.fullmatch(image):
        raise StepExecutionError(
            "setup",
            f"container image {image[:80]!r} is not a plain image reference "
            "(registry/name[:tag][@digest])",
        )
    return image


def _s3_env_args(s3_config: dict[str, Any]) -> list[str]:
    """S3 endpoint/region/credential env passthrough for the worker pod."""
    args: list[str] = []
    endpoint = (
        s3_config.get("endpoint_url")
        or os.environ.get("S3_ENDPOINT_URL")
        or os.environ.get("MINIO_ENDPOINT")
    )
    if endpoint:
        # S3_ENDPOINT_URL is canonical (ADR-055); MINIO_ENDPOINT kept for older workers.
        args.extend(
            [
                "-e",
                shlex.quote(f"S3_ENDPOINT_URL={endpoint}"),
                "-e",
                shlex.quote(f"MINIO_ENDPOINT={endpoint}"),
            ]
        )
    region = s3_config.get("region_name") or os.environ.get("AWS_REGION")
    if region:
        args.extend(["-e", shlex.quote(f"AWS_REGION={region}")])
    for cred_var in (
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
    ):
        val = os.environ.get(cred_var)
        if val:
            args.extend(["-e", shlex.quote(f"{cred_var}={val}")])
    return args


def _podman_run_argv(
    *,
    network: str,
    workdir: str,
    input_rel: str,
    output_rel: str,
    source_dir: str,
    needs_gpu: bool,
    s3_config: dict[str, Any],
    image: str,
) -> list[str]:
    """The step worker's ``podman run`` argv (env-var file transport —
    argv paths alone are ignored by ``step_worker.main()`` unless
    ``UMRS_STEP_*`` is set; same contract as PodmanExecutor)."""
    parts = [
        "podman",
        "run",
        "--rm",
        f"--network={shlex.quote(network)}",
        "-v",
        f"{shlex.quote(workdir)}:/work",
        "-e",
        f"UMRS_STEP_INPUT=/work/{input_rel}",
        "-e",
        f"UMRS_STEP_OUTPUT=/work/{output_rel}",
        # ADR-074 D-6: the reply the server reads is data (JSON); output.pkl
        # stays for images that predate it.
        "-e",
        f"UMRS_STEP_OUTPUT_JSON=/work/{os.path.splitext(output_rel)[0]}.json",
    ]
    if source_dir:
        # User step modules live outside the image (uploaded bundle):
        # mount the tree read-only at the same absolute path and tell
        # the worker to put it on sys.path. Requires the path to be
        # visible from the compute node (shared filesystem).
        parts.extend(
            [
                "-v",
                f"{shlex.quote(source_dir)}:{shlex.quote(source_dir)}:ro",
                "-e",
                f"{EXECUTOR_SOURCE_DIR_ENV_VAR}={shlex.quote(source_dir)}",
            ]
        )
    if needs_gpu:
        # Match PodmanExecutor device CDI form (site-dependent).
        parts.extend(["--device", "nvidia.com/gpu=all"])
    parts.extend(_s3_env_args(s3_config))
    parts.extend(
        [
            shlex.quote(image),
            "python",
            "-m",
            "umrs.server.step_worker",
        ]
    )
    return parts


@dataclass
class SlurmStepExecutor(StepExecutor):
    """Run workflow steps as Slurm jobs on GPU compute nodes.

    This executor is designed for the RCCS deployment pattern where
    UMRS runs on a login node and submits Slurm jobs to GPU compute
    nodes. Each step runs inside a podman container with GPU access.

    Configuration via environment variables:

    * ``UMRS_SLURM_PARTITION`` — Slurm partition (default: ``gpu``)
    * ``UMRS_SLURM_GPUS`` — Number of GPUs to request (default: ``1``);
      ``0`` omits the ``--gres`` directive entirely (required on
      partitions that reject explicit GPU requests, e.g. RCCS
      ``ng-dgx-m*``) while podman GPU passthrough still applies when
      the step wants a GPU
    * ``UMRS_SLURM_CPUS`` — CPUs per task (default: ``8``)
    * ``UMRS_SLURM_MEM`` — Memory per node (default: ``32G``); an
      **empty** value omits the ``--mem`` directive entirely (required
      on partitions that reject explicit memory requests)
    * ``UMRS_SLURM_TIME`` — Time limit (default: ``01:00:00``)
    * ``UMRS_SLURM_ACCOUNT`` — Slurm account (optional)
    * ``UMRS_SLURM_WORKDIR_ROOT`` — Parent directory for the per-step
      working directories (default: system temp). On clusters where
      the submit host's local temp is not visible from compute nodes
      this MUST point at a shared filesystem (e.g. Lustre), otherwise
      the worker never sees ``input.pkl``.
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
    workdir_root: str = field(
        default_factory=lambda: os.environ.get("UMRS_SLURM_WORKDIR_ROOT", "").strip()
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

        # Uploaded workflow source tree (user step modules). The worker
        # runs in a fresh container on a compute node, so the modules
        # must be carried in explicitly — see _generate_sbatch_script.
        source_dir = self._resolve_source_dir(ctx)

        workdir = self._new_workdir()
        logger.info(
            "SlurmStepExecutor: workdir=%s, image=%s, entrypoint=%s, source_dir=%s",
            workdir,
            image,
            entrypoint,
            source_dir or "<none>",
        )

        try:
            # Serialize the step payload — must match step_worker / PodmanExecutor
            # nested ``ctx`` shape (including parent_results) so multi-step
            # DAGs can hand off via input_keys and parent_results.
            input_path = os.path.join(workdir, "input.pkl")
            output_path = os.path.join(workdir, "output.pkl")

            _INTERNAL_KEYS = {
                EXECUTOR_IMAGE_METADATA_KEY,
                EXECUTOR_PAYLOAD_S3_METADATA_KEY,
                EXECUTOR_GPU_METADATA_KEY,
                EXECUTOR_SOURCE_DIR_METADATA_KEY,
            }
            sanitised_metadata = {
                k: v for k, v in ctx.metadata.items() if k not in _INTERNAL_KEYS
            }
            ctx_payload = _serialise_ctx_payload(ctx, metadata=sanitised_metadata)
            payload = {
                "entrypoint": entrypoint,
                "config": dict(config),
                "ctx": ctx_payload,
                "s3_config": self._step_s3_config(ctx),
            }

            payload_blob = pickle.dumps(payload)
            if len(payload_blob) > _PAYLOAD_MAX_BYTES:
                raise StepExecutionError(
                    "setup",
                    f"payload size {len(payload_blob)} bytes exceeds hard cap "
                    f"({_PAYLOAD_MAX_BYTES} bytes); refactor the step to "
                    "pass large data via S3 keys, not parent_results",
                )

            self._write_bytes(input_path, payload_blob)
            source_dir = self._stage_source_dir(source_dir, workdir)

            # Generate the sbatch script
            sbatch_script = self._generate_sbatch_script(
                workdir=workdir,
                image=image,
                input_path=input_path,
                output_path=output_path,
                ctx=ctx,
                source_dir=source_dir,
            )

            # Write the sbatch script
            sbatch_path = os.path.join(workdir, "run_step.sbatch")
            self._write_bytes(sbatch_path, sbatch_script.encode("utf-8"))

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
                self._remove_workdir(workdir)

    # ── I/O hooks (local submit host; RemoteSlurmStepExecutor routes them
    #    over SSH to the cluster login node) ────────────────────────────────

    def _run_cmd(self, cmd: list[str], *, timeout: int) -> subprocess.CompletedProcess:
        return _run_local(cmd, timeout=timeout)

    def _step_s3_config(self, ctx: StepContext) -> dict[str, Any]:
        """S3 connection settings as the compute node must use them."""
        return _serialise_s3_client_config(ctx.s3)

    def _new_workdir(self) -> str:
        """Per-step working directory. When workdir_root is configured it
        must exist and be on a filesystem the compute nodes can see (e.g.
        Lustre) — the whole point of the knob."""
        if not self.workdir_root:
            return tempfile.mkdtemp(prefix="slurm-step-")
        if not os.path.isdir(self.workdir_root):
            raise StepExecutionError(
                "setup",
                f"UMRS_SLURM_WORKDIR_ROOT={self.workdir_root!r} does not "
                "exist; create it (on a shared filesystem visible from "
                "compute nodes) or unset the variable",
            )
        return tempfile.mkdtemp(prefix="slurm-step-", dir=self.workdir_root)

    def _write_bytes(self, path: str, data: bytes) -> None:
        with open(path, "wb") as f:
            f.write(data)

    def _read_bytes(self, path: str, max_bytes: int | None = None) -> bytes | None:
        """File content (at most *max_bytes*), or None when it does not exist."""
        if not os.path.isfile(path):
            return None
        with open(path, "rb") as f:
            return f.read() if max_bytes is None else f.read(max_bytes)

    def _read_reply(self, job_id: str, output_path: str) -> StepResult:
        """The step's reply as data (ADR-074 D-6): ``output.json`` first; an
        image that predates it leaves only ``output.pkl``, read through the
        restricted loader. Never ``pickle.loads`` on step-written bytes."""
        cap = reply_max_bytes()
        json_path = json_reply_path(output_path)
        blob = self._read_bytes(json_path, max_bytes=cap + 1)
        if blob is not None:
            return result_from_json(check_reply_size(blob, json_path, cap))
        blob = self._read_bytes(output_path, max_bytes=cap + 1)
        if blob is None:
            raise StepExecutionError(
                "worker",
                f"Job {job_id} completed but no reply was written: {json_path}",
            )
        logger.warning(
            "SlurmStepExecutor: job %s wrote only a legacy pickle reply "
            "(image predates JSON replies); read restricted",
            job_id,
        )
        return result_from_legacy_pickle(check_reply_size(blob, output_path, cap))

    def _stage_source_dir(self, source_dir: str, workdir: str) -> str:
        """Where the compute node sees the step source tree (same path locally)."""
        return source_dir

    def _remove_workdir(self, workdir: str) -> None:
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
            return _checked_image(override)

        # Fall back to deployment-wide default
        if not self.image:
            raise StepExecutionError(
                "setup",
                "SlurmStepExecutor requires a container image. "
                "Set UMRS_STEP_IMAGE or routing_hints.image.",
            )
        return _checked_image(self.image)

    @staticmethod
    def _resolve_source_dir(ctx: StepContext) -> str:
        """Resolve the uploaded workflow source dir (user step modules).

        Returns ``""`` when the runner did not provide one (legacy
        callers, image-baked steps). When provided it must be an
        absolute existing directory — the worker bind-mounts it
        read-only at the same path and puts it on ``sys.path`` via
        ``UMRS_STEP_SOURCE_DIR``; a wrong value would surface as a
        confusing import error deep in the worker, so fail fast here.
        """
        raw = ctx.metadata.get(EXECUTOR_SOURCE_DIR_METADATA_KEY)
        if raw is None:
            return ""
        if not isinstance(raw, str) or not raw.strip():
            raise StepExecutionError(
                "setup",
                f"workflow source dir hint must be a non-empty string, got "
                f"{type(raw).__name__}",
            )
        candidate = raw.strip()
        if not os.path.isabs(candidate):
            raise StepExecutionError(
                "setup",
                f"workflow source dir must be an absolute path, got {candidate!r}",
            )
        if not os.path.isdir(candidate):
            raise StepExecutionError(
                "setup",
                f"workflow source dir {candidate!r} does not exist on the "
                "submit host; re-upload the workflow or fix the registration",
            )
        return candidate

    def _generate_sbatch_script(
        self,
        *,
        workdir: str,
        image: str,
        input_path: str,
        output_path: str,
        ctx: StepContext,
        source_dir: str = "",
    ) -> str:
        """Generate the sbatch script for the Slurm job."""
        # Determine if GPU is needed (metadata hint → env default).
        gpu_hint = ctx.metadata.get(EXECUTOR_GPU_METADATA_KEY)
        if gpu_hint is None:
            needs_gpu = os.environ.get("UMRS_STEP_GPU", "0").strip() == "1"
        else:
            needs_gpu = bool(gpu_hint)

        time_limit = self._format_time_limit(self.timeout_seconds) or self.time_limit

        # Build sbatch directives
        lines = ["#!/bin/bash"]
        lines.append("#SBATCH --job-name=umrs-step")
        lines.append(f"#SBATCH --partition={self.partition}")
        lines.append(f"#SBATCH --time={time_limit}")
        lines.append(f"#SBATCH --cpus-per-task={self.cpus}")
        # Empty memory means "let the partition default apply" — several
        # partitions (e.g. RCCS ng-dgx-m*) reject explicit --mem requests.
        if self.memory.strip():
            lines.append(f"#SBATCH --mem={self.memory}")
        lines.append(f"#SBATCH --output={workdir}/slurm-%j.out")
        lines.append(f"#SBATCH --error={workdir}/slurm-%j.err")

        if self.account:
            lines.append(f"#SBATCH --account={self.account}")

        # gpus=0 means "no explicit GPU request" — required on partitions
        # that reject --gres (e.g. RCCS ng-dgx-m*). Podman GPU passthrough
        # below is independent: the node may still expose its GPU to the
        # container via CDI when the partition allocates the whole node.
        if needs_gpu and self.gpus > 0:
            lines.append(f"#SBATCH --gres=gpu:{self.gpus}")

        lines.append("")
        lines.append("set -e")
        lines.append('echo "Starting step at $(date)"')
        # Everything interpolated into this script is quoted: a
        # double-quoted echo still expands $(...) (2026-09-25).
        lines.append(f"echo Workdir: {shlex.quote(workdir)}")
        lines.append(f"echo Image: {shlex.quote(image)}")
        lines.append("")

        input_rel = os.path.relpath(input_path, workdir)
        output_rel = os.path.relpath(output_path, workdir)
        network = os.environ.get("UMRS_STEP_PODMAN_NETWORK", "host").strip() or "host"
        s3_config = self._step_s3_config(ctx)

        parts = _podman_run_argv(
            network=network,
            workdir=workdir,
            input_rel=input_rel,
            output_rel=output_rel,
            source_dir=source_dir,
            needs_gpu=needs_gpu,
            s3_config=s3_config,
            image=image,
        )
        podman_cmd = " ".join(parts)

        # Never echo the command: it carries forwarded S3 credentials, and the
        # .out log is readable from the step's own workdir.
        lines.append('echo "Running the step worker container"')
        lines.append(podman_cmd)
        lines.append('echo "Step completed at $(date)"')
        lines.append("")

        return "\n".join(lines)

    @staticmethod
    def _format_time_limit(timeout_seconds: float | None) -> str | None:
        """Convert timeout seconds to Slurm ``HH:MM:SS`` when usable."""
        if timeout_seconds is None:
            return None
        try:
            total = int(timeout_seconds)
        except (TypeError, ValueError):
            return None
        if total <= 0:
            return None
        hours, rem = divmod(total, 3600)
        minutes, seconds = divmod(rem, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def _submit_sbatch(self, sbatch_path: str, workdir: str) -> str:
        """Submit the sbatch script and return the job ID."""
        cmd = ["sbatch", sbatch_path]
        logger.info("Submitting: %s", " ".join(cmd))

        result = self._run_cmd(cmd, timeout=60)

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
                result = self._read_reply(job_id, output_path)

                logger.info(
                    "SlurmStepExecutor: job %s completed successfully",
                    job_id,
                )
                # D-36 leg B: the only GPU-second number that counts is a
                # measured one — stamp the sacct accounting onto the result
                # so it rides the step record into the execution ledger.
                stamp_slurm_accounting(
                    result, measure_slurm_accounting(job_id, run=self._run_cmd)
                )
                return result

            elif status in ("FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY"):
                # Read the Slurm error log for diagnostics
                error_log = os.path.join(workdir, f"slurm-{job_id}.err")
                raw_log = self._read_bytes(error_log) or b""
                error_msg = raw_log.decode("utf-8", "replace")[-1000:]

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
            result = self._run_cmd(cmd, timeout=15)
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
            result = self._run_cmd(cmd, timeout=15)
            if result.returncode == 0 and result.stdout.strip():
                state = result.stdout.strip().split("\n")[0].upper()
                # Remove any extra info like "COMPLETED by 12345"
                state = state.split()[0]
                return state
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        # Can't determine status, assume running
        return "RUNNING"
