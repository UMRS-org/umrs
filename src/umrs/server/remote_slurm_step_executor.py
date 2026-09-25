"""Remote Slurm step executor — workflow steps from a front-end host.

:class:`SlurmStepExecutor` assumes the UMRS server runs where ``sbatch``
lives. On a front-end deployment (umrs-front, ``scheduler.kind:
slurm-remote``) there is no Slurm CLI and no shared filesystem, so
``UMRS_STEP_EXECUTOR=slurm`` could never run a step (production
enablement 2026-09-25). This subclass keeps the whole step flow and
routes its I/O hooks over the site's SSH ControlMaster to the login node:

- the per-step workdir is created under ``UMRS_SLURM_WORKDIR_ROOT`` on the
  CLUSTER shared filesystem (required — compute nodes must see it);
- ``input.pkl`` / the sbatch script are streamed with ``cat >``;
- sbatch / squeue / sacct run on the login node;
- ``output.pkl`` and the Slurm error log are read back with ``cat``;
- an uploaded workflow source tree is tar-streamed into ``<workdir>/src``;
- ``UMRS_STEP_S3_ENDPOINT`` rewrites the step's object-store endpoint to the
  one compute nodes can reach (the login-node tunnel).

SSH identity/target come from ``site.yaml`` ``extra`` (the same block the
``slurm-remote`` scheduler reads), so there is one SSH path, not two.
Selected with ``UMRS_STEP_EXECUTOR=slurm-remote``.
"""

from __future__ import annotations

import logging
import os
import shlex
import subprocess
from dataclasses import dataclass, field
from typing import Any

from umrs.orchestrator.ssh_control import SSHControlConfig, SSHControlManager
from umrs.pipeline_sdk.step import StepContext

from .slurm_step_executor import SlurmStepExecutor
from .step_executor import StepExecutionError

logger = logging.getLogger(__name__)

_WORKDIR_PREFIX = "slurm-step-"
_TRANSFER_TIMEOUT = 300


def _site_ssh_config() -> SSHControlConfig:
    """SSH config from ``site.yaml`` ``extra`` (the slurm-remote block)."""
    from umrs.orchestrator.site_config import load_site_config

    return SSHControlConfig.from_site_config(load_site_config().extra or {})


@dataclass
class RemoteSlurmStepExecutor(SlurmStepExecutor):
    """Run workflow steps as Slurm jobs submitted over SSH."""

    ssh_config: SSHControlConfig = field(default_factory=lambda: _site_ssh_config())
    #: Object-store endpoint as seen FROM the compute node (the front's store
    #: is reached through the login-node reverse tunnel, not the front-local
    #: URL the server's client uses). Empty keeps the server's endpoint.
    s3_endpoint: str = field(
        default_factory=lambda: os.environ.get("UMRS_STEP_S3_ENDPOINT", "").strip()
    )
    _ssh: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.ssh_config.ssh_target:
            raise RuntimeError(
                "UMRS_STEP_EXECUTOR=slurm-remote needs site.yaml extra.ssh_target "
                "(the login node the slurm-remote scheduler already uses)"
            )
        self._ssh = SSHControlManager(self.ssh_config)

    # ── transport ───────────────────────────────────────────────────────────

    def _ssh_run(
        self,
        command: str,
        *,
        timeout: int,
        stdin: bytes | None = None,
    ) -> subprocess.CompletedProcess:
        """Run a remote shell command; bytes in, bytes out."""
        return subprocess.run(
            self._ssh.command_argv(command),
            input=stdin,
            stdin=None if stdin is not None else subprocess.DEVNULL,
            capture_output=True,
            timeout=timeout,
            check=False,
        )

    # ── SlurmStepExecutor I/O hooks ─────────────────────────────────────────

    def _step_s3_config(self, ctx: StepContext) -> dict[str, Any]:
        config = super()._step_s3_config(ctx)
        if self.s3_endpoint:
            config["endpoint_url"] = self.s3_endpoint
        return config

    def _run_cmd(self, cmd: list[str], *, timeout: int) -> subprocess.CompletedProcess:
        proc = self._ssh_run(shlex.join(cmd), timeout=timeout)
        return subprocess.CompletedProcess(
            proc.args,
            proc.returncode,
            proc.stdout.decode("utf-8", "replace"),
            proc.stderr.decode("utf-8", "replace"),
        )

    def _new_workdir(self) -> str:
        root = self.workdir_root.rstrip("/")
        if not root.startswith("/"):
            raise StepExecutionError(
                "setup",
                "UMRS_STEP_EXECUTOR=slurm-remote requires UMRS_SLURM_WORKDIR_ROOT: "
                "an absolute directory on the cluster's shared filesystem "
                "(step_executor.workdir_root in site.yaml)",
            )
        template = shlex.quote(f"{root}/{_WORKDIR_PREFIX}XXXXXXXX")
        proc = self._ssh_run(
            f"mkdir -p {shlex.quote(root)} && mktemp -d {template}", timeout=30
        )
        workdir = proc.stdout.decode("utf-8", "replace").strip()
        if proc.returncode != 0 or not workdir.startswith(f"{root}/{_WORKDIR_PREFIX}"):
            raise StepExecutionError(
                "setup",
                f"remote workdir under {root!r} could not be created "
                f"(rc={proc.returncode}): {proc.stderr.decode('utf-8', 'replace').strip()}",
            )
        return workdir

    def _write_bytes(self, path: str, data: bytes) -> None:
        proc = self._ssh_run(
            f"cat > {shlex.quote(path)}", timeout=_TRANSFER_TIMEOUT, stdin=data
        )
        if proc.returncode != 0:
            raise StepExecutionError(
                "setup",
                f"writing {path} on the login node failed (rc={proc.returncode}): "
                f"{proc.stderr.decode('utf-8', 'replace').strip()}",
            )

    def _read_bytes(self, path: str, max_bytes: int | None = None) -> bytes | None:
        quoted = shlex.quote(path)
        reader = (
            f"cat {quoted}"
            if max_bytes is None
            else f"head -c {int(max_bytes)} {quoted}"
        )
        proc = self._ssh_run(f"test -f {quoted} && {reader}", timeout=_TRANSFER_TIMEOUT)
        return proc.stdout if proc.returncode == 0 else None

    def _stage_source_dir(self, source_dir: str, workdir: str) -> str:
        """Tar-stream the uploaded step modules next to the payload."""
        if not source_dir:
            return ""
        tar = subprocess.run(
            ["tar", "-czf", "-", "-C", source_dir, "."],
            capture_output=True,
            timeout=_TRANSFER_TIMEOUT,
            check=False,
        )
        if tar.returncode != 0:
            raise StepExecutionError(
                "setup", f"packing workflow source {source_dir!r} failed"
            )
        dest = f"{workdir}/src"
        quoted = shlex.quote(dest)
        proc = self._ssh_run(
            f"mkdir -p {quoted} && tar -xzf - -C {quoted}",
            timeout=_TRANSFER_TIMEOUT,
            stdin=tar.stdout,
        )
        if proc.returncode != 0:
            raise StepExecutionError(
                "setup",
                f"unpacking workflow source on the login node failed: "
                f"{proc.stderr.decode('utf-8', 'replace').strip()}",
            )
        return dest

    def _remove_workdir(self, workdir: str) -> None:
        # Only ever a directory this executor created (mktemp under the root).
        if not workdir.startswith(f"{self.workdir_root.rstrip('/')}/{_WORKDIR_PREFIX}"):
            logger.warning("refusing to remove unexpected remote path %r", workdir)
            return
        try:
            self._ssh_run(f"rm -rf -- {shlex.quote(workdir)}", timeout=60)
        except subprocess.TimeoutExpired:
            logger.warning("remote workdir cleanup timed out: %s", workdir)
