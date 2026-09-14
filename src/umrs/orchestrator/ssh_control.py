"""
SSH ControlMaster Manager — persistent multiplexed SSH for remote Slurm operations.

Maintains a ControlMaster connection to the login node and provides
``run_remote()`` for executing commands without per-call SSH handshake overhead.

Used by the ``slurm-remote`` scheduler kind when UMRS runs on a front-end
server that does not have direct Slurm CLI access.

Usage::

    from umrs.orchestrator.ssh_control import SSHControlManager

    mgr = SSHControlManager(ssh_target="rccs-cloud")
    await mgr.ensure_connected()
    stdout, stderr, rc = await mgr.run_remote("squeue -j 12345 -h -o %T")
    await mgr.close()
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Maximum time to wait for SSH connection establishment.
_DEFAULT_CONNECT_TIMEOUT = 15
# Maximum time to wait for a single remote command.
_DEFAULT_COMMAND_TIMEOUT = 30
# Interval between ControlMaster health checks.
_DEFAULT_HEALTH_CHECK_INTERVAL = 60
# Maximum reconnection attempts before giving up.
_MAX_RECONNECT_ATTEMPTS = 5
# Base backoff in seconds for reconnection.
_RECONNECT_BACKOFF_BASE = 2.0


@dataclass
class SSHControlConfig:
    """Configuration for SSH ControlMaster connections."""

    ssh_target: str = ""  # SSH alias or user@host
    ssh_user: str = ""  # Optional explicit user
    ssh_port: int = 22
    connect_timeout: int = _DEFAULT_CONNECT_TIMEOUT
    command_timeout: int = _DEFAULT_COMMAND_TIMEOUT
    control_path: str = ""  # Auto-generated if empty
    server_alive_interval: int = 30
    server_alive_count_max: int = 3
    # Reverse tunnels to establish on connection (list of "bind:host:port")
    reverse_tunnels: list[str] = field(default_factory=list)
    # ProxyJump for multi-hop SSH (e.g., "masaru.nagaso@rccs-cloud")
    proxy_jump: str = ""
    # Explicit identity file path (e.g., "/home/umrs/.ssh/ai4s_ed25519")
    identity_file: str = ""
    # Extra environment variables to inject before every remote command.
    # Useful for clusters that require e.g. SLURM_CONF_SERVER in non-login shells.
    remote_env: dict = field(default_factory=dict)

    @classmethod
    def from_site_config(cls, scheduler_extra: dict) -> SSHControlConfig:
        """Build from site.yaml scheduler.extra section."""
        return cls(
            ssh_target=scheduler_extra.get("ssh_target", ""),
            ssh_user=scheduler_extra.get("ssh_user", ""),
            ssh_port=int(scheduler_extra.get("ssh_port", 22)),
            connect_timeout=int(
                scheduler_extra.get("ssh_connect_timeout", _DEFAULT_CONNECT_TIMEOUT)
            ),
            command_timeout=int(
                scheduler_extra.get("ssh_command_timeout", _DEFAULT_COMMAND_TIMEOUT)
            ),
            control_path=scheduler_extra.get("ssh_control_path", ""),
            server_alive_interval=int(
                scheduler_extra.get("ssh_server_alive_interval", 30)
            ),
            server_alive_count_max=int(
                scheduler_extra.get("ssh_server_alive_count_max", 3)
            ),
            reverse_tunnels=scheduler_extra.get("ssh_reverse_tunnels", []),
            proxy_jump=scheduler_extra.get("ssh_proxy_jump", ""),
            identity_file=scheduler_extra.get("ssh_identity_file", ""),
            remote_env=scheduler_extra.get("ssh_remote_env", {}),
        )


@dataclass
class CommandResult:
    """Result of a remote SSH command."""

    stdout: str
    stderr: str
    returncode: int

    @property
    def ok(self) -> bool:
        return self.returncode == 0


class SSHControlManager:
    """Manages a persistent SSH ControlMaster connection for remote commands.

    Provides:
    - Persistent multiplexed SSH (avoids per-command handshake)
    - Auto-reconnect with exponential backoff
    - Health checking via ``ssh -O check``
    - Reverse tunnel establishment for MinIO access
    - Graceful shutdown
    """

    def __init__(self, config: SSHControlConfig) -> None:
        if not config.ssh_target:
            raise ValueError("ssh_target is required for SSHControlManager")
        self._config = config
        self._connected = False
        self._lock = asyncio.Lock()
        self._reconnect_attempts = 0
        self._health_task: asyncio.Task | None = None

        # Determine control socket path.
        if config.control_path:
            self._control_path = config.control_path
        else:
            # Use a stable path based on target to survive restarts.
            run_dir = os.environ.get("XDG_RUNTIME_DIR", tempfile.gettempdir())
            self._control_path = os.path.join(
                run_dir, f"umrs-ssh-{config.ssh_target}-%p"
            )

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def control_path(self) -> str:
        return self._control_path

    def _ssh_target(self) -> str:
        """Build the SSH target string."""
        if self._config.ssh_user:
            return f"{self._config.ssh_user}@{self._config.ssh_target}"
        return self._config.ssh_target

    def _base_ssh_args(self) -> list[str]:
        """Common SSH arguments for all commands."""
        args = [
            "ssh",
            "-o",
            f"ControlPath={self._control_path}",
            "-o",
            "BatchMode=yes",
            "-o",
            f"ConnectTimeout={self._config.connect_timeout}",
            "-o",
            f"ServerAliveInterval={self._config.server_alive_interval}",
            "-o",
            f"ServerAliveCountMax={self._config.server_alive_count_max}",
        ]
        if self._config.ssh_port != 22:
            args.extend(["-p", str(self._config.ssh_port)])
        if self._config.proxy_jump:
            args.extend(["-J", self._config.proxy_jump])
        if self._config.identity_file:
            args.extend(["-i", self._config.identity_file])
        return args

    async def ensure_connected(self) -> None:
        """Ensure the ControlMaster is established. Idempotent."""
        async with self._lock:
            if self._connected and await self._check_health_unlocked():
                return
            await self._connect_unlocked()

    async def _connect_unlocked(self) -> None:
        """Establish ControlMaster connection (caller holds lock)."""
        target = self._ssh_target()
        args = self._base_ssh_args() + [
            "-o",
            "ControlMaster=auto",
            "-o",
            "ControlPersist=yes",
            "-N",  # No remote command
        ]

        # Add reverse tunnels.
        for tunnel_spec in self._config.reverse_tunnels:
            args.extend(["-R", tunnel_spec])

        args.append(target)

        logger.info(
            "Establishing SSH ControlMaster to %s (socket: %s)",
            target,
            self._control_path,
        )

        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            _, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=self._config.connect_timeout + 5,
            )
        except asyncio.TimeoutError:
            proc.kill()
            raise ConnectionError(
                f"SSH ControlMaster connection to {target} timed out "
                f"after {self._config.connect_timeout}s"
            )

        if proc.returncode != 0:
            err_msg = stderr.decode().strip() if stderr else "unknown error"
            raise ConnectionError(
                f"SSH ControlMaster to {target} failed (rc={proc.returncode}): {err_msg}"
            )

        self._connected = True
        self._reconnect_attempts = 0
        logger.info("SSH ControlMaster established to %s", target)

        # Start background health checker.
        if self._health_task is None or self._health_task.done():
            self._health_task = asyncio.create_task(self._health_loop())

    async def _check_health_unlocked(self) -> bool:
        """Check if ControlMaster socket is alive (caller may hold lock)."""
        target = self._ssh_target()
        args = self._base_ssh_args() + [
            "-O",
            "check",
            target,
        ]

        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()
        alive = proc.returncode == 0
        if not alive:
            self._connected = False
        return alive

    async def check_health(self) -> bool:
        """Public health check."""
        async with self._lock:
            return await self._check_health_unlocked()

    async def _health_loop(self) -> None:
        """Periodic health check with auto-reconnect."""
        while True:
            await asyncio.sleep(_DEFAULT_HEALTH_CHECK_INTERVAL)
            try:
                async with self._lock:
                    alive = await self._check_health_unlocked()
                    if not alive:
                        logger.warning(
                            "SSH ControlMaster to %s lost, attempting reconnect",
                            self._config.ssh_target,
                        )
                        await self._reconnect_unlocked()
            except Exception as exc:
                logger.error("SSH health check error: %s", exc)

    async def _reconnect_unlocked(self) -> None:
        """Reconnect with exponential backoff (caller holds lock)."""
        if self._reconnect_attempts >= _MAX_RECONNECT_ATTEMPTS:
            logger.error(
                "SSH reconnect exhausted (%d attempts) to %s",
                _MAX_RECONNECT_ATTEMPTS,
                self._config.ssh_target,
            )
            return

        self._reconnect_attempts += 1
        backoff = _RECONNECT_BACKOFF_BASE**self._reconnect_attempts
        logger.info(
            "SSH reconnect attempt %d/%d (backoff %.1fs)",
            self._reconnect_attempts,
            _MAX_RECONNECT_ATTEMPTS,
            backoff,
        )
        await asyncio.sleep(backoff)

        try:
            await self._connect_unlocked()
        except ConnectionError as exc:
            logger.warning("SSH reconnect failed: %s", exc)

    async def run_remote(
        self,
        command: str,
        *,
        timeout: int | None = None,
        stdin_data: bytes | None = None,
    ) -> CommandResult:
        """Execute a command on the remote host via the ControlMaster.

        Args:
            command: Shell command to execute remotely.
            timeout: Override command timeout (seconds).
            stdin_data: Optional data to pipe to stdin.

        Returns:
            CommandResult with stdout, stderr, and return code.

        Raises:
            ConnectionError: If SSH connection is down and reconnect fails.
            asyncio.TimeoutError: If command exceeds timeout.
        """
        await self.ensure_connected()

        target = self._ssh_target()
        # Prefix command with any required env vars (e.g. SLURM_CONF_SERVER).
        if self._config.remote_env:
            env_prefix = " ".join(
                f"{k}={v}" for k, v in self._config.remote_env.items()
            )
            command = f"{env_prefix} {command}"
        args = self._base_ssh_args() + [
            "-o",
            "ControlMaster=no",
            target,
            command,
        ]

        cmd_timeout = timeout or self._config.command_timeout

        proc = await asyncio.create_subprocess_exec(
            *args,
            stdin=asyncio.subprocess.PIPE if stdin_data else asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(input=stdin_data),
                timeout=cmd_timeout,
            )
        except asyncio.TimeoutError:
            proc.kill()
            raise asyncio.TimeoutError(
                f"Remote command timed out after {cmd_timeout}s: {command[:80]}"
            )

        return CommandResult(
            stdout=stdout.decode() if stdout else "",
            stderr=stderr.decode() if stderr else "",
            returncode=proc.returncode or 0,
        )

    async def run_remote_with_stdin(
        self,
        command: str,
        stdin_data: bytes,
        *,
        timeout: int | None = None,
    ) -> CommandResult:
        """Execute a remote command with stdin piped (e.g., sbatch script)."""
        return await self.run_remote(command, timeout=timeout, stdin_data=stdin_data)

    async def read_remote_file(self, path: str) -> str | None:
        """Read a file on the remote host. Returns None if file doesn't exist."""
        result = await self.run_remote(f"cat {path}")
        if result.returncode != 0:
            return None
        return result.stdout

    async def file_exists(self, path: str) -> bool:
        """Check if a file exists on the remote host."""
        result = await self.run_remote(f"test -f {path} && echo yes || echo no")
        return result.ok and result.stdout.strip() == "yes"

    async def close(self) -> None:
        """Gracefully shut down the ControlMaster connection."""
        if self._health_task and not self._health_task.done():
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass

        if self._connected:
            target = self._ssh_target()
            args = self._base_ssh_args() + ["-O", "exit", target]
            proc = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.communicate()
            self._connected = False
            logger.info("SSH ControlMaster closed for %s", self._config.ssh_target)

        # Clean up socket file.
        socket_path = Path(self._control_path)
        if socket_path.exists():
            try:
                socket_path.unlink()
            except OSError:
                pass
