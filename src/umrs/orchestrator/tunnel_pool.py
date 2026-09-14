"""
Tunnel Pool — dynamic SSH forward tunnels for LLM endpoint access.

When UMRS runs on a front-end server that cannot directly reach compute nodes,
each active Mini-RiVault endpoint needs an SSH forward tunnel from the
front-end through the login node to the compute node's LiteLLM port.

This module manages the lifecycle of those tunnels:
- Creates a forward tunnel when a new endpoint becomes READY
- Removes the tunnel when the endpoint is torn down
- Provides local port mapping for Gateway to proxy through

Architecture::

    Gateway (front-end) ---> localhost:{local_port}
                                   |
                              SSH -L tunnel
                                   |
    login-node -----> compute-node:{litellm_port}

Usage::

    from umrs.orchestrator.tunnel_pool import TunnelPool

    pool = TunnelPool(ssh_target="rccs-cloud")
    local_port = await pool.open_tunnel("ng-dgx-m0", remote_port=8001)
    # Gateway can now proxy to http://127.0.0.1:{local_port}/v1/...
    await pool.close_tunnel("ng-dgx-m0", remote_port=8001)
    await pool.shutdown()
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Local port range for dynamically allocated forward tunnels.
_DEFAULT_LOCAL_PORT_MIN = 12000
_DEFAULT_LOCAL_PORT_MAX = 12999


@dataclass
class TunnelSpec:
    """Specification for a single forward tunnel."""

    remote_host: str  # Compute node hostname
    remote_port: int  # Port on compute node (e.g., 8001)
    local_port: int  # Allocated local port on front-end
    process: asyncio.subprocess.Process | None = field(
        default=None, repr=False, compare=False
    )


@dataclass
class TunnelPoolConfig:
    """Configuration for the tunnel pool."""

    ssh_target: str = ""  # SSH alias or user@host for login node
    ssh_user: str = ""
    ssh_port: int = 22
    local_port_min: int = _DEFAULT_LOCAL_PORT_MIN
    local_port_max: int = _DEFAULT_LOCAL_PORT_MAX
    control_path: str = ""  # Reuse existing ControlMaster socket
    bind_address: str = "127.0.0.1"  # Local bind address


class TunnelPool:
    """
    Manages dynamic SSH forward tunnels for compute node endpoint access.

    Each tunnel is an SSH -L process that forwards a local port to a
    compute node's LiteLLM port through the login node.
    """

    def __init__(self, config: TunnelPoolConfig) -> None:
        if not config.ssh_target:
            raise ValueError("ssh_target is required for TunnelPool")
        self._config = config
        self._tunnels: dict[str, TunnelSpec] = {}  # key: "host:port"
        self._next_port = config.local_port_min
        self._lock = asyncio.Lock()

    def _tunnel_key(self, remote_host: str, remote_port: int) -> str:
        return f"{remote_host}:{remote_port}"

    def _ssh_target(self) -> str:
        if self._config.ssh_user:
            return f"{self._config.ssh_user}@{self._config.ssh_target}"
        return self._config.ssh_target

    def _allocate_local_port(self) -> int:
        """Allocate the next available local port."""
        port = self._next_port
        if port > self._config.local_port_max:
            raise RuntimeError(
                f"Tunnel pool exhausted: all ports in range "
                f"{self._config.local_port_min}-{self._config.local_port_max} in use"
            )
        self._next_port += 1
        return port

    async def open_tunnel(
        self,
        remote_host: str,
        remote_port: int,
        *,
        local_port: int | None = None,
    ) -> int:
        """Open a forward tunnel to a compute node endpoint.

        Args:
            remote_host: Compute node hostname.
            remote_port: Port on compute node (e.g., 8001 for LiteLLM).
            local_port: Optional specific local port. Auto-allocated if None.

        Returns:
            The local port that forwards to the remote endpoint.
        """
        key = self._tunnel_key(remote_host, remote_port)

        async with self._lock:
            # Return existing tunnel if already open AND its local listener
            # still accepts connections. A ControlMaster restart silently
            # drops master-owned forwards (``process is None``), so trusting
            # ``_spec_alive`` alone would treat them as alive forever.
            if key in self._tunnels:
                existing = self._tunnels[key]
                if self._spec_alive(existing) and await self._listener_up(
                    existing.local_port
                ):
                    return existing.local_port
                # Tunnel dead (slave process exited or listener gone):
                # best-effort teardown, then re-establish below.
                logger.warning(
                    "Tunnel localhost:%d -> %s:%d is dead; re-establishing",
                    existing.local_port,
                    remote_host,
                    remote_port,
                )
                await self._teardown_spec(existing)
                del self._tunnels[key]

            port = local_port or self._allocate_local_port()
            bind = self._config.bind_address

            # Build SSH -L command.
            args = ["ssh", "-N", "-o", "ExitOnForwardFailure=yes"]

            # Reuse ControlMaster socket if available.
            if self._config.control_path:
                args.extend(["-o", f"ControlPath={self._config.control_path}"])
                args.extend(["-o", "ControlMaster=no"])

            if self._config.ssh_port != 22:
                args.extend(["-p", str(self._config.ssh_port)])

            args.extend(
                [
                    "-L",
                    f"{bind}:{port}:{remote_host}:{remote_port}",
                    self._ssh_target(),
                ]
            )

            logger.info(
                "Opening tunnel %s:%d -> %s:%d (via %s)",
                bind,
                port,
                remote_host,
                remote_port,
                self._config.ssh_target,
            )

            proc = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )

            # Wait briefly to ensure tunnel established.
            await asyncio.sleep(0.5)
            rc = proc.returncode
            if rc is not None and rc != 0:
                stderr = await proc.stderr.read() if proc.stderr else b""
                raise ConnectionError(
                    f"Tunnel to {remote_host}:{remote_port} failed: "
                    f"{stderr.decode().strip()}"
                )

            # When multiplexing over an existing ControlMaster, the slave
            # ``ssh -N -L`` process delegates the forward to the master and
            # exits cleanly (returncode 0); the master process owns the
            # listening socket. Track such tunnels without a live slave
            # process. Without multiplexing the slave stays alive
            # (returncode None) and owns the listener itself.
            tracked_proc = proc if rc is None else None

            spec = TunnelSpec(
                remote_host=remote_host,
                remote_port=remote_port,
                local_port=port,
                process=tracked_proc,
            )
            self._tunnels[key] = spec

            logger.info(
                "Tunnel established: localhost:%d -> %s:%d",
                port,
                remote_host,
                remote_port,
            )
            return port

    async def close_tunnel(self, remote_host: str, remote_port: int) -> None:
        """Close a specific forward tunnel."""
        key = self._tunnel_key(remote_host, remote_port)

        async with self._lock:
            spec = self._tunnels.pop(key, None)
            if spec is None:
                return
            await self._teardown_spec(spec)
            logger.info(
                "Tunnel closed: localhost:%d -> %s:%d",
                spec.local_port,
                remote_host,
                remote_port,
            )

    @staticmethod
    def _spec_alive(spec: TunnelSpec) -> bool:
        """Return True if the tunnel is considered active.

        Multiplexed tunnels (``process is None``) are owned by the
        ControlMaster connection and are assumed alive until explicitly
        closed. Standalone tunnels are alive while their slave process runs.
        """
        if spec.process is None:
            return True
        return spec.process.returncode is None

    async def _listener_up(self, local_port: int, *, timeout: float = 1.0) -> bool:
        """Return True if something accepts TCP connections on the local port.

        Detects forwards silently dropped by a ControlMaster restart: the
        pool still tracks the spec but nothing listens on the local port.
        """
        try:
            _reader, writer = await asyncio.wait_for(
                asyncio.open_connection(self._config.bind_address, local_port),
                timeout=timeout,
            )
        except (OSError, asyncio.TimeoutError):
            return False
        writer.close()
        try:
            await writer.wait_closed()
        except OSError:
            pass
        return True

    def get_local_port(self, remote_host: str, remote_port: int) -> int | None:
        """Get the local port for an active tunnel. Returns None if not open."""
        key = self._tunnel_key(remote_host, remote_port)
        spec = self._tunnels.get(key)
        if spec and self._spec_alive(spec):
            return spec.local_port
        return None

    def get_local_url(self, remote_host: str, remote_port: int) -> str | None:
        """Get the local URL (http://127.0.0.1:{port}) for an active tunnel."""
        port = self.get_local_port(remote_host, remote_port)
        if port is None:
            return None
        return f"http://{self._config.bind_address}:{port}"

    @property
    def active_tunnels(self) -> list[TunnelSpec]:
        """List all active tunnels."""
        return [spec for spec in self._tunnels.values() if self._spec_alive(spec)]

    async def cleanup_dead(self) -> int:
        """Remove tunnels whose SSH processes have died."""
        dead_keys = []
        for key, spec in self._tunnels.items():
            if spec.process and spec.process.returncode is not None:
                dead_keys.append(key)

        for key in dead_keys:
            del self._tunnels[key]

        if dead_keys:
            logger.warning("Cleaned up %d dead tunnels", len(dead_keys))
        return len(dead_keys)

    async def _teardown_spec(self, spec: TunnelSpec) -> None:
        """Tear down a single tunnel, handling both standalone and
        ControlMaster-owned forwards."""
        if spec.process is not None:
            if spec.process.returncode is None:
                spec.process.terminate()
                try:
                    await asyncio.wait_for(spec.process.wait(), timeout=5)
                except asyncio.TimeoutError:
                    spec.process.kill()
            return
        # Multiplexed forward owned by the ControlMaster; cancel it.
        await self._cancel_master_forward(spec)

    async def _cancel_master_forward(self, spec: TunnelSpec) -> None:
        """Cancel a forward that is owned by the shared ControlMaster."""
        if not self._config.control_path:
            return
        args = [
            "ssh",
            "-O",
            "cancel",
            "-o",
            f"ControlPath={self._config.control_path}",
        ]
        if self._config.ssh_port != 22:
            args.extend(["-p", str(self._config.ssh_port)])
        args.extend(
            [
                "-L",
                f"{self._config.bind_address}:{spec.local_port}:"
                f"{spec.remote_host}:{spec.remote_port}",
                self._ssh_target(),
            ]
        )
        try:
            proc = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(proc.wait(), timeout=5)
        except (asyncio.TimeoutError, OSError):
            logger.warning(
                "Failed to cancel multiplexed forward localhost:%d -> %s:%d",
                spec.local_port,
                spec.remote_host,
                spec.remote_port,
            )

    async def shutdown(self) -> None:
        """Close all tunnels and clean up."""
        async with self._lock:
            for key, spec in list(self._tunnels.items()):
                await self._teardown_spec(spec)
            self._tunnels.clear()
            logger.info("Tunnel pool shut down, all tunnels closed")
