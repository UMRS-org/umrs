"""
Port Allocation Manager.

Manages port allocation for Mini-RiVault instances to prevent collisions
on shared nodes (especially relevant for Slurm).

Strategies:
- static_range: Allocate from a static range (11434-11534)
- registry_lock: Use endpoint registry to track allocations
- k8s_service: Let Kubernetes Service allocate ports (K8s only)
"""

from __future__ import annotations

import logging
import os
import random
import socket
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class PortAllocationConfig:
    """Configuration for port allocation."""

    # Port range
    min_port: int = 11434  # Start of allocation range
    max_port: int = 11534  # End of allocation range (100 ports)

    # Strategy
    strategy: str = "static_range"  # static_range, registry_lock, k8s_service

    # Retry settings
    max_retries: int = 10  # Max attempts to find free port

    @classmethod
    def from_env(cls) -> "PortAllocationConfig":
        """Create config from environment variables."""
        return cls(
            min_port=int(os.getenv("RIVAULT_PORT_MIN", "11434")),
            max_port=int(os.getenv("RIVAULT_PORT_MAX", "11534")),
            strategy=os.getenv("RIVAULT_PORT_STRATEGY", "static_range"),
            max_retries=int(os.getenv("RIVAULT_PORT_MAX_RETRIES", "10")),
        )


@dataclass
class PortAllocation:
    """A port allocation."""

    port: int
    job_id: str
    host: str
    allocated_at: float = field(default_factory=lambda: __import__("time").time())

    def __hash__(self):
        return hash((self.port, self.host))


class PortAllocator(ABC):
    """Abstract base class for port allocators."""

    @abstractmethod
    def allocate(self, job_id: str, host: str) -> Optional[int]:
        """
        Allocate a port for a job on a specific host.

        Args:
            job_id: Job identifier.
            host: Target hostname.

        Returns:
            Allocated port number, or None if allocation failed.
        """
        pass

    @abstractmethod
    def release(self, port: int, host: str) -> bool:
        """
        Release a previously allocated port.

        Args:
            port: Port number to release.
            host: Target hostname.

        Returns:
            True if released, False if not found.
        """
        pass

    @abstractmethod
    def is_allocated(self, port: int, host: str) -> bool:
        """Check if a port is allocated on a host."""
        pass


class StaticRangeAllocator(PortAllocator):
    """
    Allocates ports from a static range.

    Simple strategy that tracks allocations in memory.
    Suitable for single-node or when UMRS manages all instances.
    """

    def __init__(self, config: Optional[PortAllocationConfig] = None):
        self.config = config or PortAllocationConfig.from_env()
        self._allocations: Dict[tuple[str, int], PortAllocation] = {}
        self._lock = threading.RLock()

    def allocate(self, job_id: str, host: str) -> Optional[int]:
        """Allocate a port from the static range."""
        with self._lock:
            # Get currently allocated ports for this host
            allocated_on_host = {
                alloc.port
                for (h, _), alloc in self._allocations.items()
                if h == host
            }

            # Try to find a free port
            available = [
                p
                for p in range(self.config.min_port, self.config.max_port + 1)
                if p not in allocated_on_host
            ]

            if not available:
                logger.error(
                    f"No ports available in range "
                    f"{self.config.min_port}-{self.config.max_port} on {host}"
                )
                return None

            # Pick a random port from available
            port = random.choice(available)

            # Verify port is actually free on the host
            if not self._is_port_free(host, port):
                # Try other ports
                for retry_port in available:
                    if self._is_port_free(host, retry_port):
                        port = retry_port
                        break
                else:
                    logger.error(f"All ports in range are in use on {host}")
                    return None

            # Record allocation
            allocation = PortAllocation(port=port, job_id=job_id, host=host)
            self._allocations[(host, port)] = allocation

            logger.info(f"Allocated port {port} for job {job_id} on {host}")
            return port

    def release(self, port: int, host: str) -> bool:
        """Release a port allocation."""
        with self._lock:
            key = (host, port)
            if key in self._allocations:
                del self._allocations[key]
                logger.info(f"Released port {port} on {host}")
                return True
            return False

    def is_allocated(self, port: int, host: str) -> bool:
        """Check if port is allocated."""
        with self._lock:
            return (host, port) in self._allocations

    def _is_port_free(self, host: str, port: int) -> bool:
        """Check if port is actually free on the host."""
        # For local host, we can check directly
        if host in ("localhost", "127.0.0.1", socket.gethostname()):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(1)
                    result = s.connect_ex(("127.0.0.1", port))
                    return result != 0  # 0 means connected (port in use)
            except Exception:
                return True  # Assume free if we can't check

        # For remote hosts, we trust our allocation tracking
        # (actual availability will be verified during readiness)
        return True

    def list_allocations(self, host: Optional[str] = None) -> list[PortAllocation]:
        """List all allocations, optionally filtered by host."""
        with self._lock:
            allocations = list(self._allocations.values())
            if host:
                allocations = [a for a in allocations if a.host == host]
            return allocations

    def clear_all(self) -> int:
        """Clear all allocations."""
        with self._lock:
            count = len(self._allocations)
            self._allocations.clear()
            logger.warning(f"Cleared {count} port allocations")
            return count


class RegistryLockAllocator(PortAllocator):
    """
    Allocates ports using the endpoint registry as the source of truth.

    Uses the endpoint registry to track which ports are in use,
    providing distributed coordination when multiple UMRS instances exist.
    """

    def __init__(self, config: Optional[PortAllocationConfig] = None):
        self.config = config or PortAllocationConfig.from_env()
        self._lock = threading.RLock()
        # Import here to avoid circular imports
        from .endpoint_registry import get_registry

        self._get_registry = get_registry

    def allocate(self, job_id: str, host: str) -> Optional[int]:
        """Allocate a port using registry information."""
        with self._lock:
            registry = self._get_registry()

            # Get ports in use from registry
            endpoints = registry.list_all()
            used_ports = {e.port for e in endpoints if e.host == host}

            # Find available port
            for port in range(self.config.min_port, self.config.max_port + 1):
                if port not in used_ports:
                    logger.info(
                        f"Allocated port {port} for job {job_id} on {host} "
                        "(registry-based)"
                    )
                    return port

            logger.error(f"No ports available on {host}")
            return None

    def release(self, port: int, host: str) -> bool:
        """Release is handled by registry endpoint removal."""
        # Port is released when endpoint is removed from registry
        logger.debug(f"Port {port} release delegated to endpoint registry")
        return True

    def is_allocated(self, port: int, host: str) -> bool:
        """Check registry for port allocation."""
        registry = self._get_registry()
        endpoints = registry.list_all()
        return any(e.port == port and e.host == host for e in endpoints)


class K8sServiceAllocator(PortAllocator):
    """
    Port allocation for Kubernetes.

    In K8s, ports are managed by Services. This allocator doesn't actually
    allocate ports but tracks the intended port for documentation.
    """

    def __init__(self, config: Optional[PortAllocationConfig] = None):
        self.config = config or PortAllocationConfig.from_env()
        self._default_port = self.config.min_port  # Usually 11434

    def allocate(self, job_id: str, host: str) -> Optional[int]:
        """Return default port (K8s Service handles actual allocation)."""
        logger.info(
            f"K8s port allocation: using default port {self._default_port} "
            f"for {job_id} (Service handles actual routing)"
        )
        return self._default_port

    def release(self, port: int, host: str) -> bool:
        """No-op for K8s (Service handles cleanup)."""
        return True

    def is_allocated(self, port: int, host: str) -> bool:
        """Always return False (K8s Services manage this)."""
        return False


# =============================================================================
# Port Allocation Manager
# =============================================================================


class PortManager:
    """
    Port allocation manager.

    Provides a unified interface for port allocation across different strategies.
    """

    def __init__(self, config: Optional[PortAllocationConfig] = None):
        self.config = config or PortAllocationConfig.from_env()
        self._allocator = self._create_allocator()

    def _create_allocator(self) -> PortAllocator:
        """Create allocator based on strategy."""
        strategy = self.config.strategy.lower()

        if strategy == "static_range":
            return StaticRangeAllocator(self.config)
        elif strategy == "registry_lock":
            return RegistryLockAllocator(self.config)
        elif strategy == "k8s_service":
            return K8sServiceAllocator(self.config)
        else:
            logger.warning(
                f"Unknown port strategy '{strategy}', falling back to static_range"
            )
            return StaticRangeAllocator(self.config)

    def allocate(self, job_id: str, host: str) -> Optional[int]:
        """Allocate a port."""
        return self._allocator.allocate(job_id, host)

    def release(self, port: int, host: str) -> bool:
        """Release a port."""
        return self._allocator.release(port, host)

    def is_allocated(self, port: int, host: str) -> bool:
        """Check if port is allocated."""
        return self._allocator.is_allocated(port, host)


# Global singleton
_port_manager: Optional[PortManager] = None
_manager_lock = threading.Lock()


def get_port_manager() -> PortManager:
    """Get the global port manager singleton."""
    global _port_manager
    if _port_manager is None:
        with _manager_lock:
            if _port_manager is None:
                _port_manager = PortManager()
    return _port_manager
