"""
Endpoint Registry for Mini-RiVault instances.

This module provides a registry of active inference endpoints,
keyed by (project_id, model). Gateway uses this to discover which endpoint
to route requests to.

UMRS readiness polling (U-107) is the authoritative source of endpoint status.
Backends only provide discovery coordinates (host/port).

Storage backends (U-109):
- InMemoryBackend: Default, for single-instance deployment
- PostgresBackend: For HA deployment with persistence
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .registry_backend import RegistryBackend

logger = logging.getLogger(__name__)


class EndpointStatus(str, Enum):
    """Status of a registered endpoint."""

    PENDING = "pending"  # Request received, not yet submitted
    PROVISIONING = "provisioning"  # Job submitted, waiting for resources
    STARTING = "starting"  # Resources allocated, container starting
    READY = "ready"  # Endpoint is responding to /v1/models
    DEGRADED = "degraded"  # Endpoint has issues but may still work
    EXPIRED = "expired"  # Past expires_at, pending cleanup
    FAILED = "failed"  # Job failed, pending cleanup
    TERMINATED = "terminated"  # Explicitly stopped


@dataclass
class RegisteredEndpoint:
    """A registered inference endpoint."""

    # Identity
    project_id: str
    model: str
    job_id: str

    # Location (discovery coordinates from backend)
    host: str
    port: int

    # Status
    status: EndpointStatus

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    ready_at: Optional[datetime] = None
    last_used_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None

    # Backend info
    backend: str = "unknown"  # slurm, kubernetes, compose

    # Error tracking
    error_message: Optional[str] = None
    consecutive_failures: int = 0

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def litellm_url(self) -> str:
        """Base URL for the LiteLLM proxy (no trailing /v1)."""
        return f"http://{self.host}:{self.port}"

    @property
    def models_url(self) -> str:
        """Models list URL for readiness checks."""
        return f"http://{self.host}:{self.port}/v1/models"

    @property
    def is_usable(self) -> bool:
        """Check if endpoint can serve requests."""
        return self.status in (EndpointStatus.READY, EndpointStatus.DEGRADED)

    @property
    def ttl_remaining_sec(self) -> Optional[float]:
        """Seconds until expiration, or None if no expiry set."""
        if not self.expires_at:
            return None
        delta = self.expires_at - datetime.utcnow()
        return max(0, delta.total_seconds())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        d = {
            "project_id": self.project_id,
            "model": self.model,
            "job_id": self.job_id,
            "host": self.host,
            "port": self.port,
            "status": self.status.value,
            "created_at": self.created_at.isoformat() + "Z",
            "ready_at": self.ready_at.isoformat() + "Z" if self.ready_at else None,
            "last_used_at": self.last_used_at.isoformat() + "Z" if self.last_used_at else None,
            "expires_at": self.expires_at.isoformat() + "Z" if self.expires_at else None,
            "backend": self.backend,
            "error_message": self.error_message,
            "litellm_url": self.litellm_url,
        }
        # Include served_model_name if available in metadata
        served = self.metadata.get("served_model_name")
        if served:
            d["served_model_name"] = served
        return d


class EndpointRegistry:
    """
    Registry of active Mini-RiVault endpoints.

    Thread-safe. Supports warm endpoint reuse and TTL-based expiration.
    Uses pluggable storage backends (U-109) for persistence.

    Key design decisions:
    - Keyed by (project_id, model) for multi-tenant isolation
    - UMRS readiness polling is authoritative (not backend)
    - Warm reuse: existing ready endpoint reused until expires_at
    - Pluggable storage: in-memory (default) or persistent (Postgres/Redis)
    """

    def __init__(
        self,
        default_ttl_sec: int = 3600,  # 1 hour default TTL
        min_ttl_for_reuse_sec: int = 60,  # Min 1 minute remaining for reuse
        backend: Optional["RegistryBackend"] = None,
    ):
        # In-memory cache (always used for fast lookups)
        self._endpoints: Dict[Tuple[str, str], RegisteredEndpoint] = {}
        self._by_job_id: Dict[str, RegisteredEndpoint] = {}
        self._lock = threading.RLock()
        self._default_ttl_sec = default_ttl_sec
        self._min_ttl_for_reuse_sec = min_ttl_for_reuse_sec

        # Persistent storage backend (U-109)
        self._backend = backend
        if backend:
            self._load_from_backend()

    def _load_from_backend(self) -> None:
        """Load existing endpoints from persistent backend on startup."""
        if not self._backend:
            return

        try:
            endpoints = self._backend.list_all()
            with self._lock:
                for ep in endpoints:
                    key = (ep.project_id, ep.model)
                    self._endpoints[key] = ep
                    self._by_job_id[ep.job_id] = ep
            logger.info(f"Loaded {len(endpoints)} endpoints from backend")
        except Exception as e:
            logger.error(f"Failed to load from backend: {e}")

    def _persist(self, endpoint: RegisteredEndpoint) -> None:
        """Persist endpoint to backend if configured."""
        if self._backend:
            try:
                self._backend.save(endpoint)
            except Exception as e:
                logger.error(f"Failed to persist endpoint {endpoint.job_id}: {e}")

    def _delete_from_backend(self, job_id: str) -> None:
        """Delete endpoint from backend if configured."""
        if self._backend:
            try:
                self._backend.delete(job_id)
            except Exception as e:
                logger.error(f"Failed to delete endpoint {job_id} from backend: {e}")

    def register(
        self,
        project_id: str,
        model: str,
        job_id: str,
        host: str,
        port: int,
        backend: str = "unknown",
        ttl_sec: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> RegisteredEndpoint:
        """
        Register a new endpoint (status=PROVISIONING).

        Returns:
            The registered endpoint.
        """
        ttl = ttl_sec if ttl_sec is not None else self._default_ttl_sec
        expires_at = datetime.utcnow() + timedelta(seconds=ttl)

        endpoint = RegisteredEndpoint(
            project_id=project_id,
            model=model,
            job_id=job_id,
            host=host,
            port=port,
            status=EndpointStatus.PROVISIONING,
            backend=backend,
            expires_at=expires_at,
            metadata=metadata or {},
        )

        key = (project_id, model)

        with self._lock:
            # Check if there's an existing endpoint to replace
            old = self._endpoints.get(key)
            if old:
                logger.info(
                    f"Replacing endpoint for ({project_id}, {model}): "
                    f"{old.job_id} -> {job_id}"
                )
                # Remove old from both in-memory index and backend
                self._by_job_id.pop(old.job_id, None)
                self._delete_from_backend(old.job_id)

            self._endpoints[key] = endpoint
            self._by_job_id[job_id] = endpoint

        # Persist to backend
        self._persist(endpoint)

        logger.info(f"Registered endpoint: {project_id}/{model} -> {host}:{port} (job={job_id})")
        return endpoint

    def update_status(
        self,
        job_id: str,
        status: EndpointStatus,
        error_message: Optional[str] = None,
    ) -> Optional[RegisteredEndpoint]:
        """
        Update endpoint status by job_id.

        Returns:
            Updated endpoint, or None if not found.
        """
        with self._lock:
            endpoint = self._by_job_id.get(job_id)
            if not endpoint:
                return None

            old_status = endpoint.status
            endpoint.status = status
            endpoint.error_message = error_message

            if status == EndpointStatus.READY and not endpoint.ready_at:
                endpoint.ready_at = datetime.utcnow()
                endpoint.consecutive_failures = 0
            elif status == EndpointStatus.FAILED:
                endpoint.consecutive_failures += 1

            # Persist status change
            self._persist(endpoint)

            logger.info(f"Endpoint {job_id} status: {old_status.value} -> {status.value}")
            return endpoint

    def update_location(
        self,
        job_id: str,
        host: Optional[str] = None,
        port: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[RegisteredEndpoint]:
        """Update backend-discovered endpoint coordinates by job_id."""
        with self._lock:
            endpoint = self._by_job_id.get(job_id)
            if not endpoint:
                return None

            if host is not None:
                endpoint.host = host
            if port is not None:
                endpoint.port = port
            if metadata:
                endpoint.metadata.update(metadata)

            self._persist(endpoint)
            logger.info(
                "Endpoint %s location updated to %s:%s",
                job_id,
                endpoint.host,
                endpoint.port,
            )
            return endpoint

    def mark_ready(self, job_id: str) -> Optional[RegisteredEndpoint]:
        """Mark endpoint as ready (passed readiness check)."""
        return self.update_status(job_id, EndpointStatus.READY)

    def mark_failed(self, job_id: str, error: str) -> Optional[RegisteredEndpoint]:
        """Mark endpoint as failed."""
        return self.update_status(job_id, EndpointStatus.FAILED, error)

    def mark_used(self, job_id: str) -> Optional[RegisteredEndpoint]:
        """Update last_used_at timestamp."""
        with self._lock:
            endpoint = self._by_job_id.get(job_id)
            if endpoint:
                endpoint.last_used_at = datetime.utcnow()
                self._persist(endpoint)
            return endpoint

    def get(
        self,
        project_id: str,
        model: str,
        prefer_warm: bool = True,
        min_ttl_sec: Optional[int] = None,
    ) -> Optional[RegisteredEndpoint]:
        """
        Get an endpoint for (project_id, model).

        Args:
            project_id: Project/tenant identifier.
            model: Model name.
            prefer_warm: If True, only return ready endpoints.
            min_ttl_sec: Minimum TTL required (default: min_ttl_for_reuse_sec).

        Returns:
            RegisteredEndpoint if found and usable, None otherwise.

        R-15 (Phase 2k): Falls back to a `served_model_name` lookup when the
        primary `(project_id, model)` key misses. The mini-rivault stack
        sometimes registers under the user-facing model name (``Qwen/Qwen3-8B``)
        and sometimes under the vLLM-served name (a filesystem basename). Both
        callers should resolve to the same endpoint.
        """
        min_ttl = min_ttl_sec if min_ttl_sec is not None else self._min_ttl_for_reuse_sec
        key = (project_id, model)

        with self._lock:
            endpoint = self._endpoints.get(key)
            if endpoint is None:
                # R-15: fall back to served_model_name index.
                for candidate in self._endpoints.values():
                    if candidate.project_id != project_id:
                        continue
                    served = candidate.metadata.get("served_model_name")
                    if served and served == model:
                        endpoint = candidate
                        break
            if not endpoint:
                return None

            # Check if usable
            if prefer_warm and not endpoint.is_usable:
                return None

            # Check TTL
            ttl_remaining = endpoint.ttl_remaining_sec
            if ttl_remaining is not None and ttl_remaining < min_ttl:
                return None

            return endpoint

    def get_by_job_id(self, job_id: str) -> Optional[RegisteredEndpoint]:
        """Get endpoint by job_id."""
        with self._lock:
            return self._by_job_id.get(job_id)

    def remove(self, job_id: str) -> Optional[RegisteredEndpoint]:
        """
        Remove an endpoint by job_id.

        Returns:
            Removed endpoint, or None if not found.
        """
        with self._lock:
            endpoint = self._by_job_id.pop(job_id, None)
            if endpoint:
                key = (endpoint.project_id, endpoint.model)
                # Only remove from main index if it's the current endpoint
                if self._endpoints.get(key) is endpoint:
                    del self._endpoints[key]
                # Delete from backend
                self._delete_from_backend(job_id)
                logger.info(f"Removed endpoint: {job_id}")
            return endpoint

    def list_all(
        self,
        project_id: Optional[str] = None,
        status_filter: Optional[List[EndpointStatus]] = None,
    ) -> List[RegisteredEndpoint]:
        """
        List all registered endpoints.

        Args:
            project_id: Filter by project (None = all projects).
            status_filter: Filter by status (None = all statuses).

        Returns:
            List of endpoints.
        """
        with self._lock:
            endpoints = list(self._endpoints.values())

        if project_id:
            endpoints = [e for e in endpoints if e.project_id == project_id]

        if status_filter:
            endpoints = [e for e in endpoints if e.status in status_filter]

        return endpoints

    def list_expired(self) -> List[RegisteredEndpoint]:
        """List endpoints that have expired."""
        now = datetime.utcnow()
        with self._lock:
            return [
                e for e in self._endpoints.values()
                if e.expires_at and e.expires_at < now
            ]

    def list_idle(self, idle_threshold_sec: int = 1800) -> List[RegisteredEndpoint]:
        """List endpoints that have been idle for too long."""
        threshold = datetime.utcnow() - timedelta(seconds=idle_threshold_sec)
        with self._lock:
            return [
                e for e in self._endpoints.values()
                if e.is_usable and e.last_used_at and e.last_used_at < threshold
            ]

    def clear_all(self) -> int:
        """
        Remove all endpoints (used on UMRS restart if backend not persistent).

        Returns:
            Number of endpoints removed.
        """
        with self._lock:
            count = len(self._endpoints)
            self._endpoints.clear()
            self._by_job_id.clear()
            # Also clear backend if configured
            if self._backend:
                try:
                    self._backend.clear_all()
                except Exception as e:
                    logger.error(f"Failed to clear backend: {e}")
        logger.warning(f"Cleared all {count} endpoints from registry")
        return count

    def stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        with self._lock:
            endpoints = list(self._endpoints.values())

        by_status = {}
        for e in endpoints:
            by_status[e.status.value] = by_status.get(e.status.value, 0) + 1

        by_backend = {}
        for e in endpoints:
            by_backend[e.backend] = by_backend.get(e.backend, 0) + 1

        return {
            "total": len(endpoints),
            "by_status": by_status,
            "by_backend": by_backend,
            "ready_count": sum(1 for e in endpoints if e.is_usable),
        }


# Global singleton instance
_registry: Optional[EndpointRegistry] = None
_registry_lock = threading.Lock()


def get_registry() -> EndpointRegistry:
    """
    Get the global endpoint registry singleton.

    Uses pluggable backend based on UMRS_REGISTRY_BACKEND environment variable:
    - "memory" (default): In-memory only, no persistence
    - "postgres": PostgreSQL backend for HA deployment

    For Postgres, also set UMRS_REGISTRY_POSTGRES_DSN.
    """
    global _registry
    if _registry is None:
        with _registry_lock:
            if _registry is None:
                backend = None
                try:
                    from .registry_backend import InMemoryBackend, get_backend

                    backend = get_backend()
                    if isinstance(backend, InMemoryBackend):
                        backend = None
                except Exception as e:
                    logger.error(f"Failed to initialize registry backend: {e}")
                    logger.warning("Falling back to in-memory registry")

                _registry = EndpointRegistry(backend=backend)
    return _registry
