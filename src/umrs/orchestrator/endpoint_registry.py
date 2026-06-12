"""
Endpoint Registry for Mini-RiVault instances.

This module provides a registry of active inference endpoints,
keyed by (project_id, job_id). Gateway uses this to discover which endpoint
to route requests to.

Multiple endpoints per model are supported via the ``variant`` field, allowing
parameter-differentiated deployments (e.g. different ctx lengths, node counts,
parallel strategies) and load-based scaling of identical replicas.

UMRS readiness polling (U-107) is the authoritative source of endpoint status.
Backends only provide discovery coordinates (host/port).

Storage backends (U-109):
- InMemoryBackend: Default, for single-instance deployment
- PostgresBackend: For HA deployment with persistence
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .registry_backend import RegistryBackend

logger = logging.getLogger(__name__)


def _default_owner_type(project_id: str) -> str:
    """Infer the effective owner type for legacy registry records."""
    if project_id == "rivault":
        return "system"
    return "user"


def _positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


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
    ready_at: datetime | None = None
    last_used_at: datetime | None = None
    expires_at: datetime | None = None

    # Backend info
    backend: str = "unknown"  # slurm, kubernetes, compose

    # Error tracking
    error_message: str | None = None
    consecutive_failures: int = 0

    # Variant — differentiates multiple endpoints for the same model.
    # Derived from the pool YAML ``name`` field (e.g. "128k", "tep-128k").
    # ``None`` means the default/sole deployment for this model.
    variant: str | None = None

    # Active request counter for least-connections routing.
    # Incremented before proxying a request, decremented after completion.
    active_requests: int = 0

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

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
    def ttl_remaining_sec(self) -> float | None:
        """Seconds until expiration, or None if no expiry set."""
        if not self.expires_at:
            return None
        delta = self.expires_at - datetime.utcnow()
        return max(0, delta.total_seconds())

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API response."""
        owner_type = self.metadata.get("owner_type") or self.metadata.get("tier")
        if owner_type is None:
            owner_type = _default_owner_type(self.project_id)

        tier = self.metadata.get("tier") or owner_type
        d = {
            "project_id": self.project_id,
            "model": self.model,
            "job_id": self.job_id,
            "host": self.host,
            "port": self.port,
            "status": self.status.value,
            "created_at": self.created_at.isoformat() + "Z",
            "ready_at": self.ready_at.isoformat() + "Z" if self.ready_at else None,
            "last_used_at": self.last_used_at.isoformat() + "Z"
            if self.last_used_at
            else None,
            "expires_at": self.expires_at.isoformat() + "Z"
            if self.expires_at
            else None,
            "backend": self.backend,
            "error_message": self.error_message,
            "litellm_url": self.litellm_url,
            "owner_type": str(owner_type),
            "tier": str(tier),
            "active_requests": self.active_requests,
        }
        if self.variant is not None:
            d["variant"] = self.variant
        owner_id = self.metadata.get("owner_id")
        if owner_id is not None:
            d["owner_id"] = str(owner_id)
        # Include served_model_name if available in metadata
        served = self.metadata.get("served_model_name")
        if served:
            d["served_model_name"] = served
        for key in (
            "model_type",
            "serving_backend",
            "service_kind",
            "service_url",
            "invoke_base_path",
            "readiness_probe",
            "readiness_path",
            "launch_profiles",
            "container_image",
            "vision_adapter",
            "vision_task",
            "output_format",
        ):
            value = self.metadata.get(key)
            if value not in (None, "", []):
                d[key] = value
        if "service_url" not in d:
            d["service_url"] = self.litellm_url
        detail = self.metadata.get("detail")
        if detail:
            d["detail"] = detail
        job_status = self.metadata.get("job_status")
        if job_status:
            d["job_status"] = job_status
        return d


class EndpointRegistry:
    """
    Registry of active Mini-RiVault endpoints.

    Thread-safe. Supports warm endpoint reuse and TTL-based expiration.
    Uses pluggable storage backends (U-109) for persistence.

    Key design decisions:
    - Keyed by (project_id, job_id) for multi-tenant isolation
    - Multiple endpoints per model supported via the ``variant`` field
    - UMRS readiness polling is authoritative (not backend)
    - Warm reuse: existing ready endpoint reused until expires_at
    - Pluggable storage: in-memory (default) or persistent (Postgres/Redis)
    - Least-connections routing: ``select_endpoint()`` picks the endpoint
      with the fewest active_requests among candidates for a model/variant
    """

    def __init__(
        self,
        default_ttl_sec: int = 3600,  # 1 hour default TTL
        min_ttl_for_reuse_sec: int = 60,  # Min 1 minute remaining for reuse
        backend: RegistryBackend | None = None,
    ):
        # In-memory cache (always used for fast lookups)
        self._endpoints: dict[tuple[str, str], RegisteredEndpoint] = {}
        self._by_job_id: dict[str, RegisteredEndpoint] = {}
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
                    key = (ep.project_id, ep.job_id)
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
        ttl_sec: int | None = None,
        metadata: dict[str, Any] | None = None,
        variant: str | None = None,
    ) -> RegisteredEndpoint:
        """
        Register a new endpoint (status=PROVISIONING).

        Args:
            project_id: Tenant/project identifier.
            model: Model name (e.g. "DeepSeek-V4-Pro").
            job_id: Unique job identifier from the backend.
            host: Compute node hostname.
            port: LiteLLM proxy port.
            backend: Execution backend ("slurm", "kubernetes", "compose").
            ttl_sec: Time-to-live in seconds. 0 = never expires, None = default.
            metadata: Optional metadata dict (tier, owner_id, served_model_name, etc.).
            variant: Deployment variant name (e.g. "128k", "tep-128k").
                     Differentiates multiple endpoints serving the same model
                     with different parameters (ctx length, node count, etc.).
                     ``None`` means the default/sole deployment.

        Returns:
            The registered endpoint.
        """
        metadata = dict(metadata or {})
        idle_ttl_sec = _positive_int(metadata.get("idle_ttl_sec"))
        if idle_ttl_sec is not None:
            metadata["idle_ttl_sec"] = idle_ttl_sec
        # ``ttl_sec == 0`` is an explicit request for a non-expiring endpoint
        # (e.g. system-tier pool models on unlimited-walltime partitions that
        # must never be auto-recycled). ``None`` falls back to the idle/default
        # TTL; positive values are finite TTLs. A non-expiring endpoint stores
        # ``expires_at = None``, which downstream TTL checks treat as "never
        # expired" (see ``ttl_remaining_sec`` / ``get`` / ``list_expired``).
        if ttl_sec is not None and ttl_sec <= 0:
            expires_at = None
        else:
            ttl = _positive_int(ttl_sec) or idle_ttl_sec or self._default_ttl_sec
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
            variant=variant,
            metadata=metadata,
        )

        key = (project_id, job_id)

        with self._lock:
            # Check by job_id — allow multiple deployments of same model
            old = self._endpoints.get(key)
            if old:
                logger.info(
                    f"Replacing endpoint for ({project_id}, {model}): "
                    f"{old.job_id} -> {job_id}"
                )
                self._by_job_id.pop(old.job_id, None)
                self._delete_from_backend(old.job_id)

            self._endpoints[key] = endpoint
            self._by_job_id[job_id] = endpoint
            # Persist inside the lock so in-memory and backend states stay
            # consistent when concurrent register() calls race for the same
            # (project_id, model).  NOTE: this still does not protect against
            # multi-process races; the caller serialises duplicates.
            self._persist(endpoint)

        logger.info(
            f"Registered endpoint: {project_id}/{model} -> {host}:{port} (job={job_id})"
        )
        return endpoint

    def update_status(
        self,
        job_id: str,
        status: EndpointStatus,
        error_message: str | None = None,
    ) -> RegisteredEndpoint | None:
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

            logger.info(
                f"Endpoint {job_id} status: {old_status.value} -> {status.value}"
            )
            return endpoint

    def update_location(
        self,
        job_id: str,
        host: str | None = None,
        port: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> RegisteredEndpoint | None:
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

    def update_metadata(
        self,
        job_id: str,
        metadata: dict[str, Any],
    ) -> RegisteredEndpoint | None:
        """Merge progress metadata without changing endpoint coordinates."""
        with self._lock:
            endpoint = self._by_job_id.get(job_id)
            if not endpoint:
                return None

            endpoint.metadata.update(metadata)
            self._persist(endpoint)
            return endpoint

    def mark_ready(self, job_id: str) -> RegisteredEndpoint | None:
        """Mark endpoint as ready (passed readiness check)."""
        return self.update_status(job_id, EndpointStatus.READY)

    def mark_failed(self, job_id: str, error: str) -> RegisteredEndpoint | None:
        """Mark endpoint as failed."""
        return self.update_status(job_id, EndpointStatus.FAILED, error)

    def mark_used(self, job_id: str) -> RegisteredEndpoint | None:
        """Update last_used_at timestamp."""
        with self._lock:
            endpoint = self._by_job_id.get(job_id)
            if endpoint:
                now = datetime.utcnow()
                endpoint.last_used_at = now
                idle_ttl_sec = _positive_int(endpoint.metadata.get("idle_ttl_sec"))
                if idle_ttl_sec:
                    endpoint.expires_at = now + timedelta(seconds=idle_ttl_sec)
                self._persist(endpoint)
            return endpoint

    def extend_ttl(
        self, job_id: str, new_expires_at: datetime
    ) -> RegisteredEndpoint | None:
        """Extend endpoint TTL to a new expiration time.

        Only extends (never shortens). Returns the updated endpoint or None
        if not found. Used by the janitor to align endpoint TTL with the
        backend job's remaining lifetime.
        """
        with self._lock:
            endpoint = self._by_job_id.get(job_id)
            if not endpoint:
                return None
            if endpoint.expires_at is None or new_expires_at > endpoint.expires_at:
                endpoint.expires_at = new_expires_at
                self._persist(endpoint)
                logger.debug(
                    "Extended TTL for %s to %s",
                    job_id,
                    new_expires_at.isoformat(),
                )
            return endpoint

    def _find_candidate(
        self, project_id: str, model: str
    ) -> RegisteredEndpoint | None:
        """Find an endpoint by (project_id, model) or served_model_name fallback."""
        for candidate in self._endpoints.values():
            if candidate.project_id != project_id:
                continue
            if candidate.model == model:
                return candidate
        # R-15: fall back to served_model_name index.
        for candidate in self._endpoints.values():
            if candidate.project_id != project_id:
                continue
            served = candidate.metadata.get("served_model_name")
            if served and served == model:
                return candidate
        return None

    def _find_candidates(
        self,
        project_id: str,
        model: str,
        variant: str | None = None,
        status_filter: list[EndpointStatus] | None = None,
    ) -> list[RegisteredEndpoint]:
        """Find all endpoints matching (project_id, model[, variant]).

        Args:
            project_id: Tenant/project identifier.
            model: Model name to match.
            variant: If provided, only return endpoints with this variant.
                     If ``None``, return endpoints regardless of variant.
            status_filter: If provided, only return endpoints with one of
                           these statuses. Typical: ``[READY, DEGRADED]``.

        Returns:
            List of matching endpoints (may be empty).
        """
        results: list[RegisteredEndpoint] = []
        for candidate in self._endpoints.values():
            if candidate.project_id != project_id:
                continue
            if candidate.model != model:
                # R-15: also match by served_model_name in metadata
                served = candidate.metadata.get("served_model_name")
                if not served or served != model:
                    continue
            if variant is not None and candidate.variant != variant:
                continue
            if status_filter and candidate.status not in status_filter:
                continue
            results.append(candidate)
        return results

    def get(
        self,
        project_id: str,
        model: str,
        prefer_warm: bool = True,
        min_ttl_sec: int | None = None,
        variant: str | None = None,
    ) -> RegisteredEndpoint | None:
        """Get an endpoint for (project_id, model[, variant]).

        Supports multiple deployments of the same model within a project.
        When ``variant`` is specified, only considers endpoints with that
        variant. Otherwise returns the first ready, usable match regardless
        of variant (backwards compatible).
        """
        min_ttl = (
            min_ttl_sec if min_ttl_sec is not None else self._min_ttl_for_reuse_sec
        )
        with self._lock:
            if variant is not None:
                # Variant-aware: search candidates directly
                candidates = self._find_candidates(
                    project_id, model, variant=variant
                )
                for ep in candidates:
                    if prefer_warm and not ep.is_usable:
                        continue
                    ttl_remaining = ep.ttl_remaining_sec
                    if ttl_remaining is not None and ttl_remaining < min_ttl:
                        continue
                    return ep
                return None
            # Legacy path: first match (backwards compatible)
            endpoint = self._find_candidate(project_id, model)
            if not endpoint:
                return None
            if prefer_warm and not endpoint.is_usable:
                return None
            ttl_remaining = endpoint.ttl_remaining_sec
            if ttl_remaining is not None and ttl_remaining < min_ttl:
                return None
            return endpoint

    def select_endpoint(
        self,
        project_id: str,
        model: str,
        variant: str | None = None,
        strategy: str = "least_connections",
        min_ttl_sec: int | None = None,
    ) -> RegisteredEndpoint | None:
        """Select the best endpoint for (project_id, model[, variant]).

        Unlike ``get()`` (which returns the first match), this method
        considers all usable candidates and applies a selection strategy.

        Args:
            project_id: Tenant/project identifier.
            model: Model name.
            variant: If provided, only consider endpoints with this variant.
            strategy: Selection strategy:
                - ``"least_connections"``: Pick the endpoint with the
                  fewest ``active_requests`` among usable candidates.
                - ``"first"``: First usable candidate (same as ``get()``).
            min_ttl_sec: Minimum remaining TTL for candidates.

        Returns:
            Selected endpoint, or ``None`` if no usable candidate exists.
        """
        min_ttl = (
            min_ttl_sec if min_ttl_sec is not None else self._min_ttl_for_reuse_sec
        )
        with self._lock:
            candidates = self._find_candidates(
                project_id,
                model,
                variant=variant,
                status_filter=[EndpointStatus.READY, EndpointStatus.DEGRADED],
            )
            # Filter by TTL
            usable: list[RegisteredEndpoint] = []
            for ep in candidates:
                ttl_remaining = ep.ttl_remaining_sec
                if ttl_remaining is not None and ttl_remaining < min_ttl:
                    continue
                usable.append(ep)
            if not usable:
                return None

            if strategy == "least_connections":
                return min(usable, key=lambda e: e.active_requests)
            # "first" or unknown strategy
            return usable[0]

    def increment_requests(self, job_id: str) -> None:
        """Atomically increment active_requests for an endpoint.

        Called before proxying a request to track load for least-connections
        routing. No-op if job_id is not found.
        """
        with self._lock:
            endpoint = self._by_job_id.get(job_id)
            if endpoint:
                endpoint.active_requests += 1

    def decrement_requests(self, job_id: str) -> None:
        """Atomically decrement active_requests for an endpoint.

        Called after a proxied request completes (success or failure).
        No-op if job_id is not found. Never decrements below zero.
        """
        with self._lock:
            endpoint = self._by_job_id.get(job_id)
            if endpoint and endpoint.active_requests > 0:
                endpoint.active_requests -= 1

    def get_by_job_id(self, job_id: str) -> RegisteredEndpoint | None:
        """Get endpoint by job_id."""
        with self._lock:
            return self._by_job_id.get(job_id)

    def remove(self, job_id: str) -> RegisteredEndpoint | None:
        """
        Remove an endpoint by job_id.

        Returns:
            Removed endpoint, or None if not found.
        """
        with self._lock:
            endpoint = self._by_job_id.pop(job_id, None)
            if endpoint:
                key = (endpoint.project_id, endpoint.job_id)
                # Only remove from main index if it's the current endpoint
                if self._endpoints.get(key) is endpoint:
                    del self._endpoints[key]
                # Delete from backend
                self._delete_from_backend(job_id)
                logger.info(f"Removed endpoint: {job_id}")
            return endpoint

    def list_all(
        self,
        project_id: str | None = None,
        status_filter: list[EndpointStatus] | None = None,
    ) -> list[RegisteredEndpoint]:
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

    def list_expired(self) -> list[RegisteredEndpoint]:
        """List endpoints that have expired."""
        now = datetime.utcnow()
        with self._lock:
            return [
                e
                for e in self._endpoints.values()
                if e.expires_at and e.expires_at < now
            ]

    def list_idle(self, idle_threshold_sec: int = 1800) -> list[RegisteredEndpoint]:
        """List endpoints that have been idle for too long."""
        threshold = datetime.utcnow() - timedelta(seconds=idle_threshold_sec)
        with self._lock:
            return [
                e
                for e in self._endpoints.values()
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

    def stats(self) -> dict[str, Any]:
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
_registry: EndpointRegistry | None = None
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
