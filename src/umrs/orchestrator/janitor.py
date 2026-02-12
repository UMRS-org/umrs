"""
Janitor: Background cleanup task.

Responsibilities:
- U-111: Idle timeout teardown (TTL enforcement)
- U-112: Orphan job cleanup (stale/stuck jobs)

This runs as a background task in the UMRS service.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Set

from .endpoint_registry import EndpointRegistry, EndpointStatus, get_registry
from .base import OrchestratorBackend
from .factory import get_orchestrator
from .detection import detect_backend

logger = logging.getLogger(__name__)


@dataclass
class JanitorConfig:
    """Configuration for janitor task."""

    # How often to run cleanup (seconds)
    interval_sec: int = 60

    # Grace period before removing expired endpoints
    ttl_grace_sec: int = 30

    # How long an endpoint can be stuck in provisioning/starting
    stuck_timeout_sec: int = 600  # 10 minutes

    # Maximum cleanup batch size
    batch_size: int = 10

    # Enable/disable specific cleanup tasks
    enable_ttl_cleanup: bool = True
    enable_orphan_cleanup: bool = True
    enable_stuck_cleanup: bool = True

    @classmethod
    def from_env(cls) -> "JanitorConfig":
        """Create config from environment variables."""
        import os

        return cls(
            interval_sec=int(os.getenv("JANITOR_INTERVAL_SEC", "60")),
            ttl_grace_sec=int(os.getenv("JANITOR_TTL_GRACE_SEC", "30")),
            stuck_timeout_sec=int(os.getenv("JANITOR_STUCK_TIMEOUT_SEC", "600")),
            batch_size=int(os.getenv("JANITOR_BATCH_SIZE", "10")),
            enable_ttl_cleanup=os.getenv("JANITOR_ENABLE_TTL", "true").lower() == "true",
            enable_orphan_cleanup=os.getenv("JANITOR_ENABLE_ORPHAN", "true").lower() == "true",
            enable_stuck_cleanup=os.getenv("JANITOR_ENABLE_STUCK", "true").lower() == "true",
        )


@dataclass
class CleanupResult:
    """Result of a cleanup run."""

    ttl_expired: int = 0
    stuck_cleaned: int = 0
    orphans_cleaned: int = 0
    errors: List[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []

    @property
    def total_cleaned(self) -> int:
        """Total endpoints cleaned."""
        return self.ttl_expired + self.stuck_cleaned + self.orphans_cleaned

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "ttl_expired": self.ttl_expired,
            "stuck_cleaned": self.stuck_cleaned,
            "orphans_cleaned": self.orphans_cleaned,
            "total_cleaned": self.total_cleaned,
            "errors": self.errors,
        }


class Janitor:
    """
    Background janitor for endpoint cleanup.

    Handles:
    - TTL expiration (idle timeout)
    - Stuck job cleanup
    - Orphan detection
    """

    def __init__(
        self,
        config: Optional[JanitorConfig] = None,
        registry: Optional[EndpointRegistry] = None,
        backend: Optional[OrchestratorBackend] = None,
    ):
        self.config = config or JanitorConfig.from_env()
        self._registry = registry
        self._backend = backend
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self._cleanup_callbacks: List[Callable[[str], None]] = []
        self._known_jobs: Set[str] = set()  # For orphan detection

    @property
    def registry(self) -> EndpointRegistry:
        """Get endpoint registry (lazy init)."""
        if self._registry is None:
            self._registry = get_registry()
        return self._registry

    @property
    def backend(self) -> OrchestratorBackend:
        """Get orchestrator backend (lazy init)."""
        if self._backend is None:
            detection = detect_backend()
            self._backend = get_orchestrator(detection.backend.value)
        return self._backend

    def add_cleanup_callback(self, callback: Callable[[str], None]):
        """Add callback to be called when an endpoint is cleaned up."""
        self._cleanup_callbacks.append(callback)

    def start(self):
        """Start the background janitor task."""
        if self._running:
            logger.warning("Janitor already running")
            return

        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info(f"Janitor started (interval={self.config.interval_sec}s)")

    def stop(self):
        """Stop the background janitor task."""
        self._running = False
        if self._task:
            self._task.cancel()
            self._task = None
        logger.info("Janitor stopped")

    async def run_once(self) -> CleanupResult:
        """Run cleanup once (for testing or manual trigger)."""
        return await self._cleanup()

    async def _run_loop(self):
        """Main janitor loop."""
        while self._running:
            try:
                result = await self._cleanup()
                if result.total_cleaned > 0:
                    logger.info(f"Janitor cleanup: {result.to_dict()}")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Janitor error: {e}")

            await asyncio.sleep(self.config.interval_sec)

    async def _cleanup(self) -> CleanupResult:
        """Run all cleanup tasks."""
        result = CleanupResult()

        if self.config.enable_ttl_cleanup:
            ttl_result = await self._cleanup_ttl_expired()
            result.ttl_expired = ttl_result

        if self.config.enable_stuck_cleanup:
            stuck_result = await self._cleanup_stuck()
            result.stuck_cleaned = stuck_result

        if self.config.enable_orphan_cleanup:
            orphan_result = await self._cleanup_orphans()
            result.orphans_cleaned = orphan_result

        return result

    async def _cleanup_ttl_expired(self) -> int:
        """
        Clean up endpoints that have exceeded their TTL.

        U-111: Idle timeout teardown.
        """
        cleaned = 0
        now = datetime.utcnow()
        grace = timedelta(seconds=self.config.ttl_grace_sec)

        endpoints = self.registry.list_all()
        for endpoint in endpoints:
            if endpoint.status not in (EndpointStatus.READY, EndpointStatus.WARM):
                continue

            # Check if TTL has expired (with grace period)
            expiry = endpoint.created_at + timedelta(seconds=endpoint.ttl_sec) + grace
            if now > expiry:
                logger.info(f"TTL expired for endpoint {endpoint.job_id}, cleaning up")
                success = await self._cleanup_endpoint(endpoint.job_id)
                if success:
                    cleaned += 1

                if cleaned >= self.config.batch_size:
                    break

        return cleaned

    async def _cleanup_stuck(self) -> int:
        """
        Clean up endpoints stuck in provisioning/starting state.

        Handles jobs that failed to reach ready state within timeout.
        """
        cleaned = 0
        now = datetime.utcnow()
        stuck_threshold = timedelta(seconds=self.config.stuck_timeout_sec)

        endpoints = self.registry.list_all()
        for endpoint in endpoints:
            if endpoint.status not in (EndpointStatus.PROVISIONING, EndpointStatus.STARTING):
                continue

            # Check if stuck for too long
            if now - endpoint.created_at > stuck_threshold:
                logger.warning(
                    f"Endpoint {endpoint.job_id} stuck in {endpoint.status.value} "
                    f"for >{self.config.stuck_timeout_sec}s, cleaning up"
                )
                success = await self._cleanup_endpoint(endpoint.job_id)
                if success:
                    cleaned += 1

                if cleaned >= self.config.batch_size:
                    break

        return cleaned

    async def _cleanup_orphans(self) -> int:
        """
        Clean up orphan jobs (jobs in backend but not in registry).

        U-112: Janitor orphan detection.
        """
        cleaned = 0

        try:
            # Get jobs from backend
            backend_jobs = await self.backend.list_jobs()
            backend_job_ids = {job.job_id for job in backend_jobs}

            # Get jobs from registry
            registry_endpoints = self.registry.list_all()
            registry_job_ids = {ep.job_id for ep in registry_endpoints}

            # Find orphans (in backend but not in registry)
            orphan_ids = backend_job_ids - registry_job_ids

            # Filter out newly created jobs (not yet registered)
            # Use known_jobs to track what we've seen before
            old_orphans = orphan_ids & self._known_jobs
            self._known_jobs = backend_job_ids  # Update for next run

            for job_id in old_orphans:
                logger.warning(f"Cleaning up orphan job: {job_id}")
                try:
                    await self.backend.cleanup(job_id)
                    cleaned += 1
                except Exception as e:
                    logger.error(f"Failed to cleanup orphan {job_id}: {e}")

                if cleaned >= self.config.batch_size:
                    break

        except Exception as e:
            logger.error(f"Orphan cleanup failed: {e}")

        return cleaned

    async def _cleanup_endpoint(self, job_id: str) -> bool:
        """
        Clean up a specific endpoint.

        Args:
            job_id: Job identifier.

        Returns:
            True if cleanup was successful.
        """
        endpoint = self.registry.get_by_job_id(job_id)
        if not endpoint:
            return True  # Already gone

        try:
            # Stop backend job
            await self.backend.cleanup(job_id)
        except Exception as e:
            logger.error(f"Backend cleanup failed for {job_id}: {e}")
            # Continue with registry cleanup anyway

        # Remove from registry
        self.registry.remove(job_id)

        # Notify callbacks
        for callback in self._cleanup_callbacks:
            try:
                callback(job_id)
            except Exception as e:
                logger.error(f"Cleanup callback error: {e}")

        return True


# Global singleton
_janitor: Optional[Janitor] = None


def get_janitor() -> Janitor:
    """Get the global janitor singleton."""
    global _janitor
    if _janitor is None:
        _janitor = Janitor()
    return _janitor


def start_janitor():
    """Start the global janitor."""
    janitor = get_janitor()
    janitor.start()


def stop_janitor():
    """Stop the global janitor."""
    janitor = get_janitor()
    janitor.stop()
