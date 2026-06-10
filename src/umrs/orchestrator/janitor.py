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
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from .base import OrchestratorBackend
from .detection import detect_backend
from .endpoint_registry import EndpointRegistry, EndpointStatus, get_registry
from .factory import get_orchestrator

logger = logging.getLogger(__name__)


def _has_idle_ttl(endpoint: Any) -> bool:
    try:
        return int((endpoint.metadata or {}).get("idle_ttl_sec") or 0) > 0
    except (TypeError, ValueError):
        return False


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
    enable_liveness_cleanup: bool = True

    # Auto-extend TTL for endpoints whose backend jobs are still alive.
    # When enabled, each janitor cycle checks READY endpoints near expiry
    # and extends their expires_at by ttl_auto_extend_sec if the backend
    # (Slurm/K8s/Compose) job is still running. This prevents the registry
    # from killing endpoints whose Slurm allocation is still valid.
    enable_ttl_auto_extend: bool = True
    ttl_auto_extend_sec: int = 600  # extend by 10 minutes each cycle
    ttl_auto_extend_threshold_sec: int = 120  # extend when <2 min remain

    @classmethod
    def from_env(cls) -> JanitorConfig:
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
            enable_liveness_cleanup=os.getenv("JANITOR_ENABLE_LIVENESS", "true").lower() == "true",
            enable_ttl_auto_extend=os.getenv("JANITOR_ENABLE_TTL_AUTO_EXTEND", "true").lower() == "true",
            ttl_auto_extend_sec=int(os.getenv("JANITOR_TTL_AUTO_EXTEND_SEC", "600")),
            ttl_auto_extend_threshold_sec=int(os.getenv("JANITOR_TTL_AUTO_EXTEND_THRESHOLD_SEC", "120")),
        )


@dataclass
class CleanupResult:
    """Result of a cleanup run."""

    ttl_expired: int = 0
    ttl_extended: int = 0
    stuck_cleaned: int = 0
    orphans_cleaned: int = 0
    ghosts_pruned: int = 0
    errors: list[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []

    @property
    def total_cleaned(self) -> int:
        """Total endpoints cleaned."""
        return self.ttl_expired + self.stuck_cleaned + self.orphans_cleaned + self.ghosts_pruned

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "ttl_expired": self.ttl_expired,
            "ttl_extended": self.ttl_extended,
            "stuck_cleaned": self.stuck_cleaned,
            "orphans_cleaned": self.orphans_cleaned,
            "ghosts_pruned": self.ghosts_pruned,
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
        config: JanitorConfig | None = None,
        registry: EndpointRegistry | None = None,
        backend: OrchestratorBackend | None = None,
    ):
        self.config = config or JanitorConfig.from_env()
        self._registry = registry
        self._backend = backend
        self._task: asyncio.Task | None = None
        self._running = False
        self._cleanup_callbacks: list[Callable[[str], None]] = []
        self._known_jobs: set[str] = set()  # For orphan detection

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

        if self.config.enable_liveness_cleanup:
            liveness_result = await self._cleanup_ghost_ready()
            result.ghosts_pruned = liveness_result

        # Auto-extend TTL before checking expiry — ensures we don't kill
        # endpoints whose backend Slurm/K8s jobs are still alive.
        if self.config.enable_ttl_auto_extend:
            extend_result = await self._auto_extend_ttl()
            result.ttl_extended = extend_result

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

    async def _cleanup_ghost_ready(self) -> int:
        """Remove READY/DEGRADED endpoints whose backend jobs are no longer alive.

        F-012: Ghost endpoint pruning.  Stale "ready" entries in the registry
        make ``r4 endpoint list`` untrustworthy.  This task runs every janitor
        cycle and verifies each READY/DEGRADED endpoint against the backend.
        Endpoints whose backing job has terminated are removed from the
        registry so users never see them.

        Uses the same liveness semantics as the request-time filter in
        ``_filter_live_ready_endpoints``: when the backend cannot be reached
        or returns an inconclusive answer, the endpoint is preserved (fail-safe).
        """
        from .base import JobStatus

        pruned = 0
        endpoints = self.registry.list_all(
            status_filter=[EndpointStatus.READY, EndpointStatus.DEGRADED],
        )
        if not endpoints:
            return 0

        for endpoint in endpoints:
            try:
                info = await self.backend.get_job_status(endpoint.job_id)
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "Liveness check for %s skipped (backend error): %s",
                    endpoint.job_id,
                    exc,
                )
                continue  # Inconclusive — preserve the endpoint

            status_value = getattr(info, "status", None)
            if status_value in (
                JobStatus.PENDING,
                JobStatus.STARTING,
                JobStatus.READY,
                JobStatus.RUNNING,
            ):
                continue  # Job is alive

            # Backend reports job as terminal — remove ghost endpoint.
            logger.warning(
                "Pruning ghost endpoint %s/%s (job_id=%s, backend_status=%s): "
                "backend job is no longer alive",
                endpoint.project_id,
                endpoint.model,
                endpoint.job_id,
                status_value,
            )
            self.registry.remove(endpoint.job_id)
            pruned += 1

            if pruned >= self.config.batch_size:
                break

        return pruned

    async def _auto_extend_ttl(self) -> int:
        """Extend TTL for READY endpoints whose backend jobs are still alive.

        When an endpoint's TTL is about to expire but the backend
        Slurm/K8s/Compose job is still running, the registry TTL should
        not kill it prematurely.  This task checks each near-expiry endpoint
        against the backend and extends ``expires_at`` when the job is alive.
        """
        from .base import JobStatus

        extended = 0
        now = datetime.utcnow()
        threshold = timedelta(seconds=self.config.ttl_auto_extend_threshold_sec)

        endpoints = self.registry.list_all(
            status_filter=[EndpointStatus.READY, EndpointStatus.DEGRADED],
        )
        for endpoint in endpoints:
            if _has_idle_ttl(endpoint):
                continue
            if not endpoint.expires_at:
                continue

            remaining = endpoint.expires_at - now
            if remaining > threshold:
                continue  # Not near expiry yet

            # Near expiry — check backend liveness
            try:
                info = await self.backend.get_job_status(endpoint.job_id)
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "TTL auto-extend check for %s skipped (backend error): %s",
                    endpoint.job_id,
                    exc,
                )
                continue  # Inconclusive — don't extend, but don't kill either

            status_value = getattr(info, "status", None)
            if status_value not in (
                JobStatus.PENDING,
                JobStatus.STARTING,
                JobStatus.READY,
                JobStatus.RUNNING,
            ):
                continue  # Job is terminal — let TTL cleanup handle it

            # Backend job is alive — extend TTL
            new_expires = now + timedelta(seconds=self.config.ttl_auto_extend_sec)
            self.registry.extend_ttl(endpoint.job_id, new_expires)
            extended += 1
            logger.info(
                "Auto-extended TTL for %s/%s (job_id=%s) by %ds",
                endpoint.project_id,
                endpoint.model,
                endpoint.job_id,
                self.config.ttl_auto_extend_sec,
            )

        return extended

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
            if endpoint.status not in (EndpointStatus.READY, EndpointStatus.DEGRADED):
                continue

            # Check if TTL has expired (with grace period)
            if not endpoint.expires_at:
                continue
            if now > endpoint.expires_at + grace:
                logger.info(f"TTL expired for endpoint {endpoint.job_id}, cleaning up")
                success = await self._cleanup_endpoint(endpoint.job_id)
                if success:
                    cleaned += 1

                if cleaned >= self.config.batch_size:
                    break

        return cleaned

    async def _cleanup_stuck(self) -> int:
        """
        Annotate and clean up endpoints stuck in provisioning/starting state.

        For every PROVISIONING/STARTING endpoint this records the scheduler's
        current reason (e.g. Slurm ``MaxMemPerLimit``, ``Resources``,
        ``Priority``) into the endpoint ``detail`` so users can see *why* a
        provision is still pending instead of it failing silently.

        A job that is still legitimately queued (benign reason such as
        ``Resources`` or ``Priority``) is left alone even past the stuck
        timeout — large multi-node, infinite-TTL endpoints can sit in the queue
        for a long time before nodes free up. Endpoints whose job is held for an
        unsatisfiable allocation, or that never reached ready for any other
        reason, are cleaned up once they exceed ``stuck_timeout_sec``.
        """
        from .base import JobStatus

        # Slurm reasons that mean "legitimately waiting in the queue". These are
        # not failures, so a stuck endpoint reporting one is preserved.
        benign_reasons = (
            "Resources",
            "Priority",
            "BeginTime",
            "Dependency",
            "JobArrayTaskLimit",
        )

        cleaned = 0
        now = datetime.utcnow()
        stuck_threshold = timedelta(seconds=self.config.stuck_timeout_sec)

        endpoints = self.registry.list_all()
        for endpoint in endpoints:
            if endpoint.status not in (EndpointStatus.PROVISIONING, EndpointStatus.STARTING):
                continue

            # Best-effort: query the backend for the live state + reason so we
            # can surface it and make an informed cleanup decision.
            reason: str | None = None
            job_status = None
            try:
                info = await self.backend.get_job_status(endpoint.job_id)
                reason = getattr(info, "state_reason", None)
                job_status = getattr(info, "status", None)
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "Stuck-check for %s skipped (backend error): %s",
                    endpoint.job_id, exc,
                )
                continue  # Inconclusive — preserve the endpoint

            # Surface the reason to users via the endpoint detail field.
            if reason:
                self.registry.update_metadata(
                    endpoint.job_id, {"detail": f"queued: {reason}"}
                )

            age = now - endpoint.created_at
            if age <= stuck_threshold:
                continue  # Not stuck yet — leave it to keep provisioning

            # Past the stuck timeout. Preserve jobs that are simply waiting in
            # the queue for resources; only reap unsatisfiable / failed ones.
            if (
                job_status == JobStatus.PENDING
                and reason
                and any(b in reason for b in benign_reasons)
            ):
                logger.info(
                    "Endpoint %s still queued after %ds (reason=%s); preserving",
                    endpoint.job_id, int(age.total_seconds()), reason,
                )
                continue

            logger.warning(
                "Endpoint %s stuck in %s for >%ds (reason=%s), cleaning up",
                endpoint.job_id, endpoint.status.value,
                self.config.stuck_timeout_sec, reason or "unknown",
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
_janitor: Janitor | None = None


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
