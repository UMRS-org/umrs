"""
AttemptMonitor - Monitor running attempts (U-714).

Polls backend status for SUBMITTED/RUNNING attempts and marks
them as SUCCEEDED/FAILED based on results.

For endpoint_provision jobs, success = readiness poll success + registry entry.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from datetime import datetime
from typing import Any, Protocol

from .models import AttemptStatus, Job, JobAttempt, JobType
from .store import JobStore
from umrs.metrics import METRICS

logger = logging.getLogger(__name__)


class BackendStatusChecker(Protocol):
    """Protocol for checking backend job status."""

    async def check_status(
        self,
        attempt: JobAttempt,
        job: Job,
    ) -> AttemptCheckResult:
        """
        Check status of an attempt.

        Returns AttemptCheckResult with current state.
        """
        ...


class AttemptCheckResult:
    """Result of checking an attempt's status."""

    def __init__(
        self,
        status: str,  # "running", "succeeded", "failed", "unknown"
        exit_reason: str | None = None,
        metrics: dict[str, Any] | None = None,
        status_detail: str | None = None,
        should_commit: bool = True,  # Whether to set commit_attempt_id
    ):
        self.status = status
        self.exit_reason = exit_reason
        self.metrics = metrics or {}
        self.status_detail = status_detail
        self.should_commit = should_commit

    @classmethod
    def running(cls, detail: str = "running") -> AttemptCheckResult:
        return cls(status="running", status_detail=detail)

    @classmethod
    def succeeded(
        cls,
        metrics: dict | None = None,
        commit: bool = True,
    ) -> AttemptCheckResult:
        return cls(
            status="succeeded",
            exit_reason="success",
            metrics=metrics,
            should_commit=commit,
        )

    @classmethod
    def failed(cls, reason: str) -> AttemptCheckResult:
        return cls(status="failed", exit_reason=reason)

    @classmethod
    def unknown(cls) -> AttemptCheckResult:
        return cls(status="unknown", status_detail="status unknown")


class AttemptMonitor:
    """
    Monitor for running attempts.

    Periodically checks status of SUBMITTED/RUNNING attempts
    and updates their state based on backend feedback.
    """

    # Maximum time to wait for a single status check before treating it as
    # a timeout.  Prevents one hanging backend from blocking all monitoring.
    STATUS_CHECK_TIMEOUT: float = 60.0

    def __init__(
        self,
        job_store: JobStore,
        status_checker: BackendStatusChecker,
        poll_interval: float = 5.0,
    ):
        """
        Initialize AttemptMonitor.

        Args:
            job_store: JobStore for state management
            status_checker: Checker to query backend status
            poll_interval: Seconds between poll cycles
        """
        self.job_store = job_store
        self.status_checker = status_checker
        self.poll_interval = poll_interval

        self._running = False
        self._task: asyncio.Task | None = None
        self._stats = {
            "cycles": 0,
            "attempts_checked": 0,
            "attempts_succeeded": 0,
            "attempts_failed": 0,
            "last_cycle_at": None,
        }

    async def start(self) -> None:
        """Start the monitor loop."""
        if self._running:
            logger.warning("AttemptMonitor already running")
            return

        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info("AttemptMonitor started")

    async def stop(self, timeout: float = 10.0) -> None:
        """Stop the monitor loop."""
        if not self._running:
            return

        self._running = False

        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout=timeout)
            except asyncio.TimeoutError:
                logger.warning("AttemptMonitor stop timeout, cancelling")
                self._task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._task

        logger.info("AttemptMonitor stopped")

    async def _run_loop(self) -> None:
        """Main monitor loop."""
        logger.info("AttemptMonitor loop starting")

        while self._running:
            try:
                await self._poll_cycle()
            except Exception as e:
                logger.error(f"AttemptMonitor cycle error: {e}", exc_info=True)

            await asyncio.sleep(self.poll_interval)

        logger.info("AttemptMonitor loop exited")

    async def _poll_cycle(self) -> None:
        """
        One poll cycle.

        1. Get all active attempts (SUBMITTED/RUNNING)
        2. Check status of each
        3. Update state based on results
        """
        self._stats["cycles"] += 1
        self._stats["last_cycle_at"] = datetime.utcnow().isoformat()

        # Get active attempts
        active_attempts = self.job_store.get_active_attempts()

        if not active_attempts:
            return

        logger.debug(f"Checking {len(active_attempts)} active attempts")

        # Check each attempt (can be parallelized if needed)
        for attempt in active_attempts:
            await self._check_attempt(attempt)

    async def _check_attempt(self, attempt: JobAttempt) -> None:
        """
        Check and update a single attempt.
        """
        self._stats["attempts_checked"] += 1

        try:
            # Get parent job
            job = self.job_store.get_job(attempt.job_id)

            # Check status via backend with timeout guard
            try:
                result = await asyncio.wait_for(
                    self.status_checker.check_status(attempt, job),
                    timeout=self.STATUS_CHECK_TIMEOUT,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    f"Status check timed out for attempt {attempt.attempt_id} "
                    f"after {self.STATUS_CHECK_TIMEOUT}s"
                )
                METRICS.monitor_check("timeout")
                return  # Skip this attempt; retry next cycle

            # Update based on result
            if result.status == "succeeded":
                self.job_store.mark_attempt_succeeded(
                    attempt.attempt_id,
                    commit=result.should_commit,
                    metrics=result.metrics,
                )
                self._stats["attempts_succeeded"] += 1
                METRICS.monitor_check("succeeded")
                logger.info(f"Attempt {attempt.attempt_id} succeeded")

            elif result.status == "failed":
                requeued = self.job_store.mark_attempt_failed(
                    attempt.attempt_id,
                    exit_reason=result.exit_reason or "unknown failure",
                )
                self._stats["attempts_failed"] += 1
                METRICS.monitor_check("failed")
                logger.warning(
                    f"Attempt {attempt.attempt_id} failed: {result.exit_reason} "
                    f"(requeued={requeued})"
                )

            elif result.status == "running":
                # Update status detail if changed
                if result.status_detail and attempt.status == AttemptStatus.SUBMITTED:
                    self.job_store.mark_attempt_running(
                        attempt.attempt_id,
                        status_detail=result.status_detail,
                    )
                elif result.status_detail:
                    self.job_store.update_attempt_status_detail(
                        attempt.attempt_id,
                        result.status_detail,
                    )

                # Update metrics if provided
                if result.metrics:
                    self.job_store.update_attempt_metrics(
                        attempt.attempt_id,
                        result.metrics,
                    )

            # unknown status - no action, wait for next cycle

        except Exception as e:
            logger.error(
                f"Error checking attempt {attempt.attempt_id}: {e}",
                exc_info=True
            )

    @property
    def stats(self) -> dict[str, Any]:
        """Get monitor statistics."""
        return self._stats.copy()

    @property
    def is_running(self) -> bool:
        """Check if monitor is running."""
        return self._running


class EndpointProvisionChecker:
    """
    Status checker for endpoint_provision jobs.

    For endpoint jobs, success = readiness poll success.
    Uses existing UMRS readiness polling infrastructure.
    """

    def __init__(
        self,
        readiness_poller,  # ReadinessPoller from orchestrator
        endpoint_registry,  # EndpointRegistry
    ):
        self.readiness_poller = readiness_poller
        self.endpoint_registry = endpoint_registry

    async def check_status(
        self,
        attempt: JobAttempt,
        job: Job,
    ) -> AttemptCheckResult:
        """
        Check endpoint provision status.

        1. Check if endpoint is in registry and ready
        2. If not, poll readiness
        3. Return appropriate result
        """
        if job.job_type != JobType.ENDPOINT_PROVISION:
            raise ValueError("EndpointProvisionChecker only handles endpoint jobs")

        project_id = job.project_id
        model = job.spec.get("model", "rivault-gpt-oss-static")

        # Check registry first
        endpoint = self.endpoint_registry.get_endpoint(project_id, model)

        if endpoint and endpoint.status == "ready":
            return AttemptCheckResult.succeeded(
                metrics={
                    "ready_at": datetime.utcnow().isoformat(),
                    "endpoint_host": endpoint.host,
                    "endpoint_port": endpoint.port,
                },
                commit=True,  # Endpoint jobs commit on success
            )

        # Try readiness poll if we have backend_ref
        if attempt.backend_ref:
            try:
                # Get endpoint info from backend
                # This depends on the specific backend implementation
                is_ready = await self._poll_readiness(attempt)

                if is_ready:
                    return AttemptCheckResult.succeeded(
                        metrics={"ready_at": datetime.utcnow().isoformat()},
                        commit=True,
                    )
                else:
                    return AttemptCheckResult.running(
                        detail=f"polling readiness ({attempt.status_detail or 'waiting'})"
                    )

            except Exception as e:
                # Check if this is a permanent failure
                if self._is_permanent_failure(e):
                    return AttemptCheckResult.failed(str(e))
                else:
                    return AttemptCheckResult.running(
                        detail=f"readiness check failed: {e}"
                    )

        return AttemptCheckResult.running(detail="waiting for backend")

    async def _poll_readiness(self, attempt: JobAttempt) -> bool:
        """Poll readiness endpoint."""
        # This would integrate with the existing readiness polling
        # For now, return False to indicate not ready
        return False

    def _is_permanent_failure(self, error: Exception) -> bool:
        """Check if error indicates permanent failure."""
        # Certain errors are permanent (container crash, bad config, etc.)
        error_str = str(error).lower()
        permanent_indicators = [
            "image not found",
            "container crashed",
            "gpu allocation failed",
            "model not found",
            "permission denied",
        ]
        return any(ind in error_str for ind in permanent_indicators)


class CompositeStatusChecker:
    """
    Status checker that routes to job-type-specific checkers.
    """

    def __init__(self):
        self._checkers: dict[JobType, BackendStatusChecker] = {}

    def register(self, job_type: JobType, checker: BackendStatusChecker) -> None:
        """Register a checker for a job type."""
        self._checkers[job_type] = checker

    async def check_status(
        self,
        attempt: JobAttempt,
        job: Job,
    ) -> AttemptCheckResult:
        """Route to appropriate checker."""
        checker = self._checkers.get(job.job_type)

        if checker:
            return await checker.check_status(attempt, job)
        else:
            # No specific checker - return unknown
            logger.warning(f"No checker registered for job type {job.job_type}")
            return AttemptCheckResult.unknown()
