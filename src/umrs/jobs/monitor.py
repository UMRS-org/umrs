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
import os
from datetime import datetime, timezone
from typing import Any, Protocol

from umrs.metrics import METRICS

from .models import AttemptStatus, Job, JobAttempt, JobType
from .store import JobStore

logger = logging.getLogger(__name__)


def _open_knowledge_conn():
    """Best-effort knowledge-DSN psycopg2 conn; None when unavailable.

    Mirrors ``jobs.workflow_run._open_artifact_store`` (same DSN + kwargs).
    """
    try:
        import psycopg2

        from umrs.shared.pg_settings import pg_connect_kwargs

        from .knowledge_adapters import _resolve_knowledge_dsn

        return psycopg2.connect(
            _resolve_knowledge_dsn(), **pg_connect_kwargs("job-monitor")
        )
    except Exception as exc:  # noqa: BLE001 — availability probe, never a gate
        logger.warning("knowledge registry unavailable: %s", exc)
        return None


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

    # Default maximum time to wait for a single status check before treating
    # it as a timeout.  Prevents one hanging backend from blocking all
    # monitoring.  May be overridden via UMRS_STATUS_CHECK_TIMEOUT env var.
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
        # Env-overridable timeout (default 60s)
        self.status_check_timeout = float(
            os.environ.get("UMRS_STATUS_CHECK_TIMEOUT", str(self.STATUS_CHECK_TIMEOUT))
        )

        self._running = False
        self._task: asyncio.Task | None = None
        self._stats = {
            "cycles": 0,
            "attempts_checked": 0,
            "attempts_succeeded": 0,
            "attempts_failed": 0,
            "last_cycle_at": None,
        }
        # Track consecutive timeout streaks per attempt.
        # Cleared on any successful check or terminal state.
        self._timeout_streaks: dict[str, int] = {}

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
            except Exception:
                logger.exception("AttemptMonitor cycle error")

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
        self._stats["last_cycle_at"] = datetime.now(timezone.utc).isoformat()

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
            result = await self._fetch_status(attempt, job)
            if result is None:
                self._record_unknown(attempt)
                return

            # Update based on result
            if result.status == "succeeded":
                self._apply_success(attempt, result)
            elif result.status == "failed":
                self._apply_failure(attempt, result)
            elif result.status == "running":
                self._apply_running(attempt, result)
            # unknown status - no action, wait for next cycle

        except Exception:
            logger.exception(f"Error checking attempt {attempt.attempt_id}")

    async def _fetch_status(self, attempt: JobAttempt, job) -> Any:
        """Call the backend status checker with a timeout guard."""
        try:
            return await asyncio.wait_for(
                self.status_checker.check_status(attempt, job),
                timeout=self.status_check_timeout,
            )
        except asyncio.TimeoutError:
            self._record_timeout(attempt)
            return None

    def _record_timeout(self, attempt: JobAttempt) -> None:
        """Log + streak-track a status-check timeout."""
        logger.warning(
            f"Status check timed out for attempt {attempt.attempt_id} "
            f"after {self.status_check_timeout}s"
        )
        METRICS.monitor_check("timeout")
        # Track consecutive-timeout streaks for escalation (≥10 → degradation)
        streak = self._timeout_streaks.get(attempt.attempt_id, 0) + 1
        self._timeout_streaks[attempt.attempt_id] = streak
        if streak >= 10:
            METRICS.degradation("monitor", 1)
            logger.error(
                f"Attempt {attempt.attempt_id}: {streak} consecutive "
                f"status-check timeouts — monitor degradation"
            )

    def _record_unknown(self, attempt: JobAttempt) -> None:
        """Log + count the checker-returned-None (unknown execution) case.

        Checker could not resolve this attempt's execution (e.g. execution
        registry record missing after a restart). Skip without crashing and
        without falsely marking failed; lost attempts belong to the reconciler.
        """
        self._stats["attempts_unknown"] = self._stats.get("attempts_unknown", 0) + 1
        logger.warning(
            f"Attempt {attempt.attempt_id} status unknown "
            f"(checker returned None; execution record unavailable) — skipping this cycle"
        )
        METRICS.monitor_check("unknown")

    def _apply_success(self, attempt: JobAttempt, result) -> None:
        """Mark attempt succeeded + commit if requested."""
        self._timeout_streaks.pop(attempt.attempt_id, None)
        self.job_store.mark_attempt_succeeded(
            attempt.attempt_id,
            commit=result.should_commit,
            metrics=result.metrics,
        )
        self._stats["attempts_succeeded"] += 1
        METRICS.monitor_check("succeeded")
        logger.info(f"Attempt {attempt.attempt_id} succeeded")
        self._autopin_result_artifacts(attempt)

    def _autopin_result_artifacts(self, attempt: JobAttempt) -> None:
        """D-43 result-kind auto-pin: on attempt success, pin the job's
        result artifacts onto the project's single active catalog
        (kind=result, ref=registry artifact id). Best-effort — a pin
        never blocks completion.
        """
        try:
            job = self.job_store.get_job(attempt.job_id)
            project_id = (getattr(job, "project_id", "") or "").strip()
            if not project_id:
                return
            artifacts = self._result_artifacts(attempt.job_id)
            if not artifacts:
                return
            self._pin_results_onto_single_catalog(project_id, attempt.job_id, artifacts)
        except Exception as exc:  # noqa: BLE001 — pin is best-effort, never a gate
            logger.warning("result autopin skipped for job %s: %s", attempt.job_id, exc)

    def _pin_results_onto_single_catalog(
        self, project_id: str, job_id: str, artifacts: list
    ) -> None:
        """Append one pin per result artifact ref (deduped, never guessed).

        Same never-guess rule as the submit-time auto-pin: zero catalogs →
        nothing to associate against; several → ambiguous, pin by hand
        with `r4 catalog pin`. Refs already on the catalog are skipped.
        """
        from umrs.scientific_catalogs.store import ScientificCatalogStore

        with self.job_store._get_conn() as conn:  # jobs DB conn, same seam as api/jobs._autopin_catalog
            store = ScientificCatalogStore(conn)
            catalogs = store.list(project_id=project_id)
            if len(catalogs) != 1:
                return
            catalog_id = catalogs[0]["catalog_id"]
            row = store.get(catalog_id)
            already = {
                p.get("ref") for p in (row.get("provenance") or {}).get("pins", [])
            }
            pinned = 0
            for art in artifacts:
                ref = str(getattr(art, "id", "") or "")
                if not ref or ref in already:
                    continue
                store.add_pin(catalog_id, kind="result", ref=ref, note=f"job {job_id}")
                pinned += 1
        if pinned:
            logger.info(
                "result autopin: %d artifact(s) of job %s onto %s",
                pinned,
                job_id,
                catalog_id,
            )

    @staticmethod
    def _result_artifacts(job_id: str) -> list:
        """Registry result artifacts for the job (knowledge DSN, best-effort)."""
        try:
            from umrs.artifact_registry.models import ArtifactQuery
            from umrs.artifact_registry.store import ArtifactStore

            conn = _open_knowledge_conn()
            if conn is None:
                return []
            try:
                return ArtifactStore(lambda: conn).list(
                    ArtifactQuery(artifact_type="result", job_id=job_id, limit=50)
                )
            finally:
                conn.close()
        except Exception as exc:  # noqa: BLE001
            logger.warning("result artifact lookup failed for job %s: %s", job_id, exc)
            return []

    def _apply_failure(self, attempt: JobAttempt, result) -> None:
        """Mark attempt failed (job store decides on requeue)."""
        self._timeout_streaks.pop(attempt.attempt_id, None)
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

    def _apply_running(self, attempt: JobAttempt, result) -> None:
        """Refresh status detail / metrics for an in-flight attempt."""
        self._timeout_streaks.pop(attempt.attempt_id, None)
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
        if result.metrics:
            self.job_store.update_attempt_metrics(
                attempt.attempt_id,
                result.metrics,
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
                    "ready_at": datetime.now(timezone.utc).isoformat(),
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
                        metrics={"ready_at": datetime.now(timezone.utc).isoformat()},
                        commit=True,
                    )
                else:
                    return AttemptCheckResult.running(
                        detail=f"polling readiness ({attempt.status_detail or 'waiting'})"
                    )

            except Exception as e:  # noqa: BLE001 - readiness poll must classify any backend error
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
