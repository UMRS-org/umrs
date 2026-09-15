"""Tests for AttemptMonitor timeout handling and consecutive-timeout escalation.

Covers:
- UMRS_STATUS_CHECK_TIMEOUT env var override
- Streak tracking for consecutive timeouts
- Streak degradation at >=10
- Streak clearing on non-timeout outcomes
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from umrs.jobs.monitor import AttemptCheckResult, AttemptMonitor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_job():
    return MagicMock(job_id="job-1", attempts=[])


@pytest.fixture
def fake_attempt():
    attempt = MagicMock()
    attempt.attempt_id = "attempt-1"
    attempt.job_id = "job-1"
    attempt.status = "submitted"
    return attempt


@pytest.fixture
def job_store():
    store = MagicMock()
    store.get_active_attempts.return_value = []
    store.get_job.return_value = MagicMock(job_id="job-1")
    store.mark_attempt_succeeded.return_value = True
    store.mark_attempt_failed.return_value = False
    store.mark_attempt_running.return_value = True
    return store


@pytest.fixture
def status_checker():
    checker = MagicMock()
    checker.check_status = AsyncMock()
    return checker


@pytest.fixture
def monitor(job_store, status_checker):
    return AttemptMonitor(
        job_store=job_store,
        status_checker=status_checker,
        poll_interval=0.01,
    )


# ---------------------------------------------------------------------------
# Env-var timeout override
# ---------------------------------------------------------------------------


class TestTimeoutConfig:
    """Verify UMRS_STATUS_CHECK_TIMEOUT env override and default."""

    def test_default_timeout_60(self):
        checker = MagicMock()
        checker.check_status = AsyncMock()
        m = AttemptMonitor(job_store=MagicMock(), status_checker=checker)
        assert m.status_check_timeout == 60.0

    @patch.dict("os.environ", {"UMRS_STATUS_CHECK_TIMEOUT": "12.5"})
    def test_env_override(self):
        checker = MagicMock()
        checker.check_status = AsyncMock()
        m = AttemptMonitor(job_store=MagicMock(), status_checker=checker)
        assert m.status_check_timeout == 12.5

    @patch.dict("os.environ", {"UMRS_STATUS_CHECK_TIMEOUT": "0.01"})
    def test_subsecond_timeout(self):
        checker = MagicMock()
        checker.check_status = AsyncMock()
        m = AttemptMonitor(job_store=MagicMock(), status_checker=checker)
        assert m.status_check_timeout == 0.01


# ---------------------------------------------------------------------------
# Timeout branch execution
# ---------------------------------------------------------------------------


class TestTimeoutBranch:
    """Verify that timeout branches are taken and streaks tracked."""

    @pytest.mark.asyncio
    async def test_single_timeout_tracked(self, monitor, fake_attempt):
        monitor.status_checker.check_status = AsyncMock(side_effect=asyncio.TimeoutError)
        monitor.status_check_timeout = 0.01
        monitor._timeout_streaks.clear()

        with patch("umrs.jobs.monitor.METRICS"):
            await monitor._check_attempt(fake_attempt)

        # Streak should be 1 after one timeout
        assert monitor._timeout_streaks.get("attempt-1") == 1

    @pytest.mark.asyncio
    async def test_sixty_timeout_triggers_degradation(
        self, job_store, status_checker, fake_attempt
    ):
        """>=10 consecutive timeouts should trigger degradation."""
        monitor = AttemptMonitor(
            job_store=job_store,
            status_checker=status_checker,
            poll_interval=0.001,
        )
        # Pre-seed streak at 8 so two more timeouts reach 10
        monitor._timeout_streaks["attempt-1"] = 8
        monitor.status_check_timeout = 0.001
        status_checker.check_status = AsyncMock(side_effect=asyncio.TimeoutError)

        with patch("umrs.jobs.monitor.METRICS") as mock_metrics:
            await monitor._check_attempt(fake_attempt)
            # First timeout → streak 9, no degradation yet
            assert monitor._timeout_streaks["attempt-1"] == 9
            mock_metrics.degradation.assert_not_called()

            await monitor._check_attempt(fake_attempt)
            # Second timeout → streak 10, degradation fires
            assert monitor._timeout_streaks["attempt-1"] == 10
            assert mock_metrics.degradation.call_count == 1

    @pytest.mark.asyncio
    async def test_succeeded_clears_streak(self, monitor, fake_attempt):
        monitor._timeout_streaks["attempt-1"] = 5
        monitor.status_checker.check_status = AsyncMock(
            return_value=AttemptCheckResult(status="succeeded")
        )
        with patch("umrs.jobs.monitor.METRICS"):
            await monitor._check_attempt(fake_attempt)

        # Streak should be cleared and success recorded
        assert monitor._timeout_streaks.get("attempt-1") is None
        monitor.job_store.mark_attempt_succeeded.assert_called_once()


# ---------------------------------------------------------------------------
# Integration: start/stop with timeout config
# ---------------------------------------------------------------------------


class TestMonitorLifecycle:
    """End-to-end start/stop with env-configured timeout."""

    @pytest.mark.asyncio
    async def test_monitor_runs_with_custom_timeout(self):
        job_store = MagicMock()
        job_store.get_active_attempts.return_value = []
        checker = AsyncMock()
        monitor = AttemptMonitor(
            job_store=job_store,
            status_checker=checker,
            poll_interval=0.01,
        )

        await monitor.start()
        await asyncio.sleep(0.05)
        await monitor.stop(timeout=1.0)

        assert monitor._running is False
        monitor._task.cancel()

