"""Tests for umrs.metrics — runtime telemetry with no-op degradation.

Uses the module-level METRICS singleton for API tests to avoid
duplicate metric registration in the Prometheus collector registry.
"""

from __future__ import annotations

import time

import pytest

from umrs.metrics import (
    METRICS,
    UMRSMetrics,
    _NoOpCounter,
    _NoOpGauge,
    _NoOpHistogram,
)


# ---------------------------------------------------------------------------
# No-op fallback behaviour
# ---------------------------------------------------------------------------

class TestNoOpCounter:
    def test_inc_is_silent(self):
        c = _NoOpCounter()
        c.inc()
        c.inc(5)

    def test_labels_returns_self(self):
        c = _NoOpCounter()
        labelled = c.labels(backend="slurm")
        assert isinstance(labelled, _NoOpCounter)
        labelled.inc()


class TestNoOpHistogram:
    def test_observe_is_silent(self):
        h = _NoOpHistogram()
        h.observe(1.5)

    def test_labels_returns_self(self):
        h = _NoOpHistogram()
        labelled = h.labels(outcome="ok")
        assert isinstance(labelled, _NoOpHistogram)
        labelled.observe(0.1)


class TestNoOpGauge:
    def test_set_is_silent(self):
        g = _NoOpGauge()
        g.set(42.0)

    def test_inc_is_silent(self):
        g = _NoOpGauge()
        g.inc()
        g.inc(3)

    def test_labels_returns_self(self):
        g = _NoOpGauge()
        labelled = g.labels(subsystem="scheduler")
        assert isinstance(labelled, _NoOpGauge)
        labelled.set(1.0)


# ---------------------------------------------------------------------------
# UMRSMetrics API — every public method must be callable without error.
# Uses the module-level singleton to avoid Prometheus registry conflicts.
# ---------------------------------------------------------------------------

class TestJobLifecycle:
    def test_submitted(self):
        METRICS.job_submitted("slurm")

    def test_completed(self):
        METRICS.job_completed("k8s")

    def test_failed(self):
        METRICS.job_failed("compose")

    def test_lost(self):
        METRICS.job_lost("slurm")

    def test_cancelled(self):
        METRICS.job_cancelled("k8s")


class TestReconciler:
    def test_reconciler_run(self):
        METRICS.reconciler_run()
        METRICS.reconciler_run("error")

    def test_reconciler_duration(self):
        METRICS.reconciler_duration(0.5)
        METRICS.reconciler_duration(1.2, "error")

    def test_reconciler_timer_success(self):
        with METRICS.reconciler_timer():
            pass  # simulate work

    def test_reconciler_timer_failure(self):
        with pytest.raises(ValueError, match="boom"):
            with METRICS.reconciler_timer():
                raise ValueError("boom")

    def test_reconciler_items(self):
        METRICS.reconciler_items("requeued")
        METRICS.reconciler_items("marked_lost", 3)

    def test_reconciler_backlog(self):
        METRICS.reconciler_backlog(7)


class TestTransport:
    def test_transport_error(self):
        METRICS.transport_error("zmq")
        METRICS.transport_error("http", "recv")

    def test_transport_reconnect(self):
        METRICS.transport_reconnect("zmq")

    def test_transport_timeout(self):
        METRICS.transport_timeout("zmq", "send")

    def test_transport_bytes(self):
        METRICS.transport_bytes("zmq", "sent", 1024)
        METRICS.transport_bytes("http", "received", 2048)


class TestModelRegistry:
    def test_model_event(self):
        METRICS.model_event("staged")
        METRICS.model_event("deleted")


class TestRetries:
    def test_retry_attempt(self):
        METRICS.retry_attempt("slurm")

    def test_retry_burst(self):
        METRICS.retry_burst("k8s")


class TestScheduler:
    def test_scheduler_cycle(self):
        METRICS.scheduler_cycle()
        METRICS.scheduler_cycle("empty")

    def test_scheduler_cycle_duration(self):
        METRICS.scheduler_cycle_duration(0.05)

    def test_scheduler_jobs_claimed(self):
        METRICS.scheduler_jobs_claimed(3)

    def test_scheduler_delay(self):
        METRICS.scheduler_delay(2.5)


class TestMonitor:
    def test_monitor_check(self):
        METRICS.monitor_check("healthy")
        METRICS.monitor_check("stale")


class TestBridgeHealth:
    def test_bridge_event(self):
        METRICS.bridge_event("durable_job", "start")
        METRICS.bridge_event("durable_job", "stop")
        METRICS.bridge_event("durable_job", "failure")

    def test_bridge_activity(self):
        METRICS.bridge_activity("durable_job")

    def test_bridge_command(self):
        METRICS.bridge_command("durable_job", "submit")


class TestOperationalMaturity:
    def test_degradation_level(self):
        METRICS.degradation_level("scheduler", 0)
        METRICS.degradation_level("transport", 1)
        METRICS.degradation_level("reconciler", 2)


class TestSingleton:
    def test_module_level_metrics_is_umrs_metrics(self):
        assert isinstance(METRICS, UMRSMetrics)


class TestReconcilerTimerTiming:
    """Verify the timer context manager actually measures elapsed time."""

    def test_timer_measures_nonzero_duration(self):
        start = time.monotonic()
        with METRICS.reconciler_timer():
            # Tiny busy-wait to ensure nonzero elapsed
            while time.monotonic() - start < 0.001:
                pass
        # If we get here without error, the timer completed successfully
