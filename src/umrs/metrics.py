"""UMRS operational metrics — baseline telemetry for runtime health.

Gracefully degrades to no-ops when ``prometheus_client`` is not installed.

All labels are low-cardinality constants (backend, status, transport).
No user-derived identifiers are ever used as label values.

Usage::

    from umrs.metrics import METRICS
    METRICS.job_submitted("slurm")
    METRICS.job_completed("slurm")
    METRICS.job_failed("slurm")
    METRICS.reconciler_run()
    METRICS.transport_error("zmq")
    METRICS.model_event("staged")
    METRICS.retry_attempt("slurm")

Scraping:
    If ``prometheus_client`` is available, expose via its built-in HTTP
    server or integrate with an existing ASGI/WSGI app using
    ``prometheus_client.make_asgi_app()`` or ``generate_latest()``.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager

logger = logging.getLogger(__name__)

try:
    from prometheus_client import Counter as _Counter, Histogram as _Histogram

    _HAS_PROMETHEUS = True
except ImportError:  # pragma: no cover
    _HAS_PROMETHEUS = False


class _NoOpCounter:
    """Drop-in replacement when prometheus_client is unavailable."""

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        pass

    def labels(self, **_kw: object) -> "_NoOpCounter":
        return self

    def inc(self, amount: float = 1) -> None:  # noqa: ARG002
        pass


class _NoOpHistogram:
    """Drop-in replacement when prometheus_client is unavailable."""

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        pass

    def labels(self, **_kw: object) -> "_NoOpHistogram":
        return self

    def observe(self, amount: float) -> None:  # noqa: ARG002
        pass


def _counter(name: str, doc: str, labels: list[str]) -> object:
    if _HAS_PROMETHEUS:
        return _Counter(name, doc, labels)
    return _NoOpCounter()


def _histogram(name: str, doc: str, labels: list[str]) -> object:
    if _HAS_PROMETHEUS:
        return _Histogram(name, doc, labels)
    return _NoOpHistogram()


class _NoOpGauge:
    """Drop-in replacement when prometheus_client is unavailable."""

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        pass

    def labels(self, **_kw: object) -> "_NoOpGauge":
        return self

    def set(self, value: float) -> None:  # noqa: ARG002, A003
        pass

    def inc(self, amount: float = 1) -> None:  # noqa: ARG002
        pass


def _gauge(name: str, doc: str, labels: list[str]) -> object:
    if _HAS_PROMETHEUS:
        from prometheus_client import Gauge as _Gauge

        return _Gauge(name, doc, labels)
    return _NoOpGauge()


class UMRSMetrics:
    """Singleton metric registry for UMRS runtime health."""

    def __init__(self) -> None:
        self._jobs = _counter(
            "umrs_job_total",
            "Job lifecycle events",
            ["backend", "event"],
        )
        self._reconciler = _counter(
            "umrs_reconciler_runs_total",
            "Reconciler execution count",
            ["outcome"],
        )
        # -- Batch 4: Observability expansion ----------------------------
        # Reconciler latency with explicit buckets: 0.1s, 0.5s, 1s, 5s, 10s+
        self._reconciler_latency = _histogram(
            "umrs_reconciler_latency_seconds",
            "Reconciler loop iteration latency",
            ["outcome"],
        )
        # Queue depth: pending reconciliations
        self._reconciler_queue_depth = _gauge(
            "umrs_reconciler_queue_depth",
            "Number of pending reconciliation tasks",
            [],
        )
        # Durable job lifecycle metrics
        self._job_created = _counter(
            "umrs_jobs_created_total",
            "Jobs created (durable)",
            ["authority"],
        )
        self._attempt_started = _counter(
            "umrs_job_attempts_started_total",
            "Job attempts started (durable)",
            ["authority"],
        )
        self._attempt_completed = _counter(
            "umrs_job_attempts_completed_total",
            "Job attempts completed (durable)",
            ["authority", "status"],
        )
        self._job_status_gauge = _gauge(
            "umrs_jobs_status_count",
            "Active jobs grouped by status (durable)",
            ["status"],
        )
        self._attempt_status_gauge = _gauge(
            "umrs_job_attempts_status_count",
            "Active attempts grouped by status (durable)",
            ["status"],
        )
        # Reconciler reconnect counter (ZMQ reconnects)
        self._reconciler_reconnects = _counter(
            "umrs_reconciler_reconnects_total",
            "Reconciler ZMQ reconnect events",
            ["transport"],
        )
        # Health check duration gauge
        self._health_check_duration = _histogram(
            "umrs_health_check_duration_seconds",
            "Health check probe duration",
            ["check_name"],
        )

        self._reconciler_duration = _histogram(
            "umrs_reconciler_duration_seconds",
            "Time spent in reconciliation",
            ["outcome"],
        )
        self._transport = _counter(
            "umrs_transport_errors_total",
            "Transport-layer error count",
            ["transport", "operation"],
        )
        self._model = _counter(
            "umrs_model_events_total",
            "Model registry events",
            ["event"],
        )
        self._retries = _counter(
            "umrs_retry_attempts_total",
            "Job retry attempts",
            ["backend"],
        )
        self._scheduler = _counter(
            "umrs_scheduler_cycles_total",
            "Scheduler loop iterations",
            ["outcome"],
        )
        self._monitor = _counter(
            "umrs_monitor_checks_total",
            "Monitor attempt check outcomes",
            ["outcome"],
        )

        # -- Phase 5A: Bridge health -----------------------------------
        self._bridge_events = _counter(
            "umrs_bridge_events_total",
            "Bridge lifecycle events (start, stop, failure, reconnect)",
            ["bridge", "event"],
        )
        self._bridge_activity = _gauge(
            "umrs_bridge_last_activity_timestamp",
            "Unix timestamp of last activity through bridge",
            ["bridge"],
        )
        self._bridge_commands = _counter(
            "umrs_bridge_commands_total",
            "Commands dispatched through bridges",
            ["bridge", "command_class"],
        )

        # -- Phase 5B: Reconciliation maturity -------------------------
        self._reconciler_items = _counter(
            "umrs_reconciler_items_total",
            "Items processed during reconciliation",
            ["action"],
        )
        self._reconciler_backlog = _gauge(
            "umrs_reconciler_backlog_size",
            "Number of stale attempts found at boot",
            [],
        )

        # -- Phase 5C: Transport health --------------------------------
        self._transport_reconnects = _counter(
            "umrs_transport_reconnects_total",
            "Transport reconnect attempts",
            ["transport"],
        )
        self._transport_timeouts = _counter(
            "umrs_transport_timeouts_total",
            "Transport operation timeouts",
            ["transport", "operation"],
        )
        self._transport_bytes = _counter(
            "umrs_transport_bytes_total",
            "Bytes transferred through transport",
            ["transport", "direction"],
        )

        # -- Phase 5D: Scheduler pressure ------------------------------
        self._scheduler_duration = _histogram(
            "umrs_scheduler_cycle_duration_seconds",
            "Time spent in one scheduler poll cycle",
            [],
        )
        self._scheduler_claimed = _histogram(
            "umrs_scheduler_jobs_claimed",
            "Jobs claimed per scheduler cycle",
            [],
        )
        self._scheduler_delay = _histogram(
            "umrs_scheduler_delay_seconds",
            "Delay between job becoming ready and being scheduled",
            [],
        )

        # -- Phase 6C: Operational maturity ----------------------------
        self._retry_bursts = _counter(
            "umrs_retry_bursts_total",
            "Bursts of retries exceeding threshold in a window",
            ["backend"],
        )
        self._degradation = _gauge(
            "umrs_degradation_level",
            "Operational degradation level (0=healthy, 1=degraded, 2=unhealthy)",
            ["subsystem"],
        )

        # -- Queue depth instrumentation ---------------------------------
        self._queue_depth = _gauge(
            "umrs_queue_depth",
            "Number of messages currently queued",
            ["queue_type"],  # control | data
        )

        # -- LLM usage metering ----------------------------------------
        # source label enables cross-HPC profiling (bdrt, rccs, ai4s, atom, …).
        self._llm_tokens = _counter(
            "umrs_llm_tokens_total",
            "LLM tokens processed through the proxy, by direction and source",
            ["source", "model", "direction"],  # direction = prompt | completion
        )
        self._llm_request_duration = _histogram(
            "umrs_llm_request_duration_seconds",
            "End-to-end LLM proxy latency in seconds",
            ["source", "model"],
        )
        self._llm_requests = _counter(
            "umrs_llm_requests_total",
            "Total LLM proxy requests by source, model, streaming",
            ["source", "model", "streaming"],
        )
        # Per-user token breakdown (bounded cardinality on HPC — user set is finite).
        # direction: prompt | completion | cache_read | cache_write
        self._llm_tokens_user = _counter(
            "umrs_llm_tokens_user_total",
            "LLM tokens per user/project (prompt|completion|cache_read|cache_write)",
            ["source", "model", "user_id", "project_id", "direction"],
        )

        # Operational source — set once at startup by http_sidecar.
        self._source: str = ""

    # -- Job lifecycle -------------------------------------------------

    def job_submitted(self, backend: str) -> None:
        self._jobs.labels(backend=backend, event="submitted").inc()

    def job_completed(self, backend: str) -> None:
        self._jobs.labels(backend=backend, event="completed").inc()

    def job_failed(self, backend: str) -> None:
        self._jobs.labels(backend=backend, event="failed").inc()

    def job_lost(self, backend: str) -> None:
        self._jobs.labels(backend=backend, event="lost").inc()

    def job_cancelled(self, backend: str) -> None:
        self._jobs.labels(backend=backend, event="cancelled").inc()

    # -- Reconciler ----------------------------------------------------

    def reconciler_run(self, outcome: str = "ok") -> None:
        self._reconciler.labels(outcome=outcome).inc()

    def reconciler_duration(self, seconds: float, outcome: str = "ok") -> None:
        self._reconciler_duration.labels(outcome=outcome).observe(seconds)

    # -- Reconciler latency ------------------------------------------------

    def reconciler_latency(self, seconds: float, outcome: str = "ok") -> None:
        self._reconciler_latency.labels(outcome=outcome).observe(seconds)

    def set_reconciler_queue_depth(self, depth: int) -> None:
        self._reconciler_queue_depth.set(depth)

    def jobs_created(self, authority: str = "slurm") -> None:
        self._job_created.labels(authority=authority).inc()

    def job_attempts_started(self, authority: str = "slurm") -> None:
        self._attempt_started.labels(authority=authority).inc()

    def job_attempts_completed(
        self, authority: str = "slurm", status: str = "completed"
    ) -> None:
        self._attempt_completed.labels(authority=authority, status=status).inc()

    def set_job_status_count(self, status: str, count: int) -> None:
        self._job_status_gauge.labels(status=status).set(count)

    def set_attempt_status_count(self, status: str, count: int) -> None:
        self._attempt_status_gauge.labels(status=status).set(count)

    def reconciler_reconnect(self, transport: str = "zmq") -> None:
        self._reconciler_reconnects.labels(transport=transport).inc()

    def health_check_duration(self, check_name: str, seconds: float) -> None:
        self._health_check_duration.labels(check_name=check_name).observe(seconds)

    @contextmanager
    def reconciler_timer(self):
        """Context manager that records reconciler duration and outcome."""
        start = time.monotonic()
        outcome = "ok"
        try:
            yield
        except Exception:
            outcome = "error"
            raise
        finally:
            elapsed = time.monotonic() - start
            self.reconciler_run(outcome)
            self.reconciler_duration(elapsed, outcome)

    # -- Transport -----------------------------------------------------

    def transport_error(self, transport: str, operation: str = "send") -> None:
        self._transport.labels(transport=transport, operation=operation).inc()

    # -- Model registry ------------------------------------------------

    def model_event(self, event: str) -> None:
        self._model.labels(event=event).inc()

    # -- Retries -------------------------------------------------------

    def retry_attempt(self, backend: str) -> None:
        self._retries.labels(backend=backend).inc()

    # -- Scheduler -----------------------------------------------------

    def scheduler_cycle(self, outcome: str = "ok") -> None:
        self._scheduler.labels(outcome=outcome).inc()

    # -- Monitor -------------------------------------------------------

    def monitor_check(self, outcome: str) -> None:
        self._monitor.labels(outcome=outcome).inc()

    # -- Phase 5A: Bridge health ----------------------------------------

    def bridge_event(self, bridge: str, event: str) -> None:
        """Record a bridge lifecycle event (start, stop, failure, reconnect)."""
        self._bridge_events.labels(bridge=bridge, event=event).inc()

    def bridge_activity(self, bridge: str) -> None:
        """Update last-activity timestamp for a bridge."""
        self._bridge_activity.labels(bridge=bridge).set(time.time())

    def bridge_command(self, bridge: str, command_class: str) -> None:
        """Record a command dispatched through a bridge."""
        self._bridge_commands.labels(bridge=bridge, command_class=command_class).inc()

    # -- Phase 5B: Reconciliation maturity ------------------------------

    def reconciler_items(self, action: str, count: int = 1) -> None:
        """Record items processed during reconciliation."""
        self._reconciler_items.labels(action=action).inc(count)

    def reconciler_backlog(self, size: int) -> None:
        """Report the reconciliation backlog size found at boot."""
        self._reconciler_backlog.set(size)

    # -- Phase 5C: Transport health ------------------------------------

    def transport_reconnect(self, transport: str) -> None:
        """Record a transport reconnect attempt."""
        self._transport_reconnects.labels(transport=transport).inc()

    def transport_timeout(self, transport: str, operation: str) -> None:
        """Record a transport timeout event."""
        self._transport_timeouts.labels(transport=transport, operation=operation).inc()

    def transport_bytes(self, transport: str, direction: str, nbytes: int) -> None:
        """Record bytes transferred."""
        self._transport_bytes.labels(transport=transport, direction=direction).inc(
            nbytes
        )

    # -- Phase 5D: Scheduler pressure ----------------------------------

    def scheduler_cycle_duration(self, seconds: float) -> None:
        """Record the duration of one scheduler poll cycle."""
        self._scheduler_duration.observe(seconds)

    def scheduler_jobs_claimed(self, count: int) -> None:
        """Record how many jobs were claimed in a cycle."""
        self._scheduler_claimed.observe(count)

    def scheduler_delay(self, seconds: float) -> None:
        """Record delay between job ready time and scheduling time."""
        self._scheduler_delay.observe(seconds)

    # -- Phase 6C: Operational maturity --------------------------------

    def retry_burst(self, backend: str) -> None:
        """Record a retry burst event (many retries in short window)."""
        self._retry_bursts.labels(backend=backend).inc()

    def degradation_level(self, subsystem: str, level: int) -> None:
        """Set degradation level for a subsystem (0=healthy, 1=degraded, 2=unhealthy)."""
        self._degradation.labels(subsystem=subsystem).set(level)

    # -- LLM usage metering --------------------------------------------

    def set_source(self, source: str) -> None:
        """Set the operational source identifier (set once at startup).

        Called by http_sidecar during startup with the site name so that
        all LLM metrics carry the ``source`` label for cross-HPC profiling.
        """
        self._source = source

    def llm_tokens(
        self,
        model: str,
        prompt: int,
        completion: int,
        *,
        user_id: str = "",
        project_id: str = "",
        cached_read: int = 0,
        cached_write: int = 0,
        source: str = "",
    ) -> None:
        """Record prompt and completion tokens for an LLM request.

        ``user_id`` / ``project_id`` / ``cached_read`` / ``cached_write`` /
        ``source`` are optional keyword-only arguments.  When ``source`` is
        omitted the instance-level default set via :meth:`set_source` is used.
        When ``user_id`` is provided, the per-user counter
        ``umrs_llm_tokens_user_total`` is also updated with direction labels:
        prompt | completion | cache_read | cache_write.
        """
        s = source or self._source
        if prompt:
            self._llm_tokens.labels(source=s, model=model, direction="prompt").inc(
                prompt
            )
        if completion:
            self._llm_tokens.labels(source=s, model=model, direction="completion").inc(
                completion
            )
        if user_id:
            pid = project_id or "default"
            if prompt:
                self._llm_tokens_user.labels(
                    source=s,
                    model=model,
                    user_id=user_id,
                    project_id=pid,
                    direction="prompt",
                ).inc(prompt)
            if completion:
                self._llm_tokens_user.labels(
                    source=s,
                    model=model,
                    user_id=user_id,
                    project_id=pid,
                    direction="completion",
                ).inc(completion)
            if cached_read:
                self._llm_tokens_user.labels(
                    source=s,
                    model=model,
                    user_id=user_id,
                    project_id=pid,
                    direction="cache_read",
                ).inc(cached_read)
            if cached_write:
                self._llm_tokens_user.labels(
                    source=s,
                    model=model,
                    user_id=user_id,
                    project_id=pid,
                    direction="cache_write",
                ).inc(cached_write)

    def llm_request_duration(
        self, model: str, seconds: float, source: str = ""
    ) -> None:
        """Record end-to-end proxy latency."""
        s = source or self._source
        self._llm_request_duration.labels(source=s, model=model).observe(seconds)

    def llm_request(self, model: str, streaming: bool, source: str = "") -> None:
        """Increment LLM request counter."""
        s = source or self._source
        self._llm_requests.labels(
            source=s, model=model, streaming=str(streaming).lower()
        ).inc()

    def set_queue_depth(self, queue_type: str, depth: float) -> None:
        """Set the current message depth for a queue (control | data)."""
        self._queue_depth.labels(queue_type=queue_type).set(depth)


METRICS = UMRSMetrics()
