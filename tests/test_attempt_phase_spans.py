"""WP-12 round 2 — attempt phase spans on OTel (fail-without-change).

Span rules: no-op without OTEL_EXPORTER_OTLP_ENDPOINT; span window = the
durable phase window (ns timestamps, skew-floored); ids ride span
attributes (never metric labels).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

from umrs.jobs.timing import maybe_emit_attempt_span


class TestMaybeEmitAttemptSpan:
    def _base_kwargs(self):
        start = datetime(2026, 9, 11, 0, 0, tzinfo=timezone.utc)
        end = start + timedelta(seconds=7)
        return dict(
            job_id="job-1",
            attempt_id="att-1",
            attempt_no=2,
            backend="slurm",
            job_type="endpoint_provision",
            started_at=start,
            transitioned_at=end,
            fields={"queue_wait_ms": 7000},
        )

    def test_noop_without_otlp_endpoint(self, monkeypatch):
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
        assert maybe_emit_attempt_span(**self._base_kwargs()) is False

    def test_noop_on_empty_fields(self):
        kwargs = self._base_kwargs()
        kwargs["fields"] = {}
        assert maybe_emit_attempt_span(**kwargs) is False

    def test_noop_on_unparseable_window(self, monkeypatch):
        kwargs = self._base_kwargs()
        kwargs["started_at"] = "not-a-date"
        assert maybe_emit_attempt_span(**kwargs) is False

    def test_emits_span_with_window_and_ids(self, monkeypatch):
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://otel:4318")
        spans = []
        calls = []

        fake_span = MagicMock()
        spans.append(fake_span)

        def _start(name, *, start_time, attributes):
            calls.append((name, start_time, attributes))
            return fake_span

        fake_tracer = MagicMock()
        fake_tracer.start_span.side_effect = _start
        fake_trace = MagicMock()
        fake_trace.get_tracer.return_value = fake_tracer

        import sys
        import types

        otel_mod = types.ModuleType("umrs.server.otel")
        otel_mod.otel_active = lambda: True
        monkeypatch.setitem(sys.modules, "umrs.server.otel", otel_mod)
        fake_ot = types.SimpleNamespace(get_tracer=fake_trace.get_tracer)
        otel_pkg = types.ModuleType("opentelemetry")
        otel_pkg.trace = fake_ot.trace if hasattr(fake_ot, "trace") else fake_trace
        monkeypatch.setitem(sys.modules, "opentelemetry", otel_pkg)

        kwargs = self._base_kwargs()
        assert maybe_emit_attempt_span(**kwargs) is True

        assert len(calls) == 1
        name, start_ns, attrs = calls[0]
        assert name == "umrs.attempt.queue_wait"
        start = kwargs["started_at"]
        end = kwargs["transitioned_at"]
        assert start_ns == int(start.timestamp() * 1e9)
        assert fake_span.end.call_args.kwargs["end_time"] == int(end.timestamp() * 1e9)
        assert attrs["umrs.job_id"] == "job-1"
        assert attrs["umrs.attempt_id"] == "att-1"
        assert attrs["umrs.attempt_no"] == 2
        assert attrs["umrs.backend"] == "slurm"
        assert attrs["umrs.job_type"] == "endpoint_provision"

    def test_tracer_failure_still_noop(self, monkeypatch):
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://otel:4318")
        import sys
        import types

        otel_mod = types.ModuleType("umrs.server.otel")
        otel_mod.otel_active = lambda: True
        monkeypatch.setitem(sys.modules, "umrs.server.otel", otel_mod)
        boom_trace = MagicMock()
        boom_trace.get_tracer.side_effect = RuntimeError("sdk broken")
        otel_pkg = types.ModuleType("opentelemetry")
        otel_pkg.trace = boom_trace
        monkeypatch.setitem(sys.modules, "opentelemetry", otel_pkg)

        assert maybe_emit_attempt_span(**self._base_kwargs()) is False
