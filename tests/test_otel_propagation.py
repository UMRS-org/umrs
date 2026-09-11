"""WP-12 r2: OTel propagation hooks (job env + HTTP write-back injection)."""

from __future__ import annotations

import sys
import types

from umrs.jobs.cluster_venue import _otel_job_env
from umrs.knowledge.http_loop_store import _inject_trace_context


def _stub_ot(carrier_out: dict[str, str]):
    propagate = types.ModuleType("opentelemetry.propagate")
    propagate.inject = lambda carrier: carrier.update(carrier_out)
    return propagate


def test_job_env_empty_without_endpoint(monkeypatch):
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    assert _otel_job_env() == {}


def test_job_env_carries_endpoint_and_traceparent(monkeypatch):
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector:4318")
    monkeypatch.setitem(
        sys.modules, "opentelemetry.propagate",
        _stub_ot({"traceparent": "00-ab12-ef34-01"}),
    )
    env = _otel_job_env()
    assert env["OTEL_EXPORTER_OTLP_ENDPOINT"] == "http://collector:4318"
    assert env["TRACEPARENT"] == "00-ab12-ef34-01"


def test_job_env_skips_traceparent_when_no_parent_span(monkeypatch):
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector:4318")
    monkeypatch.setitem(sys.modules, "opentelemetry.propagate", _stub_ot({}))
    env = _otel_job_env()
    assert "TRACEPARENT" not in env  # never fabricated


def test_http_store_no_injection_without_endpoint(monkeypatch):
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    headers: dict[str, str] = {}
    _inject_trace_context(headers)
    assert headers == {}


def test_http_store_injects_traceparent_when_live(monkeypatch):
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector:4318")
    monkeypatch.setitem(
        sys.modules, "opentelemetry.propagate",
        _stub_ot({"traceparent": "00-feed-beef-99"}),
    )
    headers: dict[str, str] = {}
    _inject_trace_context(headers)
    assert headers["traceparent"] == "00-feed-beef-99"


def test_http_store_import_failure_keeps_headers_unchanged(monkeypatch):
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector:4318")
    monkeypatch.item_deleted = True
    monkeypatch.setitem(sys.modules, "opentelemetry.propagate", None)
    headers: dict[str, str] = {"X-User-ID": "u"}
    _inject_trace_context(headers)
    assert headers == {"X-User-ID": "u"}
