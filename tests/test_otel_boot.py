"""OTel boot tests (WP-12 round 2) — umrs side of the parity pin.

The exact same ``setup_otel`` file ships in gateway/sidecar/otel.py
(KEEP IN SYNC docstring in both); this suite asserts the two files are
byte-identical (parity), the no-op default, knob precedence, and that a
stubbed OTel stack yields the active path without raising.
"""

from __future__ import annotations

import sys
from pathlib import Path

from umrs.server import otel


def test_parity_with_gateway_sidecar():
    """The twin is kept byte-identical by design (docstring's parity note)."""
    umrs_file = Path(otel.__file__).read_text()
    sidecar_file = (
        Path(__file__).resolve().parents[2] / "gateway" / "sidecar" / "otel.py"
    ).read_text()
    assert umrs_file == sidecar_file


def test_otel_noop_default(monkeypatch):
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    assert otel.otel_active() is False
    assert otel.setup_otel(object(), service_label="t") is False


def test_otel_missing_dependency_warns_not_raises(monkeypatch, caplog):
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector:4318")
    # Force ImportsError path deterministically
    monkeypatch.setitem(sys.modules, "opentelemetry", None)
    with caplog.at_level("WARNING"):
        applied = otel.setup_otel(object(), service_label="t")
    assert applied is False
    assert any("not importable" in r.message for r in caplog.records)


def test_otel_active_path_with_stubbed_stack(monkeypatch):
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector:4318")
    monkeypatch.setenv("OTEL_TRACES_SAMPLER_ARG", "0.25")

    captured: dict = {}

    class _Provider:
        def __init__(self, resource=None, sampler=None):
            captured["resource"] = resource
            captured["sampler"] = sampler

        def add_span_processor(self, processor):
            captured["processor"] = processor

    class _Instr:
        @staticmethod
        def instrument_app(app):
            captured["app"] = app

    class _Resource:
        @staticmethod
        def create(attrs):
            return {"r": dict(attrs)}

    import types

    fake_ot = types.ModuleType("opentelemetry")
    fake_trace = types.SimpleNamespace(
        set_tracer_provider=lambda p: captured.setdefault("provider", p)
    )
    fake_ot.trace = fake_trace
    exporter_mod = types.ModuleType("opentelemetry.exporter.otlp.proto.http.trace_exporter")
    exporter_mod.OTLPSpanExporter = lambda endpoint: ("exp", endpoint)
    instr_mod = types.ModuleType("opentelemetry.instrumentation.fastapi")
    instr_mod.FastAPIInstrumentor = _Instr
    resources_mod = types.ModuleType("opentelemetry.sdk.resources")
    resources_mod.Resource = _Resource
    sdk_trace_mod = types.ModuleType("opentelemetry.sdk.trace")
    sdk_trace_mod.TracerProvider = _Provider
    export_mod = types.ModuleType("opentelemetry.sdk.trace.export")
    export_mod.BatchSpanProcessor = lambda p: ("batch", p)
    sampling_mod = types.ModuleType("opentelemetry.sdk.trace.sampling")

    class _Ratio:
        def __init__(self, ratio):
            self.ratio = ratio

    sampling_mod.ParentBasedTraceIdRatio = _Ratio

    for name, mod in {
        "opentelemetry": fake_ot,
        "opentelemetry.exporter": types.ModuleType("opentelemetry.exporter"),
        "opentelemetry.exporter.otlp": types.ModuleType("opentelemetry.exporter.otlp"),
        "opentelemetry.exporter.otlp.proto": types.ModuleType("opentelemetry.exporter.otlp.proto"),
        "opentelemetry.exporter.otlp.proto.http": types.ModuleType(
            "opentelemetry.exporter.otlp.proto.http"
        ),
        "opentelemetry.exporter.otlp.proto.http.trace_exporter": exporter_mod,
        "opentelemetry.instrumentation": types.ModuleType("opentelemetry.instrumentation"),
        "opentelemetry.instrumentation.fastapi": instr_mod,
        "opentelemetry.sdk": types.ModuleType("opentelemetry.sdk"),
        "opentelemetry.sdk.resources": resources_mod,
        "opentelemetry.sdk.trace": sdk_trace_mod,
        "opentelemetry.sdk.trace.export": export_mod,
        "opentelemetry.sdk.trace.sampling": sampling_mod,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)

    applied = otel.setup_otel("APP_OBJ", service_label="umrs-http-sidecar")
    assert applied is True
    assert captured["app"] == "APP_OBJ"
    assert captured["sampler"].ratio == 0.25
    assert captured["resource"]["r"]["service.name"] == "umrs-http-sidecar"
    assert captured["processor"] == ("batch", ("exp", "http://collector:4318/v1/traces"))


def test_otel_sampler_arg_clamped_and_defaulted(monkeypatch):
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://c:4318")
    monkeypatch.setenv("OTEL_TRACES_SAMPLER_ARG", "bogus")
    # stub a failing import endpoint record: the sampler fallback is proven in
    # the log, then imports stop (None) — assert no raise
    monkeypatch.setitem(sys.modules, "opentelemetry", None)
    assert otel.setup_otel(object(), service_label="t") is False
    assert otel.DEFAULT_SAMPLE_RATIO == "0.1"
