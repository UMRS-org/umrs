"""Optional OpenTelemetry tracing boot (WP-12 round 2).

No-op by default: tracing activates only when BOTH
- ``OTEL_EXPORTER_OTLP_ENDPOINT`` is set (e.g. ``http://otel-collector:4318``), and
- an opentelemetry stack is importable (the ``umrs[otel]`` extra).

Then the app is instrumented (FastAPI spans, W3C traceparent/tracestate
propagation, OTLP/HTTP JSON exporter), sampling at
``OTEL_TRACES_SAMPLER_ARG`` (default 0.1 = 10%). Service name comes from
``OTEL_SERVICE_NAME`` (fallback: the caller's label).

KEEP IN SYNC — a byte-identical copy of this module ships under both
``umrs/src/umrs/server/otel.py`` and ``gateway/sidecar/otel.py`` (the sidecar
image does not package umrs). umrs/tests/test_otel_boot.py pins parity.
"""


from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

DEFAULT_SAMPLE_RATIO = "0.1"


def otel_active() -> bool:
    """Feature knob visible to probes/tests (set endpoint = intent)."""
    return bool(os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"))


def setup_otel(app, *, service_label: str) -> bool:
    """Instrument ``app`` when enabled; returns True when applied.

    Never raises: missing dependency, unusable endpoint, or an OTel import
    hiccup all log a warning and leave the app untraced (deployment posture:
    observability must never take the platform down).
    """
    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()
    if not endpoint:
        logger.info("OTel tracing disabled (OTEL_EXPORTER_OTLP_ENDPOINT unset)")
        return False
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.sdk.trace.sampling import ParentBasedTraceIdRatio
    except ImportError as exc:
        logger.warning(
            "OTEL_EXPORTER_OTLP_ENDPOINT set but opentelemetry not importable (%s); "
            "install the otel extra to enable tracing",
            exc.name,
        )
        return False
    try:
        ratio_raw = os.environ.get("OTEL_TRACES_SAMPLER_ARG", DEFAULT_SAMPLE_RATIO)
        ratio = max(0.0, min(1.0, float(ratio_raw)))
    except ValueError:
        logger.warning(
            "OTEL_TRACES_SAMPLER_ARG=%r unparseable — falling back to %s",
            os.environ.get("OTEL_TRACES_SAMPLER_ARG"),
            DEFAULT_SAMPLE_RATIO,
        )
        ratio = float(DEFAULT_SAMPLE_RATIO)
    service_name = os.environ.get("OTEL_SERVICE_NAME") or service_label
    try:
        provider = TracerProvider(
            resource=Resource.create(
                {
                    "service.name": service_name,
                    "service.instance.id": os.uname().nodename,
                }
            ),
            sampler=ParentBasedTraceIdRatio(ratio),
        )
        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=f"{endpoint}/v1/traces"))
        )
        trace.set_tracer_provider(provider)
        FastAPIInstrumentor.instrument_app(app)
    except Exception:  # noqa: BLE001 — never take the app down for tracing
        logger.exception("OTel setup failed; continuing untraced")
        return False
    logger.info(
        "OTel tracing active: service=%s endpoint=%s ratio=%.3f",
        service_name,
        endpoint,
        ratio,
    )
    return True


__all__ = ["DEFAULT_SAMPLE_RATIO", "otel_active", "setup_otel"]
