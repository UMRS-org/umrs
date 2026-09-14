"""Foundation-facing accessor for the OTel feature knob (ADR-053).

`umrs.server.otel.otel_active` was lazily imported by `jobs/timing.py` — a
foundation back-edge. The feature-knob definition is pure env reading and
lives here in the foundation layer as the duplicate pair of
`umrs.server.otel.otel_active` (the umrs/src/umrs/server/otel.py copy keeps
byte-parity with `gateway/sidecar/otel.py` pinned in
umrs/tests/test_otel_boot.py; THIS file is the foundation address for
foundation consumers building from `umrs.shared`).

Parity contract: this definition must string-match the twin definitions in
umrs/src/umrs/server/otel.py and gateway/sidecar/otel.py.
"""

from __future__ import annotations

import os


def otel_active() -> bool:
    """Feature knob visible to probes/tests (endpoint set = intent)."""
    return bool(os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"))


__all__ = ["otel_active"]
