"""Attempt/loop phase timing (WP-12, G-P11-1, G-P1-4).

Computes per-attempt and per-iteration wall-clock phases from the durable
rows (pure functions, tolerant of absent timestamps — an unknown phase is
recorded as None, never invented). The job store observes these into the
metric registry and persists phase records under attempts.metrics['timing'];
the loop runtime does the same per iteration (results['timing_ms']).

Phases measurable without schema changes:
- ``queue_wait``: attempt created → running
- ``wall``: attempt created → terminal end (succeeded/failed)
- ``iteration``: loop iteration started_at → completed_at
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def _parse_iso(value: Any) -> datetime | None:
    """Tolerant ISO-8601 parse (aware or naive → aware UTC)."""
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str) and value:
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            return None
    else:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def duration_ms(started: Any, ended: Any) -> int | None:
    """Milliseconds between two timestamps, or None when either is unknown.

   ``ended`` never precedes ``started`` on the wire (clock skew floors at 0).
    """
    start = _parse_iso(started)
    end = _parse_iso(ended)
    if start is None or end is None:
        return None
    return max(0, int((end - start).total_seconds() * 1000))


def attempt_phases(
    *,
    started_at: Any,
    transitioned_at: Any,
    phase: str,
) -> dict[str, int | None]:
    """One transition's phase record.

    - ``queue_wait_ms`` when phase == "running"
    - ``wall_ms`` when phase == "ended"
    Nothing is fabricated: unknown sides → None field.
    """
    ms = duration_ms(started_at, transitioned_at)
    if phase == "running":
        return {"queue_wait_ms": ms}
    return {"wall_ms": ms}


def maybe_emit_span(
    name: str,
    *,
    started_at: Any,
    transitioned_at: Any,
    attributes: dict[str, Any] | None = None,
) -> bool:
    """Emit ONE OTel span covering [started_at, transitioned_at] (WP-12 r2).

    Generic hook shared by attempt-phase and loop-iteration emissions:
    window comes from the durable rows (skew-floored at 0, ns stamps);
    attributes are passed verbatim (ids ride spans — the cardinality ban
    is metrics-only). Hard no-ops: empty/incoherent window, otlp endpoint
    unset, otel SDK absent, tracer failure. Returns True only when a span
    was actually exported-scheduled.
    """
    try:
        from umrs.shared.otel_flag import otel_active

        if not otel_active():
            return False
        from opentelemetry import trace
    except (ImportError, AttributeError, RuntimeError):
        # tracer stack absent/broken — never block the runtime path
        return False
    start = _parse_iso(started_at)
    end = _parse_iso(transitioned_at)
    if start is None or end is None:
        return False
    start_ns = int(start.timestamp() * 1e9)
    end_ns = max(int(end.timestamp() * 1e9), start_ns)
    try:
        tracer = trace.get_tracer("umrs")
        span = tracer.start_span(name, start_time=start_ns, attributes=attributes or {})
        span.end(end_time=end_ns)
        return True
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return False


def maybe_emit_attempt_span(
    *,
    job_id: str,
    attempt_id: str,
    attempt_no: int,
    backend: str,
    job_type: str = "",
    started_at: Any,
    transitioned_at: Any,
    fields: dict[str, int],
) -> bool:
    """Emit ONE OTel span per recorded phase field (WP-12 round 2, G-P11-1).

    Span timing = the measured phase window [started_at, transitioned_at]
    (ns timestamps from the durable rows — skew-floored at 0 just like the
    metric path). Span-carried ids are the same ones the lineage tables
    carry (job/attempt ids are fine on spans; they were forbidden on
    *metrics* for cardinality, never on traces).

    Hard no-ops besides the fields check: OTel disabled, otel SDK absent,
    tracer failure — observability never blocks attempt state transitions.
    Returns True only when all spans were exported-scheduled.
    """
    if not fields:
        return False
    all_ok = True
    for field_name in fields:
        ok = maybe_emit_span(
            f"umrs.attempt.{field_name.removesuffix('_ms')}",
            started_at=started_at,
            transitioned_at=transitioned_at,
            attributes={
                "umrs.job_id": str(job_id),
                "umrs.attempt_id": str(attempt_id),
                "umrs.attempt_no": int(attempt_no),
                "umrs.backend": str(backend or "unknown"),
                "umrs.job_type": str(job_type or "unknown"),
                "umrs.authority_bounded": True,
            },
        )
        all_ok = all_ok and ok
    return all_ok
