"""Shared harness for the G-003 transport-parity audit (Q-04).

Approach (evidence-driven, never "assume parity"):

- Run each P0 operation through the HTTP (gateway-shaped) route using a real
  FastAPI TestClient against a spy `AuditStore`.
- Run the same logical operation through the ZMQ-direct handler with the
  generic ZMQ payload shape the server expects.
- Compare rows on the fields the transport parity matrix says should match:
  event name, project binding, actor binding, outcome. Transport tag and
  request_id correlation are EXPECTED to differ and are NOT compared.

Findings semantic:

- ``expected==parity`` means the matrix's invariant held; regression-proof.
- ``expected==divergence(<reason>)`` means we prove the invariant is violated IN
  CODE TODAY — register the code-resilient gap rather than hiding it behind
  ``pytest.xfail`` noise. Each divergence is keyed as ``D-<n>`` so future M10
  items suck it up by path patch (closing the gap flips the test green).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SpyAuditStore:
    """Record every persistent-write row the platform emits."""

    writes: list[dict[str, Any]] = field(default_factory=list)

    def write(self, record: dict[str, Any]) -> None:
        self.writes.append(dict(record))

    # AuditEvents-only helpers used by the API
    def write_many(self, records: list[dict[str, Any]]) -> None:
        for r in records:
            self.write(r)

    def rows_for_event(self, event: str) -> list[dict[str, Any]]:
        return [w for w in self.writes if w.get("event") == event]


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    """The audit fields the parity matrix says must match between transports."""
    return {
        "event": row.get("event"),
        "project_id": row.get("project_id") or "",
        "actor": row.get("actor") or row.get("user_id") or "",
        "outcome": row.get("outcome") or "success",
    }


def compare_rows(
    http_rows: list[dict[str, Any]],
    zmq_rows: list[dict[str, Any]],
) -> tuple[bool, str]:
    """Return (matches, explain-string) for a transport pair.

    Rules: both sides must emit at least one row (zero-row on either side counts
    as a called-out divergence with the exact offending transport). Otherwise
    every normalized HTTP row must have a normalized ZMQ twin.
    """
    if not http_rows and not zmq_rows:
        return False, "neither transport emitted audit rows (both silent?)"
    if not http_rows:
        return False, "HTTP emitted ZERO audit rows for the operation"
    if not zmq_rows:
        return False, "ZMQ emitted ZERO audit rows for the operation"
    nz_http = [normalize_row(r) for r in http_rows]
    nz_zmq = [normalize_row(r) for r in zmq_rows]
    missing = [h for h in nz_http if h not in nz_zmq]
    extra = [z for z in nz_zmq if z not in nz_http]
    if not missing and not extra:
        return True, "parity: rows match on the matrix-normalized shape"
    why = []
    if missing:
        why.append(f"ZMQ missing rows the HTTP emitted: {missing}")
    if extra:
        why.append(f"HTTP missing rows the ZMQ emitted: {extra}")
    return False, "; ".join(why)
