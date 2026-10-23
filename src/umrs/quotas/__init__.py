"""Remote-session quotas (ADR-099 D-4 to D-7, RS-1 lane Q).

- ``usage_tap``: token usage read from the inference proxy's relayed
  responses and written to the existing ``llm_usage_log`` ledger
  (``origin='infer_proxy'``, migration 0126).
- ``enforcer``: the proxy's daily-token and requests-per-minute checks,
  refused as a typed 429 (``quota.tokens_per_day_exceeded``,
  ``quota.rpm_exceeded``), fail closed as ``quota.unavailable`` (503).
- ``session_cap``: the per-user (and optional per-project) cap on live
  remote sessions at start (``quota.concurrent_sessions_exceeded``).

The limits come from the session policy (ADR-099 D-3); with no limit set the
proxy makes no extra query.
"""

from __future__ import annotations

__all__ = ["enforcer", "session_cap", "usage_tap"]
