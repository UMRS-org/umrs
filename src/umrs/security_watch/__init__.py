"""UMRS Security Watch: rules and detectors (ADR-088 W1, slice S-D61a).

A foundation module (ADR-088 D-1): it never imports ``umrs.api`` or
``umrs.server``, so the watch's rules and detectors carry no route,
lifespan or request state. This slice holds:

- :mod:`.rules`: the versioned rules pack, the packaged default
  (``rules/default.yaml``) plus a site overlay by rule id.
- :mod:`.detectors`: one parameterised ``SELECT`` per rule, run through a
  read-only role inside a read-only transaction; hits carry ids and counts.

Incidents, notification sinks, the shadow T1 log, the kill switch and the
admin route come in slice S-D61b. Nothing here acts on a detection: the
watch is monotone (D-2) and starts in shadow mode (D-5).
"""

from __future__ import annotations

from umrs.security_watch.detectors import (
    DetectionResult,
    Hit,
    WriterRoleRefused,
    assert_read_only,
    build_query,
    run_pack,
    run_rule,
)
from umrs.security_watch.rules import (
    SITE_RULES_ENV,
    Rule,
    RulePack,
    RulePackError,
    load_rule_pack,
    site_rules_path,
)

__all__ = [
    "SITE_RULES_ENV",
    "DetectionResult",
    "Hit",
    "Rule",
    "RulePack",
    "RulePackError",
    "WriterRoleRefused",
    "assert_read_only",
    "build_query",
    "load_rule_pack",
    "run_pack",
    "run_rule",
    "site_rules_path",
]
