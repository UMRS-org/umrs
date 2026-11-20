"""Jobs that UMRS records but never hosts itself (U3 S1, ADR-117).

A ``research_loop_job`` whose spec carries ``hosting: client`` or
``hosting: venue`` is a durable record of work that another party drives:
its client, or a frontend-venue worker, opens and closes its attempts. Core
scheduling code only needs to recognise such a spec and refuse to host it;
the plan-run record format itself is r4 extension material
(``umrs.jobs.plan_run_records``, moving to ``umrs_r4`` in U3 S5).

The values and the refusal text are unchanged from ``plan_run_records``
(S1 changes no behaviour); neutral wording follows in S6.
"""

from __future__ import annotations

from typing import Any

HOSTING_KEY = "hosting"
HOSTING_CLIENT = "client"
HOSTING_VENUE = "venue"
EXTERNAL_HOSTINGS = frozenset({HOSTING_CLIENT, HOSTING_VENUE})

HOSTED_ELSEWHERE_REFUSAL = (
    "client_hosted_run: this research_loop_job records a team plan run that "
    "its client (r4) hosts (ADR-092 slice 7); UMRS never hosts or requeues it. "
    "Resume it from r4 (r4 team plan resume <run_id>)."
)


def is_externally_hosted(spec: Any) -> bool:
    """True when ``spec`` records a job another party hosts (client or venue)."""
    return isinstance(spec, dict) and spec.get(HOSTING_KEY) in EXTERNAL_HOSTINGS


def is_venue_hosted(spec: Any) -> bool:
    """True when a frontend-venue worker hosts the recorded job."""
    return isinstance(spec, dict) and spec.get(HOSTING_KEY) == HOSTING_VENUE


__all__ = [
    "EXTERNAL_HOSTINGS",
    "HOSTED_ELSEWHERE_REFUSAL",
    "HOSTING_CLIENT",
    "HOSTING_KEY",
    "HOSTING_VENUE",
    "is_externally_hosted",
    "is_venue_hosted",
]
