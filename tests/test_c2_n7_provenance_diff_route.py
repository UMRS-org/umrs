"""Cycle-2 follow-up N7 (2026-10-21): GET /umrs/provenance/diff was
unreachable — GET /umrs/provenance/{job_id}, declared first, matched "diff"
as a job id and answered "Job not found: diff".

(N8 — workflow executions answering 403 for an inaccessible id — was kept as
is on purpose: HTTP and ZMQ pin the same `workflow_access_denied` code
(test_workflow_isolation parity), and execution ids are random UUIDs.)
"""

from __future__ import annotations

from umrs.api import provenance as provenance_api


def _route_index(path: str) -> int:
    paths = [getattr(route, "path", "") for route in provenance_api.router.routes]
    return paths.index(f"/umrs/provenance{path}")


def test_diff_is_declared_before_the_job_id_route() -> None:
    assert _route_index("/diff") < _route_index("/{job_id}")
