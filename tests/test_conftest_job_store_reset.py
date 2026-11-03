"""conftest clears ``umrs.api.jobs._job_store`` after every test (2026-11-03).

Booting the real app (test_github_issue_87_features) set the module-global job
store and left it; a later test on the same worker then turned its workflow
trigger into a durable job no scheduler ran (test_workflow_gate_api timeouts
under pytest-xdist). The pair below fails without the conftest reset when run
in file order, as pytest does on one worker.
"""

import umrs.api.jobs as jobs

_LEFT_BEHIND = object()


def test_a_test_may_leave_a_job_store_behind():
    jobs._job_store = _LEFT_BEHIND  # what booting the real app does


def test_the_next_test_starts_without_it():
    assert jobs._job_store is None
