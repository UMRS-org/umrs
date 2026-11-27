"""Skip-until-implemented for lane A105-S6a (the integrator deletes this file).

The lane landed its seam with stub modules that raise
``NotImplementedError("a105-s6a-0N")``. A test that needs one marks itself
``@pending("a105-s6a-0N", ...)``: it is skipped while any named stub still
raises, and runs once every one is implemented. A worker's check sets
``A105_S6A_REQUIRE=<its task id>`` so its own tests never skip.

Integration: delete this file and every ``@pending(...)`` line.
"""

from __future__ import annotations

import os
from collections.abc import Callable

import pytest


def _probe_01() -> object:
    from umrs.jobs.read_query import JobReadFilters, where_clause

    return where_clause(JobReadFilters())


def _probe_02() -> object:
    from umrs.api.read_cursor import scope_key

    return scope_key({})


def _probe_03() -> object:
    from umrs.api.read_fields import project

    return project({}, ())


def _probe_04() -> object:
    from umrs.jobs.read_summary import fold_summary

    return fold_summary([])


def _probe_05() -> object:
    from umrs.api.workflow_execution_list import page_executions

    return page_executions([], after=None, limit=1)


_PROBES: dict[str, Callable[[], object]] = {
    "a105-s6a-01": _probe_01,
    "a105-s6a-02": _probe_02,
    "a105-s6a-03": _probe_03,
    "a105-s6a-04": _probe_04,
    "a105-s6a-05": _probe_05,
}

ALL_TASKS = tuple(_PROBES)


def _stubbed(task_id: str) -> bool:
    try:
        _PROBES[task_id]()
    except NotImplementedError:
        return True
    except Exception:  # implemented (a wrong answer is the tests' business)
        return False
    return False


def pending(*task_ids: str) -> pytest.MarkDecorator:
    """Skip while a named stub is unimplemented, unless required."""
    required = os.environ.get("A105_S6A_REQUIRE", "")
    waiting = [t for t in task_ids if t != required and required != "all" and _stubbed(t)]
    return pytest.mark.skipif(
        bool(waiting), reason=f"lane A105-S6a stub pending: {', '.join(waiting)}"
    )
