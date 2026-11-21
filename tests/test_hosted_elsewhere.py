"""U3 S1 item 1: the "hosted elsewhere" markers live in core.

Core scheduling code (the research-loop backend, the agent quota) needs only
to recognise a job another party hosts; the plan-run record format is r4
extension material that moves to ``umrs_r4`` in U3 S5. S1 changes no
behaviour: the values and the refusal text are the ones plan_run_records
always had.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from umrs.jobs import hosted_elsewhere as core
from umrs_r4.jobs import plan_run_records as ext

SRC = Path(__file__).resolve().parents[1] / "src" / "umrs"
CORE_CONSUMERS = ("jobs/research_loop_backend.py", "api/agent_quota.py")


def test_plan_run_records_reexports_the_core_values():
    assert ext.HOSTING_KEY == core.HOSTING_KEY == "hosting"
    assert (ext.HOSTING_CLIENT, ext.HOSTING_VENUE) == ("client", "venue")
    assert ext.PLAN_RUN_HOSTINGS == core.EXTERNAL_HOSTINGS == frozenset({"client", "venue"})
    assert ext.CLIENT_HOSTED_REFUSAL == core.HOSTED_ELSEWHERE_REFUSAL
    assert core.HOSTED_ELSEWHERE_REFUSAL.startswith("client_hosted_run:")


@pytest.mark.parametrize(
    ("spec", "external", "venue"),
    [
        ({"hosting": "client"}, True, False),
        ({"hosting": "venue"}, True, True),
        ({"hosting": "server"}, False, False),
        ({}, False, False),
        (None, False, False),
        ("hosting", False, False),
    ],
)
def test_externally_hosted_truth_table(spec, external, venue):
    assert core.is_externally_hosted(spec) is external
    assert ext.is_plan_run_record(spec) is external
    assert core.is_venue_hosted(spec) is venue


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
    return found


@pytest.mark.parametrize("rel", CORE_CONSUMERS)
def test_core_consumers_no_longer_import_plan_run_records(rel):
    modules = _imported_modules(SRC / rel)
    assert not {m for m in modules if m.endswith("plan_run_records")}, rel
