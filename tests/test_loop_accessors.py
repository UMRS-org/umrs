"""U3 S1 item 7: loop accessors are public core API.

Assets, the agent brief, research sessions, the companion routes and the
plan-run venue reached into ``api.loop``'s private ``_loop_store``,
``_LoopCaller``/``_may_access`` and ``_project_budget_caps``, and plan
iteration into ``engine_loop_runner._elapsed_hours``. They now use
``get_loop_store``, ``loop_access``, ``project_budget_caps`` and
``elapsed_hours``.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from umrs.api import loop
from umrs.knowledge import engine_loop_runner

SRC = Path(__file__).resolve().parents[1] / "src" / "umrs"
R4_SRC = Path(__file__).resolve().parents[1] / "packages" / "umrs-r4" / "src" / "umrs_r4"  # callers moved by U3 S5
CALLERS = (
    "api/assets.py",
    "api/agent_brief.py",
    "api/research_sessions.py",
    "api/companion.py",
    "api/plan_run_venue.py",
    "knowledge/plan_iteration.py",
)
PRIVATE = {"_loop_store", "_LoopCaller", "_may_access", "_project_budget_caps", "_elapsed_hours"}


def test_get_loop_store_reads_the_current_store(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(loop, "_loop_store", sentinel)
    assert loop.get_loop_store() is sentinel


@pytest.mark.parametrize(
    ("owner", "user", "admin", "allowed"),
    [
        ("", "u", False, True),
        ("o", "", False, True),
        ("o", "u", True, True),
        ("o", "o", False, True),
        ("o", "u", False, False),
    ],
)
def test_loop_access_rule(owner, user, admin, allowed):
    assert loop.loop_access(owner, user_id=user, is_admin=admin) is allowed


def test_private_aliases_stay_the_same_objects():
    assert engine_loop_runner._elapsed_hours is engine_loop_runner.elapsed_hours


def _private_uses(rel: str) -> set[str]:
    path = SRC / rel if (SRC / rel).exists() else R4_SRC / rel
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names}
    attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    return (names | attrs) & PRIVATE


@pytest.mark.parametrize("rel", CALLERS)
def test_callers_use_the_public_accessors(rel):
    assert not _private_uses(rel), rel


def test_always_decision_flips_the_loop_through_the_current_store(monkeypatch):
    """The flip runs inside a broad ``except``: a stale store name would only
    have logged a warning, never failed, so the save is checked here."""
    from types import SimpleNamespace

    from umrs.api import approvals

    saved = []
    record = SimpleNamespace(metadata={"mode": "manual"})
    store = SimpleNamespace(get=lambda _id: record, save=saved.append)
    monkeypatch.setattr(loop, "_loop_store", store)
    approvals._flip_loop_to_auto("L1")
    assert saved == [record] and record.metadata["mode"] == "auto"
