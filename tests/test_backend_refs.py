"""U3 S1 item 3: one public lenient backend-ref parser in core.

The research-loop, frontend-session and companion-worker backends borrowed
the private ``reindex._parse_ref``; plan_run_records kept its own copy. They
now share ``umrs.jobs.backend_refs.parse_backend_ref``.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from umrs.jobs import plan_run_records, reindex
from umrs.jobs.backend_refs import parse_backend_ref

SRC = Path(__file__).resolve().parents[1] / "src" / "umrs" / "jobs"
BORROWERS = (
    "research_loop_backend.py",
    "frontend_session_backend.py",
    "companion_worker_backend.py",
)


@pytest.mark.parametrize(
    ("ref", "expected"),
    [
        ("research_loop_job|loop_id=L1|driver=slurm", {"loop_id": "L1", "driver": "slurm"}),
        ("kind|a=1|noise|b=x=y", {"a": "1", "b": "x=y"}),
        ("", {}),
        ("kind", {}),
    ],
)
def test_parse_backend_ref(ref, expected):
    assert parse_backend_ref(ref) == expected


def test_old_private_names_are_the_core_parser():
    assert reindex._parse_ref is parse_backend_ref
    assert plan_run_records._parse_ref is parse_backend_ref


@pytest.mark.parametrize("name", BORROWERS)
def test_backends_no_longer_borrow_reindex_internals(name):
    tree = ast.parse((SRC / name).read_text(encoding="utf-8"))
    borrowed = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.ImportFrom) and (n.module or "").endswith("reindex")
    ]
    assert not borrowed, name
