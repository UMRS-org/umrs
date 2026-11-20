"""U3 S1 item 2: one core parser for ``cluster:<site>`` venues.

The job store, the loop start route and the HPC user-work check each tested
the ``cluster:`` prefix by hand, and the store borrowed the parser from the
research-loop backend (r4 loop hosting). They now share umrs.jobs.venue_names.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from umrs.jobs import research_loop_backend
from umrs.jobs.venue_names import CLUSTER_VENUE_PREFIX, is_cluster_venue, split_cluster_venue_name

SRC = Path(__file__).resolve().parents[1] / "src" / "umrs"
CONSUMERS = ("jobs/store.py", "api/loop.py", "shared/hpc_user_work_access.py")


@pytest.mark.parametrize(
    ("venue", "expected"),
    [
        ("cluster:rccs-front", True),
        ("  cluster:x", True),
        ("frontend", False),
        ("", False),
        (None, False),
    ],
)
def test_is_cluster_venue(venue, expected):
    assert is_cluster_venue(venue) is expected


def test_split_keeps_the_site_name_byte_exact():
    assert split_cluster_venue_name("cluster:rccs-front") == "rccs-front"
    assert split_cluster_venue_name("cluster: bdrt ") == "bdrt"


@pytest.mark.parametrize(
    "venue", ["cluster:", "cluster:  ", "cluster:a/b", "cluster:RCCS", "cluster:-x"]
)
def test_split_refuses_ambiguous_names(venue):
    with pytest.raises(ValueError, match="cluster"):
        split_cluster_venue_name(venue)


def test_the_backend_reexports_the_core_parser():
    assert research_loop_backend.CLUSTER_VENUE_PATTERN == CLUSTER_VENUE_PREFIX == "cluster:"
    assert research_loop_backend.split_cluster_venue_name is split_cluster_venue_name


def _prefix_literals(tree: ast.AST) -> list[str]:
    return [
        n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and n.value == "cluster:"
    ]


def _borrows_backend_parser(node: ast.AST) -> bool:
    if not isinstance(node, ast.ImportFrom) or not (node.module or "").endswith(
        "research_loop_backend"
    ):
        return False
    return any(a.name == "split_cluster_venue_name" for a in node.names)


@pytest.mark.parametrize("rel", CONSUMERS)
def test_consumers_use_the_core_parser(rel):
    text = (SRC / rel).read_text(encoding="utf-8")
    tree = ast.parse(text)
    assert not _prefix_literals(tree), f"{rel} tests the 'cluster:' prefix by hand"
    assert not any(_borrows_backend_parser(n) for n in ast.walk(tree)), rel
    assert "is_cluster_venue" in text, f"{rel} does not use the core helper"
