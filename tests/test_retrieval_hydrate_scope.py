"""A chunk's labels and object come from its own project's artifact only.

``content.artifact_id`` is node data, and a node can be imported under a
chosen id (bundle import, the legacy canonizer). Hydration joins the artifact
for its trusted label (``source_name``) and its object (``source_ref``, the
md5-verified slice). Joined on the id alone, a planted node of one project
borrows another project's file name and object (review security-3, the
hydrate leg, 2026-10-10). The backfill reads the same join.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from umrs.knowledge.retrieval import backfill
from umrs.knowledge.retrieval import hydrate as hydrate_mod
from umrs.knowledge.retrieval.types import Candidate

_PROJECT_PREDICATE = "a.project_id = n.project_id"

_NODES = {
    "node-chunk-planted-0": {
        "project_id": "p-attacker",
        "graph_id": "",
        "content": {"kind": "chunk", "artifact_id": "art-victim", "excerpt": "x y z"},
    },
    "node-chunk-own-0": {
        "project_id": "p-victim",
        "graph_id": "",
        "content": {"kind": "chunk", "artifact_id": "art-victim", "excerpt": "a b c"},
    },
}
_ARTIFACTS = {
    "art-victim": {
        "project_id": "p-victim",
        "source_ref": "s3://victim/secret-plan.pdf",
        "source_name": "secret-plan.pdf",
        "title": "Secret plan",
    }
}


class _Pool:
    """Evaluates hydrate's node read, honouring the artifact join as written:
    with the project predicate in the SQL only a same-project artifact joins."""

    def __init__(self) -> None:
        self.rows: list[tuple] = []

    def getconn(self) -> _Pool:
        return self

    def putconn(self, _conn: Any) -> None:
        return None

    def __enter__(self) -> _Pool:
        return self

    def __exit__(self, *_exc: Any) -> bool:
        return False

    def cursor(self) -> _Pool:
        return self

    def execute(self, sql: str, params: tuple = ()) -> None:
        self.rows = []
        if "LEFT JOIN knowledge_artifacts" in sql:
            scoped = _PROJECT_PREDICATE in sql
            self.rows = [self._row(nid, scoped) for nid in params[0] if nid in _NODES]

    def _row(self, node_id: str, scoped: bool) -> tuple:
        node = _NODES[node_id]
        art = _ARTIFACTS.get(node["content"]["artifact_id"], {})
        if scoped and art.get("project_id") != node["project_id"]:
            art = {}
        content = json.dumps(node["content"])
        return (
            node_id,
            node["graph_id"],
            content,
            art.get("source_ref"),
            art.get("source_name"),
            None,
            art.get("title"),
        )

    def fetchall(self) -> list[tuple]:
        return list(self.rows)


@pytest.fixture(autouse=True)
def _no_column(monkeypatch):
    monkeypatch.setattr(hydrate_mod, "_SEARCH_TEXT", False)


def _hydrated(node_id: str) -> Candidate:
    candidate = Candidate(node_id=node_id)
    hydrate_mod.hydrate(_Pool(), [candidate], lambda _ref: b"")
    return candidate


def test_a_planted_node_never_borrows_another_projects_label_or_object():
    planted = _hydrated("node-chunk-planted-0")
    assert planted.source_name != "secret-plan.pdf"
    assert planted.source_ref == ""
    assert planted.source_name == "art-victim"  # the id fallback, no metadata
    assert planted.text == "x y z" and planted.text_source == "excerpt"


def test_the_owning_projects_node_keeps_its_trusted_label():
    own = _hydrated("node-chunk-own-0")
    assert own.source_name == "secret-plan.pdf"
    assert own.source_ref == "s3://victim/secret-plan.pdf"


def test_the_backfill_reads_only_its_projects_artifacts():
    """The backfill's slice source is the same join (never another project's
    object): its row read carries the predicate."""
    assert _PROJECT_PREDICATE in backfill._ROWS_SQL
    assert _PROJECT_PREDICATE in hydrate_mod._NODE_SQL
