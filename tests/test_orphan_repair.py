"""``python -m umrs.knowledge.orphan_repair`` (ADR-093 D-11 / MD-17, wave F).

The orchestration (dry run by default, one transaction on --apply, a second
run finds nothing, a failure rolls every class back) is checked over a fake
cursor that classifies nodes the way the SQL says; the SQL itself is pinned
by its guards and, with ``UMRS_TEST_DB_URL``, run on a real Postgres inside a
rolled-back transaction.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

import pytest
from umrs.knowledge import orphan_repair as repair
from umrs.knowledge.graph_registry import default_graph_id

P = "p"


class _Db:
    def __init__(self) -> None:
        # node_id -> (kind, node_type, origin, artifact_id, has part_of edge)
        self.nodes: dict[str, tuple[str, str, str | None, str | None, bool]] = {}
        self.artifacts: set[str] = set()
        self.stale: set[str] = set()
        self.commits = 0
        self.fail_on_delete = False

    def member(self, name: str, node: tuple) -> bool:
        kind, node_type, origin, artifact, edge = node
        if artifact is None:
            return False  # only nodes carrying content.artifact_id
        alive = artifact in self.artifacts
        return {
            "rowless_chunks": kind == "chunk" and not alive,
            "rowless_files": node_type == "file" and origin is None and not alive,
            "edgeless_chunks": kind == "chunk"
            and origin == "knowledge_embed"
            and alive
            and not edge,
        }[name]


class _Cur:
    def __init__(self, db: _Db, staged: dict) -> None:
        self.db, self.staged = db, staged
        self.rows: list[tuple] = []
        self.rowcount = 0

    def __enter__(self) -> _Cur:
        return self

    def __exit__(self, *_a: Any) -> bool:
        return False

    def execute(self, sql: str, params: Any = ()) -> None:
        tag = re.search(r"/\* orphan_repair:(\w+) \*/", sql)
        name = tag.group(1) if tag else ""
        if name in repair.CLASSES:
            self._select(name)
        elif name == "delete":
            self._delete(params[1])
        elif sql.startswith("UPDATE knowledge_artifacts"):
            self.staged["stale"].update(params[2])
            self.rowcount = len(params[2])

    def _select(self, name: str) -> None:
        nodes = sorted(self.db.nodes.items())
        self.rows = [(nid, node[3]) for nid, node in nodes if self.db.member(name, node)]

    def _delete(self, node_ids: list[str]) -> None:
        if self.db.fail_on_delete:
            raise RuntimeError("lock timeout")
        gone = [nid for nid in node_ids if nid in self.db.nodes]
        self.staged["delete"].update(gone)
        self.rowcount = len(gone)

    def fetchall(self) -> list[tuple]:
        return self.rows

    def fetchone(self) -> tuple | None:
        return None


class _Conn:
    def __init__(self, db: _Db) -> None:
        self.db = db
        self.staged: dict[str, set] = {"delete": set(), "stale": set()}

    def __enter__(self) -> _Conn:
        return self

    def __exit__(self, exc_type: Any, *_a: Any) -> bool:
        if exc_type is None:  # psycopg2: commit on success, roll back on error
            for nid in self.staged["delete"]:
                del self.db.nodes[nid]
            self.db.stale |= self.staged["stale"]
            self.db.commits += 1
        return False

    def cursor(self) -> _Cur:
        return _Cur(self.db, self.staged)


class _Pool:
    def __init__(self, db: _Db) -> None:
        self.db = db

    def getconn(self) -> _Conn:
        return _Conn(self.db)

    def putconn(self, _conn: _Conn) -> None:
        return None


def _stack() -> _Db:
    """The stand-alone stack's shape: 3 rowless embed chunks, live chunks,
    rag.upload and legacy chunks keyed by document_id, an edgeless chunk."""
    db = _Db()
    db.artifacts |= {"art-live", "art-edgeless", "artifact-rag-d1"}
    embed = ("chunk", "entity", "knowledge_embed")
    for art in ("art-3de698e03d87", "art-d7a806cc4b75", "art-f262129bf8a4"):
        db.nodes[f"node-chunk-{art}-0"] = (*embed, art, False)
    db.nodes["node-chunk-art-live-0"] = (*embed, "art-live", True)
    db.nodes["node-file-art-live"] = ("", "file", None, "art-live", False)
    db.nodes["node-chunk-art-edgeless-0"] = (*embed, "art-edgeless", False)
    db.nodes["node-file-art-gone"] = ("", "file", None, "art-gone", False)
    db.nodes["ragdoc-d1"] = ("document", "file", "rag.upload", "artifact-rag-d1", False)
    db.nodes["ragdoc-d2"] = ("document", "file", "legacy", "artifact-legacy-d2", False)
    db.nodes["ragchunk-d1-0"] = ("chunk", "entity", "rag.upload", None, True)
    db.nodes["legacy-chunk-d2-0"] = ("chunk", "entity", "legacy", None, False)
    return db


def test_a_dry_run_counts_each_class_and_writes_nothing():
    db = _stack()
    before = dict(db.nodes)
    report = repair.run_repair(_Pool(db), P)
    assert report.found == {"rowless_chunks": 3, "rowless_files": 1, "edgeless_chunks": 1}
    assert report.samples["rowless_chunks"] == [
        "node-chunk-art-3de698e03d87-0",
        "node-chunk-art-d7a806cc4b75-0",
        "node-chunk-art-f262129bf8a4-0",
    ]
    assert not report.applied and report.deleted == {}
    assert db.nodes == before and db.stale == set()


def test_apply_deletes_only_rowless_and_edgeless_nodes_and_marks_stale():
    db = _stack()
    report = repair.run_repair(_Pool(db), P, apply=True)
    assert report.deleted == {"rowless_chunks": 3, "rowless_files": 1, "edgeless_chunks": 1}
    assert sorted(db.nodes) == [
        "legacy-chunk-d2-0",
        "node-chunk-art-live-0",
        "node-file-art-live",
        "ragchunk-d1-0",
        "ragdoc-d1",
        "ragdoc-d2",  # a writer's file node with an origin is never touched
    ]
    assert db.stale == {"art-edgeless"} and report.artifacts_marked_stale == 1
    assert db.commits == 1


def test_a_second_run_finds_nothing():
    db = _stack()
    repair.run_repair(_Pool(db), P, apply=True)
    again = repair.run_repair(_Pool(db), P, apply=True)
    assert set(again.found.values()) == {0} and set(again.deleted.values()) == {0}


def test_a_failure_rolls_every_class_back():
    db = _stack()
    db.fail_on_delete = True
    before = dict(db.nodes)
    with pytest.raises(RuntimeError, match="lock timeout"):
        repair.run_repair(_Pool(db), P, apply=True)
    assert db.nodes == before and db.stale == set() and db.commits == 0


def test_the_cli_is_a_dry_run_unless_told_otherwise(monkeypatch, capsys):
    db = _stack()
    monkeypatch.setattr(repair, "_build_pool", lambda: _ClosingPool(db))
    assert repair.main(["--project", P]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["applied"] is False and out["found"]["rowless_chunks"] == 3
    assert len(db.nodes) == 11  # nothing deleted


class _ClosingPool(_Pool):
    def closeall(self) -> None:
        return None


# ── the SQL's guards ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", list(repair.CLASSES))
def test_every_class_holds_only_nodes_carrying_their_artifact_id(name):
    assert "n.content ? 'artifact_id'" in repair.CLASSES[name]


def test_the_rowless_file_class_spares_writers_with_an_origin():
    assert "(n.content->>'origin') IS NULL" in repair.CLASSES["rowless_files"]


def test_the_repair_reads_the_expression_migration_0122_indexes():
    from tests.test_migration_0122 import ARTIFACT_ID_EXPR

    assert repair.ARTIFACT_ID_EXPR == ARTIFACT_ID_EXPR
    assert all(f"n.{ARTIFACT_ID_EXPR}" in sql for sql in repair.CLASSES.values())


def test_a_graph_scope_keeps_the_legacy_default_graph_rule():
    sql, params = repair.scope(P, default_graph_id(P))
    assert "n.graph_id = '' AND %s" in sql and params == [P, default_graph_id(P), True]


# ── integration: a real Postgres (skipped without a database URL) ────────────


@pytest.fixture()
def pg_pool(request: pytest.FixtureRequest) -> Any:
    psycopg2 = pytest.importorskip("psycopg2")
    url = request.config.getoption("--db-url") or os.environ.get("UMRS_TEST_DB_URL")
    if not url:
        pytest.skip("no database URL (--db-url / UMRS_TEST_DB_URL)")
    conn = psycopg2.connect(url)

    class _OneConn:
        def getconn(self) -> Any:
            return _NoCommit(conn)

        def putconn(self, _c: Any) -> None:
            return None

    try:
        yield conn, _OneConn()
    finally:
        conn.rollback()
        conn.close()


class _NoCommit:
    """The test's own transaction: the repair's ``with conn`` never commits."""

    def __init__(self, conn: Any) -> None:
        self.conn = conn

    def __enter__(self) -> Any:
        return self.conn

    def __exit__(self, *_a: Any) -> bool:
        return False

    def cursor(self) -> Any:
        return self.conn.cursor()


def _node(cur: Any, node_id: str, node_type: str, content: dict) -> None:
    cur.execute(
        "INSERT INTO knowledge_nodes (node_id, project_id, label, node_type, content, "
        "content_hash) VALUES (%s, 'orph-it', %s, %s, %s::jsonb, 'h')",
        (node_id, node_id, node_type, json.dumps(content)),
    )


@pytest.mark.integration
def test_the_sql_deletes_the_orphans_and_spares_document_keyed_chunks(pg_pool: Any) -> None:
    conn, pool = pg_pool
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO knowledge_artifacts (artifact_id, content_hash, kind, project_id, "
            "status) VALUES ('orph-live', 'h-orph-live', 'document', 'orph-it', 'active')"
        )
        embed = {"kind": "chunk", "origin": "knowledge_embed"}
        _node(cur, "orph-chunk-gone", "entity", {**embed, "artifact_id": "orph-gone"})
        _node(cur, "orph-chunk-live", "entity", {**embed, "artifact_id": "orph-live"})
        _node(cur, "orph-file-live", "file", {"artifact_id": "orph-live"})
        _node(
            cur, "orph-rag", "entity", {"kind": "chunk", "origin": "rag.upload", "document_id": "d"}
        )
        cur.execute(
            "INSERT INTO knowledge_edges (edge_id, project_id, source_id, target_id, "
            "relation_type) VALUES ('orph-e', 'orph-it', 'orph-chunk-live', 'orph-file-live', 'part_of')"
        )
    report = repair.run_repair(pool, "orph-it", apply=True)
    assert report.deleted == {"rowless_chunks": 1, "rowless_files": 0, "edgeless_chunks": 0}
    with conn.cursor() as cur:
        cur.execute("SELECT node_id FROM knowledge_nodes WHERE project_id = 'orph-it' ORDER BY 1")
        assert [row[0] for row in cur.fetchall()] == [
            "orph-chunk-live",
            "orph-file-live",
            "orph-rag",
        ]
